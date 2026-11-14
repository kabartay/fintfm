"""Regression by asking binary questions: is the target above each of K thresholds?

:class:`~fintfm.inference.regressor.FinancialTFMRegressor` cuts the target into at most
``max_classes`` quantile bins (10 for every multiclass checkpoint here), so the network never
sees the difference between two rows in the same decile, and its resolution is fixed by the
head. Partitioned contexts (``PartitionedTFMRegressor``, FINDINGS §231) bought regression Elo
by giving each leaf a finer grid of its own, which says the grid is a binding limit.

This estimator removes the head's cap instead. It cuts the target at ``K`` quantile
thresholds and fits one binary :class:`~fintfm.inference.classifier.FinancialTFMClassifier`
per threshold on the label ``y > t_k``. Each is the task the binary checkpoints were trained
on, and the one where FinTFM ranks best on TabArena-Lite (997 Elo, §232). Together they give
the survival function ``P(y > t_k | x)`` at ``K`` points, so the resolution is ``K`` rather
than ``max_classes``, and any checkpoint with ``max_classes >= 2`` serves.

The point prediction is the mean of the implied step distribution: the probability of each
interval between consecutive thresholds times the mean training target in that interval. The
survival probabilities are made non-increasing in ``k`` first, since ``K`` separately fitted
classifiers need not agree. :meth:`ThresholdTFMRegressor.readout_features` returns their logits
for a fitted readout (the TabArena fork's ridge probe reads them beside the raw features).

Cost: ``K`` classifier fits and ``K`` forward passes per prediction, against one for the binned
regressor. Like the binned regressor, it cannot predict outside the training targets' range.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from fintfm.inference.classifier import FinancialTFMClassifier
from fintfm.modeling.model import FinancialTFM

#: Thresholds when none are asked for. A starting point for the first measurement, not a
#: tuned value; 15 gives 16 intervals, a finer grid than any 10-class head.
DEFAULT_N_THRESHOLDS = 15


class ThresholdTFMRegressor(BaseEstimator, RegressorMixin):
    """In-context regressor from ``K`` binary classifiers on ``y > t_k``.

    Args:
        model: A pretrained :class:`FinancialTFM` or a path to a checkpoint. Loaded once and
            shared by every threshold's classifier.
        n_thresholds: Thresholds asked for, at the training targets' quantiles
            ``k / (K + 1)``. Fewer are used when the target has ties (duplicate quantiles are
            dropped, and so is any threshold with no training row above it).
        classifier_params: Keyword arguments forwarded verbatim to every
            :class:`FinancialTFMClassifier` (``device``, ``max_context``, ``feature_transform``,
            ``random_state``, ...). ``None`` means the classifier's defaults.

    Attributes:
        thresholds_: ``(K,)`` increasing thresholds actually used, each a training target value.
        interval_means_: ``(K + 1,)`` mean training target in ``(-inf, t_0]``,
            ``(t_0, t_1]``, ..., ``(t_{K-1}, inf)``; every interval holds at least one row.
        classifiers_: The ``K`` fitted binary classifiers, in threshold order.
    """

    def __init__(
        self,
        model: FinancialTFM | str,
        n_thresholds: int = DEFAULT_N_THRESHOLDS,
        classifier_params: dict[str, object] | None = None,
    ) -> None:
        """Configure the regressor. See the class docstring for what each argument means."""
        self.model = model
        self.n_thresholds = n_thresholds
        self.classifier_params = classifier_params

    def fit(self, X: np.ndarray, y: np.ndarray) -> ThresholdTFMRegressor:
        """Choose the thresholds on ``y`` and fit one binary classifier per threshold.

        Args:
            X: ``(n, n_features)`` training features.
            y: ``(n,)`` continuous training targets.

        Returns:
            self.

        Raises:
            ValueError: If ``n_thresholds < 1``, if ``y`` holds a NaN or an infinity, or if
                ``y`` has a single distinct value (no threshold splits it).
        """
        if int(self.n_thresholds) < 1:
            raise ValueError(f"n_thresholds must be at least 1, got {self.n_thresholds}")
        y = np.asarray(y, dtype=np.float64).ravel()
        if not np.isfinite(y).all():
            raise ValueError(
                f"y must be finite; got {int(np.isnan(y).sum())} NaN and "
                f"{int(np.isinf(y).sum())} infinite value(s)"
            )
        k = int(self.n_thresholds)
        # "inverted_cdf" returns training values, so every interval (t_j, t_{j+1}] holds the
        # row at t_{j+1}; thresholds come from the training targets only, never the queries.
        cand = np.unique(np.quantile(y, np.arange(1, k + 1) / (k + 1), method="inverted_cdf"))
        cand = cand[cand < y.max()]
        if cand.size == 0:
            raise ValueError("y has a single distinct value; no threshold can split it")
        self.thresholds_ = cand
        interval = np.searchsorted(cand, y, side="left")
        self.interval_means_ = np.array([y[interval == j].mean() for j in range(cand.size + 1)])
        network = FinancialTFM.load(self.model) if isinstance(self.model, str) else self.model
        params = dict(self.classifier_params or {})
        self.classifiers_ = [
            FinancialTFMClassifier(network, **params).fit(X, (y > t).astype(np.int64))  # type: ignore[arg-type]
            for t in cand
        ]
        return self

    def predict_survival(self, X: np.ndarray) -> np.ndarray:
        """``P(y > t_k | x)`` for every threshold, made non-increasing in ``k``.

        Args:
            X: ``(n, n_features)`` query features.

        Returns:
            ``(n, K)`` probabilities. Column ``k`` is the classifier's probability of the
            label 1 at threshold ``k``, capped by every lower threshold's (a running minimum),
            so the implied interval probabilities are never negative.
        """
        self._check_fitted()
        cols = []
        for clf in self.classifiers_:
            proba = clf.predict_proba(X)
            classes = list(clf.classes_)
            cols.append(proba[:, classes.index(1)])
        return np.minimum.accumulate(np.column_stack(cols).astype(np.float64), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of each interval between consecutive thresholds.

        Args:
            X: ``(n, n_features)`` query features.

        Returns:
            ``(n, K + 1)`` non-negative probabilities summing to 1, in interval order.
        """
        s = self.predict_survival(X)
        ones = np.ones((s.shape[0], 1))
        return -np.diff(np.hstack([ones, s, np.zeros_like(ones)]), axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point estimate: the step distribution's mean.

        Args:
            X: ``(n, n_features)`` query features.

        Returns:
            ``(n,)`` predictions inside the training targets' range.
        """
        return self.predict_proba(X) @ self.interval_means_

    def readout_features(self, X: np.ndarray, clip: float = 1e-4) -> np.ndarray:
        """Logits of :meth:`predict_survival`, the input a fitted readout reads.

        Args:
            X: ``(n, n_features)`` query features.
            clip: Probabilities are clipped to ``[clip, 1 - clip]`` before the logit.

        Returns:
            ``(n, K)`` float64 logits.
        """
        s = np.clip(self.predict_survival(X), clip, 1.0 - clip)
        return np.log(s) - np.log1p(-s)

    def _check_fitted(self) -> None:
        """Raise unless :meth:`fit` has run.

        Raises:
            RuntimeError: If the regressor is unfitted.
        """
        if not hasattr(self, "classifiers_"):
            raise RuntimeError("ThresholdTFMRegressor is not fitted; call fit first")
