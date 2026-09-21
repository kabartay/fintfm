"""Regression by de-binning a classifier's distribution over quantile bins (task 46.3).

Why this is a wrapper and not a model
-------------------------------------

:mod:`fintfm.inference.binning` argues the case in full: a continuous target cut into ``K``
quantile bins *is* an integer index over ``K`` outcomes, so the existing head, loss and
per-cell label injection already do the work. What was missing was the object at the boundary
that closes the loop — bin the target on fit, run the classifier, turn the predicted
distribution back into a number on predict. That is this file, and it needs no retraining of
anything: any checkpoint with ``max_classes >= 2`` can regress today.

What it buys, concretely
------------------------

TabArena's 51 datasets are 30 binary, 13 regression and 8 multiclass; 46 of them fall inside
the 136-feature cap (27 / 12 / 7). Declaring ``regression`` takes the suite from 34/51 to
**46/51 (90%)**, which is the difference between a score that carries a coverage caveat in
every sentence and one that does not (``docs/TABARENA.md``).

What it is not
--------------

**Not a calibrated conditional density.** The output is a discrete distribution over at most
``K`` representatives, so its resolution is bounded by ``K`` no matter how confident the model
is — an 80% interval cannot be narrower than a bin. Task 46.5's coverage check
(:func:`fintfm.inference.binning.interval_coverage`) exists to measure that rather than assume
it, and a coverage number from it must be read against ``K``, not in isolation.

**Not a claim that regression works.** Nothing here has been scored on a real regression
panel yet. This makes the measurement possible; it does not anticipate it.

The distributional output is the point, though
----------------------------------------------

:meth:`FinancialTFMRegressor.predict` returns the mean, because sklearn's contract demands a
point estimate, and it is the *least* interesting thing this produces.
:meth:`predict_proba`, :meth:`predict_quantile` and :meth:`predict_interval` expose the full
distribution — multimodal where the target is multimodal, which is what loss given default
actually looks like and what a Gaussian head cannot represent at all.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from fintfm.inference.binning import DEFAULT_N_BINS, QuantileBinner
from fintfm.inference.classifier import FinancialTFMClassifier
from fintfm.modeling.model import FinancialTFM


class FinancialTFMRegressor(BaseEstimator, RegressorMixin):
    """In-context tabular regressor: quantile-bin the target, classify, de-bin.

    Args:
        model: A pretrained :class:`FinancialTFM` or a path to a checkpoint.
        n_bins: Bins the target is cut into. Capped at the checkpoint's ``max_classes``,
            because the head has exactly that many logits — passing more is a silent
            truncation otherwise, and this raises the cap into view instead. More bins means
            finer resolution and fewer rows per bin; ``DEFAULT_N_BINS`` is a starting point,
            not a measured optimum.
        **classifier_kwargs: Forwarded verbatim to :class:`FinancialTFMClassifier`
            (``device``, ``max_context``, ``feature_transform``, ``n_ensemble``, ...), so
            every context and conditioning choice measured for classification applies here
            unchanged.

    Attributes:
        binner_: The fitted :class:`QuantileBinner`. Its ``representatives_`` are the only
            values :meth:`predict` can ever return a weighted average of.
        classifier_: The underlying :class:`FinancialTFMClassifier`.
    """

    def __init__(
        self,
        model: FinancialTFM | str,
        n_bins: int = DEFAULT_N_BINS,
        **classifier_kwargs: object,
    ) -> None:
        self.model = model
        self.n_bins = int(n_bins)
        self.classifier_kwargs = classifier_kwargs
        self.binner_: QuantileBinner | None = None
        self.classifier_: FinancialTFMClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> FinancialTFMRegressor:
        """Bin ``y`` on the training rows and fit the classifier on the bin indices.

        Args:
            X: ``(n, n_features)`` training features.
            y: ``(n,)`` continuous training targets.

        Returns:
            self.

        Raises:
            ValueError: If ``n_bins`` exceeds the checkpoint's ``max_classes``.
        """
        clf = FinancialTFMClassifier(self.model, **self.classifier_kwargs)  # type: ignore[arg-type]
        cap = clf.model.cfg.max_classes
        if self.n_bins > cap:
            raise ValueError(
                f"n_bins={self.n_bins} exceeds the checkpoint's max_classes={cap}; "
                "the head has one logit per bin, so the extra bins have nowhere to go"
            )
        y = np.asarray(y, dtype=np.float64).ravel()
        # Fitted on the training targets only. The query rows' targets are what is being
        # predicted, so letting them set the edges would leak the answer into the grid the
        # answer is read off -- the same reason `prior/base.py` bins on context rows alone.
        self.binner_ = QuantileBinner(self.n_bins).fit(y)
        self.classifier_ = clf.fit(X, self.binner_.transform(y))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicted distribution over bins.

        Args:
            X: ``(n, n_features)`` query features.

        Returns:
            ``(n, binner_.n_bins_)`` probabilities, one column per bin **in bin order**,
            with zeros in bins that held no training row. The classifier only ever emits
            columns for the labels it saw, so this widens its output back to the full grid —
            without which :meth:`expected_value` would silently pair probabilities with the
            wrong representatives.
        """
        self._check_fitted()
        assert self.classifier_ is not None and self.binner_ is not None
        proba = self.classifier_.predict_proba(X)
        full = np.zeros((proba.shape[0], self.binner_.n_bins_), dtype=np.float64)
        full[:, self.classifier_.classes_.astype(int)] = proba
        return full

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point estimate: the predicted distribution's mean.

        Args:
            X: ``(n, n_features)`` query features.

        Returns:
            ``(n,)`` predictions, each a convex combination of the bin representatives and
            therefore inside the training target's range by construction. That is a real
            limitation, not a safety feature: this cannot extrapolate beyond the targets it
            was fitted on, and on a trending series it will under-predict the trend.
        """
        self._check_fitted()
        assert self.binner_ is not None
        return self.binner_.expected_value(self.predict_proba(X))

    def predict_quantile(self, X: np.ndarray, q: float) -> np.ndarray:
        """The predicted distribution's ``q``-quantile.

        Args:
            X: ``(n, n_features)`` query features.
            q: Quantile in ``(0, 1)``.

        Returns:
            ``(n,)`` quantile estimates.
        """
        self._check_fitted()
        assert self.binner_ is not None
        return self.binner_.quantile(self.predict_proba(X), q)

    def predict_interval(
        self, X: np.ndarray, level: float = 0.8
    ) -> tuple[np.ndarray, np.ndarray]:
        """Central prediction interval.

        Args:
            X: ``(n, n_features)`` query features.
            level: Nominal coverage, e.g. 0.8.

        Returns:
            ``(lower, upper)``, each ``(n,)``. Both are bin representatives, so the interval
            is quantised at the bin grid and cannot be narrower than one bin — check it with
            :func:`fintfm.inference.binning.interval_coverage` rather than trusting it.
        """
        self._check_fitted()
        assert self.binner_ is not None
        return self.binner_.interval(self.predict_proba(X), level)

    def _check_fitted(self) -> None:
        if self.classifier_ is None or self.binner_ is None:
            raise RuntimeError("fit must be called before predicting")
