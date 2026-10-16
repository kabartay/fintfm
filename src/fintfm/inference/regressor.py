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
every sentence and one that does not (``docs/results/TABARENA.md``).

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

import warnings

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from fintfm.inference.binning import DEFAULT_N_BINS, QuantileBinner
from fintfm.inference.classifier import FinancialTFMClassifier
from fintfm.modeling.model import FinancialTFM


def _pretrained_with_permuted_classes(model: FinancialTFM | str) -> bool:
    """Whether a checkpoint's provenance says its prior permuted class labels.

    Read from the file, since :meth:`FinancialTFM.load` deliberately ignores the record
    (design review DR1.2). An in-memory model has no file, so it is read from a
    ``provenance`` attribute when a caller attached one (the mapping
    :func:`fintfm.provenance.read_checkpoint_provenance` returns).

    Args:
        model: A checkpoint path, or a model.

    Returns:
        ``PriorConfig.permute_classes`` as recorded, ``False`` when nothing is recorded.
    """
    from fintfm.provenance import read_checkpoint_provenance

    record = (
        read_checkpoint_provenance(model)
        if isinstance(model, str)
        else getattr(model, "provenance", None)
    ) or {}
    return bool((record.get("prior_config") or {}).get("permute_classes", False))


class FinancialTFMRegressor(BaseEstimator, RegressorMixin):
    """In-context tabular regressor: quantile-bin the target, classify, de-bin.

    Args:
        model: A pretrained :class:`FinancialTFM` or a path to a checkpoint.
        n_bins: Bins the target is cut into. Capped at the checkpoint's ``max_classes``,
            because the head has exactly that many logits — passing more is a silent
            truncation otherwise, and this raises the cap into view instead. More bins means
            finer resolution and fewer rows per bin; ``DEFAULT_N_BINS`` is a starting point,
            not a measured optimum.
        classifier_params: Keyword arguments forwarded verbatim to
            :class:`FinancialTFMClassifier` (``device``, ``max_context``,
            ``feature_transform``, ``n_ensemble``, ...), so every context and conditioning
            choice measured for classification applies here unchanged. ``None`` means the
            classifier's defaults. One explicit parameter rather than ``**kwargs`` so that
            ``get_params``, and with it ``sklearn.clone`` and ``GridSearchCV``, carries the
            settings (audit AU1.10).
        **classifier_kwargs: Deprecated spelling of ``classifier_params``: still accepted,
            with a ``DeprecationWarning``, and merged over it.

    Attributes:
        binner_: The fitted :class:`QuantileBinner`. Its ``representatives_`` are the only
            values :meth:`predict` can ever return a weighted average of.
        classifier_: The underlying :class:`FinancialTFMClassifier`.
    """

    def __init__(
        self,
        model: FinancialTFM | str,
        n_bins: int = DEFAULT_N_BINS,
        classifier_params: dict[str, object] | None = None,
        **classifier_kwargs: object,
    ) -> None:
        """Configure the regressor. See the class docstring for what each argument means."""
        self.model = model
        self.n_bins = int(n_bins)
        if classifier_kwargs:
            warnings.warn(
                "passing classifier settings as bare keywords is deprecated, and they were "
                "lost by sklearn.clone; pass classifier_params={...} instead",
                DeprecationWarning,
                stacklevel=2,
            )
            classifier_params = {**(classifier_params or {}), **classifier_kwargs}
        # Stored as passed, not copied: sklearn.clone checks the attribute *is* the object
        # it handed to the constructor.
        self.classifier_params = classifier_params
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
            ValueError: If ``n_bins`` exceeds the checkpoint's ``max_classes``, if ``y``
                holds a NaN or an infinity, or for a combination
                :meth:`_validate_combination` refuses.
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        # np.digitize puts NaN and +inf in the top bin and -inf in bin 0, so a non-finite
        # target would otherwise be trained on as an ordinary extreme value (audit AU1.5).
        if not np.isfinite(y).all():
            raise ValueError(
                f"y must be finite; got {int(np.isnan(y).sum())} NaN and "
                f"{int(np.isinf(y).sum())} infinite value(s)"
            )
        clf = FinancialTFMClassifier(self.model, **(self.classifier_params or {}))  # type: ignore[arg-type]
        cap = clf.model.cfg.max_classes
        if self.n_bins > cap:
            raise ValueError(
                f"n_bins={self.n_bins} exceeds the checkpoint's max_classes={cap}; "
                "the head has one logit per bin, so the extra bins have nowhere to go"
            )
        # Fitted on the training targets only. The query rows' targets are what is being
        # predicted, so letting them set the edges would leak the answer into the grid the
        # answer is read off -- the same reason `prior/base.py` bins on context rows alone.
        self.binner_ = QuantileBinner(self.n_bins).fit(y)
        self._validate_combination(clf, self.binner_.n_bins_)
        self.classifier_ = clf.fit(X, self.binner_.transform(y))
        return self

    def _validate_combination(self, clf: FinancialTFMClassifier, n_bins: int) -> None:
        """Refuse classifier levers that misread ordinal bins as classes (design review DR1.5).

        The wrapped classifier validates its own combinations at its ``fit``; this adds the
        one that only the regressor can see, because only it knows the classes are bins.

        Args:
            clf: The unfitted classifier about to be fitted on the bin indices.
            n_bins: Bins the binner produced, which can be fewer than ``n_bins`` asked.

        Raises:
            ValueError: For ``ensemble_class_shift`` over more than two bins on a checkpoint
                not pretrained with ``permute_classes``. A shifted member relabels bin ``j``
                as bin ``j + i`` modulo the count, so the top bin lands beside the bottom
                one, and a model that never saw permuted labels reads the order of its
                classes (``PriorConfig.permute_classes``, audit AU5.5). Two bins are a label
                swap, which is allowed. A checkpoint with no provenance record, or a model
                in memory without a ``provenance`` attribute, counts as not permuted.
        """
        if not clf.ensemble_class_shift or n_bins <= 2:
            return
        if _pretrained_with_permuted_classes(self.model):
            return
        raise ValueError(
            f"ensemble_class_shift relabels the {n_bins} ordinal bins by a cyclic shift, and "
            "this checkpoint was not pretrained with permute_classes, so it reads the bin "
            "order a shifted member scrambles; drop ensemble_class_shift, or use a "
            "checkpoint pretrained with permute_classes"
        )

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
        """Raise if :meth:`fit` has not run.

        Raises:
            RuntimeError: If either the binner or the wrapped classifier is missing.
        """
        if self.classifier_ is None or self.binner_ is None:
            raise RuntimeError("fit must be called before predicting")
