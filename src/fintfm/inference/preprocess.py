"""Tame heavy-tailed features before the model z-scores them.

Why this exists
---------------
:func:`fintfm.modeling.model.normalize_features` standardises each feature by its **context
mean and standard deviation**, then clips to ±10. That is the standard PFN treatment and it
is fragile in exactly the way financial data punishes: a ratio is a quotient, so a small
denominator produces an arbitrarily large value, and one such firm inflates the standard
deviation enough to collapse every other firm toward zero. The clip bounds the outlier; it
does nothing about the collapse of everyone else.

Corporate financial ratios are heavy-tailed by construction rather than by accident — a firm
about to default is precisely where denominators go small — so this is the common case here,
not the tail case.

Both transforms below are **fitted on training rows only** and applied to context and query
alike, which keeps a query out of its own normalisation. Both preserve NaN, because the model
reads missingness as a signal through a separate mask and destroying it would lose
information.

Neither requires retraining, which is why they are tried before anything that does.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np

#: Which transform to apply to features before the model sees them.
FeatureTransform = Literal["none", "winsor", "rank", "power"]


class FeatureConditioner:
    """Fit a feature transform on training rows and apply it to any rows.

    Args:
        kind: ``"none"`` leaves the data alone. ``"winsor"`` clips each feature to a central
            quantile range, which keeps the shape of the distribution and removes only the
            extremes. ``"rank"`` replaces each value by its quantile mapped to a normal
            distribution, which discards the shape entirely and is therefore immune to tails
            but also to genuine magnitude information. ``"power"`` applies a Yeo-Johnson power
            transform (task 48.21, surfaced from Neuralk-AI's benchmark harness, see
            ``docs/paper/RELATED_WORK.md``): unlike ``"rank"`` it preserves the relative
            *magnitude* within a monotonic reshaping rather than discarding it for pure order,
            so it is worth comparing against ``"rank"`` rather than assumed better or worse.
        quantile: Tail fraction clipped at each end by ``"winsor"``.
        subsample: Rows sampled to estimate the quantiles, bounding cost on a large panel.
        random_state: Seed for that subsample.
    """

    def __init__(
        self,
        kind: FeatureTransform = "none",
        quantile: float = 0.01,
        subsample: int = 20_000,
        random_state: int = 0,
    ) -> None:
        """Configure the conditioner. See the class docstring for what each argument means."""
        if not 0.0 <= quantile < 0.5:
            raise ValueError(f"quantile must lie in [0, 0.5), got {quantile}")
        self.kind = kind
        self.quantile = quantile
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> FeatureConditioner:
        """Estimate the transform from training rows.

        Args:
            X: ``(n, F)`` training features, NaN for missing.

        Returns:
            ``self``.
        """
        X = np.asarray(X, dtype=np.float32)
        if self.kind == "none":
            return self
        rng = np.random.default_rng(self.random_state)
        rows = (
            rng.choice(X.shape[0], size=self.subsample, replace=False)
            if X.shape[0] > self.subsample
            else np.arange(X.shape[0])
        )
        S = X[rows]
        if self.kind == "winsor":
            with np.errstate(all="ignore"):
                lo = np.nanquantile(S, self.quantile, axis=0)
                hi = np.nanquantile(S, 1.0 - self.quantile, axis=0)
            # an all-NaN column has no bounds; leave it untouched rather than zeroing it
            self._lo = np.where(np.isfinite(lo), lo, -np.inf).astype(np.float32)
            self._hi = np.where(np.isfinite(hi), hi, np.inf).astype(np.float32)
            return self
        if self.kind == "power":
            from sklearn.preprocessing import PowerTransformer

            # A ratio with a near-zero denominator produces a genuine +-inf, not merely a
            # large value; replace it with the finite extreme per column, fitted on training
            # rows only and reused at transform time -- the same discipline "winsor" applies
            # -- so power is a genuine drop-in alternative rather than one that crashes on
            # this data's tails.
            finite = np.where(np.isfinite(S), S, np.nan)
            # An all-NaN column (handled below) makes nanmax/nanmin warn; errstate does not
            # cover it, since it is a RuntimeWarning rather than a floating-point exception.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self._pt_hi = np.nanmax(finite, axis=0)
                self._pt_lo = np.nanmin(finite, axis=0)
            # A column that is entirely +-inf in the fitting subsample has no finite value to
            # clamp to; nanmax/nanmin then return NaN, which the formula below also rejects.
            # Such a column carries no information in this subsample regardless, so 0.0 is an
            # arbitrary but finite and harmless placeholder rather than a value someone chose
            # to mean something.
            self._pt_hi = np.where(np.isfinite(self._pt_hi), self._pt_hi, 0.0)
            self._pt_lo = np.where(np.isfinite(self._pt_lo), self._pt_lo, 0.0)
            S = np.where(np.isposinf(S), self._pt_hi, S)
            S = np.where(np.isneginf(S), self._pt_lo, S)

            # standardize=False: sklearn's Yeo-Johnson formula itself overflows to +-inf on a
            # small fraction of cells (measured: 5 of 78,015,600 on this project's real V4
            # panel) when a column's fitted lambda is poorly conditioned, and PowerTransformer's
            # *internal* standardizing scaler then raises on its own output before this
            # function ever sees it. Standardizing by hand below lets those rare cells be
            # clipped the same way every other tail value here is, rather than crashing a run
            # over 5 cells in 78 million.
            with warnings.catch_warnings():
                # scipy's own yeo-johnson math overflows on a poorly-conditioned lambda for a
                # rare cell (measured: 5 of 78,015,600 on this project's real V4 panel); the
                # result is clipped below the same way every other tail value here is, so the
                # warning describes a case already handled rather than an unhandled one.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self._pt = PowerTransformer(method="yeo-johnson", standardize=False).fit(S)
                yj = self._pt.transform(S)
            yj_finite = np.where(np.isfinite(yj), yj, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self._pt_yj_hi = np.nanmax(yj_finite, axis=0)
                self._pt_yj_lo = np.nanmin(yj_finite, axis=0)
                self._pt_mean = np.nanmean(yj_finite, axis=0)
                self._pt_std = np.nanstd(yj_finite, axis=0)
            self._pt_yj_hi = np.where(np.isfinite(self._pt_yj_hi), self._pt_yj_hi, 0.0)
            self._pt_yj_lo = np.where(np.isfinite(self._pt_yj_lo), self._pt_yj_lo, 0.0)
            self._pt_mean = np.where(np.isfinite(self._pt_mean), self._pt_mean, 0.0)
            # A column with zero (or unmeasurable) spread would divide by zero; 1.0 leaves it
            # merely uncentred rather than turning a real value into nan or inf.
            self._pt_std = np.where(
                np.isfinite(self._pt_std) & (self._pt_std > 0), self._pt_std, 1.0
            )
            return self

        from sklearn.preprocessing import QuantileTransformer

        self._qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(1000, S.shape[0]),
            subsample=self.subsample,
            random_state=self.random_state,
        ).fit(S)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted transform.

        Args:
            X: ``(n, F)`` features, NaN for missing.

        Returns:
            ``(n, F)`` transformed features, with NaN preserved in place.
        """
        X = np.asarray(X, dtype=np.float32)
        if self.kind == "none":
            return X
        if self.kind == "winsor":
            return np.clip(X, self._lo, self._hi).astype(np.float32)
        if self.kind == "power":
            X = np.where(np.isposinf(X), self._pt_hi, X)
            X = np.where(np.isneginf(X), self._pt_lo, X)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                yj = self._pt.transform(X)
            yj = np.where(np.isposinf(yj), self._pt_yj_hi, yj)
            yj = np.where(np.isneginf(yj), self._pt_yj_lo, yj)
            return ((yj - self._pt_mean) / self._pt_std).astype(np.float32)
        return self._qt.transform(X).astype(np.float32)
