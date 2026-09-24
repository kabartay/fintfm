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

from typing import Literal

import numpy as np

#: Which transform to apply to features before the model sees them.
FeatureTransform = Literal["none", "winsor", "rank"]


class FeatureConditioner:
    """Fit a feature transform on training rows and apply it to any rows.

    Args:
        kind: ``"none"`` leaves the data alone. ``"winsor"`` clips each feature to a central
            quantile range, which keeps the shape of the distribution and removes only the
            extremes. ``"rank"`` replaces each value by its quantile mapped to a normal
            distribution, which discards the shape entirely and is therefore immune to tails
            but also to genuine magnitude information.
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
        return self._qt.transform(X).astype(np.float32)
