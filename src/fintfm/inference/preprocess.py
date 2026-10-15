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

Audit AU5.12 added two pieces used only by ensemble members that ask for them, so the
default path is untouched: a ``"squash"`` transform (robust scaling then a soft clip), and
:class:`SVDAugmenter`, which appends a few singular-vector projections of the context.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np

#: Which transform to apply to features before the model sees them.
FeatureTransform = Literal["none", "winsor", "rank", "power", "squash"]

#: Scale of the ``"squash"`` soft clip, in robust units (IQRs from the median): a value at
#: this distance maps to ``3 / sqrt(2)``, and every value maps inside ``(-3, 3)``.
SQUASH_SCALE = 3.0


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
            ``"squash"`` (audit AU5.12) centres each feature on its median, divides by its
            interquartile range, then soft-clips with ``z / sqrt(1 + (z / 3)^2)``: linear near
            the centre, so ordinary magnitudes survive as they are, and bounded inside
            ``(-3, 3)``, so a tail value keeps its sign and its order but cannot inflate the
            model's per-feature standard deviation. A feature with zero IQR is scaled by its
            standard deviation instead, and left unscaled if that is zero too.
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
        # A ratio with a near-zero denominator produces a genuine +-inf, not merely a large
        # value; replace it with the finite extreme per column, fitted on training rows only
        # and reused at transform time -- the same discipline "winsor" applies -- so neither
        # "power" nor "rank" crashes on this data's tails. ``QuantileTransformer`` raises on
        # an infinity, which made the default "rank" fail where "power" did not (audit
        # AU1.3). On finite data the clamp is the identity.
        finite = np.where(np.isfinite(S), S, np.nan)
        # An all-NaN column (handled below) makes nanmax/nanmin warn; errstate does not
        # cover it, since it is a RuntimeWarning rather than a floating-point exception.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self._inf_hi = np.nanmax(finite, axis=0)
            self._inf_lo = np.nanmin(finite, axis=0)
        # A column that is entirely +-inf in the fitting subsample has no finite value to
        # clamp to; nanmax/nanmin then return NaN, which the transforms below also reject.
        # Such a column carries no information in this subsample regardless, so 0.0 is an
        # arbitrary but finite and harmless placeholder rather than a value someone chose
        # to mean something.
        self._inf_hi = np.where(np.isfinite(self._inf_hi), self._inf_hi, 0.0)
        self._inf_lo = np.where(np.isfinite(self._inf_lo), self._inf_lo, 0.0)
        S = self._clamp_infinities(S)
        if self.kind == "squash":
            with warnings.catch_warnings():
                # an all-NaN column has no median or quantiles; the fallbacks below handle it
                warnings.simplefilter("ignore", category=RuntimeWarning)
                med = np.nanmedian(S, axis=0)
                iqr = np.nanquantile(S, 0.75, axis=0) - np.nanquantile(S, 0.25, axis=0)
                std = np.nanstd(S, axis=0)
            # a column that is mostly one value (a zero-inflated ratio, a flag) has IQR 0;
            # its standard deviation still measures the rare departures, and a column with
            # neither is constant, so 1.0 leaves it merely centred
            scale = np.where(np.isfinite(iqr) & (iqr > 0), iqr, std)
            self._sq_centre = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
            self._sq_scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0).astype(
                np.float32
            )
            return self
        if self.kind == "power":
            from sklearn.preprocessing import PowerTransformer

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
                # A column with no value in the training rows gives Yeo-Johnson nothing to fit
                # and sklearn raises on it, failing the whole fit over one empty column. Fit the
                # rest; transform() returns the empty ones as NaN, which is what "rank" already
                # yields for them and all the training rows say about them. With no empty
                # column, ``_pt_cols`` is None and the path below is the one it always was.
                fittable = ~np.isnan(S).all(axis=0)
                self._pt_cols = None if fittable.all() else np.flatnonzero(fittable)
                if self._pt_cols is not None:
                    S = S[:, self._pt_cols]
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
        X = self._clamp_infinities(X)
        if self.kind == "squash":
            z = (X - self._sq_centre) / self._sq_scale
            return (z / np.sqrt(1.0 + (z / SQUASH_SCALE) ** 2)).astype(np.float32)
        if self.kind == "power":
            cols = self._pt_cols
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                yj = self._pt.transform(X if cols is None else X[:, cols])
            yj = np.where(np.isposinf(yj), self._pt_yj_hi, yj)
            yj = np.where(np.isneginf(yj), self._pt_yj_lo, yj)
            out = ((yj - self._pt_mean) / self._pt_std).astype(np.float32)
            if cols is None:
                return out
            full = np.full(X.shape, np.nan, dtype=np.float32)
            full[:, cols] = out
            return full
        return self._qt.transform(X).astype(np.float32)

    def _clamp_infinities(self, X: np.ndarray) -> np.ndarray:
        """Replace ``+-inf`` with each column's finite extreme from the fitted rows.

        Args:
            X: ``(n, F)`` features, possibly holding ``+-inf``.

        Returns:
            ``(n, F)`` features with no infinity; NaN and finite values untouched.
        """
        X = np.where(np.isposinf(X), self._inf_hi, X)
        return np.where(np.isneginf(X), self._inf_lo, X)


class SVDAugmenter:
    """Append a context's leading singular-vector projections to every row (audit AU5.12).

    A truncated SVD of the conditioned context rows, uncentred as in the usual truncated-SVD
    convention: the right singular vectors ``V_k`` are fitted on the context and every row,
    context or query, gains the ``k`` extra columns ``x @ V_k``. Used only by the ensemble
    members that ``ensemble_svd_components`` selects, so a member sees a few dense linear
    summaries of the table beside its raw columns while its siblings do not.

    Missing and non-finite cells count as 0 in the projection (the centre of every
    conditioning transform except ``"none"``); the original columns, NaN included, are passed
    through untouched, so the model's missingness mask is unaffected. Each singular vector's
    sign is fixed so its largest-magnitude loading is positive, which makes the appended
    columns a function of the context rather than of the LAPACK build.

    Args:
        n_components: Projections to append. Capped at the context's row and column counts.
    """

    def __init__(self, n_components: int) -> None:
        """Configure the augmenter. See the class docstring."""
        if n_components < 0:
            raise ValueError(f"n_components must be >= 0, got {n_components}")
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> SVDAugmenter:
        """Fit the singular vectors on context rows.

        Args:
            X: ``(n, F)`` conditioned context rows, NaN for missing.

        Returns:
            ``self``, with ``components_`` of shape ``(k, F)``.
        """
        X0 = self._filled(X)
        k = min(self.n_components, *X0.shape)
        if k == 0:
            self.components_ = np.zeros((0, X0.shape[1]), dtype=np.float32)
            return self
        _, _, vt = np.linalg.svd(X0.astype(np.float64), full_matrices=False)
        vt = vt[:k]
        signs = np.sign(vt[np.arange(k), np.abs(vt).argmax(axis=1)])
        self.components_ = (vt * np.where(signs == 0, 1.0, signs)[:, None]).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Append the fitted projections.

        Args:
            X: ``(n, F)`` conditioned rows, matching the width seen by :meth:`fit`.

        Returns:
            ``(n, F + k)``: ``X`` unchanged, then its ``k`` projections.
        """
        X = np.asarray(X, dtype=np.float32)
        return np.hstack([X, self._filled(X) @ self.components_.T]).astype(np.float32)

    @staticmethod
    def _filled(X: np.ndarray) -> np.ndarray:
        """``X`` with NaN and ``+-inf`` set to 0, for the projection only."""
        X = np.asarray(X, dtype=np.float32)
        return np.where(np.isfinite(X), X, 0.0).astype(np.float32)
