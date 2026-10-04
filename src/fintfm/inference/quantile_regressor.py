"""In-context regression from a genuine pinball-loss head (task 48.3).

Why this is a separate class from :class:`fintfm.inference.regressor.FinancialTFMRegressor`
----------------------------------------------------------------------------------------

That class wraps :class:`fintfm.inference.classifier.FinancialTFMClassifier`: it bins a
continuous target into quantile bins and reuses the classification path unchanged, which is
the trick ``fintfm.inference.binning`` argues for in full. A ``head_type="quantile"``
checkpoint has no classification path at all -- its label injection takes a continuous
scalar, not a one-hot class vector, and its head emits raw quantile values rather than
logits -- so there is nothing of :class:`FinancialTFMClassifier` left to wrap. This file is
the minimal context-assembly and de-normalisation logic such a checkpoint actually needs.

Scope, deliberately narrower than the classifier
-------------------------------------------------

Only ``context_strategy="uniform"`` is supported (random subsampling to ``max_context``, no
class-balance concept since there is no class), and only a single column-identity draw (no
``n_ensemble``). Both match what :func:`fintfm.experiments.capability.regression_sweep`
already does for the binned-head arm, which this exists to compare against -- extending
either axis is possible later but would not change that comparison.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin

from fintfm.inference.preprocess import FeatureConditioner, FeatureTransform
from fintfm.modeling.model import FinancialTFM


class ContinuousTargetScaler:
    """Z-score a continuous target using **training-row statistics only**.

    The quantile head's analogue of :class:`fintfm.inference.binning.QuantileBinner`'s
    context-only bin edges: fit on the rows that become in-context evidence, never on the
    rows being predicted, so the normalisation inference can reproduce (context-only) matches
    the one training saw (:func:`fintfm.prior.base.collate`, context-only).

    Attributes:
        mean_: Fitted mean.
        std_: Fitted standard deviation, floored away from zero.
    """

    def __init__(self) -> None:
        """Construct an unfitted scaler; call :meth:`fit` before use."""
        self.mean_: float = 0.0
        self.std_: float = 1.0
        self._fitted = False

    def fit(self, y: np.ndarray) -> ContinuousTargetScaler:
        """Fit mean and standard deviation.

        Args:
            y: ``(n,)`` continuous targets.

        Returns:
            self.
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        self.mean_ = float(np.mean(y))
        self.std_ = float(np.std(y)) + 1e-6
        self._fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Targets to the normalised space the model was trained in.

        Args:
            y: ``(n,)`` continuous targets.

        Returns:
            ``(n,)`` z-scored targets.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        self._check_fitted()
        return (np.asarray(y, dtype=np.float64).ravel() - self.mean_) / self.std_

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        """Normalised model output back to the target's own units.

        Args:
            z: Values in the normalised space (any shape).

        Returns:
            Same shape, in the target's original units.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        self._check_fitted()
        return np.asarray(z, dtype=np.float64) * self.std_ + self.mean_

    def _check_fitted(self) -> None:
        """Raise if :meth:`fit` has not run.

        Raises:
            RuntimeError: If the scaler has no fitted mean/std yet.
        """
        if not self._fitted:
            raise RuntimeError("fit must be called before this method")


class FinancialTFMQuantileRegressor(BaseEstimator, RegressorMixin):
    """In-context regressor for a ``head_type="quantile"`` checkpoint.

    ``fit`` only stores the training table as context and fits the target scaler -- no
    gradient steps, exactly as :class:`fintfm.inference.classifier.FinancialTFMClassifier`.
    ``predict``/``predict_quantile``/``predict_interval`` run it through the frozen
    pretrained network alongside the query rows and de-normalise the result.

    Args:
        model: A pretrained :class:`FinancialTFM` with ``cfg.head_type == "quantile"``, or a
            path to such a checkpoint.
        device: Torch device.
        max_context: Training rows kept as context; a uniform random subsample above this.
        feature_transform: Conditioning applied to features before the model sees them. See
            :mod:`fintfm.inference.preprocess`.
        random_state: Seed for the context subsample and the column-identity draw.
        query_chunk: Queries scored per forward pass. Exact, not an approximation -- see
            :meth:`fintfm.inference.classifier.FinancialTFMClassifier._predict_chunk`'s
            docstring for why splitting queries changes nothing here either.

    Attributes:
        scaler_: The fitted :class:`ContinuousTargetScaler`.
        taus_: ``(n_quantiles,)`` the quantile levels the head's output columns correspond
            to, read off the loaded checkpoint.
    """

    def __init__(
        self,
        model: FinancialTFM | str,
        device: str = "cpu",
        max_context: int = 2000,
        feature_transform: FeatureTransform = "rank",
        random_state: int = 0,
        query_chunk: int = 2048,
    ) -> None:
        """Configure the regressor. See the class docstring for what each argument means.

        Raises:
            ValueError: If the checkpoint's ``cfg.head_type`` is not ``"quantile"``.
        """
        loaded = FinancialTFM.load(model, map_location=device) if isinstance(model, str) else model
        if loaded.cfg.head_type != "quantile":
            raise ValueError(
                f"FinancialTFMQuantileRegressor needs a head_type='quantile' checkpoint, "
                f"got {loaded.cfg.head_type!r} -- use FinancialTFMRegressor for a binned head"
            )
        loaded.assert_trained_for("quantile")
        self.model = loaded
        self.device = device
        self.max_context = max_context
        self.feature_transform = feature_transform
        self.random_state = random_state
        self.query_chunk = query_chunk
        self.model.to(device).eval()
        self.scaler_ = ContinuousTargetScaler()
        self.taus_ = self.model.quantile_taus.detach().cpu().numpy()

    def fit(self, X: np.ndarray, y: np.ndarray) -> FinancialTFMQuantileRegressor:
        """Condition features, subsample to ``max_context``, and fit the target scaler.

        Args:
            X: ``(n, n_features)`` training features.
            y: ``(n,)`` continuous training targets.

        Returns:
            self.

        Raises:
            ValueError: If the conditioned feature width exceeds the checkpoint's
                ``max_features``.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64).ravel()
        self._conditioner = FeatureConditioner(
            self.feature_transform, random_state=self.random_state
        ).fit(X)
        Xc = self._conditioner.transform(X)
        if Xc.shape[1] > self.model.cfg.max_features:
            raise ValueError(
                f"model supports at most {self.model.cfg.max_features} features, "
                f"got {Xc.shape[1]}"
            )
        rng = np.random.default_rng(self.random_state)
        n = Xc.shape[0]
        idx = np.sort(rng.choice(n, size=self.max_context, replace=False)) if n > self.max_context else np.arange(n)
        self._ctx_X = Xc[idx]
        ctx_y = y[idx]
        self.scaler_.fit(ctx_y)
        self._ctx_y = self.scaler_.transform(ctx_y)
        return self

    @torch.no_grad()
    def _predict_chunk(self, X: np.ndarray) -> np.ndarray:
        """Score one chunk of queries against the stored context, in one forward pass.

        Args:
            X: ``(n, n_features)`` conditioned query rows.

        Returns:
            ``(n, n_quantiles)`` raw (normalised-space) quantile predictions.
        """
        n_ctx = self._ctx_X.shape[0]
        cfg = self.model.cfg
        Xp = np.full((1, n_ctx + X.shape[0], cfg.max_features), np.nan, dtype=np.float32)
        Xp[0, :n_ctx, : self._ctx_X.shape[1]] = self._ctx_X
        Xp[0, n_ctx:, : X.shape[1]] = X
        yp = np.zeros((1, n_ctx + X.shape[0]), dtype=np.float32)
        yp[0, :n_ctx] = self._ctx_y
        Xt = torch.from_numpy(Xp).to(self.device)
        yt = torch.from_numpy(yp).to(self.device)
        out = self.model(Xt, yt, n_ctx, column_id_seed=int(self.random_state))
        return out[0, n_ctx:].cpu().numpy()

    def predict_quantile_grid(self, X: np.ndarray) -> np.ndarray:
        """Every predicted quantile level, de-normalised to the target's own units.

        Args:
            X: ``(n, n_features)`` query features.

        Returns:
            ``(n, n_quantiles)`` predictions, column ``j`` corresponding to ``taus_[j]``.
        """
        self._check_fitted()
        Xc = self._conditioner.transform(np.asarray(X, dtype=np.float32))
        if Xc.shape[0] > self.query_chunk:
            raw = np.concatenate(
                [
                    self._predict_chunk(Xc[i : i + self.query_chunk])
                    for i in range(0, Xc.shape[0], self.query_chunk)
                ]
            )
        else:
            raw = self._predict_chunk(Xc)
        return self.scaler_.inverse_transform(raw)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point estimate: the predicted value at the quantile level closest to 0.5.

        Args:
            X: ``(n, n_features)`` query features.

        Returns:
            ``(n,)`` predictions, the median rather than the mean -- the natural point
            estimate of a quantile head, and the one robust to the asymmetric tails a
            pinball-trained head is specifically meant to capture.
        """
        grid = self.predict_quantile_grid(X)
        idx = int(np.argmin(np.abs(self.taus_ - 0.5)))
        return grid[:, idx]

    def predict_quantile(self, X: np.ndarray, q: float) -> np.ndarray:
        """The predicted ``q``-quantile, interpolated across the head's own quantile grid.

        Unlike :meth:`fintfm.inference.binning.QuantileBinner.quantile`, this is a direct
        lookup rather than a CDF inversion: each output column already *is* the model's
        estimate of ``F^{-1}(tau_j)``, so an intermediate ``q`` is linear interpolation in
        ``tau``-space between the two nearest trained columns, not a search over a predicted
        probability mass.

        Args:
            X: ``(n, n_features)`` query features.
            q: Quantile in ``(0, 1)``.

        Returns:
            ``(n,)`` quantile estimates.

        Raises:
            ValueError: If ``q`` is not in ``(0, 1)``.
        """
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must be in (0, 1), got {q}")
        grid = self.predict_quantile_grid(X)
        return np.array([np.interp(q, self.taus_, row) for row in grid])

    def predict_interval(
        self, X: np.ndarray, level: float = 0.8
    ) -> tuple[np.ndarray, np.ndarray]:
        """Central prediction interval at ``level`` coverage.

        Args:
            X: ``(n, n_features)`` query features.
            level: Nominal coverage, e.g. 0.8 for an 80% interval.

        Returns:
            ``(lower, upper)``, each ``(n,)``.

        Raises:
            ValueError: If ``level`` is not in ``(0, 1)``.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        tail = (1.0 - level) / 2.0
        return self.predict_quantile(X, tail), self.predict_quantile(X, 1.0 - tail)

    def _check_fitted(self) -> None:
        """Raise if :meth:`fit` has not run.

        Raises:
            RuntimeError: If the context has not been assembled yet.
        """
        if not hasattr(self, "_ctx_X"):
            raise RuntimeError("fit must be called before predicting")
