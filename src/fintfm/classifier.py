"""scikit-learn-compatible in-context classifier wrapping :class:`FinancialTFM`.

``fit`` only stores the training table as context (no gradient steps);
``predict``/``predict_proba`` run it through the frozen pretrained network
alongside the query rows. This mirrors how TabPFN-style models are used.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

from fintfm.model import FinancialTFM


class FinancialTFMClassifier(BaseEstimator, ClassifierMixin):
    """In-context tabular classifier.

    Args:
        model: A pretrained :class:`FinancialTFM` (or path to a checkpoint).
        device: Torch device for inference.
        max_context: Cap on stored training rows (subsampled if exceeded) to
            bound the O(n^2) attention cost at inference time.
    """

    def __init__(
        self,
        model: FinancialTFM | str,
        device: str = "cpu",
        max_context: int = 2000,
    ) -> None:
        self.model = FinancialTFM.load(model, map_location=device) if isinstance(model, str) else model
        self.device = device
        self.max_context = max_context
        self.model.to(device).eval()

    def fit(self, X: np.ndarray, y: np.ndarray) -> FinancialTFMClassifier:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) > self.model.cfg.max_classes:
            raise ValueError(
                f"model supports at most {self.model.cfg.max_classes} classes, got {len(self.classes_)}"
            )
        if X.shape[1] > self.model.cfg.max_features:
            raise ValueError(f"model supports at most {self.model.cfg.max_features} features, got {X.shape[1]}")
        if X.shape[0] > self.max_context:
            idx = np.random.default_rng(0).choice(X.shape[0], size=self.max_context, replace=False)
            X, y = X[idx], y[idx]
        label_map = {c: i for i, c in enumerate(self.classes_)}
        self._ctx_X = X
        self._ctx_y = np.array([label_map[v] for v in y], dtype=np.int64)
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self._ctx_X.shape[1]:
            raise ValueError("feature width at predict time must match fit time")
        n_ctx = self._ctx_X.shape[0]
        cfg = self.model.cfg
        Xp = np.full((1, n_ctx + X.shape[0], cfg.max_features), np.nan, dtype=np.float32)
        Xp[0, :n_ctx, : self._ctx_X.shape[1]] = self._ctx_X
        Xp[0, n_ctx:, : X.shape[1]] = X
        yp = np.zeros((1, n_ctx + X.shape[0]), dtype=np.int64)
        yp[0, :n_ctx] = self._ctx_y
        Xt = torch.from_numpy(Xp).to(self.device)
        yt = torch.from_numpy(yp).to(self.device)
        nc = torch.tensor([len(self.classes_)], device=self.device)
        logits = self.model(Xt, yt, n_ctx, nc)[0, n_ctx:, : len(self.classes_)]
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]
