"""scikit-learn-compatible in-context classifier wrapping :class:`FinancialTFM`.

``fit`` only stores the training table as context (no gradient steps);
``predict``/``predict_proba`` run it through the frozen pretrained network
alongside the query rows. This mirrors how TabPFN-style models are used.

**How the context is chosen matters more than the architecture.** Tanna et al. (2026),
*Data Presentation Over Architecture* (arXiv:2605.18635), benchmark seven context-construction
strategies for credit-risk TFMs and find balanced and hybrid sampling worth 3-4 AUC points
over uniform sampling — a gap wider than the spread between model families. Credit default
rates run at a few percent, so uniform subsampling of a capped context spends almost all of
it on non-defaulters. Hence :class:`ContextStrategy` and a default of ``"balanced"``.
See ``docs/FINDINGS.md`` §5.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

from fintfm.model import FinancialTFM

ContextStrategy = Literal["balanced", "hybrid", "uniform"]


def _select_context(
    y: np.ndarray, max_context: int, strategy: ContextStrategy, rng: np.random.Generator
) -> np.ndarray:
    """Choose which training rows become the model's context.

    Args:
        y: Integer class labels of the full training set.
        max_context: Maximum number of rows to keep.
        strategy: ``"balanced"`` draws as evenly across classes as their sizes allow;
            ``"hybrid"`` splits the budget between a balanced half and a uniform half,
            preserving some of the true base rate; ``"uniform"`` samples at random and is
            the baseline the literature reports as worst on imbalanced credit data.
        rng: Random generator.

    Returns:
        Indices into ``y``, sorted for reproducibility.
    """
    n = y.shape[0]
    if n <= max_context:
        return np.arange(n)
    if strategy == "uniform":
        return np.sort(rng.choice(n, size=max_context, replace=False))

    by_class = {c: np.flatnonzero(y == c) for c in np.unique(y)}
    balanced_budget = max_context if strategy == "balanced" else max_context // 2

    # Water-filling: classes smaller than an equal share contribute everything they have,
    # and the remaining budget is redistributed over the classes that still have rows.
    quotas: dict[int, int] = {}
    remaining, pending = balanced_budget, dict(by_class)
    while pending:
        share = remaining // len(pending)
        if share == 0:
            break
        exhausted = {c: idx for c, idx in pending.items() if len(idx) <= share}
        if not exhausted:
            for c in pending:
                quotas[c] = share
            remaining -= share * len(pending)
            break
        for c, idx in exhausted.items():
            quotas[c] = len(idx)
            remaining -= len(idx)
            del pending[c]

    chosen = [rng.choice(by_class[c], size=k, replace=False) for c, k in quotas.items() if k]
    picked = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)

    if strategy == "hybrid" or picked.shape[0] < max_context:
        rest = np.setdiff1d(np.arange(n), picked, assume_unique=False)
        top_up = min(max_context - picked.shape[0], rest.shape[0])
        if top_up > 0:
            picked = np.concatenate([picked, rng.choice(rest, size=top_up, replace=False)])
    return np.sort(picked.astype(np.int64))


class FinancialTFMClassifier(BaseEstimator, ClassifierMixin):
    """In-context tabular classifier.

    Args:
        model: A pretrained :class:`FinancialTFM` (or path to a checkpoint).
        device: Torch device for inference.
        max_context: Cap on stored training rows (subsampled if exceeded) to bound the
            O(n^2) attention cost at inference time. Tanna et al. report 5000-10000 as the
            sweet spot on credit data; the default here is deliberately lower because
            nothing in this repository has measured that trade-off yet on CPU.
        context_strategy: How to subsample when the training set exceeds ``max_context``.
            See :func:`_select_context`. Irrelevant when it does not.
        random_state: Seed for context subsampling, so a stored context is reproducible.
    """

    def __init__(
        self,
        model: FinancialTFM | str,
        device: str = "cpu",
        max_context: int = 2000,
        context_strategy: ContextStrategy = "balanced",
        random_state: int = 0,
    ) -> None:
        self.model = FinancialTFM.load(model, map_location=device) if isinstance(model, str) else model
        self.device = device
        self.max_context = max_context
        self.context_strategy = context_strategy
        self.random_state = random_state
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
        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_coded = np.array([label_map[v] for v in y], dtype=np.int64)
        idx = _select_context(
            y_coded, self.max_context, self.context_strategy, np.random.default_rng(self.random_state)
        )
        X, y_coded = X[idx], y_coded[idx]
        self._ctx_X = X
        self._ctx_y = y_coded
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
