"""Mixture over priors plus the batch sampler used for pretraining."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fintfm.prior.base import Task, TaskBatch, collate
from fintfm.prior.financial import sample_financial_task
from fintfm.prior.scm import sample_scm_task


@dataclass
class PriorConfig:
    """Hyper-parameters of the pretraining task distribution.

    Attributes:
        max_features: Feature width the model is built for.
        max_classes: Class-head width of the model.
        p_financial: Probability of drawing a financial task instead of an SCM task.
        n_rows: Rows per task (context + query).
        min_ctx_frac / max_ctx_frac: Range for the context fraction of ``n_rows``.
    """

    max_features: int = 24
    max_classes: int = 10
    p_financial: float = 0.7
    n_rows: int = 256
    min_ctx_frac: float = 0.3
    max_ctx_frac: float = 0.9


def sample_task(rng: np.random.Generator, cfg: PriorConfig, n_rows: int | None = None) -> Task:
    """Draw one task from the mixture prior."""
    n = cfg.n_rows if n_rows is None else n_rows
    if rng.random() < cfg.p_financial:
        return sample_financial_task(rng, n, max_features=cfg.max_features)
    return sample_scm_task(rng, n, max_features=cfg.max_features, max_classes=cfg.max_classes)


def sample_batch(rng: np.random.Generator, cfg: PriorConfig, batch_size: int) -> TaskBatch:
    """Draw a padded batch with a shared random context/query split."""
    tasks = [sample_task(rng, cfg) for _ in range(batch_size)]
    n_ctx = int(rng.integers(int(cfg.min_ctx_frac * cfg.n_rows), int(cfg.max_ctx_frac * cfg.n_rows) + 1))
    n_ctx = min(max(n_ctx, 2), cfg.n_rows - 1)
    return collate(tasks, n_ctx=n_ctx, max_features=cfg.max_features)
