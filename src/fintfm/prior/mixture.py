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
        n_rows_choices: When set, each *batch* draws its task size from this tuple instead of
            using ``n_rows``. This matters because the base-rate floor is derived from task
            size (``prior/financial.py``): a 256-row task cannot carry a 0.2% default rate,
            so a fixed small ``n_rows`` silently caps how imbalanced the model ever sees —
            which is exactly how §26 happened. Varying it across batches lets one run cover
            both the dense-small and sparse-large regimes. All tasks *within* a batch share a
            size, since :func:`fintfm.prior.base.collate` requires it.
        n_horizons: When set, financial tasks additionally carry a default **period** so a
            hazard head can be trained on the survival likelihood (``docs/FINDINGS.md`` §20).
            The generic SCM prior has no notion of time, so **``p_financial`` must be 1.0**
            when this is set — a batch mixing survival and binary-only tasks is refused by
            :func:`fintfm.prior.base.collate` rather than silently padded.
    """

    max_features: int = 24
    max_classes: int = 10
    p_financial: float = 0.7
    n_rows: int = 256
    min_ctx_frac: float = 0.3
    max_ctx_frac: float = 0.9
    n_rows_choices: tuple[int, ...] | None = None
    n_horizons: int | None = None


def sample_task(rng: np.random.Generator, cfg: PriorConfig, n_rows: int | None = None) -> Task:
    """Draw one task from the mixture prior.

    Raises:
        ValueError: If ``n_horizons`` is set with ``p_financial < 1``, which would produce
            batches mixing survival and binary-only tasks.
    """
    n = cfg.n_rows if n_rows is None else n_rows
    if cfg.n_horizons is not None and cfg.p_financial < 1.0:
        raise ValueError(
            "n_horizons requires p_financial=1.0; the SCM prior has no time axis and a "
            "mixed batch cannot carry a coherent survival likelihood"
        )
    if rng.random() < cfg.p_financial:
        return sample_financial_task(
            rng, n, max_features=cfg.max_features, n_horizons=cfg.n_horizons
        )
    return sample_scm_task(rng, n, max_features=cfg.max_features, max_classes=cfg.max_classes)


def sample_batch(rng: np.random.Generator, cfg: PriorConfig, batch_size: int) -> TaskBatch:
    """Draw a padded batch with a shared task size and context/query split.

    The task size is drawn per batch from ``cfg.n_rows_choices`` when set, because the
    base-rate floor scales with it and a fixed size caps how imbalanced any task can be.
    """
    n_rows = (
        int(rng.choice(cfg.n_rows_choices)) if cfg.n_rows_choices else cfg.n_rows
    )
    tasks = [sample_task(rng, cfg, n_rows=n_rows) for _ in range(batch_size)]
    n_ctx = int(rng.integers(int(cfg.min_ctx_frac * n_rows), int(cfg.max_ctx_frac * n_rows) + 1))
    n_ctx = min(max(n_ctx, 2), n_rows - 1)
    return collate(tasks, n_ctx=n_ctx, max_features=cfg.max_features)
