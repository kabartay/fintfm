"""Mixture over priors plus the batch sampler used for pretraining."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fintfm.prior.base import Task, TaskBatch, collate
from fintfm.prior.crossed import sample_scm_features_financial_label
from fintfm.prior.financial import (
    _ABSOLUTE_RATE_FLOOR,
    _N_SECTORS,
    _RATE_CEILING,
    _SHARPNESS_MAX,
    _SHARPNESS_MIN,
    MIN_EXPECTED_POSITIVES,
    sample_financial_task,
)
from fintfm.prior.scm import sample_scm_task
from fintfm.prior.trivial import sample_trivial_task


@dataclass
class PriorConfig:
    """Hyper-parameters of the pretraining task distribution.

    Attributes:
        max_features: Feature width the model is built for.
        max_classes: Class-head width of the model.
        p_financial: Probability of drawing a financial task instead of an SCM task.
        p_trivial: Probability of drawing a **trivial** task instead — few clean features, a
            deterministic linear rule, no noise. Zero by default. This is the diagnostic
            control of ``prior/trivial.py``: it answers whether the architecture can learn
            in-context prediction at all (``docs/FINDINGS.md`` §53), and a prior made only of
            trivial tasks would teach nothing about abstention.
        p_crossed: Probability of drawing a **crossed-design** task instead —
            ``prior/crossed.py``'s ``sample_scm_features_financial_label``: the generic SCM
            prior's feature-generating computational graph, labelled with the financial
            prior's signed-linear-driver mechanism instead of a fresh SCM label node. Zero by
            default. Diagnostic for ``docs/FINDINGS.md`` §66: isolates whether the financial
            prior's failure to teach column-specific in-context inference tracks its
            *features* or its *label function*, after seven other candidates were eliminated.
            A prior made only of crossed tasks is not a candidate for production; see
            ``prior/crossed.py``'s module docstring.
        n_rows: Rows per task (context + query).
        min_ctx_frac / max_ctx_frac: Range for the context fraction of ``n_rows``.
        n_rows_choices: When set, each *batch* draws its task size from this tuple instead of
            using ``n_rows``. This matters because the base-rate floor is derived from task
            size (``prior/financial.py``): a 256-row task cannot carry a 0.2% default rate,
            so a fixed small ``n_rows`` silently caps how imbalanced the model ever sees —
            which is exactly how §26 happened. Varying it across batches lets one run cover
            both the dense-small and sparse-large regimes. All tasks *within* a batch share a
            size, since :func:`fintfm.prior.base.collate` requires it.
        sharpness_min / sharpness_max: The financial prior's signal-to-noise range, drawn
            log-uniformly per task. ``docs/FINDINGS.md`` §64: what a prior teaches tracks how
            learnable its tasks are, not how much column identity they demand.
        min_expected_positives / absolute_rate_floor / rate_ceiling / n_sectors_max: The
            financial prior's default-rate envelope, defaulting to the measured constants in
            ``prior/financial.py``. Populated from the ``prior`` section of the configuration
            by the training entry point, so a run can widen or narrow the regime it covers
            without a code change — the gap that produced §26.
        n_horizons: When set, financial tasks additionally carry a default **period** so a
            hazard head can be trained on the survival likelihood (``docs/FINDINGS.md`` §20).
            The generic SCM prior has no notion of time, so **``p_financial`` must be 1.0**
            when this is set — a batch mixing survival and binary-only tasks is refused by
            :func:`fintfm.prior.base.collate` rather than silently padded.
    """

    max_features: int = 24
    max_classes: int = 10
    p_financial: float = 0.7
    p_trivial: float = 0.0
    p_crossed: float = 0.0
    n_rows: int = 256
    min_ctx_frac: float = 0.3
    max_ctx_frac: float = 0.9
    n_rows_choices: tuple[int, ...] | None = None
    n_horizons: int | None = None
    sharpness_min: float = _SHARPNESS_MIN
    sharpness_max: float = _SHARPNESS_MAX
    min_expected_positives: float = MIN_EXPECTED_POSITIVES
    absolute_rate_floor: float = _ABSOLUTE_RATE_FLOOR
    rate_ceiling: float = _RATE_CEILING
    n_sectors_max: int = _N_SECTORS


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
    if cfg.p_trivial and rng.random() < cfg.p_trivial:
        return sample_trivial_task(rng, n, max_features=min(8, cfg.max_features))
    if cfg.p_crossed and rng.random() < cfg.p_crossed:
        return sample_scm_features_financial_label(
            rng, n, max_features=cfg.max_features,
            min_expected_positives=cfg.min_expected_positives,
            absolute_rate_floor=cfg.absolute_rate_floor, rate_ceiling=cfg.rate_ceiling,
            sharpness_min=cfg.sharpness_min, sharpness_max=cfg.sharpness_max,
        )
    if rng.random() < cfg.p_financial:
        return sample_financial_task(
            rng,
            n,
            max_features=cfg.max_features,
            n_horizons=cfg.n_horizons,
            min_expected_positives=cfg.min_expected_positives,
            absolute_rate_floor=cfg.absolute_rate_floor,
            rate_ceiling=cfg.rate_ceiling,
            n_sectors_max=cfg.n_sectors_max,
            sharpness_min=cfg.sharpness_min,
            sharpness_max=cfg.sharpness_max,
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
