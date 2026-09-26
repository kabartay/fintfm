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
from fintfm.prior.learnability import is_learnable
from fintfm.prior.scm import (
    sample_scm_regression_task,
    sample_scm_task,
    sample_scm_task_group,
)
from fintfm.prior.tree import sample_tree_task
from fintfm.prior.trivial import sample_trivial_task


@dataclass
class PriorConfig:
    """Hyper-parameters of the pretraining task distribution.

    Attributes:
        max_features: Feature width the model is built for.
        max_classes: Class-head width of the model.
        p_tree: Probability of drawing a **tree-structured** task
            (:func:`fintfm.prior.tree.sample_tree_task`) instead. Zero by default so existing
            checkpoints are unaffected. Selected on *distinctiveness*, not performance: §111
            measures (as corrected by §112) that `tree` is the **only** prior in this
            mixture with positive tree-versus-linear distinctiveness, +0.0225, against −0.0292
            for the financial prior and −0.0021 for the SCM one — so the mixture had no member
            generating the axis-aligned structure every tree baseline exploits. Draws from the financial
            budget, since it is a general-structure prior like the SCM one.
        scm_reuse_graph: Tasks drawn per SCM graph within a batch (task 48.12). ``1`` keeps
            the original one-graph-one-task behaviour. Above 1, a graph is built once and
            several of its nodes are used as targets in turn. **This does not change how many
            tasks a batch contains** — 8 either way — so it does not address the task shortfall
            `CLAUDE.md`'s "count the tasks, not the steps" records; §113 corrects an earlier
            claim here that it did. What it changes is generation cost (~20% of GPU step time,
            two-thirds saved, worth roughly 13% more steps per dollar) against **a quarter as
            many independent graphs per batch**, which is a diversity loss nothing has yet
            measured. Defensible only because the siblings are genuinely distinct problems: a
            model fitted on one target scores 0.4775 — chance — on another from the same graph
            (§113).
        scm_legacy: Draw SCM tasks from the pre-48.17/48.19 prior. The control arm for the
            widened prior; see :func:`fintfm.prior.scm.sample_scm_task`. **Note that this knob
            is inert at ``p_financial=1.0``**, where no SCM task is ever drawn -- the mistake
            that wasted one 6,000-step run.
        p_financial: Probability of drawing a financial task instead of an SCM task.
        p_trivial: Probability of drawing a **trivial** task instead — few clean features, a
            deterministic linear rule, no noise. Zero by default. This is the diagnostic
            control of ``prior/trivial.py``: it answers whether the architecture can learn
            in-context prediction at all (``docs/results/FINDINGS.md`` §53), and a prior made only of
            trivial tasks would teach nothing about abstention.
        identity_shuffle: When True, every financial task drawn (whether via ``p_financial``
            or as the default source) exposes its named/ratio columns from an
            independently-per-account-permuted copy of the accounts, breaking cross-account
            identities (``equity = assets - liabilities``) while leaving the label and each
            column's own marginal distribution untouched. Diagnostic for ``docs/results/FINDINGS.md``
            §67-§71 (task 38.11); default False reproduces every prior checkpoint's behaviour.
        p_regression: Probability of drawing a **regression** task instead — the SCM prior's
            continuous latent kept rather than thresholded, binned on context quantiles by
            :func:`fintfm.prior.base.collate`. Mixing freely with classification tasks in one
            batch is safe: each task carries its own bin count in ``TaskBatch.n_classes``,
            exactly as a classification task carries its own class count.
        p_crossed: Probability of drawing a **crossed-design** task instead —
            ``prior/crossed.py``'s ``sample_scm_features_financial_label``: the generic SCM
            prior's feature-generating computational graph, labelled with the financial
            prior's signed-linear-driver mechanism instead of a fresh SCM label node. Zero by
            default. Diagnostic for ``docs/results/FINDINGS.md`` §66: isolates whether the financial
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
            log-uniformly per task. ``docs/results/FINDINGS.md`` §64: what a prior teaches tracks how
            learnable its tasks are, not how much column identity they demand.
        min_expected_positives / absolute_rate_floor / rate_ceiling / n_sectors_max: The
            financial prior's default-rate envelope, defaulting to the measured constants in
            ``prior/financial.py``. Populated from the ``prior`` section of the configuration
            by the training entry point, so a run can widen or narrow the regime it covers
            without a code change — the gap that produced §26.
        n_horizons: When set, financial tasks additionally carry a default **period** so a
            hazard head can be trained on the survival likelihood (``docs/results/FINDINGS.md`` §20).
            The generic SCM prior has no notion of time, so **``p_financial`` must be 1.0**
            when this is set — a batch mixing survival and binary-only tasks is refused by
            :func:`fintfm.prior.base.collate` rather than silently padded.
    """

    max_features: int = 24
    max_classes: int = 10
    scm_legacy: bool = False
    scm_reuse_graph: int = 1
    p_tree: float = 0.0
    p_financial: float = 0.7
    p_trivial: float = 0.0
    p_crossed: float = 0.0
    p_regression: float = 0.0
    identity_shuffle: bool = False
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
    p_learnability_filter: float = 0.0
    """Reject a drawn task and resample when a cheap held-out learner cannot beat
    :attr:`learnability_auc_floor` AUC on it. Nori rejects unlearnable synthetic datasets with
    an ExtraTrees signal-quality filter; §125 measured that the naive version here would reject
    35% of the production mixture's tasks, including a fifth that are single-class or otherwise
    unscorable, and that on the rejected subset the model itself scores 0.520 against the
    filter's judge at 0.392 -- the filter would discard signal this model can use. Named as a
    probability rather than a bool so the filter's own effect can be swept (0.0 = off, 1.0 =
    every task is judged) without a second flag; ``0.0`` reproduces every prior checkpoint's
    behaviour. Costs about **0.03-0.04 s/task** to judge, measured directly rather than assumed:
    a naive extrapolation from §125's per-task wall time (which included sampling the task, not
    just judging it) would have priced a full run's filtering at roughly 20 hours. Judging every
    one of 48,000 draws once is under 30 minutes; :attr:`learnability_max_resamples` bounds the
    added cost of resampling the rejects."""
    learnability_auc_floor: float = 0.51
    """AUC threshold below which :attr:`p_learnability_filter` rejects a task. §125's value,
    chosen to match a signal an ExtraTrees can barely distinguish from chance."""
    learnability_max_resamples: int = 8
    """Give up and keep the task rather than resample forever. A batch member must never be
    silently dropped -- :func:`sample_batch` always returns exactly ``batch_size`` tasks -- and
    a task family with a rejection rate above this bound would otherwise spin without limit."""


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
    if cfg.p_tree and rng.random() < cfg.p_tree:
        if cfg.n_horizons is not None:
            raise ValueError("n_horizons requires p_financial=1.0; the tree prior has no time axis")
        return sample_tree_task(rng, n, max_features=cfg.max_features, max_classes=cfg.max_classes)
    if cfg.p_regression and rng.random() < cfg.p_regression:
        if cfg.n_horizons is not None:
            raise ValueError(
                "n_horizons requires a survival label; a regression task has no event time"
            )
        return sample_scm_regression_task(
            rng, n, max_features=cfg.max_features, n_bins=cfg.max_classes
        )
    if cfg.p_trivial and rng.random() < cfg.p_trivial:
        return sample_trivial_task(rng, n, max_features=min(8, cfg.max_features))
    if cfg.p_crossed and rng.random() < cfg.p_crossed:
        return sample_scm_features_financial_label(
            rng,
            n,
            max_features=cfg.max_features,
            min_expected_positives=cfg.min_expected_positives,
            absolute_rate_floor=cfg.absolute_rate_floor,
            rate_ceiling=cfg.rate_ceiling,
            sharpness_min=cfg.sharpness_min,
            sharpness_max=cfg.sharpness_max,
        )
    if rng.random() < cfg.p_financial:
        return sample_financial_task(
            rng,
            n,
            max_features=cfg.max_features,
            n_horizons=cfg.n_horizons,
            identity_shuffle=cfg.identity_shuffle,
            min_expected_positives=cfg.min_expected_positives,
            absolute_rate_floor=cfg.absolute_rate_floor,
            rate_ceiling=cfg.rate_ceiling,
            n_sectors_max=cfg.n_sectors_max,
            sharpness_min=cfg.sharpness_min,
            sharpness_max=cfg.sharpness_max,
        )
    return sample_scm_task(
        rng,
        n,
        max_features=cfg.max_features,
        max_classes=cfg.max_classes,
        legacy=cfg.scm_legacy,
    )


def _sample_tasks_reusing_graphs(
    rng: np.random.Generator, cfg: PriorConfig, n_rows: int, batch_size: int
) -> list[Task]:
    """Fill a batch, drawing several SCM tasks per graph (task 48.12).

    Non-SCM draws are unaffected and go through :func:`sample_task` one at a time; only the
    SCM slots are grouped, since the financial and tree priors have no shared-graph structure
    to amortise.

    Args:
        rng: NumPy random generator.
        cfg: Prior configuration.
        n_rows: Rows per task, already drawn for this batch.
        batch_size: Tasks required.

    Returns:
        Exactly ``batch_size`` tasks, in draw order.
    """
    tasks: list[Task] = []
    while len(tasks) < batch_size:
        # Decide the slot's source the same way sample_task would, then either group it or
        # fall through. Drawing the source first keeps the mixture proportions intact.
        if (
            (cfg.p_tree and rng.random() < cfg.p_tree)
            or (cfg.p_regression and rng.random() < cfg.p_regression)
            or rng.random() < cfg.p_financial
        ):
            tasks.append(sample_task(rng, cfg, n_rows=n_rows))
            continue
        want = min(cfg.scm_reuse_graph, batch_size - len(tasks))
        tasks.extend(
            sample_scm_task_group(
                rng,
                n_rows,
                n_targets=want,
                max_features=cfg.max_features,
                max_classes=cfg.max_classes,
                legacy=cfg.scm_legacy,
            )[:want]
        )
    return tasks[:batch_size]


def _resample_until_learnable(
    rng: np.random.Generator, cfg: PriorConfig, n_rows: int, task: Task
) -> Task:
    """Return ``task`` if a cheap held-out learner clears the floor, else a resampled draw.

    Bounded by :attr:`PriorConfig.learnability_max_resamples`: a batch member is always
    returned, learnable or not, because silently shrinking a batch is a worse failure than
    keeping one hard task in it.
    """
    for _ in range(cfg.learnability_max_resamples):
        if is_learnable(task, auc_floor=cfg.learnability_auc_floor):
            return task
        task = sample_task(rng, cfg, n_rows=n_rows)
    return task


def sample_batch(rng: np.random.Generator, cfg: PriorConfig, batch_size: int) -> TaskBatch:
    """Draw a padded batch with a shared task size and context/query split.

    The task size is drawn per batch from ``cfg.n_rows_choices`` when set, because the
    base-rate floor scales with it and a fixed size caps how imbalanced any task can be.
    """
    n_rows = int(rng.choice(cfg.n_rows_choices)) if cfg.n_rows_choices else cfg.n_rows
    if cfg.scm_reuse_graph > 1:
        tasks = _sample_tasks_reusing_graphs(rng, cfg, n_rows, batch_size)
    else:
        tasks = [sample_task(rng, cfg, n_rows=n_rows) for _ in range(batch_size)]
    if cfg.p_learnability_filter > 0:
        tasks = [
            _resample_until_learnable(rng, cfg, n_rows, t)
            if rng.random() < cfg.p_learnability_filter
            else t
            for t in tasks
        ]
    n_ctx = int(rng.integers(int(cfg.min_ctx_frac * n_rows), int(cfg.max_ctx_frac * n_rows) + 1))
    n_ctx = min(max(n_ctx, 2), n_rows - 1)
    return collate(tasks, n_ctx=n_ctx, max_features=cfg.max_features)
