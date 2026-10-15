"""Mixture over priors plus the batch sampler used for pretraining."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fintfm.prior.base import Task, TaskBatch, collate, ensure_context_classes
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
from fintfm.prior.real_edgar import DEFAULT_PANEL_PATH, sample_real_edgar_task
from fintfm.prior.real_lendingclub import DEFAULT_PANEL_PATH as DEFAULT_LENDINGCLUB_PANEL_PATH
from fintfm.prior.real_lendingclub import sample_real_lendingclub_task
from fintfm.prior.real_mortgage import DEFAULT_PANEL_PATH as DEFAULT_MORTGAGE_PANEL_PATH
from fintfm.prior.real_mortgage import sample_real_mortgage_task
from fintfm.prior.real_panel import parse_real_panels, sample_real_panel_task
from fintfm.prior.scm import (
    sample_scm_regression_task,
    sample_scm_task,
    sample_scm_task_group,
)
from fintfm.prior.task_families import FAMILIES, sample_composition_task, sample_family_task
from fintfm.prior.tree import sample_tree_task
from fintfm.prior.trivial import sample_trivial_task

#: The constructions `PriorConfig.p_task_family` draws uniformly from: task 40.2's nine named
#: families plus the AND-composition (`sample_composition_task(mode="composed")`, task 40.4).
#: Confound-collider (task 40.5) is deliberately excluded -- its paired
#: observed/intervened output is a diagnostic instrument, not a single training task, and
#: folding in only its "observed" half would silently drop the property (an intact confound
#: path) the whole construction exists to test.
_TASK_FAMILY_CONSTRUCTIONS: tuple[str, ...] = (*FAMILIES, "composition_composed")

#: Bayes-AUC range `PriorConfig.p_task_family` draws from, per task -- spans the same
#: chance-to-near-certainty range `docs/results/FINDINGS.md` §42 established a prior must span
#: to teach anything, rather than fixing one difficulty for every draw.
_TASK_FAMILY_AUC_RANGE: tuple[float, float] = (0.6, 0.97)


def _sample_task_family_mixture(rng: np.random.Generator, n_rows: int, max_features: int) -> Task:
    """Draw one task from `_TASK_FAMILY_CONSTRUCTIONS`, uniformly, at a randomly-spanned AUC.

    Args:
        rng: NumPy random generator.
        n_rows: Rows to generate.
        max_features: Feature width to draw at; every construction here supports it.
    """
    construction = _TASK_FAMILY_CONSTRUCTIONS[int(rng.integers(len(_TASK_FAMILY_CONSTRUCTIONS)))]
    target_auc = float(rng.uniform(*_TASK_FAMILY_AUC_RANGE))
    if construction == "composition_composed":
        return sample_composition_task(
            rng, n_rows, n_features=max_features, target_component_auc=target_auc, mode="composed"
        )
    return sample_family_task(
        rng, construction, n_rows, n_features=max_features, target_auc=target_auc
    )


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
            (§113). Since audit AU1.13 every task's source is drawn from the mixture before
            any grouping, so the reuse value no longer changes the batch's source shares;
            before it, an SCM-heavy distortion (0.198 SCM share at reuse 1, 0.558 at reuse
            4) rode along with every reuse comparison.
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
        discretize_frac / n_noise_features / n_correlated_block_features / label_noise_rate:
            Four cheap realism augmentations Nori lists (task 48.6), passed straight through
            to every financial task drawn; see :func:`fintfm.prior.financial.sample_financial_task`
            for what each does. All zero/off by default, reproducing every prior checkpoint's
            behaviour.
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
        p_task_family: Probability of drawing one of ``prior/task_families.py``'s task 40.2-40.4
            constructions instead — uniformly among the nine named families
            (:data:`~fintfm.prior.task_families.FAMILIES`) plus the AND-composition
            (task 40.4), at a Bayes-AUC target drawn uniformly from ``(0.6, 0.97)`` per task, so
            difficulty spans a range rather than fixing one value. Task 40.7
            (``mechanism-diverse-prior``): the first pretraining use of that module, scoped only
            after 40.2-40.6 each validated on existing checkpoints (§137-§140). Zero by default,
            reproducing every checkpoint trained before this existed. Confound-collider
            (task 40.5) is deliberately excluded — its paired observed/intervened output is a
            diagnostic instrument, not a single training task.
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
        head_type: Forwarded to :func:`fintfm.prior.base.collate` — ``"binned"`` (default)
            bins a continuous target, ``"quantile"`` (task 48.3) z-scores it instead. Must
            match the checkpoint's own ``ModelConfig.head_type``, and ``"quantile"`` requires
            every sampled task to be a regression task (``p_regression=1.0``), since that
            head has no class-index path — :func:`fintfm.prior.base.collate` raises rather
            than silently coercing a classification task.
    """

    max_features: int = 24
    max_classes: int = 10
    scm_legacy: bool = False
    scm_reuse_graph: int = 1
    p_tree: float = 0.0
    p_financial: float = 0.7
    p_trivial: float = 0.0
    p_crossed: float = 0.0
    p_task_family: float = 0.0
    p_regression: float = 0.0
    identity_shuffle: bool = False
    discretize_frac: float = 0.0
    n_noise_features: int = 0
    n_correlated_block_features: int = 0
    label_noise_rate: float = 0.0
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
    min_ctx_per_class: int = 0
    """Context rows guaranteed to every class of a classification task, by moving rows between
    query and context once the batch's split is drawn
    (:func:`fintfm.prior.base.ensure_context_classes`). Audit AU1.16: random cut-points in the
    SCM and tree priors leave a class with two rows or fewer in about 31% of 256-row tasks, and
    such a class can land only in the query, where the loss asks for a label the context never
    showed. Applies to every classification source, not only SCM and tree. ``0`` (default)
    moves nothing and consumes no random draw, reproducing every checkpoint trained before this
    field existed; any other value changes the pretraining distribution."""
    head_type: str = "binned"
    p_real_edgar: float = 0.0
    """Probability of drawing a task sampled from real SEC EDGAR firm-quarters instead of a
    synthetic one (FinTFM-R, task A2/A3). Unlike every other ``p_*`` field, the task this
    source produces is not generated per draw -- it is a random subset of a fixed, finite real
    panel built by ``scripts/edgar/build_panel.py``; see
    :func:`fintfm.prior.real_edgar.sample_real_edgar_task` for why that is a sound design
    rather than a limitation. ``0.0`` (default) reproduces every checkpoint trained before
    this field existed, byte for byte: the branch is never taken, and no panel file is ever
    read. Checked before :attr:`p_financial` in :func:`sample_task`'s chain, so a non-zero
    value draws from the real panel at exactly the stated rate regardless of the other
    weights' sum."""
    real_edgar_panel_path: str = DEFAULT_PANEL_PATH
    """Where :func:`sample_task` reads the real EDGAR panel from when :attr:`p_real_edgar` is
    non-zero. Ignored otherwise -- a misconfigured path with ``p_real_edgar=0.0`` causes no
    error, since the file is never opened."""
    p_real_mortgage: float = 0.0
    """Probability of drawing a task from Freddie Mac's real loan-level mortgage panel instead
    of a synthetic one (FinTFM-R, task B3/B4). Present from the same commit as
    :attr:`p_real_edgar` per D17's "both real sources from the first commit" design, so adding
    a second real domain is additive rather than a rework. ``0.0`` (default) reproduces every
    checkpoint trained before this field existed, byte for byte. Checked alongside
    :attr:`p_real_edgar`, before :attr:`p_financial`, in :func:`sample_task`'s chain."""
    real_mortgage_panel_path: str = DEFAULT_MORTGAGE_PANEL_PATH
    """Where :func:`sample_task` reads the real mortgage panel from when
    :attr:`p_real_mortgage` is non-zero. Ignored otherwise -- a misconfigured path with
    ``p_real_mortgage=0.0`` causes no error, since the file is never opened."""
    p_real_lendingclub: float = 0.0
    """Probability of drawing a task from real LendingClub loans instead of a synthetic one
    (D17's third real source). Present alongside :attr:`p_real_edgar`/:attr:`p_real_mortgage`
    rather than a rework. ``0.0`` (default) reproduces every checkpoint trained before this
    field existed, byte for byte. Checked alongside the other two real sources, before
    :attr:`p_financial`, in :func:`sample_task`'s chain."""
    real_lendingclub_panel_path: str = DEFAULT_LENDINGCLUB_PANEL_PATH
    """Where :func:`sample_task` reads the real LendingClub panel from when
    :attr:`p_real_lendingclub` is non-zero. Ignored otherwise."""
    real_panels: str = ""
    """Further real sources as ``"path=prob,path=prob"``, read by the generic
    :func:`fintfm.prior.real_panel.sample_real_panel_task`. Each entry is an independent
    draw with its stated probability, checked after the dedicated ``p_real_*`` sources and before
    :attr:`p_financial`. ``""`` (default) parses nothing and opens nothing, so every checkpoint
    trained before this field existed is reproduced exactly."""


def _draw_source(rng: np.random.Generator, cfg: PriorConfig) -> tuple[str, str | None]:
    """Decide which prior the next task comes from, consuming only the source draws.

    Split out of :func:`sample_task` (audit AU1.13) so the graph-reusing batch path can draw a
    slot's source **once** and then dispatch on it. Before this, that path drew the source to
    decide whether to group a slot, left ``p_trivial``, ``p_crossed`` and ``p_task_family``
    out of the check, and then let :func:`sample_task` draw the source a second time -- so
    the realised SCM share at ``scm_reuse_graph=4`` was 0.558 against 0.198 at 1 for the
    same weights.

    The draws are made in exactly the order :func:`sample_task` always made them, and a
    weight of zero still consumes no draw, so ``sample_task`` reproduces its pre-split random
    stream byte for byte.

    Args:
        rng: NumPy random generator.
        cfg: Prior configuration.

    Returns:
        ``(source, panel_path)``. ``source`` is one of ``"tree"``, ``"regression"``,
        ``"trivial"``, ``"crossed"``, ``"task_family"``, ``"real_edgar"``,
        ``"real_mortgage"``, ``"real_lendingclub"``, ``"real_panel"``, ``"financial"`` or
        ``"scm"``; ``panel_path`` is the chosen panel for ``"real_panel"`` and ``None``
        otherwise.

    Raises:
        ValueError: If ``n_horizons`` is set with ``p_financial < 1``, checked before any
            draw exactly as :func:`sample_task` always did.
    """
    if cfg.n_horizons is not None and cfg.p_financial < 1.0:
        raise ValueError(
            "n_horizons requires p_financial=1.0; the SCM prior has no time axis and a "
            "mixed batch cannot carry a coherent survival likelihood"
        )
    for name, p in (
        ("tree", cfg.p_tree),
        ("regression", cfg.p_regression),
        ("trivial", cfg.p_trivial),
        ("crossed", cfg.p_crossed),
        ("task_family", cfg.p_task_family),
        ("real_edgar", cfg.p_real_edgar),
        ("real_mortgage", cfg.p_real_mortgage),
        ("real_lendingclub", cfg.p_real_lendingclub),
    ):
        if p and rng.random() < p:
            return name, None
    for path, p in parse_real_panels(cfg.real_panels):
        if rng.random() < p:
            return "real_panel", path
    if rng.random() < cfg.p_financial:
        return "financial", None
    return "scm", None


def _sample_from_source(
    rng: np.random.Generator, cfg: PriorConfig, source: str, panel_path: str | None, n: int
) -> Task:
    """Draw one task from the prior :func:`_draw_source` chose.

    Args:
        rng: NumPy random generator.
        cfg: Prior configuration.
        source: A source name returned by :func:`_draw_source`.
        panel_path: The panel for ``source="real_panel"``; ignored otherwise.
        n: Rows to generate.

    Returns:
        The task.

    Raises:
        ValueError: If ``n_horizons`` is set and the chosen prior has no time axis, or if
            ``source`` is unknown.
    """
    if cfg.n_horizons is not None and source not in ("financial", "trivial", "crossed"):
        reasons = {
            "tree": "n_horizons requires p_financial=1.0; the tree prior has no time axis",
            "regression": "n_horizons requires a survival label; a regression task has no event time",
            "task_family": "n_horizons requires p_financial=1.0; task_families has no time axis",
            "real_edgar": "n_horizons requires p_financial=1.0; real_edgar carries no horizon grid",
            "real_mortgage": (
                "n_horizons requires p_financial=1.0; real_mortgage carries no horizon grid"
            ),
            "real_lendingclub": (
                "n_horizons requires p_financial=1.0; real_lendingclub carries no horizon grid"
            ),
            "real_panel": "n_horizons requires p_financial=1.0; real panels carry no horizon grid",
        }
        if source in reasons:
            raise ValueError(reasons[source])
    if source == "tree":
        return sample_tree_task(rng, n, max_features=cfg.max_features, max_classes=cfg.max_classes)
    if source == "regression":
        return sample_scm_regression_task(
            rng, n, max_features=cfg.max_features, n_bins=cfg.max_classes
        )
    if source == "trivial":
        return sample_trivial_task(rng, n, max_features=min(8, cfg.max_features))
    if source == "crossed":
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
    if source == "task_family":
        return _sample_task_family_mixture(rng, n, max_features=cfg.max_features)
    # Every real source gets the synthetic financial prior's positive floor (audit AU1.14), met
    # with real positive rows, so a rare-label panel is enriched rather than relabelled.
    min_positives = math.ceil(cfg.min_expected_positives)
    if source == "real_edgar":
        return sample_real_edgar_task(
            rng, n, max_features=cfg.max_features, panel_path=cfg.real_edgar_panel_path,
            min_positives=min_positives,
        )
    if source == "real_mortgage":
        return sample_real_mortgage_task(
            rng, n, max_features=cfg.max_features, panel_path=cfg.real_mortgage_panel_path,
            min_positives=min_positives,
        )
    if source == "real_lendingclub":
        return sample_real_lendingclub_task(
            rng, n, max_features=cfg.max_features, panel_path=cfg.real_lendingclub_panel_path,
            min_positives=min_positives,
        )
    if source == "real_panel":
        assert panel_path is not None
        return sample_real_panel_task(
            rng, n, panel_path=panel_path, max_features=cfg.max_features,
            min_positives=min_positives,
        )
    if source == "financial":
        return sample_financial_task(
            rng,
            n,
            max_features=cfg.max_features,
            n_horizons=cfg.n_horizons,
            identity_shuffle=cfg.identity_shuffle,
            discretize_frac=cfg.discretize_frac,
            n_noise_features=cfg.n_noise_features,
            n_correlated_block_features=cfg.n_correlated_block_features,
            label_noise_rate=cfg.label_noise_rate,
            min_expected_positives=cfg.min_expected_positives,
            absolute_rate_floor=cfg.absolute_rate_floor,
            rate_ceiling=cfg.rate_ceiling,
            n_sectors_max=cfg.n_sectors_max,
            sharpness_min=cfg.sharpness_min,
            sharpness_max=cfg.sharpness_max,
        )
    if source == "scm":
        return sample_scm_task(
            rng,
            n,
            max_features=cfg.max_features,
            max_classes=cfg.max_classes,
            legacy=cfg.scm_legacy,
        )
    raise ValueError(f"unknown prior source {source!r}")


def sample_task(rng: np.random.Generator, cfg: PriorConfig, n_rows: int | None = None) -> Task:
    """Draw one task from the mixture prior.

    Raises:
        ValueError: If ``n_horizons`` is set with ``p_financial < 1``, which would produce
            batches mixing survival and binary-only tasks.
    """
    n = cfg.n_rows if n_rows is None else n_rows
    source, panel_path = _draw_source(rng, cfg)
    return _sample_from_source(rng, cfg, source, panel_path, n)


def _sample_tasks_reusing_graphs(
    rng: np.random.Generator, cfg: PriorConfig, n_rows: int, batch_size: int
) -> list[Task]:
    """Fill a batch, drawing several SCM tasks per graph (task 48.12).

    Every task's source is drawn **first and once**, by :func:`_draw_source`, exactly as
    :func:`sample_task` would draw it (audit AU1.13). Only then are the tasks that came up SCM
    filled, ``scm_reuse_graph`` at a time from one shared graph; every other source is sampled
    one task at a time, since the financial and tree priors have no shared-graph structure to
    amortise. So the batch's source mix follows the configured mixture whatever the reuse
    value, and reuse changes only which SCM tasks share a graph.

    Before AU1.13 this function decided whether a *slot* was SCM, left ``p_trivial``,
    ``p_crossed`` and ``p_task_family`` out of that check, drew the source a second time inside
    :func:`sample_task`, and filled each SCM slot with several tasks -- measured at p_financial
    0.7, p_tree 0.2, p_task_family 0.2, an SCM share of 0.198 at reuse 1 against 0.558 at
    reuse 4. Any comparison across reuse values made before the fix changed two things at once.

    Args:
        rng: NumPy random generator.
        cfg: Prior configuration.
        n_rows: Rows per task, already drawn for this batch.
        batch_size: Tasks required.

    Returns:
        Exactly ``batch_size`` tasks, in draw order.
    """
    sources = [_draw_source(rng, cfg) for _ in range(batch_size)]
    tasks: list[Task | None] = [None] * batch_size
    for i, (source, panel_path) in enumerate(sources):
        if source != "scm":
            tasks[i] = _sample_from_source(rng, cfg, source, panel_path, n_rows)
    scm_slots = [i for i, (source, _) in enumerate(sources) if source == "scm"]
    while scm_slots:
        want = min(cfg.scm_reuse_graph, len(scm_slots))
        group = sample_scm_task_group(
            rng,
            n_rows,
            n_targets=want,
            max_features=cfg.max_features,
            max_classes=cfg.max_classes,
            legacy=cfg.scm_legacy,
        )[:want]
        # A group comes back short only if a graph has too few nodes for distinct targets;
        # the loop then draws another graph for the slots still empty.
        for task in group:
            tasks[scm_slots.pop(0)] = task
    return [t for t in tasks if t is not None]


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
    if cfg.min_ctx_per_class > 0:
        tasks = [ensure_context_classes(rng, t, n_ctx, cfg.min_ctx_per_class) for t in tasks]
    return collate(tasks, n_ctx=n_ctx, max_features=cfg.max_features, head_type=cfg.head_type)
