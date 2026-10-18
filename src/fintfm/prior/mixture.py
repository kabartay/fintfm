"""Mixture over priors plus the batch sampler used for pretraining."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from fintfm.prior.base import (
    Task,
    TaskBatch,
    collate,
    condition_task_features,
    ensure_context_classes,
    permute_task_classes,
    truncate_query,
)
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
from fintfm.prior.gated import sample_gated_task
from fintfm.prior.knobs import (
    TaskKnobs,
    apply_task_knobs,
    check_knobs,
    draw_extra_counts,
    draw_width,
    knobs_from,
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
    permute_classes: bool = False
    """Relabel every classification task's classes by a fresh random permutation once the
    batch's split is drawn (audit AU5.5; :func:`fintfm.prior.base.permute_task_classes`).
    The SCM prior's classes are ordinal bands and the financial prior's default is always
    class 1, so without it a model can learn which class *index* is rare instead of reading
    the context. Context and query labels share the permutation; regression tasks are left
    alone because their bins are ordinal. Refused with :attr:`n_horizons` (a survival label is
    tied to its period) and with ``head_type="quantile"`` (every task is regression, so the
    flag could never act). ``False`` (default) consumes no draw and reproduces every existing
    checkpoint's batches byte for byte."""
    feature_transforms: tuple[str, ...] = ()
    """Inference-time feature transforms to train under (audit AU5.6). Each task draws one
    uniformly from this tuple and has it fitted on its **context rows only** and applied to all
    its rows (:func:`fintfm.prior.base.condition_task_features`), mirroring
    :class:`~fintfm.inference.preprocess.FeatureConditioner` at ``fit``. Names are
    ``"none"``, ``"rank"``, ``"power"`` and ``"winsor"``. ``()`` (default) transforms nothing
    and consumes no draw -- today's behaviour, where the model's own context z-score is the
    only normalisation. A one-element tuple applies that transform to every task without a
    draw, so ``("none",)`` is also byte-identical to the default."""
    max_query_rows: int | None = None
    """Keep at most this many query rows per task once the split is drawn (audit AU5.4;
    :func:`fintfm.prior.base.truncate_query`). Contexts up to 4,096 rows are already reachable
    through :attr:`n_rows_choices` -- ``int(0.9 * 4552) == 4096`` at the default
    :attr:`max_ctx_frac` -- but at that size a task also carries 456-3,187 query rows, and
    the row encoder's cost is paid on context plus query. This caps the query so a long
    context costs what its context costs. ``None`` (default) keeps every row and consumes no
    draw, reproducing every existing checkpoint's batches byte for byte."""
    source_weights: dict[str, float] | None = None
    """Draw each task's source from one categorical over these weights, normalised to sum to
    1, instead of the sequential chain (design review DR2.1). The chain tries each ``p_*`` in
    turn on what the earlier ones left, so every ``p_*`` is a *conditional* share: at
    ``p_tree`` 0.3 and ``p_financial`` 0.7 the financial share is 0.49, not 0.7
    (``docs/results/FINDINGS.md`` §166). Here a weight is the share. Keys are
    :data:`SOURCE_NAMES` or ``"real_panel:<path>"``; an unknown name, a negative or
    non-finite weight, or an all-zero mapping is refused. Every task costs exactly one
    ``rng.random()``. While it is set the ``p_*`` fields and :attr:`real_panels` are not
    read, so a non-default value in any of them is refused rather than left silently inert,
    and with :attr:`n_horizons` only ``financial``, ``trivial`` and ``crossed`` may carry
    weight. ``None`` (default) keeps the sequential chain and its random stream byte for
    byte."""
    pad_to_batch_width: bool = False
    """Pad each batch's feature axis to its widest task rather than to :attr:`max_features`
    (design review DR6.1; :func:`fintfm.prior.base.collate`). The financial prior draws 4-136
    features per task, so at ``max_features=136`` about half of all cell tokens are padding,
    and the row-within-feature and column attention stages run on them. Padding is masked
    everywhere in the model, and ``fintfm-train`` also sets
    :attr:`fintfm.modeling.model.FinancialTFM.column_ids_full_width` so column identities are
    drawn as at full width; a task's outputs are then the same at either width (to float
    rounding). It saves compute, not tasks: the batch's tasks and random stream are
    unchanged. ``False`` (default) pads to ``max_features``, byte for byte as before."""
    financial_gate_prob: float = 0.0
    """Probability a financial task's label is **gated** (design review DR2.2;
    :func:`fintfm.prior.financial.sample_financial_task`'s ``gate_prob``): one of negative
    equity, negative working capital, interest coverage below one, or a random ratio
    threshold holds nearly every default, the existing driver score ranks firms inside it,
    and a small share of defaults leaks outside. V4FinBench's label has that shape and the
    financial prior's sigmoid label does not (``docs/design/DESIGN_REVIEW_2026-10-16.md``
    §1.1, §1.5). Refused with :attr:`n_horizons`. ``0.0`` (default) consumes no draw and
    reproduces every checkpoint's batches byte for byte. The generic gated prior is a source
    of its own, ``"gated"``, reached through :attr:`source_weights` only."""
    feature_ratio_bands: tuple[tuple[float, float, float], ...] = ()
    """``(weight, lo, hi)`` bands of features per row (design review DR2.3,
    :mod:`fintfm.prior.knobs`). Each SCM, regression, tree, task-family, gated and financial
    task draws a band by weight, a ratio uniformly inside it, and takes that ratio times its
    rows as its width, clipped to the source's limits. Today every width is uniform on
    ``[2, max_features]`` at every row count, so a 256-row task reaches 0.53 features per row
    while peers skew narrow (§1.5). ``()`` (default) keeps each source's own draw."""
    p_binary: float | None = None
    """Probability an SCM or tree task is binary, its class count otherwise uniform on
    ``[3, max_classes]`` (DR2.3; the review proposes 0.7, against SCM's 11% today). ``None``
    (default) keeps the uniform draw on ``[2, max_classes]``."""
    imbalance_draws: bool = False
    """Draw SCM and tree class shares explicitly (DR2.3): a binary minority share from
    LogU(4/N, 0.5) on a random side, multiclass shares from a Dirichlet over a floor of 4 rows
    per class, cut at the latent's quantiles. Today imbalance comes only from random cut
    points, leaving a class with two rows or fewer in 5 of 20 SCM and 7 of 20 tree tasks
    (§1.5, MEASURED). ``False`` (default) keeps the quantile-or-random-cut coin."""
    categorical_mode: str = "ordered"
    """``"unordered"`` replaces the SCM and tree priors' ordered 2-7-level quantile codes with
    permuted codes of LogInt(2, 64) levels at Zipf frequencies, on a drawn share of columns,
    and applies the same to task-family and gated tasks (DR2.3). ``"ordered"`` (default)
    keeps each source's own coding."""
    missing_mode: str = "mcar"
    """``"mixed"`` replaces the general priors' MCAR step with one of MCAR, MAR or MNAR per
    task at a LogU(0.01, 0.3) rate (DR2.3). ``"mcar"`` (default) keeps each source's own."""
    tie_column_prob: float = 0.0
    """Per numeric column probability, in the general priors, of ties, a point mass or zero
    inflation (DR2.3) -- the structure that survives the rank transform, where heavy tails do
    not (§1.5). ``0.0`` (default) touches no column."""
    irrelevant_frac_max: float = 0.0
    """Upper end of the share of a general task's width added as row-permuted copies of its
    own columns, irrelevant to the label (DR2.3). ``0.0`` (default) adds none."""
    duplicate_frac_max: float = 0.0
    """Upper end of the share of a general task's width added as near-duplicate columns
    (DR2.3). ``0.0`` (default) adds none."""
    label_flip_max: float = 0.0
    """Upper end of the per-task rate at which a general classification task's labels flip to
    another class (DR2.3). The financial prior keeps its own :attr:`label_noise_rate`.
    ``0.0`` (default) flips nothing."""


#: Every source :attr:`PriorConfig.source_weights` may name besides a real panel, in the order
#: the weighted draw lays them out: the sequential chain's order, then ``financial`` and
#: ``scm``. A real panel is named ``"real_panel:<path>"`` and sits before ``financial``, in
#: path order, so two equal mappings written in different orders draw the same stream.
SOURCE_NAMES: tuple[str, ...] = (
    "tree", "regression", "trivial", "crossed", "task_family", "real_edgar", "real_mortgage",
    "real_lendingclub", "gated", "financial", "scm",
)

#: Sources only :attr:`PriorConfig.source_weights` can reach: they have no ``p_*`` field, so
#: the sequential chain never draws them.
_WEIGHTED_ONLY_SOURCES: tuple[str, ...] = ("gated",)

#: Sources the DR2.3 column- and label-side knobs apply to (:func:`apply_task_knobs`), and
#: the narrowest width each supports, for the band draw. The financial prior takes the band
#: width only; its observation model and label noise are its own.
_KNOB_SOURCES: frozenset[str] = frozenset({"scm", "regression", "tree", "task_family", "gated"})
_MIN_WIDTH: dict[str, int] = {
    "scm": 2, "regression": 2, "tree": 2, "task_family": 2, "gated": 2, "financial": 4,
}

#: Prefix of a real panel's key in :attr:`PriorConfig.source_weights`.
REAL_PANEL_PREFIX = "real_panel:"

#: The sources a weighted draw may give weight to when :attr:`PriorConfig.n_horizons` is set:
#: exactly the ones the sequential chain can reach at its required ``p_financial`` 1.0 and
#: :func:`_sample_from_source` does not refuse under ``n_horizons``.
_HORIZON_SOURCES: frozenset[str] = frozenset({"financial", "trivial", "crossed"})


#: The names :attr:`PriorConfig.feature_transforms` accepts -- the values of
#: :data:`fintfm.inference.preprocess.FeatureTransform`, so pretraining can draw exactly the
#: transforms inference can apply.
FEATURE_TRANSFORMS: tuple[str, ...] = ("none", "rank", "power", "winsor")


def _check_levers(cfg: PriorConfig) -> None:
    """Refuse audit Phase 5 lever settings that are malformed or could never take effect.

    Checked before any draw, so a bad configuration fails at the first batch rather than
    training on a knob that silently did nothing -- the failure ``scm_legacy`` at
    ``p_financial=1.0`` once cost a 6,000-step run.

    Args:
        cfg: Prior configuration.

    Raises:
        ValueError: On an unknown transform name, a non-positive ``max_query_rows``,
            ``permute_classes`` combined with ``n_horizons`` or ``head_type="quantile"``, any
            ``source_weights`` refusal of :func:`_weighted_shares`, a malformed DR2.3 knob
            (:func:`fintfm.prior.knobs.check_knobs`), or ``financial_gate_prob`` outside
            ``[0, 1]`` or combined with ``n_horizons``.
    """
    if cfg.source_weights is not None:
        _weighted_shares(cfg)
    knobs = knobs_from(cfg)
    if knobs is not None:
        check_knobs(knobs)
    if not (math.isfinite(cfg.financial_gate_prob) and 0.0 <= cfg.financial_gate_prob <= 1.0):
        raise ValueError(f"financial_gate_prob must lie in [0, 1], got {cfg.financial_gate_prob}")
    if cfg.financial_gate_prob > 0 and cfg.n_horizons is not None:
        raise ValueError(
            "financial_gate_prob cannot be combined with n_horizons: the gated label has no "
            "survival form"
        )
    unknown = [k for k in cfg.feature_transforms if k not in FEATURE_TRANSFORMS]
    if unknown:
        raise ValueError(
            f"unknown feature_transforms {unknown}; valid names are {list(FEATURE_TRANSFORMS)}"
        )
    if cfg.max_query_rows is not None and cfg.max_query_rows < 1:
        raise ValueError(f"max_query_rows must be at least 1, got {cfg.max_query_rows}")
    if cfg.permute_classes and cfg.n_horizons is not None:
        raise ValueError(
            "permute_classes cannot be combined with n_horizons: a survival task's label is "
            "exactly period != CENSORED, and relabelling it would break that identity"
        )
    if cfg.permute_classes and cfg.head_type == "quantile":
        raise ValueError(
            "permute_classes is inert with head_type='quantile': every task is regression, "
            "and regression bins are never permuted"
        )


def _sequential_sources(cfg: PriorConfig) -> tuple[tuple[str, float], ...]:
    """The named sources :func:`_draw_source` tries first, in its draw order, with weights.

    One list for the draw and for :func:`source_shares`, so the realised shares a checkpoint
    records cannot drift from the order the draw actually uses.

    Args:
        cfg: Prior configuration.

    Returns:
        ``(source, weight)`` pairs, before the real panels, ``financial`` and ``scm``.
    """
    return (
        ("tree", cfg.p_tree),
        ("regression", cfg.p_regression),
        ("trivial", cfg.p_trivial),
        ("crossed", cfg.p_crossed),
        ("task_family", cfg.p_task_family),
        ("real_edgar", cfg.p_real_edgar),
        ("real_mortgage", cfg.p_real_mortgage),
        ("real_lendingclub", cfg.p_real_lendingclub),
    )


def parse_source_weights(spec: str) -> dict[str, float] | None:
    """Parse a ``--source-weights`` value, ``"tree=0.25,scm=0.4,financial=0.35"``.

    Only the syntax is checked here; names and weights are checked by :func:`_weighted_shares`
    when the prior is used, so a file and a flag are refused by the same code.

    Args:
        spec: Comma-separated ``source=weight`` entries. A real panel's key is
            ``real_panel:<path>``, split at its last ``=``.

    Returns:
        The mapping in the order written, or ``None`` for an empty spec, which keeps the
        sequential draw.

    Raises:
        ValueError: On an entry without ``=``, a weight that is not a number, or a source
            named twice.
    """
    out: dict[str, float] = {}
    for raw in spec.split(","):
        entry = raw.strip()
        if not entry:
            continue
        name, sep, value = entry.rpartition("=")
        name = name.strip()
        if not sep or not name:
            raise ValueError(f"source_weights entry {entry!r} is not of the form source=weight")
        try:
            weight = float(value)
        except ValueError:
            raise ValueError(
                f"source_weights weight for {name!r} is not a number: {value!r}"
            ) from None
        if name in out:
            raise ValueError(f"source_weights names {name!r} twice")
        out[name] = weight
    return out or None


def _weighted_shares(cfg: PriorConfig) -> tuple[dict[str, float], dict[str, float]]:
    """Validate :attr:`PriorConfig.source_weights` and normalise it, in draw order.

    Args:
        cfg: Prior configuration with ``source_weights`` set.

    Returns:
        ``(configured, realised)``: the weights as given and the normalised shares, both keyed
        in :data:`SOURCE_NAMES` order with real panels, by path, before ``financial``. Zero
        weights are kept, so the record states every source the run named.

    Raises:
        ValueError: On an empty mapping, an unknown source name, a ``real_panel:`` key with no
            path, a negative or non-finite weight, weights summing to zero, a non-default
            ``p_*`` field or non-empty ``real_panels`` (neither is read in this mode), or
            ``n_horizons`` set with weight on a source that has no time axis.
    """
    weights = cfg.source_weights
    if not weights:
        raise ValueError("source_weights is empty; give it at least one source, or None")
    unknown = [
        k for k in weights
        if k not in SOURCE_NAMES
        and not (k.startswith(REAL_PANEL_PREFIX) and k[len(REAL_PANEL_PREFIX):].strip())
    ]
    if unknown:
        raise ValueError(
            f"unknown source_weights source(s) {unknown}; valid names are {list(SOURCE_NAMES)} "
            f"or '{REAL_PANEL_PREFIX}<path>'"
        )
    bad = {k: w for k, w in weights.items() if not (math.isfinite(float(w)) and float(w) >= 0)}
    if bad:
        raise ValueError(f"source_weights must be finite and non-negative, got {bad}")
    total = math.fsum(float(w) for w in weights.values())
    if total <= 0:
        raise ValueError("source_weights sum to zero, so no source could ever be drawn")
    defaults = PriorConfig()
    inert = [
        f for f in ("p_tree", "p_regression", "p_trivial", "p_crossed", "p_task_family",
                    "p_real_edgar", "p_real_mortgage", "p_real_lendingclub", "p_financial")
        if getattr(cfg, f) != getattr(defaults, f)
    ]
    if cfg.real_panels:
        inert.append("real_panels")
    if inert:
        raise ValueError(
            f"{inert} would be silently ignored: source_weights replaces the sequential draw "
            "that reads them. Give each source its weight in source_weights and leave these "
            "at their defaults"
        )
    if cfg.n_horizons is not None:
        timeless = [k for k, w in weights.items() if float(w) > 0 and k not in _HORIZON_SOURCES]
        if timeless:
            raise ValueError(
                f"n_horizons requires every weighted source to carry a survival label; "
                f"{timeless} have no time axis (allowed: {sorted(_HORIZON_SOURCES)})"
            )
    panels = sorted(k for k in weights if k.startswith(REAL_PANEL_PREFIX))
    order = [*(k for k in SOURCE_NAMES[:-2] if k in weights), *panels,
             *(k for k in SOURCE_NAMES[-2:] if k in weights)]
    configured = {k: float(weights[k]) for k in order}
    return configured, {k: w / total for k, w in configured.items()}


def source_shares(cfg: PriorConfig) -> dict[str, Any]:
    """The configured source weights and the shares :func:`_draw_source` actually produces.

    By default the draw is sequential: each source in turn takes its weight of whatever the
    earlier ones left, ``financial`` takes its weight of the rest, and ``scm`` takes the
    remainder. So a weight is a *conditional* probability, not a share. At ``p_tree`` 0.3 and
    ``p_financial`` 0.7 the financial share is 0.7 x 0.7 = 0.49, not 0.7, and the SCM share
    is 0.21 -- the confound ``docs/results/FINDINGS.md`` §166 found behind §116's tree-prior
    loss on credit. With :attr:`PriorConfig.source_weights` set (design review DR2.1) the draw
    is one categorical and the realised shares are the normalised weights. Recorded in every
    checkpoint (design review DR1.2) so a run states the mixture it saw.

    Computed analytically from the weights in draw order, so it costs nothing and consumes
    no random draw. A sequential weight is clipped to ``[0, 1]`` exactly as
    ``rng.random() < p`` clips it. The shares are per source draw:
    :attr:`PriorConfig.p_learnability_filter` resamples rejected tasks, which can move the
    shares of *accepted* tasks away from these.

    Args:
        cfg: Prior configuration.

    Returns:
        ``{"mode": "sequential" | "weighted", "configured": weights, "realised": shares}``,
        weights and shares keyed by source name in draw order (a real panel as
        ``"real_panel:<path>"``). In sequential mode ``configured`` has no ``scm`` entry,
        since SCM has no weight of its own there; ``realised`` sums to 1 in both modes.

    Raises:
        ValueError: If ``source_weights`` is set and invalid (:func:`_weighted_shares`).
    """
    if cfg.source_weights is not None:
        configured, realised = _weighted_shares(cfg)
        return {"mode": "weighted", "configured": configured, "realised": realised}
    weighted = [
        *_sequential_sources(cfg),
        *((f"real_panel:{path}", p) for path, p in parse_real_panels(cfg.real_panels)),
        ("financial", cfg.p_financial),
    ]
    remaining = 1.0
    realised: dict[str, float] = {}
    for name, p in weighted:
        share = remaining * (min(float(p), 1.0) if p > 0 else 0.0)
        realised[name] = share
        remaining -= share
    realised["scm"] = remaining
    return {
        "mode": "sequential",
        "configured": {name: float(p) for name, p in weighted},
        "realised": realised,
    }


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

    With :attr:`PriorConfig.source_weights` set (design review DR2.1) the chain is replaced by
    one categorical draw over the normalised weights: exactly one ``rng.random()``, mapped
    onto the cumulative shares in :func:`_weighted_shares`' order. Zero-weight sources are
    left out of that table, so none can be drawn through rounding at the top of it.

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
            draw exactly as :func:`sample_task` always did, or if ``source_weights`` is set
            and invalid (:func:`_weighted_shares`).
    """
    if cfg.source_weights is not None:
        _, realised = _weighted_shares(cfg)
        names = [k for k, w in realised.items() if w > 0]
        cdf = np.cumsum([realised[k] for k in names])
        i = min(int(np.searchsorted(cdf, rng.random(), side="right")), len(names) - 1)
        if names[i].startswith(REAL_PANEL_PREFIX):
            return "real_panel", names[i][len(REAL_PANEL_PREFIX):]
        return names[i], None
    if cfg.n_horizons is not None and cfg.p_financial < 1.0:
        raise ValueError(
            "n_horizons requires p_financial=1.0; the SCM prior has no time axis and a "
            "mixed batch cannot carry a coherent survival likelihood"
        )
    for name, p in _sequential_sources(cfg):
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

    With any DR2.3 knob on (:mod:`fintfm.prior.knobs`), a band width is drawn first for
    the sources that take one, the prior is called with the knobs, and the column- and
    label-side knobs are applied to the general sources' tasks. With every knob off, the
    prior is called exactly as before the knobs existed and no extra draw is made.

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
            "gated": "n_horizons requires a survival label; a gated task has no event time",
        }
        if source in reasons:
            raise ValueError(reasons[source])
    knobs = knobs_from(cfg)
    if knobs is None or source not in _MIN_WIDTH:
        return _dispatch_source(rng, cfg, source, panel_path, n, None, None)
    width: int | None = None
    extra: tuple[int, int] | None = None
    band: int | None = None
    if knobs.feature_ratio_bands:
        width, band = draw_width(rng, knobs, n, _MIN_WIDTH[source], cfg.max_features)
        if source in _KNOB_SOURCES:
            # irrelevant and duplicate columns are carved out of the band's width, not added
            extra = draw_extra_counts(rng, knobs, width)
            width = max(_MIN_WIDTH[source], width - sum(extra))
    task = _dispatch_source(rng, cfg, source, panel_path, n, knobs, width)
    if source in _KNOB_SOURCES:
        task = apply_task_knobs(
            rng, task, knobs, cfg.max_features, extra, label_knobs=source != "regression"
        )
    if band is not None:
        task.meta = {**(task.meta or {}), "width_band": band}
    return task


def _dispatch_source(
    rng: np.random.Generator,
    cfg: PriorConfig,
    source: str,
    panel_path: str | None,
    n: int,
    knobs: TaskKnobs | None,
    width: int | None,
) -> Task:
    """Call the prior for ``source``; :func:`_sample_from_source`'s dispatch table.

    Args:
        rng: NumPy random generator.
        cfg: Prior configuration.
        source: A source name returned by :func:`_draw_source`.
        panel_path: The panel for ``source="real_panel"``; ignored otherwise.
        n: Rows to generate.
        knobs: DR2.3's knobs, or ``None``, which calls every prior exactly as before them.
        width: A band-drawn width, or ``None`` for the prior's own draw.

    Returns:
        The task.

    Raises:
        ValueError: If ``source`` is unknown.
    """
    if source == "tree":
        return sample_tree_task(
            rng, n, max_features=cfg.max_features, max_classes=cfg.max_classes, knobs=knobs,
            n_features=width,
        )
    if source == "regression":
        return sample_scm_regression_task(
            rng, n, max_features=cfg.max_features, n_bins=cfg.max_classes, knobs=knobs,
            n_features=width,
        )
    if source == "gated":
        return sample_gated_task(
            rng, n, max_features=cfg.max_features, knobs=knobs, n_features=width
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
        return _sample_task_family_mixture(
            rng, n, max_features=cfg.max_features if width is None else width
        )
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
            n_features=width,
            gate_prob=cfg.financial_gate_prob,
        )
    if source == "scm":
        return sample_scm_task(
            rng, n, max_features=cfg.max_features, max_classes=cfg.max_classes,
            legacy=cfg.scm_legacy, knobs=knobs, n_features=width,
        )
    raise ValueError(f"unknown prior source {source!r}")


def sample_task(rng: np.random.Generator, cfg: PriorConfig, n_rows: int | None = None) -> Task:
    """Draw one task from the mixture prior.

    Raises:
        ValueError: If ``n_horizons`` is set with ``p_financial < 1`` (or, with
            ``source_weights``, with weight on a source that has no time axis), which would
            produce batches mixing survival and binary-only tasks; or if ``source_weights``
            is invalid.
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
    value, and reuse changes only which SCM tasks share a graph. That holds in both source
    modes, sequential and :attr:`PriorConfig.source_weights`, since both go through
    :func:`_draw_source`.

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
    knobs = knobs_from(cfg)
    tasks: list[Task | None] = [None] * batch_size
    for i, (source, panel_path) in enumerate(sources):
        if source != "scm":
            tasks[i] = _sample_from_source(rng, cfg, source, panel_path, n_rows)
    scm_slots = [i for i, (source, _) in enumerate(sources) if source == "scm"]
    while scm_slots:
        want = min(cfg.scm_reuse_graph, len(scm_slots))
        if knobs is None:
            group = sample_scm_task_group(
                rng,
                n_rows,
                n_targets=want,
                max_features=cfg.max_features,
                max_classes=cfg.max_classes,
                legacy=cfg.scm_legacy,
            )[:want]
        else:
            # DR2.3: one band width per graph, since the group's tasks share their columns
            width: int | None = None
            extra: tuple[int, int] | None = None
            if knobs.feature_ratio_bands:
                width, _ = draw_width(rng, knobs, n_rows, _MIN_WIDTH["scm"], cfg.max_features)
                extra = draw_extra_counts(rng, knobs, width)
                width = max(_MIN_WIDTH["scm"], width - sum(extra))
            group = [
                apply_task_knobs(rng, t, knobs, cfg.max_features, extra)
                for t in sample_scm_task_group(
                    rng, n_rows, n_targets=want, max_features=cfg.max_features,
                    max_classes=cfg.max_classes, legacy=cfg.scm_legacy, knobs=knobs,
                    n_features=width,
                )[:want]
            ]
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

    Once the split is drawn, the audit Phase 5 levers apply in a fixed order, each consuming
    random draws only when switched on: the class floor, query truncation
    (:attr:`PriorConfig.max_query_rows`, no draw), the feature transform
    (:attr:`PriorConfig.feature_transforms`, one draw per task when there is a choice), then
    class permutation (:attr:`PriorConfig.permute_classes`, one draw per classification task).
    With all of them off the random stream and the batch are exactly what they were before the
    levers existed.

    Args:
        rng: NumPy random generator; the only source of randomness.
        cfg: Prior configuration.
        batch_size: Tasks in the batch.

    Returns:
        The collated batch.
    """
    _check_levers(cfg)
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
    if cfg.max_query_rows is not None:
        tasks = [truncate_query(t, n_ctx, cfg.max_query_rows) for t in tasks]
    if cfg.feature_transforms:
        kinds = cfg.feature_transforms
        tasks = [
            condition_task_features(
                t, n_ctx, kinds[int(rng.integers(len(kinds)))] if len(kinds) > 1 else kinds[0]
            )
            for t in tasks
        ]
    if cfg.permute_classes:
        tasks = [permute_task_classes(rng, t) for t in tasks]
    return collate(
        tasks,
        n_ctx=n_ctx,
        max_features=cfg.max_features,
        head_type=cfg.head_type,
        pad_to_batch_width=cfg.pad_to_batch_width,
    )
