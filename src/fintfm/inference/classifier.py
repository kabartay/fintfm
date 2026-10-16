"""scikit-learn-compatible in-context classifier wrapping :class:`FinancialTFM`.

``fit`` only stores the training table as context (no gradient steps);
``predict``/``predict_proba`` run it through the frozen pretrained network
alongside the query rows. This mirrors how TabPFN-style models are used.

**How the context is chosen matters more than the architecture.** Tanna et al. (2026),
*Data Presentation Over Architecture* (arXiv:2605.18635), benchmark seven
context-construction strategies for credit-risk TFMs and find balanced and hybrid sampling
worth 3-4 AUC points over uniform sampling — a gap wider than the spread between model
families. That is why :class:`ContextStrategy` exists and why the default is ``"balanced"``
(``docs/results/FINDINGS.md`` §5).

**The premise held and the conclusion did not.** Measured on the V4FinBench out-of-time
split, the ordering is reversed and three times larger: **uniform beats balanced by 10-12
mean AUC points** at every context size tested, and twelve in-context defaults outrank 1,122
(§29). The mechanism appears to be the *majority* class — a balanced 2,000-row context spends
half its budget on 1,000 of 71,500 non-defaulters, so the model's picture of a healthy firm
comes from 1.4% of that class. The default is unchanged only because §29 measured the
six-horizon survival path and the binary evidence behind it has not been re-measured
(decision D9, ``openspec/changes/revisit-context-strategy``). **Resolved in §35:** on the binary path the strategies are
nearly tied, because these smaller panels do not have enough positives for "balanced" to
actually reach 50/50 — the effect scales with how extreme the rebalancing is, not with the
strategy's name. So the default is now ``"uniform"``: never worse than balanced in any
measurement here, and blind, so it keeps batch independence.

§31 then showed *which* rows is the only remaining lever on the dominant part of the
out-of-time gap, since more rows do not help and larger pretraining tasks are priced out.
Hence ``"retrieval"``; see :mod:`fintfm.inference.retrieval`, including the batch-independence
property it deliberately trades away.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Literal, get_args

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from fintfm.inference.preprocess import FeatureConditioner, FeatureTransform, SVDAugmenter
from fintfm.inference.retrieval import (
    distance_stats,
    group_queries,
    normalise_for_distance,
    prototype_context,
    retrieve,
)
from fintfm.modeling.hazard import base_rate_shift, shift_cumulative_pd
from fintfm.modeling.model import FinancialTFM

ContextStrategy = Literal["balanced", "hybrid", "uniform", "retrieval", "prototype"]
#: What :meth:`FinancialTFMClassifier.fit` does with a table wider than the checkpoint.
MaxFeaturesStrategy = Literal["raise", "subsample"]


def _finetune_rows(
    rng: np.random.Generator,
    groups: np.ndarray,
    uniq: np.ndarray,
    n_q_groups: int,
    n_ctx: int,
    n_q: int,
    y: np.ndarray | None = None,
    pos_frac: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Draw one fine-tuning task's context and query rows, with no group on both sides.

    Args:
        rng: Random generator.
        groups: ``(n,)`` integer code per row, indexing ``uniq``.
        uniq: The distinct groups.
        n_q_groups: Groups held out to supply queries.
        n_ctx: Context rows to draw.
        n_q: Query rows to draw.
        y: ``(n,)`` coded labels; needed only with ``pos_frac``.
        pos_frac: When set, draw this fraction of the queries from positive rows (the last
            class) and return importance weights -- true rate over sampled rate, per class --
            so a weighted loss is still an estimate of the uniform-query loss (§164).

    Returns:
        ``(context_rows, query_rows, query_weights)``. Indices are drawn with replacement only
        when a side has fewer rows than requested; weights are ``None`` for uniform queries,
        and for a query side with no positives, which falls back to uniform.
    """
    # groups arrive as integer codes into ``uniq``: membership by table lookup is O(n), where
    # np.isin over string company ids degraded to an O(n x k) object comparison that took a
    # 600k-row fold eight hours without completing one step (§163)
    held_out = np.zeros(len(uniq), dtype=bool)
    held_out[rng.choice(len(uniq), size=n_q_groups, replace=False)] = True
    is_q = held_out[groups]
    ctx_pool, q_pool = np.flatnonzero(~is_q), np.flatnonzero(is_q)
    ci = rng.choice(ctx_pool, size=n_ctx, replace=len(ctx_pool) < n_ctx)
    if pos_frac is not None and y is not None:
        pos_label = y.max()
        q_pos, q_neg = q_pool[y[q_pool] == pos_label], q_pool[y[q_pool] != pos_label]
        if len(q_pos) and len(q_neg):
            # at least one query on each side: near pos_frac=1, rounding alone gave every
            # query to the positives and the negative weight divided by zero
            n_pos = min(max(1, round(n_q * pos_frac)), n_q - 1)
            qi = np.concatenate([
                rng.choice(q_pos, size=n_pos, replace=len(q_pos) < n_pos),
                rng.choice(q_neg, size=n_q - n_pos, replace=len(q_neg) < n_q - n_pos),
            ])
            true_rate, sampled = len(q_pos) / len(q_pool), n_pos / n_q
            w = np.where(
                y[qi] == pos_label, true_rate / sampled, (1 - true_rate) / (1 - sampled)
            )
            return ci, qi, w.astype(np.float32)
    qi = rng.choice(q_pool, size=n_q, replace=len(q_pool) < n_q)
    return ci, qi, None


@contextlib.contextmanager
def _forked_rng() -> Iterator[None]:
    """Restore every generator ``torch.manual_seed`` reseeds, CPU and accelerator alike.

    ``torch.random.fork_rng(devices=[])`` restores the CPU generator only, while
    ``torch.manual_seed`` also reseeds CUDA and MPS, so fine-tuning used to leave the caller's
    accelerator generator at the fine-tuning seed. Reading the CUDA states initialises CUDA
    where it is available, as ``fork_rng`` does by default.
    """
    restore: list[tuple[Callable[[object], None], object]] = []
    if torch.cuda.is_available():
        restore.append((torch.cuda.set_rng_state_all, torch.cuda.get_rng_state_all()))
    if torch.backends.mps.is_available():
        restore.append((torch.mps.set_rng_state, torch.mps.get_rng_state()))
    with torch.random.fork_rng(devices=[]):
        try:
            yield
        finally:
            for set_state, state in restore:
                set_state(state)


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
        correct_prior: Undo the base-rate distortion that resampling introduces. An
            in-context model reads the class balance *out of its context*, so handing it a
            balanced context tells it defaults are far commoner than they are, and its
            probabilities come out inflated by roughly that factor. Measured on
            ``polish-bankruptcy-3y``: balanced context predicted a 14.9% mean against a 4.7%
            actual rate (ECE 0.10) where uniform predicted 2.9% (ECE 0.02). The correction
            shifts the log-odds by the difference between the context prior and the true
            training prior, which is exact under the standard label-shift assumption that
            ``P(x | y)`` is unchanged by resampling — resampling selects on ``y`` alone, so
            it holds by construction here. Leaves the ranking, and therefore AUC, untouched.
        random_state: Seed for context subsampling, so a stored context is reproducible.
        feature_chunk: Features processed per row-within-feature attention call, for
            checkpoints with ``n_cell_blocks > 0``. Purely a memory/throughput knob: the
            ``B*F`` attention problems in that stage are independent, so chunking them is an
            identity and changes no prediction. ``docs/results/FINDINGS.md`` §81 measured 21.0 GB and
            10.6 s unchunked against 3.4 GB and 6.8 s at ``feature_chunk=16`` on V4FinBench's
            136 features, and it is what makes ``max_context=4000`` runnable at all (§79/§80).
            ``None`` disables chunking.
        query_chunk: Queries scored per forward pass. Attention cost grows with the square
            of (context + queries), so 48,000 queries at once is not feasible. **Chunking is
            exact, not an approximation**: the row mask already forbids a query from
            attending to any other query, so a prediction depends only on the context and
            itself. Splitting them changes nothing, which is a direct benefit of that design
            choice and is asserted in the tests. **Does not hold for
            ``context_strategy="retrieval"``**, where the context is chosen per query group;
            see :mod:`fintfm.inference.retrieval`.
        retrieval_groups: Number of query groups sharing a retrieved context, used only by
            ``context_strategy="retrieval"``. Set to 0 for exact per-query retrieval, which
            is correct but costs one forward pass per query — usable for validating the
            approximation on a subsample, not for scoring a book.
        feature_transform: Conditioning applied to features before the model sees them, to
            blunt the heavy tails that break its mean/standard-deviation normalisation. See
            :mod:`fintfm.inference.preprocess`. **Defaults to ``"rank"``** on the evidence
            in ``docs/results/FINDINGS.md`` §35: it improves AUC in six of six configurations on the
            V4FinBench out-of-time split and seven of eight across two independent panels,
            because 110 of 136 features here have a standard deviation more than ten times
            their interquartile range. Every number recorded before 2026-09-09 was produced
            with ``"none"``, so pass it explicitly to reproduce those.
        n_ensemble: Predictions averaged over this many independently drawn contexts.
            Standard practice for prior-fitted networks and untried here until now
            (``openspec/changes/adopt-published-methods`` task 36.2). Costs inference time
            linearly and needs no retraining.

            **Only the context draw is varied, not the feature order.** TabPFN ensembles over
            feature permutations as well, because its predictions depend on column position.
            Ours do not: column-order invariance is a design property asserted in
            ``tests/test_model.py`` (decision D4), so permuting features would average
            identical predictions and buy nothing. The axis that remains is *which rows* the
            model conditions on.
        ensemble_label_swap: Average each member with its label-swapped twin — relabel the
            context 0<->1, predict, invert. **This cancels a measured defect rather than
            merely reducing variance.** Swapping the class names and inverting should return
            the same probabilities; on a real checkpoint it returns predictions correlated
            **−0.62** with the original (``docs/results/FINDINGS.md`` §45), so the model's output
            depends on which class occupies the "1" slot. Averaging removes that component.
            It is a workaround, not a cure, and should be reported as one. Binary tasks only:
            with more classes there is no swap, and the member is returned as it is.
        ensemble_feature_frac: Fraction of features each member sees, sampled without
            replacement. Below 1.0 this is a real diversity axis — a 70% subset moves
            predictions to correlation 0.496 with the full-feature prediction (§45).
            **Feature *permutation* is deliberately not an axis**: unlike TabPFN, our
            predictions are column-order invariant by construction (decision D4), measured at
            1.19e-07 maximum change, so permuting would average identical members.
        prototype_minority_ratio: Target minority-to-majority ratio for
            ``context_strategy="prototype"``, the published best method on this benchmark.
            0.3 is the value Kostrzewa et al. use (``docs/results/FINDINGS.md`` §36).
        retrieval_min_positive: Floor on positive-class rows in a retrieved context. A
            nearest-neighbour draw at a 0.19% default rate can return **zero** defaults, and
            a context with no positives says nothing about default. Deliberately a floor and
            not class balancing, which §29 measured as costing 10-12 AUC points.
        finetune_steps: Gradient steps of dataset-specific fine-tuning taken in :meth:`fit`
            before the context is stored. **0 (the default) takes none**, and the classifier is
            then byte-identical to one without these arguments: the in-context model as
            pretrained. Above 0, a *copy* of the model is updated and kept as ``model_`` -- the
            instance passed in is never modified, because callers (the V4FinBench protocol
            among them) reuse one loaded model across folds, and ``model`` stays that instance,
            so refitting or cloning starts again from the pretrained weights. A fine-tuned
            classifier has seen the target's training rows through gradients, so it is no
            longer the zero-shot FinTFM; label results.
        finetune_lr: AdamW learning rate for fine-tuning.
        finetune_batch: Tasks per fine-tuning step.
        finetune_query: Query rows per fine-tuning task, whose labels the loss is taken on.
        finetune_context: Context rows per fine-tuning task; ``None`` uses ``max_context``.
            Capped at the rows left outside the largest held-out query groups, so a small
            table is never padded with duplicates.
        finetune_query_group_frac: Fraction of groups (companies, when ``fit`` receives
            ``groups``) held out per task to supply its queries; the context comes from the
            rest. Mirrors a company-grouped test split, so the model is never trained to
            recognise a firm it can also see in its context.
        finetune_query_pos_frac: ``None`` (default) draws queries uniformly. A fraction, e.g.
            0.1, draws that share of each task's queries from defaults and weights the loss
            back to the true rate (§164): more default labels per step, same expected loss.
        finetune_seed: Second axis of fine-tuning randomness beside ``random_state``, so the
            same fold can be fine-tuned at several seeds. Seeds torch as well as numpy --
            column identities are drawn from torch's generator in training mode -- inside a
            forked RNG, so the caller's random state, CPU and CUDA/MPS, is untouched.
        finetune_clip: Gradient-norm clip per fine-tuning step, as pretraining applies at 1.0.
            **``None`` (the default) does not clip**, because §163 and §164 were measured
            without clipping and clipping changes the tuned weights; pass 1.0 to match
            pretraining in a new experiment.
        ensemble_transforms: Per-member feature conditioning (audit AU5.12/AU5.6). **``None``
            (the default) gives every member ``feature_transform``**, today's behaviour. A
            sequence, e.g. ``("rank", "power", "squash")``, gives member ``i`` the transform
            ``ensemble_transforms[i % len]``, so the members disagree about how a tail value
            should look rather than only about which rows they condition on. Experiment lever;
            nothing here measures its effect on a score yet.
        ensemble_svd_components: Members with an odd index (1, 3, ...) append this many
            truncated-SVD projections of their own conditioned context to every row; see
            :class:`fintfm.inference.preprocess.SVDAugmenter`. **0 (the default) appends
            none.** Member 0 never carries them, so the plain model stays one of the votes,
            and a value above 0 therefore needs ``n_ensemble >= 2``. Capped per member at the
            checkpoint's ``max_features`` minus the member's own width, so a member already
            at capacity (a wide table under ``max_features_strategy="subsample"``) gets none.
            Not supported with ``context_strategy="retrieval"``, whose distance space would
            then disagree with the model's input width.
        ensemble_class_shift: Member ``i`` relabels the context by a cyclic shift of ``i``
            modulo the number of classes (class ``c`` becomes ``c + i``), predicts, and maps
            the probabilities back, so averaging over members cancels the part of a
            prediction tied to which class occupies which output slot. **``False`` (the
            default) shifts nothing.** On a binary task this is label swap on every other
            member at no extra cost, where ``ensemble_label_swap`` doubles each member
            instead; the two compose. Through :class:`FinancialTFMRegressor` the classes are
            ordinal bins, so a shifted member sees a scrambled bin order: legal, but a
            different experiment from the one this axis was written for.
        max_features_strategy: What :meth:`fit` does with a table wider than the checkpoint's
            ``max_features`` (audit AU5.7). **``"raise"`` (the default) raises**, as before.
            ``"subsample"`` scores the table with an ensemble whose members each see a
            balanced round-robin subset of the columns: the columns are shuffled once (at
            ``random_state``) and member ``i`` takes the next ``size`` of them cyclically, so
            every column is seen by at least one member and no column by more than one
            member more than any other. ``size`` is ``max_features``, or
            ``round(F * ensemble_feature_frac)`` when that is smaller. **The member count is
            raised to ``ceil(F / size)`` when ``n_ensemble`` is below it**, so ``n_ensemble``
            is a floor on a wide table, not the count; the count used is ``n_members_``.
            Only :meth:`predict_proba` (and :meth:`predict`) can score a wide table; fine-
            tuning, :meth:`predict_term_structure` and :meth:`transform_representation`
            need the whole row in one forward pass and raise.

    Attributes:
        model_: The network predictions run through: ``model`` itself, placed on ``device``
            and in eval mode at :meth:`fit`, or its fine-tuned copy when ``finetune_steps > 0``.
        n_features_in_: Feature count seen by :meth:`fit`.
        n_members_: Ensemble members :meth:`predict_proba` averages: ``n_ensemble``, or more
            on a wide table under ``max_features_strategy="subsample"``.
        svd_components_: Projections this estimator appended to its own rows; 0 except on an
            ensemble member that ``ensemble_svd_components`` selected.
    """

    def __init__(
        self,
        model: FinancialTFM | str,
        device: str = "cpu",
        max_context: int = 2000,
        context_strategy: ContextStrategy = "uniform",
        correct_prior: bool = True,
        random_state: int = 0,
        query_chunk: int = 2048,
        retrieval_groups: int = 64,
        retrieval_min_positive: int = 8,
        feature_transform: FeatureTransform = "rank",
        prototype_minority_ratio: float = 0.3,
        n_ensemble: int = 1,
        ensemble_label_swap: bool = False,
        ensemble_feature_frac: float = 1.0,
        feature_chunk: int | None = 16,
        finetune_steps: int = 0,
        finetune_lr: float = 1e-5,
        finetune_batch: int = 2,
        finetune_query: int = 512,
        finetune_context: int | None = None,
        finetune_query_group_frac: float = 0.3,
        finetune_query_pos_frac: float | None = None,
        finetune_seed: int = 0,
        finetune_clip: float | None = None,
        ensemble_transforms: tuple[FeatureTransform, ...] | None = None,
        ensemble_svd_components: int = 0,
        ensemble_class_shift: bool = False,
        max_features_strategy: MaxFeaturesStrategy = "raise",
    ) -> None:
        """Configure the classifier. See the class docstring — every argument is documented
        there with the measurement that set its default."""
        self.model = (
            FinancialTFM.load(model, map_location=device) if isinstance(model, str) else model
        )
        self.device = device
        self.max_context = max_context
        self.context_strategy = context_strategy
        self.correct_prior = correct_prior
        self.random_state = random_state
        self.query_chunk = query_chunk
        self.retrieval_groups = retrieval_groups
        self.retrieval_min_positive = retrieval_min_positive
        self.feature_transform = feature_transform
        self.prototype_minority_ratio = prototype_minority_ratio
        self.n_ensemble = n_ensemble
        self.ensemble_label_swap = ensemble_label_swap
        self.ensemble_feature_frac = ensemble_feature_frac
        self.feature_chunk = feature_chunk
        self.finetune_steps = finetune_steps
        self.finetune_lr = finetune_lr
        self.finetune_batch = finetune_batch
        self.finetune_query = finetune_query
        self.finetune_context = finetune_context
        self.finetune_query_group_frac = finetune_query_group_frac
        self.finetune_query_pos_frac = finetune_query_pos_frac
        self.finetune_seed = finetune_seed
        self.finetune_clip = finetune_clip
        self.ensemble_transforms = ensemble_transforms
        self.ensemble_svd_components = ensemble_svd_components
        self.ensemble_class_shift = ensemble_class_shift
        self.max_features_strategy = max_features_strategy
        # Placement, eval mode and feature_chunk are applied in fit, not here: applied at
        # construction, a later set_params(device=...) moved nothing and the first forward
        # pass mixed devices. A checkpoint path is still loaded here, so model.cfg is readable
        # before fit (FinancialTFMRegressor reads max_classes from it).

    def fit(
        self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None
    ) -> FinancialTFMClassifier:
        """Store the table as in-context evidence. **No gradient steps are taken** unless
        ``finetune_steps > 0`` (see :meth:`_finetune`).

        The name is sklearn's, and it is misleading here by convention rather than by choice:
        nothing is learned. The table is conditioned, subsampled to at most ``max_context``
        rows by the configured strategy, and kept. Everything that would be "training" in
        another estimator happened during pretraining, on synthetic data.

        Args:
            X: ``(n, n_features)`` training features. Must not exceed the checkpoint's
                ``max_features``.
            y: ``(n,)`` labels. The number of distinct values must not exceed the
                checkpoint's ``max_classes``.
            groups: Optional ``(n,)`` group labels (e.g. company), used only by fine-tuning
                to keep a task's query groups out of its context.

        Returns:
            self.

        Raises:
            ValueError: If the task exceeds the checkpoint's class or feature capacity. It
                raises rather than truncating, so a caller records a skip instead of a
                meaningless score; with ``max_features_strategy="subsample"`` a wide table
                is accepted instead (and fine-tuning on it rejected). Also if ``groups``
                does not have one entry per row, for the ensemble settings
                :meth:`_check_ensemble_settings` rejects, and for the fine-tuning settings
                :meth:`_finetune` rejects.
        """
        self._check_ensemble_settings()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if groups is not None and len(groups) != len(y):
            raise ValueError(f"groups has {len(groups)} entries for {len(y)} rows")
        # kept unconditioned so an ensemble member can redraw its own context and refit the
        # conditioner from scratch, rather than inheriting this instance's draw
        self._raw_X, self._raw_y = X, y
        # fitted on training rows only, so a query never influences its own conditioning
        self._conditioner = FeatureConditioner(
            self.feature_transform, random_state=self.random_state
        ).fit(X)
        X = self._conditioner.transform(X)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        if len(self.classes_) > self.model.cfg.max_classes:
            raise ValueError(
                f"model supports at most {self.model.cfg.max_classes} classes, got {len(self.classes_)}"
            )
        wide = X.shape[1] > self.model.cfg.max_features
        if wide and self.max_features_strategy != "subsample":
            raise ValueError(
                f"model supports at most {self.model.cfg.max_features} features, got {X.shape[1]}"
            )
        if wide and self.finetune_steps > 0:
            raise ValueError(
                f"fine-tuning needs every feature in one forward pass, and this table has "
                f"{X.shape[1]} against the model's {self.model.cfg.max_features}"
            )
        # None unless the table is wide: then each member's columns, planned once per fit
        self._member_columns = self._plan_member_columns(X.shape[1]) if wide else None
        self.n_members_ = (
            self.n_ensemble if self._member_columns is None else len(self._member_columns)
        )
        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_coded = np.array([label_map[v] for v in y], dtype=np.int64)
        if self.context_strategy == "retrieval":
            # keep the whole pool; the context is chosen per query group at predict time
            self._centre, self._scale = distance_stats(X)
            self._pool_X, self._pool_y = X, y_coded
            self._pool_Z = normalise_for_distance(X, self._centre, self._scale)
            idx = np.arange(len(y_coded))
        elif self.context_strategy == "prototype":
            # blind like the other strategies -- selected once, never query-dependent -- but
            # it needs the distance space to cluster the majority class in
            self._pool_Z = None
            centre, scale = distance_stats(X)
            idx = prototype_context(
                normalise_for_distance(X, centre, scale),
                y_coded,
                self.max_context,
                self.prototype_minority_ratio,
                np.random.default_rng(self.random_state),
            )
        else:
            self._pool_Z = None
            idx = _select_context(
                y_coded,
                self.max_context,
                self.context_strategy,
                np.random.default_rng(self.random_state),
            )
        # an ensemble member selected by ensemble_svd_components appends projections fitted on
        # its own context rows; set by _clone_with_seed, absent (0) on any other estimator
        self._svd: SVDAugmenter | None = None
        self.svd_components_ = 0
        k = min(getattr(self, "_member_svd", 0), self.model.cfg.max_features - X.shape[1])
        if k > 0:
            if self.context_strategy == "retrieval":
                raise ValueError("SVD components are not supported with retrieval contexts")
            self._svd = SVDAugmenter(k).fit(X[idx])
            X = self._svd.transform(X)
            self.svd_components_ = self._svd.components_.shape[0]
        n_classes = len(self.classes_)
        # Identity-preserving, so it is safe to default on: asserted byte-for-byte by
        # tests/test_model.py::test_feature_chunking_is_an_identity. No effect on a
        # checkpoint with n_cell_blocks=0, which has no row-within-feature stage at all.
        self.model.feature_chunk = self.feature_chunk
        self.model_ = self.model.to(self.device).eval()
        if self.finetune_steps > 0:
            # on the full conditioned pool, before the stored context is cut from it
            self._finetune(X, y_coded, groups, n_classes)
        full_prior = np.bincount(y_coded, minlength=n_classes) / len(y_coded)
        X, y_coded = X[idx], y_coded[idx]
        ctx_prior = np.bincount(y_coded, minlength=n_classes) / len(y_coded)
        self._ctx_X = X
        self._ctx_y = y_coded
        # positive-class rates, kept for the term-structure correction, which needs a
        # scalar log-odds shift rather than the per-class vector below
        pos = n_classes - 1
        self._full_prior = full_prior
        self._full_rate = float(full_prior[pos])
        self._ctx_rate = float(ctx_prior[pos])
        # log P_true(y) - log P_context(y), added to logits at predict time. Zero when the
        # context was not resampled, so the correction is inert in that case.
        with np.errstate(divide="ignore"):
            self._log_prior_shift = np.where(
                (full_prior > 0) & (ctx_prior > 0),
                np.log(np.maximum(full_prior, 1e-12)) - np.log(np.maximum(ctx_prior, 1e-12)),
                0.0,
            )
        return self

    def _finetune(
        self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None, n_classes: int
    ) -> None:
        """Take ``finetune_steps`` gradient steps on tasks drawn from the training rows.

        Each task is built the way pretraining builds one -- a labelled context and query rows
        whose labels the loss is taken on, NaN-padded to ``max_features`` -- and scored with
        the model's own :meth:`FinancialTFM.loss`, so fine-tuning optimises exactly what
        pretraining did, on the target's rows. Query groups are disjoint from context groups
        within a task (``finetune_query_group_frac``). The context draw is uniform, matching
        the inference default; the stored context is drawn afterwards, independently. With any
        other ``context_strategy``, or with ``ensemble_label_swap``, inference reads a context
        distribution the tuned weights never saw, and a ``UserWarning`` says so.

        Args:
            X: ``(n, n_features)`` conditioned training rows (the whole pool).
            y: ``(n,)`` coded labels.
            groups: Optional ``(n,)`` group labels; each row is its own group when omitted.
            n_classes: Classes in this task.

        Raises:
            ValueError: If ``finetune_context`` is below 1, ``finetune_query_pos_frac`` is not
                strictly between 0 and 1, or there are fewer than two groups to split into a
                context side and a query side.
        """
        import copy
        import warnings

        if self.context_strategy != "uniform" or self.ensemble_label_swap:
            # a warning rather than an error: the combination may be what an experiment wants
            # to measure, but it is not what §163 measured, and the result should say so
            warnings.warn(
                f"fine-tuning draws a uniform context, but inference will read one chosen by "
                f"context_strategy={self.context_strategy!r}"
                + (" and label-swapped ensemble members" if self.ensemble_label_swap else "")
                + "; the tuned weights never saw that distribution",
                UserWarning,
                stacklevel=3,
            )
        if self.finetune_context is not None and self.finetune_context < 1:
            raise ValueError(
                f"finetune_context must be None (use max_context) or >= 1, "
                f"got {self.finetune_context}"
            )
        pos_frac = self.finetune_query_pos_frac
        if pos_frac is not None and not (0.0 < pos_frac < 1.0 and self.finetune_query >= 2):
            raise ValueError(
                f"finetune_query_pos_frac must lie strictly between 0 and 1, with "
                f"finetune_query >= 2, got {pos_frac} and {self.finetune_query}"
            )
        raw = np.asarray(groups) if groups is not None else np.arange(len(y))
        uniq, g = np.unique(raw, return_inverse=True)  # integer codes; see _finetune_rows
        if len(uniq) < 2:
            raise ValueError(
                "fine-tuning needs at least two groups, one to hold out for queries and one "
                f"for the context; got {len(uniq)}"
            )
        # at least one group on each side: one group, or two at a query fraction of 0.75 or
        # more, used to hold out every group and leave an empty context pool
        n_q_groups = min(max(1, round(len(uniq) * self.finetune_query_group_frac)), len(uniq) - 1)
        # no larger than the context pool left after holding out the largest query groups, so
        # a small table is not padded with duplicate rows to a size inference never uses
        held_out_max = int(np.sort(np.bincount(g))[::-1][:n_q_groups].sum())
        n_ctx = self.max_context if self.finetune_context is None else self.finetune_context
        n_ctx = min(n_ctx, len(y) - held_out_max)
        n_q, B = self.finetune_query, self.finetune_batch

        # a copy of self.model, never self.model itself: a second fit, or a clone of a fitted
        # estimator, must start again from the weights it was given, not from these
        self.model_ = copy.deepcopy(self.model)
        self.model_.feature_chunk = self.feature_chunk
        model = self.model_.to(self.device).train()
        opt = torch.optim.AdamW(model.parameters(), lr=self.finetune_lr)
        # seed 0 keeps §163's numpy stream; other seeds mix in finetune_seed
        rng = np.random.default_rng(
            self.random_state if self.finetune_seed == 0 else [self.random_state, self.finetune_seed]
        )
        F = model.cfg.max_features
        nc = torch.full((B,), n_classes, dtype=torch.long, device=self.device)
        self.finetune_losses_: list[float] = []
        try:
            with _forked_rng():
                torch.manual_seed(1_000_003 * int(self.random_state) + int(self.finetune_seed))
                for _ in range(self.finetune_steps):
                    Xb = np.full((B, n_ctx + n_q, F), np.nan, dtype=np.float32)
                    yb = np.zeros((B, n_ctx + n_q), dtype=np.int64)
                    wb = np.ones((B, n_q), dtype=np.float32)
                    weighted = False
                    for b in range(B):
                        ci, qi, w = _finetune_rows(
                            rng, g, uniq, n_q_groups, n_ctx, n_q, y, pos_frac
                        )
                        rows = np.concatenate([ci, qi])
                        Xb[b, :, : X.shape[1]] = X[rows]
                        yb[b] = y[rows]
                        if w is not None:
                            wb[b], weighted = w, True
                    loss = model.loss(
                        torch.from_numpy(Xb).to(self.device),
                        torch.from_numpy(yb).to(self.device),
                        n_ctx, nc,
                        weights=torch.from_numpy(wb).to(self.device) if weighted else None,
                    )
                    opt.zero_grad()
                    loss.backward()
                    if self.finetune_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.finetune_clip)
                    opt.step()
                    self.finetune_losses_.append(float(loss.detach()))
        finally:
            # free what training held -- gradients, optimiser state, the allocator's cache --
            # before prediction needs the memory: on a 14.5 GB T4 it was the difference between
            # scoring and an out-of-memory error at the first query chunk (§164). In a finally
            # so a failed step does not leave the model in training mode either.
            opt.zero_grad(set_to_none=True)
            del opt
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
            elif str(self.device).startswith("mps"):
                torch.mps.empty_cache()
            model.eval()

    def _pooled_prior_shift(self, pooled_rate: float) -> np.ndarray:
        """Class-prior correction from a pooled context rate, for the retrieval path.

        One shift for every query, for the reason given on :meth:`_pooled_context_rate`: a
        per-group shift inverts the cross-group ranking that retrieval exists to exploit.

        Args:
            pooled_rate: Positive-class rate across all retrieved contexts.

        Returns:
            ``(n_classes,)`` log-odds shift, zero for any class absent from either side.
            With more than two classes the pooled share of every class is used, as cached by
            :meth:`_pooled_context_rate`; one rate alone cannot say how the rest divide.
        """
        n_classes = len(self.classes_)
        if n_classes > 2:
            # an equal split of 1 - rate over the other classes was wrong whenever the
            # retrieved classes were unequal, which on an ordinal bin grid is always
            ctx_prior = self._pooled_class_shares
        else:
            ctx_prior = np.full(n_classes, (1.0 - pooled_rate) / max(n_classes - 1, 1))
            ctx_prior[n_classes - 1] = pooled_rate
        with np.errstate(divide="ignore"):
            return np.where(
                (self._full_prior > 0) & (ctx_prior > 0),
                np.log(np.maximum(self._full_prior, 1e-12)) - np.log(np.maximum(ctx_prior, 1e-12)),
                0.0,
            )

    def _pooled_context_rate(self, plan: list[tuple[np.ndarray, np.ndarray]]) -> float:
        """Size-weighted positive rate across every retrieved context.

        **Why one rate and not one per group.** The base-rate correction is exact under label
        shift — ``P(x | y)`` unchanged by resampling — which holds by construction for the
        blind strategies because they select on ``y`` alone. **Retrieval selects on ``x``, so
        the assumption is violated by construction**, and applying the correction per group
        does active harm: a risky cluster retrieves risky neighbours, so its context rate is
        high and it is shifted *down* hardest, erasing exactly the between-group risk
        differences that carry the signal. Measured on the V4FinBench out-of-time split, a
        per-group correction drove mean AUC to **0.3679 — below chance** — where the same
        contexts uncorrected score 0.7872 (``docs/results/FINDINGS.md`` §32).

        A single pooled shift is applied to every query instead. Being constant it cannot
        reorder anything, so it corrects the level while leaving ranking untouched.

        Args:
            plan: ``(query_indices, context_indices)`` pairs from :meth:`_retrieval_plan`.

        Returns:
            The positive-class rate over all retrieved contexts, weighted by how many queries
            each context serves. Also stored as ``pooled_context_rate_``, and the same
            weighting of every class's share as ``_pooled_class_shares`` for the multiclass
            correction in :meth:`_pooled_prior_shift`.
        """
        n_classes = len(self.classes_)
        pos_class = n_classes - 1
        weight = sum(q.size for q, _ in plan)
        if weight == 0:
            self._pooled_class_shares = self._full_prior
            return self._full_rate
        self._pooled_class_shares = sum(
            q.size * np.bincount(self._pool_y[c], minlength=n_classes) / c.size for q, c in plan
        ) / weight
        total = sum(q.size * float((self._pool_y[c] == pos_class).mean()) for q, c in plan)
        # cached so a caller reporting what was actually retrieved does not have to rebuild
        # the plan; recomputing it means a second k-means and a second pass over the pool,
        # which doubled the cost of every retrieval cell in the context sweep
        self.pooled_context_rate_ = total / weight
        return self.pooled_context_rate_

    def _retrieval_plan(self, X: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Group the queries and retrieve one context per group.

        Args:
            X: ``(n, F)`` query rows.

        Returns:
            A list of ``(query_indices, context_indices)`` pairs covering every query once.
            With ``retrieval_groups == 0`` each group holds a single query, which is exact
            per-query retrieval and costs one forward pass per row.
        """
        rng = np.random.default_rng(self.random_state)
        Zq = normalise_for_distance(X, self._centre, self._scale)
        if self.retrieval_groups == 0:
            groups = [np.array([i]) for i in range(X.shape[0])]
        else:
            groups = group_queries(Zq, self.retrieval_groups, self.query_chunk, rng)
        plan = []
        for g in groups:
            target = Zq[g].mean(axis=0)
            plan.append(
                (
                    g,
                    retrieve(
                        self._pool_Z,
                        self._pool_y,
                        target,
                        self.max_context,
                        self.retrieval_min_positive,
                    ),
                )
            )
        return plan

    def _run_retrieval(
        self,
        X: np.ndarray,
        chunk_fn: Callable[..., np.ndarray],
        width: int,
    ) -> np.ndarray:
        """Score every query under its group's retrieved context.

        Args:
            X: ``(n, F)`` query rows.
            chunk_fn: Either :meth:`_predict_chunk` or :meth:`_term_structure_chunk`.
            width: Output columns, so the result array can be allocated before scoring.

        Returns:
            ``(n, width)`` predictions in the original query order.
        """
        plan = self._retrieval_plan(X)
        rate = self._pooled_context_rate(plan)
        out = np.empty((X.shape[0], width), dtype=np.float32)
        for q_idx, c_idx in plan:
            ctx = (self._pool_X[c_idx], self._pool_y[c_idx])
            for start in range(0, q_idx.size, self.query_chunk):
                block = q_idx[start : start + self.query_chunk]
                out[block] = chunk_fn(X[block], ctx, rate)
        return out

    def _check_ensemble_settings(self) -> None:
        """Reject AU5.7/AU5.12 settings that would otherwise be silently ignored or misread.

        Raises:
            ValueError: For an unknown ``max_features_strategy``, an empty or unknown entry in
                ``ensemble_transforms``, a negative ``ensemble_svd_components``, or SVD
                components that no member would carry (``n_ensemble < 2``) or that the
                retrieval path cannot score.
        """
        if self.max_features_strategy not in get_args(MaxFeaturesStrategy):
            raise ValueError(
                f"max_features_strategy must be one of {get_args(MaxFeaturesStrategy)}, "
                f"got {self.max_features_strategy!r}"
            )
        if self.ensemble_transforms is not None:
            # FeatureConditioner treats any unrecognised kind as "rank", so a misspelling here
            # would run, report numbers, and have measured something else
            known = get_args(FeatureTransform)
            if len(self.ensemble_transforms) == 0 or any(
                t not in known for t in self.ensemble_transforms
            ):
                raise ValueError(
                    f"ensemble_transforms must be None or a non-empty sequence drawn from "
                    f"{known}, got {self.ensemble_transforms!r}"
                )
        if self.ensemble_svd_components < 0:
            raise ValueError(
                f"ensemble_svd_components must be >= 0, got {self.ensemble_svd_components}"
            )
        if self.ensemble_svd_components > 0 and self.n_ensemble < 2:
            raise ValueError(
                "ensemble_svd_components applies to odd-indexed members only, so it needs "
                f"n_ensemble >= 2; got n_ensemble={self.n_ensemble}"
            )
        if self.ensemble_svd_components > 0 and self.context_strategy == "retrieval":
            raise ValueError("ensemble_svd_components is not supported with retrieval contexts")

    def _plan_member_columns(self, n_features: int) -> list[np.ndarray]:
        """Balanced round-robin column subsets covering a table wider than the model.

        The columns are shuffled once at ``random_state``; member ``i`` then takes positions
        ``i * size`` to ``i * size + size - 1`` of that order, wrapping around, so consecutive
        members tile the shuffled columns and every column is covered as soon as the members'
        sizes add up to the width. Coverage counts differ by at most one between columns.

        Args:
            n_features: Width of the table, above the checkpoint's ``max_features``.

        Returns:
            One sorted column-index array per member: ``max(n_ensemble, ceil(F / size))``
            members of ``size`` columns each, ``size`` being ``max_features`` or
            ``round(F * ensemble_feature_frac)``, whichever is smaller.
        """
        size = min(
            self.model.cfg.max_features,
            max(1, round(n_features * self.ensemble_feature_frac)),
        )
        n_members = max(self.n_ensemble, -(-n_features // size))
        order = np.random.default_rng(self.random_state).permutation(n_features)
        return [
            np.sort(order[(i * size + np.arange(size)) % n_features]) for i in range(n_members)
        ]

    def _uses_members(self) -> bool:
        """Whether :meth:`predict_proba` averages ensemble members rather than one pass."""
        return (
            self.n_ensemble > 1
            or self.ensemble_label_swap
            or self.ensemble_feature_frac < 1.0
            or self.ensemble_transforms is not None
            or self.ensemble_class_shift
            or getattr(self, "_member_columns", None) is not None
        )

    def _member_proba(self, X: np.ndarray, seed: int, member: int = 0) -> np.ndarray:
        """One ensemble member: its own context draw, features, conditioning and labelling.

        Args:
            X: ``(n, n_features)`` raw query rows.
            seed: The member's seed, for its context draw and column identities.
            member: The member's index, which selects its transform, SVD components, class
                shift and, on a wide table, its column subset.

        Returns:
            ``(n, n_classes)`` probabilities in this estimator's class order.
        """
        rng = np.random.default_rng(seed)
        Xa, Xb = self._raw_X, np.asarray(X, dtype=np.float32)
        member_columns = getattr(self, "_member_columns", None)
        if member_columns is not None:
            cols = member_columns[member]
            Xa, Xb = Xa[:, cols], Xb[:, cols]
        elif self.ensemble_feature_frac < 1.0:
            k = max(1, round(Xa.shape[1] * self.ensemble_feature_frac))
            cols = np.sort(rng.choice(Xa.shape[1], size=k, replace=False))
            Xa, Xb = Xa[:, cols], Xb[:, cols]
        transform = (
            None
            if self.ensemble_transforms is None
            else self.ensemble_transforms[member % len(self.ensemble_transforms)]
        )
        svd = self.ensemble_svd_components if member % 2 == 1 else 0
        n_classes = len(np.unique(self._raw_y))
        shift = member % n_classes if self.ensemble_class_shift else 0
        y = self._raw_y if shift == 0 else self._shift_labels(self._raw_y, shift)
        twin = self._clone_with_seed(seed, Xa, y, transform, svd)
        p = twin.predict_proba(Xb)
        # the swap is defined for two classes only: on more, reversing the columns would pair
        # each class with another, and through the regressor would reverse the bin order
        if self.ensemble_label_swap and n_classes == 2:
            # the label-swapped twin: relabel the context, predict, and reverse the columns
            # back. Averaging the two removes the part of the prediction that depends on which
            # class is called "1" (§45).
            swapped = self._clone_with_seed(seed, Xa, self._flip(y), transform, svd)
            p = 0.5 * (p + swapped.predict_proba(Xb)[:, ::-1])
        # the twin's column j is the class shifted onto slot j, i.e. original class j - shift,
        # so original class c sits in column c + shift
        return p if shift == 0 else np.roll(p, -shift, axis=1)

    def _shift_labels(self, y: np.ndarray, shift: int) -> np.ndarray:
        """Relabel class ``classes[j]`` as ``classes[(j + shift) % K]``, a cyclic shift."""
        classes = np.unique(y)
        return classes[(np.searchsorted(classes, y) + shift) % len(classes)]

    def _condition(self, X: np.ndarray) -> np.ndarray:
        """Query rows as the stored context saw them: conditioned, then SVD-augmented if fit was.

        Args:
            X: ``(n, n_features_in_)`` raw query rows.

        Returns:
            ``(n, n_features_in_ + svd_components_)`` rows, the conditioner's output alone
            unless this estimator is an SVD-carrying ensemble member.
        """
        X = self._conditioner.transform(X)
        svd = getattr(self, "_svd", None)
        return X if svd is None else svd.transform(X)

    def _refuse_wide(self, method: str) -> None:
        """Raise if ``fit`` saw a table wider than the model, which only members can score.

        Raises:
            ValueError: Naming ``method``, when ``max_features_strategy="subsample"`` let a
                wide table through :meth:`fit`.
        """
        if getattr(self, "_member_columns", None) is not None:
            raise ValueError(
                f"{method} needs every feature in one forward pass; this table has "
                f"{self.n_features_in_} features against the model's "
                f"{self.model_.cfg.max_features}, which only predict_proba's column-subset "
                "members can score"
            )

    def _flip(self, y: np.ndarray) -> np.ndarray:
        """Exchange the two class labels; only meaningful for binary tasks."""
        classes = np.unique(y)
        if len(classes) != 2:
            return y
        return np.where(y == classes[0], classes[1], classes[0])

    def _clone_with_seed(
        self,
        seed: int,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        feature_transform: FeatureTransform | None = None,
        svd_components: int = 0,
    ) -> FinancialTFMClassifier:
        """A sibling estimator differing in its context draw, features or label orientation.

        Args:
            seed: The sibling's ``random_state``.
            X: Raw training rows, ``_raw_X`` when omitted.
            y: Training labels, ``_raw_y`` when omitted.
            feature_transform: The sibling's conditioning; this estimator's when ``None``.
            svd_components: SVD projections the sibling appends (see
                :class:`fintfm.inference.preprocess.SVDAugmenter`); 0 appends none.

        Returns:
            The fitted sibling, a single model (``n_ensemble=1``) with no ensemble settings.
        """
        twin = FinancialTFMClassifier(
            self.model_,
            device=self.device,
            max_context=self.max_context,
            context_strategy=self.context_strategy,
            correct_prior=self.correct_prior,
            random_state=seed,
            query_chunk=self.query_chunk,
            retrieval_groups=self.retrieval_groups,
            retrieval_min_positive=self.retrieval_min_positive,
            feature_transform=(
                self.feature_transform if feature_transform is None else feature_transform
            ),
            prototype_minority_ratio=self.prototype_minority_ratio,
            n_ensemble=1,
            feature_chunk=self.feature_chunk,
        )
        # private, not a constructor argument: a member's SVD is the parent's decision, and a
        # public single-model knob would be a second way to ask for the same thing
        twin._member_svd = svd_components
        return twin.fit(self._raw_X if X is None else X, self._raw_y if y is None else y)

    @torch.no_grad()
    def transform_representation(
        self, X: np.ndarray, stage: Literal["encode_rows", "encoder_mid", "pre_head"] = "encode_rows"
    ) -> np.ndarray:
        """Row representations for ``X``, at a chosen depth short of the classification head.

        Task 48.22: a pretrained TFM's context-conditioned embeddings, handed to a
        downstream model rather than used end-to-end, is a pattern this project had not
        tried (Neuralk-AI's ``TabPfnVectorizer``, ``docs/paper/RELATED_WORK.md``). §130
        measured that a plain classifier on the ``"encode_rows"`` stage beats the model's own
        head, 5 of 5 folds. **That stage is shallower than "the head's own input."**
        :meth:`FinancialTFM.forward` runs three more steps after ``encode_rows`` returns: it
        adds ``y_emb`` (row-level label conditioning), passes the result through
        ``self.encoder`` (a full row-to-row ``nn.TransformerEncoder``, not merely the
        within-feature attention already inside ``encode_rows``), then normalises. Only that
        final tensor -- ``"pre_head"`` here -- is what ``self.head`` actually reads. Comparing
        a downstream classifier against the two stages separately is what distinguishes "a
        linear probe reads any depth better than this head does" from "linear legibility falls
        off through this specific stack," which is a different and more common finding in
        representation learning generally.

        Uses :meth:`fit`'s stored context exactly as :meth:`predict_proba` does -- same
        conditioner, same padding, same column identities at ``self.random_state`` -- so a
        representation and a prediction for the same row come from the same forward pass in
        spirit, differing only in which layer is read. Chunked the same way, for the same
        reason: a query never attends to another query, so splitting the rows changes nothing.

        Args:
            X: ``(n, n_features)`` query rows, matching the width seen by :meth:`fit`.
            stage: ``"encode_rows"`` (§130's stage; cell/column/within-feature attention and
                pooling, no label conditioning past the cell level, no row-to-row encoder),
                ``"encoder_mid"`` (§131's stage plus ``y_emb`` and the first half of
                ``self.encoder``'s layers, rounded up -- partway through the row-to-row
                transformer rather than before or after all of it, no final ``self.norm``), or
                ``"pre_head"`` (every step ``forward`` runs short of ``self.head`` itself).

        Returns:
            ``(n, d_model)`` row representations, one per query row.
        """
        check_is_fitted(self, "_ctx_X")
        self.model_.assert_trained_for("classification")
        self._refuse_wide("transform_representation")
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("feature width at predict time must match fit time")
        X = self._condition(X)
        if X.shape[0] > self.query_chunk:
            return np.concatenate(
                [
                    self._representation_chunk(X[i : i + self.query_chunk], stage)
                    for i in range(0, X.shape[0], self.query_chunk)
                ]
            )
        return self._representation_chunk(X, stage)

    @torch.no_grad()
    def _representation_chunk(
        self, X: np.ndarray, stage: Literal["encode_rows", "encoder_mid", "pre_head"] = "encode_rows"
    ) -> np.ndarray:
        """One chunk of query rows, encoded but not classified. See :meth:`transform_representation`."""
        ctx_X, ctx_y = self._ctx_X, self._ctx_y
        n_ctx = ctx_X.shape[0]
        cfg = self.model_.cfg
        Xp = np.full((1, n_ctx + X.shape[0], cfg.max_features), np.nan, dtype=np.float32)
        Xp[0, :n_ctx, : ctx_X.shape[1]] = ctx_X
        Xp[0, n_ctx:, : X.shape[1]] = X
        yp = np.zeros((1, n_ctx + X.shape[0]), dtype=np.int64)
        yp[0, :n_ctx] = ctx_y
        Xt = torch.from_numpy(Xp).to(self.device)
        yt = torch.from_numpy(yp).to(self.device)
        N = Xt.shape[1]
        h = self.model_.encode_rows(Xt, n_ctx, column_id_seed=int(self.random_state), y=yt)
        if stage in ("encoder_mid", "pre_head"):
            import torch.nn.functional as Fnn

            B = h.shape[0]
            y_onehot = Fnn.one_hot(yt[:, :n_ctx].clamp(0, cfg.max_classes - 1), cfg.max_classes).to(
                h.dtype
            )
            y_emb = torch.cat(
                [
                    self.model_.y_proj(y_onehot),
                    self.model_.query_token.expand(B, N - n_ctx, -1),
                ],
                dim=1,
            )
            h = h + y_emb
            # through the model's own row stage, so attn_impl and attn_length_scale apply here
            # as they do in forward
            if stage == "pre_head":
                h = self.model_._row_attention(h, n_ctx)
                h = self.model_.norm(h)
            else:
                # Half the row-to-row stack, rounded up -- partway through, not before or
                # after all of it. No final `self.norm`: that normalisation is shaped for
                # `self.head`'s input, not for a linear probe reading an intermediate layer.
                n_half = (len(self.model_.encoder.layers) + 1) // 2
                h = self.model_._row_attention(h, n_ctx, n_layers=n_half)
        return h[0, n_ctx:].cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities for every row, scored in exact chunks.

        Args:
            X: ``(n, n_features)`` query rows, matching the width seen by :meth:`fit`.

        Returns:
            ``(n, n_classes)`` probabilities.
        """
        check_is_fitted(self, "_ctx_X")
        self.model_.assert_trained_for("classification")
        if self._uses_members():
            # a wide table's members each slice their own columns, so a query table of the
            # wrong width would be sliced rather than caught by the members' own check
            wide = getattr(self, "_member_columns", None) is not None
            if wide and np.shape(X)[1] != self.n_features_in_:
                raise ValueError("feature width at predict time must match fit time")
            # average probabilities, not logits: the members disagree about the base rate
            # their context implies, and averaging in logit space would let one confident
            # member dominate the mean rather than contribute one vote to it
            members = [
                self._member_proba(X, self.random_state + i, i) for i in range(self.n_members_)
            ]
            return np.mean(members, axis=0)
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("feature width at predict time must match fit time")
        X = self._condition(X)
        if self.context_strategy == "retrieval":
            return self._run_retrieval(X, self._predict_chunk, len(self.classes_))
        if X.shape[0] > self.query_chunk:
            return np.concatenate(
                [
                    self._predict_chunk(X[i : i + self.query_chunk])
                    for i in range(0, X.shape[0], self.query_chunk)
                ]
            )
        return self._predict_chunk(X)

    @torch.no_grad()
    def _predict_chunk(
        self,
        X: np.ndarray,
        ctx: tuple[np.ndarray, np.ndarray] | None = None,
        pooled_rate: float | None = None,
    ) -> np.ndarray:
        """Score one chunk of queries against a context, in a single forward pass.

        **Chunking is exact, not an approximation.** The row mask forbids a query attending to
        any other query, so a prediction depends only on the context and itself and splitting
        the queries changes nothing. That does not hold for ``context_strategy="retrieval"``,
        where the context is chosen per group.

        Args:
            X: ``(n, n_features)`` conditioned query rows.
            ctx: Optional ``(context_X, context_y)`` override, used by the retrieval path.
                Defaults to the context stored at fit time.
            pooled_rate: Optional positive rate for the base-rate correction, used when the
                retrieval path pools several groups whose contexts imply different priors.

        Returns:
            ``(n, n_classes)`` probabilities.
        """
        ctx_X, ctx_y = ctx if ctx is not None else (self._ctx_X, self._ctx_y)
        n_ctx = ctx_X.shape[0]
        cfg = self.model_.cfg
        Xp = np.full((1, n_ctx + X.shape[0], cfg.max_features), np.nan, dtype=np.float32)
        Xp[0, :n_ctx, : ctx_X.shape[1]] = ctx_X
        Xp[0, n_ctx:, : X.shape[1]] = X
        yp = np.zeros((1, n_ctx + X.shape[0]), dtype=np.int64)
        yp[0, :n_ctx] = ctx_y
        Xt = torch.from_numpy(Xp).to(self.device)
        yt = torch.from_numpy(yp).to(self.device)
        nc = torch.tensor([len(self.classes_)], device=self.device)
        # Each ensemble member draws its own column identities, keyed off its own seed, so
        # n_ensemble > 1 averages over them and recovers the column-order invariance that
        # decision D12 traded for expressiveness. A single member is still deterministic.
        logits = self.model_(Xt, yt, n_ctx, nc, column_id_seed=int(self.random_state))[
            0, n_ctx:, : len(self.classes_)
        ]
        if self.correct_prior:
            prior = (
                self._log_prior_shift
                if pooled_rate is None
                else self._pooled_prior_shift(pooled_rate)
            )
            shift = torch.from_numpy(prior).to(logits.device, logits.dtype)
            logits = logits + shift
        return torch.softmax(logits, dim=-1).cpu().numpy()

    @torch.no_grad()
    def predict_term_structure(self, X: np.ndarray) -> np.ndarray:
        """Cumulative PD across horizons for every row, corrected and coherent.

        **Use this rather than calling** :meth:`FinancialTFM.term_structure` **directly.**
        Doing the latter bypasses the base-rate correction, which cost this project a whole
        pretraining run and a wrong diagnosis: the term-structure arm of the out-of-time
        harness reached into the fitted context and ran the model itself, so it reported a
        12.8% default rate against a 0.47% truth and the error was misread as the synthetic
        prior being unable to reach low default rates (``docs/results/FINDINGS.md`` §28).

        Chunked exactly, for the reason given on :meth:`predict_proba`.

        Args:
            X: ``(n, n_features)`` query rows, matching the width seen by :meth:`fit`.

        Returns:
            ``(n, n_horizons)`` cumulative PD, non-decreasing in the horizon for every row.
            Coherence survives the correction because the shift is strictly increasing.

        Raises:
            RuntimeError: If the model has no hazard head.
            ValueError: If the feature width does not match :meth:`fit`.
        """
        check_is_fitted(self, "_ctx_X")
        if self.model_.hazard is None:
            raise RuntimeError("this checkpoint has no hazard head; pretrain with --n-horizons K")
        self.model_.assert_trained_for("survival")
        self._refuse_wide("predict_term_structure")
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("feature width at predict time must match fit time")
        X = self._condition(X)
        if self.context_strategy == "retrieval":
            plan = self._retrieval_plan(X)
            out = np.empty((X.shape[0], self.model_.hazard.n_horizons), dtype=np.float32)
            for q_idx, c_idx in plan:
                ctx = (self._pool_X[c_idx], self._pool_y[c_idx])
                for start in range(0, q_idx.size, self.query_chunk):
                    block = q_idx[start : start + self.query_chunk]
                    out[block] = self._term_structure_chunk(X[block], ctx)
            rate = self._pooled_context_rate(plan)
            if self.correct_prior and 0.0 < rate < 1.0 and 0.0 < self._full_rate < 1.0:
                out = shift_cumulative_pd(
                    torch.from_numpy(out), base_rate_shift(rate, self._full_rate)
                ).numpy()
            return out
        curves = [
            self._term_structure_chunk(X[i : i + self.query_chunk])
            for i in range(0, X.shape[0], self.query_chunk)
        ]
        curve = torch.from_numpy(np.concatenate(curves)) if curves else torch.empty(0)
        if self.correct_prior and 0.0 < self._ctx_rate < 1.0 and 0.0 < self._full_rate < 1.0:
            curve = shift_cumulative_pd(curve, base_rate_shift(self._ctx_rate, self._full_rate))
        return curve.numpy()

    @torch.no_grad()
    def _term_structure_chunk(
        self, X: np.ndarray, ctx: tuple[np.ndarray, np.ndarray] | None = None
    ) -> np.ndarray:
        """Uncorrected cumulative PD for one chunk of queries."""
        ctx_X, ctx_y = ctx if ctx is not None else (self._ctx_X, self._ctx_y)
        n_ctx = ctx_X.shape[0]
        cfg = self.model_.cfg
        Xp = np.full((1, n_ctx + X.shape[0], cfg.max_features), np.nan, dtype=np.float32)
        Xp[0, :n_ctx, : ctx_X.shape[1]] = ctx_X
        Xp[0, n_ctx:, : X.shape[1]] = X
        yp = np.zeros((1, n_ctx + X.shape[0]), dtype=np.int64)
        yp[0, :n_ctx] = ctx_y
        out = self.model_.term_structure(
            torch.from_numpy(Xp).to(self.device), torch.from_numpy(yp).to(self.device), n_ctx
        )
        return out[0].cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard class labels, as the arg-max of :meth:`predict_proba`.

        **Prefer `predict_proba` for credit work.** A lender provisions against the *level* of
        a default probability, and an arg-max at a 0.4% base rate predicts the majority class
        for nearly every row — which is accurate and useless.

        Args:
            X: ``(n, n_features)`` query rows.

        Returns:
            ``(n,)`` labels drawn from ``classes_``.
        """
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]
