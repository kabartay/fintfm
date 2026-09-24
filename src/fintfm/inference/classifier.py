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

from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

from fintfm.inference.preprocess import FeatureConditioner, FeatureTransform
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
            It is a workaround, not a cure, and should be reported as one.
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
    ) -> None:
        """Configure the classifier. See the class docstring — every argument is documented
        there with the measurement that set its default."""
        self.model = FinancialTFM.load(model, map_location=device) if isinstance(model, str) else model
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
        # Identity-preserving, so it is safe to default on: asserted byte-for-byte by
        # tests/test_model.py::test_feature_chunking_is_an_identity. No effect on a
        # checkpoint with n_cell_blocks=0, which has no row-within-feature stage at all.
        self.model.feature_chunk = feature_chunk
        self.model.to(device).eval()

    def fit(self, X: np.ndarray, y: np.ndarray) -> FinancialTFMClassifier:
        """Store the table as in-context evidence. **No gradient steps are taken.**

        The name is sklearn's, and it is misleading here by convention rather than by choice:
        nothing is learned. The table is conditioned, subsampled to at most ``max_context``
        rows by the configured strategy, and kept. Everything that would be "training" in
        another estimator happened during pretraining, on synthetic data.

        Args:
            X: ``(n, n_features)`` training features. Must not exceed the checkpoint's
                ``max_features``.
            y: ``(n,)`` labels. The number of distinct values must not exceed the
                checkpoint's ``max_classes``.

        Returns:
            self.

        Raises:
            ValueError: If the task exceeds the checkpoint's class or feature capacity. It
                raises rather than truncating, so a caller records a skip instead of a
                meaningless score.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        # kept unconditioned so an ensemble member can redraw its own context and refit the
        # conditioner from scratch, rather than inheriting this instance's draw
        self._raw_X, self._raw_y = X, y
        # fitted on training rows only, so a query never influences its own conditioning
        self._conditioner = FeatureConditioner(
            self.feature_transform, random_state=self.random_state
        ).fit(X)
        X = self._conditioner.transform(X)
        self.classes_ = np.unique(y)
        if len(self.classes_) > self.model.cfg.max_classes:
            raise ValueError(
                f"model supports at most {self.model.cfg.max_classes} classes, got {len(self.classes_)}"
            )
        if X.shape[1] > self.model.cfg.max_features:
            raise ValueError(f"model supports at most {self.model.cfg.max_features} features, got {X.shape[1]}")
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
        n_classes = len(self.classes_)
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
                (full_prior > 0) & (ctx_prior > 0), np.log(np.maximum(full_prior, 1e-12))
                - np.log(np.maximum(ctx_prior, 1e-12)), 0.0
            )
        return self

    def _pooled_prior_shift(self, pooled_rate: float) -> np.ndarray:
        """Class-prior correction from a pooled context rate, for the retrieval path.

        One shift for every query, for the reason given on :meth:`_pooled_context_rate`: a
        per-group shift inverts the cross-group ranking that retrieval exists to exploit.

        Args:
            pooled_rate: Positive-class rate across all retrieved contexts.

        Returns:
            ``(n_classes,)`` log-odds shift, zero for any class absent from either side.
        """
        n_classes = len(self.classes_)
        ctx_prior = np.full(n_classes, (1.0 - pooled_rate) / max(n_classes - 1, 1))
        ctx_prior[n_classes - 1] = pooled_rate
        with np.errstate(divide="ignore"):
            return np.where(
                (self._full_prior > 0) & (ctx_prior > 0),
                np.log(np.maximum(self._full_prior, 1e-12))
                - np.log(np.maximum(ctx_prior, 1e-12)),
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
            each context serves. Also stored as ``pooled_context_rate_``.
        """
        pos_class = len(self.classes_) - 1
        weight = sum(q.size for q, _ in plan)
        if weight == 0:
            return self._full_rate
        total = sum(
            q.size * float((self._pool_y[c] == pos_class).mean()) for q, c in plan
        )
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

    def _member_proba(self, X: np.ndarray, seed: int) -> np.ndarray:
        """One ensemble member: its own context draw, feature subset and label orientation."""
        rng = np.random.default_rng(seed)
        Xa, Xb = self._raw_X, np.asarray(X, dtype=np.float32)
        if self.ensemble_feature_frac < 1.0:
            k = max(1, round(Xa.shape[1] * self.ensemble_feature_frac))
            cols = np.sort(rng.choice(Xa.shape[1], size=k, replace=False))
            Xa, Xb = Xa[:, cols], Xb[:, cols]
        twin = self._clone_with_seed(seed, Xa, self._raw_y)
        p = twin.predict_proba(Xb)
        if not self.ensemble_label_swap:
            return p
        # the label-swapped twin: relabel the context, predict, and reverse the columns back.
        # Averaging the two removes the part of the prediction that depends on which class is
        # called "1" (§45).
        swapped = self._clone_with_seed(seed, Xa, self._flip(self._raw_y))
        return 0.5 * (p + swapped.predict_proba(Xb)[:, ::-1])

    def _flip(self, y: np.ndarray) -> np.ndarray:
        """Exchange the two class labels; only meaningful for binary tasks."""
        classes = np.unique(y)
        if len(classes) != 2:
            return y
        return np.where(y == classes[0], classes[1], classes[0])

    def _clone_with_seed(
        self, seed: int, X: np.ndarray | None = None, y: np.ndarray | None = None
    ) -> FinancialTFMClassifier:
        """A sibling estimator differing in its context draw, features or label orientation."""
        twin = FinancialTFMClassifier(
            self.model, device=self.device, max_context=self.max_context,
            context_strategy=self.context_strategy, correct_prior=self.correct_prior,
            random_state=seed, query_chunk=self.query_chunk,
            retrieval_groups=self.retrieval_groups,
            retrieval_min_positive=self.retrieval_min_positive,
            feature_transform=self.feature_transform,
            prototype_minority_ratio=self.prototype_minority_ratio,
            n_ensemble=1,
        )
        return twin.fit(
            self._raw_X if X is None else X, self._raw_y if y is None else y
        )

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities for every row, scored in exact chunks.

        Args:
            X: ``(n, n_features)`` query rows, matching the width seen by :meth:`fit`.

        Returns:
            ``(n, n_classes)`` probabilities.
        """
        self.model.assert_trained_for("classification")
        if self.n_ensemble > 1 or self.ensemble_label_swap or self.ensemble_feature_frac < 1.0:
            # average probabilities, not logits: the members disagree about the base rate
            # their context implies, and averaging in logit space would let one confident
            # member dominate the mean rather than contribute one vote to it
            members = [
                self._member_proba(X, self.random_state + i) for i in range(self.n_ensemble)
            ]
            return np.mean(members, axis=0)
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self._ctx_X.shape[1]:
            raise ValueError("feature width at predict time must match fit time")
        X = self._conditioner.transform(X)
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
        cfg = self.model.cfg
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
        logits = self.model(
            Xt, yt, n_ctx, nc, column_id_seed=int(self.random_state)
        )[0, n_ctx:, : len(self.classes_)]
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
        if self.model.hazard is None:
            raise RuntimeError(
                "this checkpoint has no hazard head; pretrain with --n-horizons K"
            )
        self.model.assert_trained_for("survival")
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self._ctx_X.shape[1]:
            raise ValueError("feature width at predict time must match fit time")
        X = self._conditioner.transform(X)
        if self.context_strategy == "retrieval":
            plan = self._retrieval_plan(X)
            out = np.empty((X.shape[0], self.model.hazard.n_horizons), dtype=np.float32)
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
            curve = shift_cumulative_pd(
                curve, base_rate_shift(self._ctx_rate, self._full_rate)
            )
        return curve.numpy()

    @torch.no_grad()
    def _term_structure_chunk(
        self, X: np.ndarray, ctx: tuple[np.ndarray, np.ndarray] | None = None
    ) -> np.ndarray:
        """Uncorrected cumulative PD for one chunk of queries."""
        ctx_X, ctx_y = ctx if ctx is not None else (self._ctx_X, self._ctx_y)
        n_ctx = ctx_X.shape[0]
        cfg = self.model.cfg
        Xp = np.full((1, n_ctx + X.shape[0], cfg.max_features), np.nan, dtype=np.float32)
        Xp[0, :n_ctx, : ctx_X.shape[1]] = ctx_X
        Xp[0, n_ctx:, : X.shape[1]] = X
        yp = np.zeros((1, n_ctx + X.shape[0]), dtype=np.int64)
        yp[0, :n_ctx] = ctx_y
        out = self.model.term_structure(
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
