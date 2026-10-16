"""Large-table wrappers around :class:`FinancialTFMClassifier`: a tree partition, and boosting.

Why this exists
---------------
The in-context model conditions on at most ``max_context`` rows, so on V4FinBench's
~600,000-row training folds it sees a fraction of a percent of the table. Retrieval
(:mod:`fintfm.inference.retrieval`) chooses *which* rows per query; the two wrappers here
are the other published routes for spending more of a large table on a fixed context
(audit AU5.8). Both are experiment levers, opt-in, and change nothing about the classifier
they wrap. Nothing in this repository has measured either on a score yet.

- :class:`PartitionedTFMClassifier` cuts the training rows with a shallow decision tree and
  gives every leaf its own context, so a query is conditioned on rows from its own region of
  feature space. With ``n_bags > 1`` it fits one tree per random row subset and averages
  their probabilities, forest style. Written from the general description of tree-partitioned
  and forest-of-PFN inference (the ``rf_pfn`` idea in the TabPFN extensions); no code from that
  or any other tabular-foundation-model project is used, per ``CLAUDE.md``.
- :class:`BoostedTFMClassifier` treats small-context models as weak learners. Each round
  draws its context with probability rising with the current ensemble's residual on each
  training row, so later rounds look at the rows the earlier ones got wrong, and the rounds
  are combined in logit space with a learning rate. Written from the description of BoostPFN
  (arXiv:2503.01256); again no code is used.

Cost, stated because it is the reason to be careful
---------------------------------------------------
A partition stores one context per leaf and scores each query once per bag, so its query
cost is ``n_bags`` times the plain classifier's. Boosting scores its residual pool once per
round to update the residuals, so ``n_rounds * pool_rows`` extra query rows are scored at
fit time; ``pool_rows`` caps that pool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from fintfm.inference.classifier import FinancialTFMClassifier

if TYPE_CHECKING:  # the config module is only needed for the annotation
    from fintfm.config import LargeNConfig

#: How :class:`BoostedTFMClassifier` turns a residual into a sampling weight.
ResidualWeighting = Literal["abs", "exp"]

#: Probability clip before a logit, so a saturated member cannot produce an infinite score.
_P_CLIP = 1e-6


def _inner(template: FinancialTFMClassifier, **overrides: Any) -> FinancialTFMClassifier:
    """An unfitted copy of ``template`` with some parameters replaced.

    Built from ``get_params(deep=False)`` rather than :func:`sklearn.base.clone`, which deep-
    copies every parameter: the wrapped network would then be copied once per leaf, bag or
    round. Here every inner estimator shares the one model instance, which the classifier
    never modifies (a fine-tuning estimator updates its own copy).

    Args:
        template: The classifier whose settings the inner estimators take.
        **overrides: Parameters to replace, e.g. ``random_state`` or ``max_context``.

    Returns:
        A new, unfitted :class:`FinancialTFMClassifier`.
    """
    return FinancialTFMClassifier(**{**template.get_params(deep=False), **overrides})


def _logit(p: np.ndarray) -> np.ndarray:
    """Log-odds of a probability, clipped away from 0 and 1."""
    p = np.clip(np.asarray(p, dtype=np.float64), _P_CLIP, 1.0 - _P_CLIP)
    return np.log(p) - np.log1p(-p)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function, written to stay finite for large ``|z|``."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=np.float64)))


class PartitionedTFMClassifier(BaseEstimator, ClassifierMixin):
    """A shallow decision tree routes rows to leaves, and each leaf has its own context.

    The tree is fitted on the training rows' labels, so its leaves are regions where the
    default rate differs, and a query is scored against a context drawn from its own leaf
    rather than from the whole table. A leaf too small to hold a useful context, or missing
    any class, falls back to a global context fitted on every training row.

    Args:
        estimator: An unfitted :class:`FinancialTFMClassifier` whose settings every leaf, and
            the global fallback, uses. Its ``random_state`` is replaced per leaf.
        max_depth: Tree depth; at most ``2 ** max_depth`` leaves. Capped at 3, because the
            point is a few large regions, not a fine partition that leaves each context
            only its own rows.
        min_samples_leaf: The tree's minimum leaf size, as a row count or a fraction of the
            (bagged) training rows. Large by default for the same reason.
        n_bags: Trees in the forest. ``1`` fits one tree on every training row; above 1 each
            tree is fitted on its own random ``bag_frac`` subset, and the bags' probabilities
            are averaged.
        bag_frac: Fraction of training rows each bag draws, without replacement. Used only
            when ``n_bags > 1``.
        min_leaf_rows: A leaf with fewer training rows than this uses the global context.
        min_class_rows: A leaf with fewer than this many rows of any class uses the global
            context; a leaf with one class could not score the others at all.
        random_state: Seed for the bags, the trees and the leaf contexts.

    Attributes:
        classes_: Class labels, from :meth:`fit`.
        trees_: One fitted :class:`~sklearn.tree.DecisionTreeClassifier` per bag.
        leaf_estimators_: Per bag, a dict from leaf id to its fitted classifier, or to
            ``None`` where the leaf falls back to ``global_estimator_``.
        global_estimator_: The fallback classifier, fitted on every training row.
        bag_rows_: Per bag, the training-row indices that bag was fitted on.
    """

    def __init__(
        self,
        estimator: FinancialTFMClassifier,
        max_depth: int = 2,
        min_samples_leaf: float = 0.05,
        n_bags: int = 1,
        bag_frac: float = 0.6,
        min_leaf_rows: int = 200,
        min_class_rows: int = 2,
        random_state: int = 0,
    ) -> None:
        """Store the settings. See the class docstring for every argument."""
        self.estimator = estimator
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_bags = n_bags
        self.bag_frac = bag_frac
        self.min_leaf_rows = min_leaf_rows
        self.min_class_rows = min_class_rows
        self.random_state = random_state

    def _check_settings(self) -> None:
        """Reject settings that would silently change what the partition means.

        Raises:
            ValueError: For a depth outside ``[1, 3]``, fewer than one bag, or a
                ``bag_frac`` outside ``(0, 1]``.
        """
        if not 1 <= self.max_depth <= 3:
            raise ValueError(f"max_depth must lie in [1, 3], got {self.max_depth}")
        if self.n_bags < 1:
            raise ValueError(f"n_bags must be at least 1, got {self.n_bags}")
        if not 0.0 < self.bag_frac <= 1.0:
            raise ValueError(f"bag_frac must lie in (0, 1], got {self.bag_frac}")

    def _leaf_is_usable(self, y_leaf: np.ndarray) -> bool:
        """Whether a leaf's rows can carry a context of their own.

        Args:
            y_leaf: The labels of the training rows the tree put in this leaf.

        Returns:
            ``True`` when the leaf has at least ``min_leaf_rows`` rows and at least
            ``min_class_rows`` rows of every class seen in :meth:`fit`.
        """
        if y_leaf.size < self.min_leaf_rows:
            return False
        counts = np.array([(y_leaf == c).sum() for c in self.classes_])
        return bool((counts >= self.min_class_rows).all())

    def fit(
        self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None
    ) -> PartitionedTFMClassifier:
        """Fit the trees, then store one context per usable leaf and a global fallback.

        Args:
            X: ``(n, F)`` training features. NaN is allowed; the tree routes it natively.
            y: ``(n,)`` labels.
            groups: Optional ``(n,)`` group labels, passed to each inner classifier's
                ``fit`` (used only when it fine-tunes).

        Returns:
            self.
        """
        from sklearn.tree import DecisionTreeClassifier

        self._check_settings()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        groups = None if groups is None else np.asarray(groups)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        base_seed = int(self.estimator.random_state)
        self.global_estimator_ = _inner(self.estimator, random_state=base_seed).fit(
            X, y, groups=groups
        )
        self.trees_: list[Any] = []
        self.leaf_estimators_: list[dict[int, FinancialTFMClassifier | None]] = []
        self.bag_rows_: list[np.ndarray] = []
        n = len(y)
        for b in range(self.n_bags):
            if self.n_bags == 1:
                rows = np.arange(n)
            else:
                size = max(1, round(self.bag_frac * n))
                rows = np.sort(rng.choice(n, size=size, replace=False))
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=int(rng.integers(1 << 31)),
            ).fit(X[rows], y[rows])
            leaf_of_row = tree.apply(X[rows])
            leaves: dict[int, FinancialTFMClassifier | None] = {}
            for leaf in np.unique(leaf_of_row):
                members = rows[leaf_of_row == leaf]
                if not self._leaf_is_usable(y[members]):
                    leaves[int(leaf)] = None
                    continue
                # a distinct seed per leaf and bag, so two leaves never share a context draw
                # or a column-identity draw by accident
                seed = base_seed + 1 + b * 1000 + int(leaf)
                leaves[int(leaf)] = _inner(self.estimator, random_state=seed).fit(
                    X[members], y[members],
                    groups=None if groups is None else groups[members],
                )
            self.trees_.append(tree)
            self.leaf_estimators_.append(leaves)
            self.bag_rows_.append(rows)
        return self

    def route(self, X: np.ndarray) -> np.ndarray:
        """The leaf each query lands in, per bag.

        Args:
            X: ``(n, F)`` query rows.

        Returns:
            ``(n_bags, n)`` leaf ids. Whether a leaf has its own context or falls back to the
            global one is ``leaf_estimators_[b][leaf] is None``.
        """
        check_is_fitted(self, "trees_")
        X = np.asarray(X, dtype=np.float32)
        return np.stack([tree.apply(X) for tree in self.trees_])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities, each query scored against its own leaf's context.

        Args:
            X: ``(n, F)`` query rows.

        Returns:
            ``(n, n_classes)`` probabilities, averaged over bags.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("feature width at predict time must match fit time")
        routes = self.route(X)
        out = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)
        for b, leaves in enumerate(self.leaf_estimators_):
            for leaf in np.unique(routes[b]):
                rows = np.flatnonzero(routes[b] == leaf)
                est = leaves[int(leaf)]
                if est is None:
                    est = self.global_estimator_
                out[rows] += est.predict_proba(X[rows])
        return (out / len(self.leaf_estimators_)).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard class labels, as the arg-max of :meth:`predict_proba`."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class BoostedTFMClassifier(BaseEstimator, ClassifierMixin):
    """Boosting rounds of small-context in-context models, combined in logit space.

    Binary tasks only. The ensemble's score starts at the training rows' prior log-odds
    ``F_0``. Round ``m`` then:

    1. computes every pool row's residual ``y - sigmoid(F)`` under the current ensemble,
    2. draws ``context_rows`` rows without replacement with probability proportional to
       ``|residual|`` (``weighting="abs"``) or ``exp(|residual|)`` (``"exp"``), forcing one
       row of each class in if the draw missed it,
    3. fits a :class:`FinancialTFMClassifier` on that context and takes its **evidence**
       ``h_m(x) = logit p_m(x) - logit(context positive rate)``, the part of its log-odds
       beyond what its own context's base rate implies (the context is resampled on purpose,
       so its rate is not the population's),
    4. picks a step ``gamma_m`` in ``[0, max_step]`` minimising log-loss on the pool rows
       **outside** that context, so a learner is never scored on the rows it conditioned on,
    5. updates ``F <- F + learning_rate * gamma_m * h_m``.

    A learner whose evidence does not help gets ``gamma_m = 0`` and contributes nothing.

    Args:
        estimator: An unfitted :class:`FinancialTFMClassifier` whose settings every round
            uses; ``max_context`` and ``random_state`` are replaced per round.
        n_rounds: Boosting rounds, one context and one forward pass over the pool each.
        context_rows: Context rows per round. Small by design: each learner is weak and cheap.
        learning_rate: Shrinkage on every round's step.
        weighting: ``"abs"`` or ``"exp"``, how a residual becomes a sampling weight.
        max_step: Upper bound of the line search for ``gamma_m``.
        pool_rows: Training rows on which residuals are tracked and contexts are drawn.
            ``None`` uses every row; a cap keeps the fit-time cost at ``n_rounds *
            pool_rows`` query rows instead of ``n_rounds * n``. The pool is a uniform
            sample, drawn once.
        random_state: Seed for the pool, every round's draw and every learner's seed.

    Attributes:
        classes_: The two class labels.
        estimators_: The fitted learner of every round.
        gammas_: The line-searched step of every round.
        contexts_: Every round's context, as indices into the training rows.
        pool_: The training-row indices residuals were tracked on.
        init_logit_: ``F_0``, the pool's prior log-odds.
        train_loss_: Pool log-loss after every round, for checking the rounds help.
    """

    def __init__(
        self,
        estimator: FinancialTFMClassifier,
        n_rounds: int = 8,
        context_rows: int = 512,
        learning_rate: float = 0.5,
        weighting: ResidualWeighting = "abs",
        max_step: float = 4.0,
        pool_rows: int | None = None,
        random_state: int = 0,
    ) -> None:
        """Store the settings. See the class docstring for every argument."""
        self.estimator = estimator
        self.n_rounds = n_rounds
        self.context_rows = context_rows
        self.learning_rate = learning_rate
        self.weighting = weighting
        self.max_step = max_step
        self.pool_rows = pool_rows
        self.random_state = random_state

    def sampling_weights(self, residual: np.ndarray) -> np.ndarray:
        """Normalised context-sampling probabilities from residuals.

        A floor of ``1e-12`` of the mean weight keeps every row drawable, so a round never
        runs short of nonzero-weight rows when the ensemble already fits most of the pool.

        Args:
            residual: ``(n,)`` residuals ``y - p``.

        Returns:
            ``(n,)`` probabilities summing to 1.

        Raises:
            ValueError: For an unknown ``weighting``.
        """
        r = np.abs(np.asarray(residual, dtype=np.float64))
        if self.weighting == "abs":
            w = r
        elif self.weighting == "exp":
            w = np.exp(r)
        else:
            raise ValueError(f"weighting must be 'abs' or 'exp', got {self.weighting!r}")
        w = w + 1e-12 * max(float(w.mean()), 1e-12)
        return w / w.sum()

    def _draw_context(
        self, y_pool: np.ndarray, prob: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """One round's context, drawn by residual weight, with both classes present.

        Args:
            y_pool: ``(n,)`` coded labels of the pool (0 or 1).
            prob: ``(n,)`` sampling probabilities from :meth:`sampling_weights`.
            rng: The fit's random generator.

        Returns:
            Sorted positions into the pool.
        """
        k = min(self.context_rows, y_pool.size)
        picked = rng.choice(y_pool.size, size=k, replace=False, p=prob)
        for c in (0, 1):
            if not (y_pool[picked] == c).any():
                # the heaviest row of the missing class replaces the lightest row drawn, so
                # the learner can score both classes and the context size is unchanged
                cand = np.flatnonzero(y_pool == c)
                picked[np.argmin(prob[picked])] = cand[np.argmax(prob[cand])]
        return np.sort(picked)

    def _line_search(self, f: np.ndarray, h: np.ndarray, y: np.ndarray) -> float:
        """The step along ``h`` that minimises log-loss from score ``f``.

        Args:
            f: ``(n,)`` current scores on held-out pool rows.
            h: ``(n,)`` the new learner's evidence on the same rows.
            y: ``(n,)`` their coded labels.

        Returns:
            ``gamma`` in ``[0, max_step]``.
        """
        from scipy.optimize import minimize_scalar

        if y.size == 0 or not np.any(h):
            return 0.0

        def loss(g: float) -> float:
            """Mean log-loss after a step of ``g`` along ``h``."""
            z = f + g * h
            # log(1 + e^z) - y z, the binary log-loss in score form
            return float(np.mean(np.logaddexp(0.0, z) - y * z))

        res = minimize_scalar(loss, bounds=(0.0, self.max_step), method="bounded")
        g = float(res.x)
        return g if loss(g) < loss(0.0) else 0.0

    def fit(
        self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None
    ) -> BoostedTFMClassifier:
        """Run the boosting rounds.

        Args:
            X: ``(n, F)`` training features.
            y: ``(n,)`` binary labels.
            groups: Optional ``(n,)`` group labels, passed to each learner's ``fit``.

        Returns:
            self.

        Raises:
            ValueError: If ``y`` does not have exactly two classes, or a setting is out of
                range.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                f"BoostedTFMClassifier is binary only, got {len(self.classes_)} classes"
            )
        if self.n_rounds < 1 or self.context_rows < 2 or self.learning_rate <= 0:
            raise ValueError("n_rounds >= 1, context_rows >= 2 and learning_rate > 0 required")
        self.n_features_in_ = X.shape[1]
        yc = (y == self.classes_[1]).astype(np.float64)
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        if self.pool_rows is not None and self.pool_rows < n:
            pool = np.sort(rng.choice(n, size=self.pool_rows, replace=False))
        else:
            pool = np.arange(n)
        y_pool = yc[pool]
        if y_pool.min() == y_pool.max():
            raise ValueError("the residual pool holds one class; raise pool_rows")
        rate = float(y_pool.mean())
        self.pool_ = pool
        self.init_logit_ = float(_logit(np.array([rate]))[0])
        f = np.full(pool.size, self.init_logit_)
        base_seed = int(self.estimator.random_state)
        self.estimators_: list[FinancialTFMClassifier] = []
        self.gammas_: list[float] = []
        self.contexts_: list[np.ndarray] = []
        self._ctx_logits: list[float] = []
        self.train_loss_: list[float] = []
        for m in range(self.n_rounds):
            prob = self.sampling_weights(y_pool - _sigmoid(f))
            ctx = self._draw_context(y_pool.astype(np.int64), prob, rng)
            rows = pool[ctx]
            learner = _inner(
                self.estimator, max_context=len(rows), random_state=base_seed + 1 + m
            ).fit(X[rows], y[rows], groups=None if groups is None else groups[rows])
            ctx_logit = float(_logit(np.array([yc[rows].mean()]))[0])
            h = self._evidence(learner, ctx_logit, X[pool])
            held = np.ones(pool.size, dtype=bool)
            held[ctx] = False
            gamma = self._line_search(f[held], h[held], y_pool[held])
            f = f + self.learning_rate * gamma * h
            self.estimators_.append(learner)
            self.gammas_.append(gamma)
            self.contexts_.append(rows)
            self._ctx_logits.append(ctx_logit)
            self.train_loss_.append(float(np.mean(np.logaddexp(0.0, f) - y_pool * f)))
        return self

    def _evidence(
        self, learner: FinancialTFMClassifier, ctx_logit: float, X: np.ndarray
    ) -> np.ndarray:
        """A learner's log-odds beyond its own context's base rate.

        Args:
            learner: A fitted round learner.
            ctx_logit: Log-odds of its context's positive rate.
            X: ``(n, F)`` rows to score.

        Returns:
            ``(n,)`` evidence ``logit p - ctx_logit``.
        """
        return _logit(learner.predict_proba(X)[:, 1]) - ctx_logit

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """The ensemble's log-odds of the positive class.

        Args:
            X: ``(n, F)`` query rows.

        Returns:
            ``(n,)`` scores ``F_0 + learning_rate * sum_m gamma_m * h_m(x)``.
        """
        check_is_fitted(self, "estimators_")
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("feature width at predict time must match fit time")
        f = np.full(X.shape[0], self.init_logit_)
        for learner, gamma, ctx_logit in zip(
            self.estimators_, self.gammas_, self._ctx_logits, strict=True
        ):
            if gamma > 0.0:
                f = f + self.learning_rate * gamma * self._evidence(learner, ctx_logit, X)
        return f

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities from :meth:`decision_function`.

        Args:
            X: ``(n, F)`` query rows.

        Returns:
            ``(n, 2)`` probabilities in ``classes_`` order.
        """
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p, p]).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard class labels, as the arg-max of :meth:`predict_proba`."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def wrap_large_n(
    estimator: FinancialTFMClassifier, cfg: LargeNConfig
) -> FinancialTFMClassifier | PartitionedTFMClassifier | BoostedTFMClassifier:
    """Wrap a classifier as the configuration's ``large_n`` section asks.

    Args:
        estimator: The unfitted classifier the wrapper's inner estimators copy.
        cfg: The ``large_n`` section; its ``partition_*`` or ``boost_*`` fields become the
            wrapper's arguments with the prefix dropped.

    Returns:
        ``estimator`` itself when ``cfg.method == "none"``, so the default path is the plain
        classifier object and nothing else; otherwise an unfitted wrapper around it, seeded
        with the classifier's own ``random_state``.

    Raises:
        ValueError: For an unknown ``cfg.method``.
    """
    from dataclasses import asdict

    if cfg.method == "none":
        return estimator
    if cfg.method not in ("partition", "boost"):
        raise ValueError(f"large_n.method must be none, partition or boost, got {cfg.method!r}")
    prefix = f"{cfg.method}_"
    kwargs = {k[len(prefix):]: v for k, v in asdict(cfg).items() if k.startswith(prefix)}
    cls = PartitionedTFMClassifier if cfg.method == "partition" else BoostedTFMClassifier
    return cls(estimator, random_state=int(estimator.random_state), **kwargs)
