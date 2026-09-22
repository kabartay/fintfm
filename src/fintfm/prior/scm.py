"""Generic structural-causal prior (TabPFN-style random MLP graphs).

Mixed into training so the model learns general tabular structure (arbitrary
nonlinear boundaries, multi-class targets, categorical inputs) rather than
only the financial story. Independent implementation of the well-known idea:
sample a random layered MLP with random activations, feed Gaussian noise,
read features and target from random hidden nodes.

Task 48.17/48.19 (`openspec/changes/learn-from-peers`): edge connectivity and the
activation set were widened against TabICLv2's published prior appendix
(arXiv:2602.11139, Appendix E), reimplemented from the description rather than
copied -- no code or weights from that project enter this repository.
"""

from __future__ import annotations

import numpy as np

from fintfm.prior.base import Task


def _rank_act(v: np.ndarray) -> np.ndarray:
    """Per-column rank, rescaled to [-1, 1]. An order-statistic activation.

    §35 measures that this project's *inference-time* rank conditioning is worth +0.086 AUC on
    real financial ratios; the prior's label mechanism had no analogous function to generate
    rank-like structure for the model to learn from. Rank is taken down each column (across
    rows) rather than elementwise, since a rank is only defined relative to a population.
    """
    order = np.argsort(np.argsort(v, axis=0), axis=0).astype(np.float64)
    n = v.shape[0]
    return (order / max(n - 1, 1) - 0.5) * 2.0


def _softmax_act(v: np.ndarray) -> np.ndarray:
    """Row-wise softmax. The other order-statistic activation named in task 48.19."""
    m = v.max(axis=1, keepdims=True)
    e = np.exp(v - m)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


_ACTS = (
    np.tanh,
    np.sin,
    lambda v: np.maximum(v, 0.0),
    lambda v: v,
    lambda v: np.sign(v) * np.sqrt(np.abs(v)),
    _rank_act,
    _softmax_act,
)


def _cauchy_edge_mask(rng: np.random.Generator, n_in: int, n_out: int) -> np.ndarray:
    """Sample a boolean edge mask with heterogeneous per-node connectivity.

    Reimplements the mechanism described in TabICLv2's Appendix E.4: edge probability
    ``sigmoid(A + B_i + C_j)`` with ``A``, ``B_i``, ``C_j`` drawn i.i.d. standard Cauchy. ``A``
    sets overall connectivity; ``B_i``/``C_j`` give each source/destination node its own
    outgoing/incoming connectivity. Cauchy's heavy tails put some node pairs near-certain and
    others near-impossible to connect -- "exceptions to the rule" -- rather than the uniform
    per-layer sparsity threshold this prior used before, which could not express that some
    nodes matter far more than others.

    Args:
        rng: NumPy random generator.
        n_in: Number of source nodes (rows of the mask).
        n_out: Number of destination nodes (columns of the mask).

    Returns:
        ``(n_in, n_out)`` boolean mask.
    """
    a = rng.standard_cauchy()
    b = rng.standard_cauchy(n_in)
    c = rng.standard_cauchy(n_out)
    logits = np.clip(a + b[:, None] + c[None, :], -30.0, 30.0)
    prob = 1.0 / (1.0 + np.exp(-logits))
    return rng.random((n_in, n_out)) < prob


def _sample_graph_pool(
    rng: np.random.Generator, n_rows: int, n_features: int, max_features: int, legacy: bool
) -> np.ndarray:
    """Build one random layered graph and return every node's activations.

    Split out of :func:`sample_scm_task` so several tasks can be drawn from **one** graph
    (task 48.12). The graph is the expensive part — layers of matrix multiplies over
    ``n_rows`` — and using it for a single target throws away every other node, each of which
    is an equally valid target with a different dependency structure over the same features.

    Args:
        rng: NumPy random generator.
        n_rows: Rows to generate.
        n_features: Features the caller intends to expose; sets the graph's minimum width.
        max_features: Upper bound used to size the layer width.
        legacy: Use the pre-48.17/48.19 edge sampling and activation set.

    Returns:
        ``(n_rows, n_nodes)`` activations of every node in the graph.
    """
    acts = _ACTS[:5] if legacy else _ACTS
    n_layers = int(rng.integers(1, 5))
    width = int(rng.integers(max(4, n_features + 1), 3 * max_features + 8))
    n_noise = int(rng.integers(2, 8))
    h = rng.normal(0, 1, size=(n_rows, n_noise)) * np.exp(rng.normal(0, 0.5, n_noise))
    nodes: list[np.ndarray] = []
    for _ in range(n_layers):
        w = rng.normal(0, 1, size=(h.shape[1], width)) / np.sqrt(h.shape[1])
        w *= (
            rng.random((h.shape[1], width)) < rng.uniform(0.3, 1.0)
            if legacy
            else _cauchy_edge_mask(rng, h.shape[1], width)
        )
        act = acts[int(rng.integers(len(acts)))]
        h = act(h @ w + rng.normal(0, 0.3, size=width)) + rng.normal(
            0, rng.uniform(0, 0.2), size=(n_rows, width)
        )
        nodes.append(h)
    return np.concatenate(nodes, axis=1)


def sample_scm_task_group(
    rng: np.random.Generator,
    n_rows: int,
    n_targets: int,
    max_features: int = 24,
    min_features: int = 2,
    max_classes: int = 10,
    legacy: bool = False,
) -> list[Task]:
    """Draw ``n_targets`` tasks that share one random graph, re-targeting a different node each.

    Task 48.12, the synthetic analogue of TabDPT's self-supervised re-targeting and the weak
    form of LimiX-2's joint objective: `CLAUDE.md`'s "count the tasks, not the steps" records
    that every checkpoint here has seen 48,000 tasks against a field norm near 10^7, and this
    is the one route to that shortfall that needs no real data.

    **The tasks are not independent draws and must not be counted as if they were.** They
    share a graph, so their features are literally the same columns; only the target node
    differs. That is the point — a different node is a genuinely different function of those
    features — but it also caps what this buys: §93 measured a 5x volume increase as inert,
    and correlated tasks are the most likely way this one reproduces that null rather than
    beating it. :func:`fintfm.experiments.prior_score.score_prior` is the instrument for
    checking whether the extra tasks carry extra signal.

    Args:
        rng: NumPy random generator.
        n_rows: Rows per task.
        n_targets: Tasks to draw from the shared graph. ``1`` reproduces the single-task path.
        max_features: Upper bound on exposed features.
        min_features: Lower bound on exposed features.
        max_classes: Upper bound on classes.
        legacy: Use the pre-48.17/48.19 prior.

    Returns:
        A list of ``n_targets`` tasks. Shorter only if the graph has too few nodes to supply
        distinct targets, which cannot happen at the widths this prior samples.

    Raises:
        ValueError: If ``n_targets`` is below one.
    """
    if n_targets < 1:
        raise ValueError(f"n_targets must be >= 1, got {n_targets}")
    n_features = int(rng.integers(min_features, max_features + 1))
    pool = _sample_graph_pool(rng, n_rows, n_features, max_features, legacy)
    n_nodes = pool.shape[1]
    # Feature columns are shared; only the target node varies. Drawing the features once is
    # what makes this cheap, and holding them fixed is what makes the tasks comparable.
    take = min(n_features + n_targets, n_nodes)
    idx = rng.choice(n_nodes, size=take, replace=False)
    feat_idx, target_idx = idx[:n_features], idx[n_features:]
    X_base = pool[:, feat_idx].astype(np.float64)
    return [
        _finish_scm_task(rng, X_base.copy(), pool[:, j], n_features, max_classes)
        for j in target_idx
    ]


def sample_scm_task(
    rng: np.random.Generator,
    n_rows: int,
    max_features: int = 24,
    min_features: int = 2,
    max_classes: int = 10,
    legacy: bool = False,
    _latent_out: list[np.ndarray] | None = None,
) -> Task:
    """Sample one classification task from a random structural causal model.

    Args:
        rng: NumPy random generator.
        n_rows: Number of rows.
        max_features: Upper bound on the number of exposed features.
        min_features: Lower bound on the number of exposed features.
        max_classes: Upper bound on the number of classes (>= 2).
        legacy: Use the pre-48.17/48.19 prior -- uniform per-layer edge sparsity and the
            original five elementwise activations. Exists so the widened prior has a control
            arm trained at the same prior *mixture*, since comparing it against a checkpoint
            trained at ``p_financial=1.0`` would confound the mechanism change with the
            mixture change. Delete once the comparison is recorded.
        _latent_out: Internal. When a list is passed, the continuous pre-threshold latent is
            appended to it, so :func:`sample_scm_regression_task` can reuse this exact
            generative process instead of approximating it.
    """
    n_features = int(rng.integers(min_features, max_features + 1))
    pool = _sample_graph_pool(rng, n_rows, n_features, max_features, legacy)
    idx = rng.choice(pool.shape[1], size=n_features + 1, replace=False)
    X = pool[:, idx[:-1]].astype(np.float64)
    t = pool[:, idx[-1]]
    if _latent_out is not None:
        # The caller wants the pre-threshold latent. It is a node of the same random graph as
        # the features -- genuinely nonlinear in them and not recoverable by a linear fit --
        # which is the property that makes a regression task drawn from it as hard as the
        # classification task drawn from the same node.
        _latent_out.append(t.copy())
    return _finish_scm_task(rng, X, t, n_features, max_classes)


def _finish_scm_task(
    rng: np.random.Generator,
    X: np.ndarray,
    t: np.ndarray,
    n_features: int,
    max_classes: int,
) -> Task:
    """Turn a continuous target node and a feature block into a finished classification Task.

    Shared by the single-task and re-targeting paths (task 48.12) so the two cannot drift.
    Discretisation, the degenerate-label repair, categorical coding and missingness all live
    here; a second copy of this in the group path is how the two would silently start
    generating different task distributions.

    Args:
        rng: NumPy random generator.
        X: ``(n_rows, n_features)`` feature block. Modified in place.
        t: ``(n_rows,)`` continuous target node.
        n_features: Number of features in ``X``.
        max_classes: Upper bound on classes.

    Returns:
        A finished :class:`Task` with ``source="scm"``.
    """
    n_rows = X.shape[0]
    n_classes = int(rng.integers(2, max_classes + 1))
    # discretise target either by quantiles (balanced) or random cut points (imbalanced)
    if rng.random() < 0.5:
        edges = np.quantile(t, np.linspace(0, 1, n_classes + 1)[1:-1])
    else:
        edges = np.sort(rng.uniform(t.min(), t.max(), size=n_classes - 1))
    y = np.searchsorted(edges, t).astype(np.int64)
    present = np.unique(y)
    if len(present) < 2:
        y[rng.choice(n_rows, size=max(1, n_rows // 20), replace=False)] = (y[0] + 1) % n_classes
        present = np.unique(y)
    # remap to a dense label space so n_classes reflects classes actually present
    remap = {c: i for i, c in enumerate(present)}
    y = np.vectorize(remap.get)(y).astype(np.int64)
    # categorical columns: quantise a few features into integer codes
    is_cat = rng.random(n_features) < rng.uniform(0.0, 0.4)
    for j in np.flatnonzero(is_cat):
        k = int(rng.integers(2, 8))
        X[:, j] = np.searchsorted(np.quantile(X[:, j], np.linspace(0, 1, k + 1)[1:-1]), X[:, j])
    if rng.random() < 0.5:
        X[rng.random(X.shape) < rng.uniform(0, 0.2)] = np.nan
    return Task(
        X=X.astype(np.float32), y=y, n_classes=len(present), is_categorical=is_cat, source="scm"
    )


#: Target shapes a regression task is drawn from. Real continuous financial targets are not
#: Gaussian: loss given default is bounded and bimodal, exposure is heavy-tailed and
#: non-negative, and a prior offering only symmetric unimodal targets would teach the model
#: that every target looks like its latent already does. Each entry maps the SCM's latent to
#: a differently-shaped target while preserving its ordering, so the task stays exactly as
#: learnable and only the *shape* of what must be predicted changes.
_TARGET_SHAPES: tuple[str, ...] = ("identity", "lognormal", "bounded", "bimodal")


def _shape_target(t: np.ndarray, shape: str, rng: np.random.Generator) -> np.ndarray:
    """Map a latent to a continuous target of the named shape, order-preserving."""
    # Rank-uniformise first so each shape is applied to a known [0, 1] marginal rather than
    # to whatever scale the sampled SCM happened to produce.
    u = (np.argsort(np.argsort(t)) + 0.5) / len(t)
    if shape == "identity":
        return t.astype(np.float64)
    if shape == "lognormal":
        from scipy.special import ndtri

        return np.exp(ndtri(u) * rng.uniform(0.5, 1.5))
    if shape == "bounded":
        return u
    # bimodal: push mass toward both ends of [0, 1], the LGD shape (task 46.6)
    sharp = rng.uniform(2.0, 6.0)
    return np.where(u < 0.5, 0.5 * (2 * u) ** sharp, 1.0 - 0.5 * (2 * (1 - u)) ** sharp)


def sample_scm_regression_task(
    rng: np.random.Generator,
    n_rows: int,
    max_features: int = 24,
    min_features: int = 2,
    n_bins: int = 10,
) -> Task:
    """Sample one **regression** task by keeping the SCM's latent instead of thresholding it.

    `sample_scm_task` computes a continuous latent and then discretises it into classes. A
    regression task is the same generative process with that final step removed, so this
    shares the classification prior's structure rather than introducing a second,
    independently-tuned generator whose difficulty would have to be matched by hand.

    **The latent is a node of the same random graph as the features**, so it is nonlinear in
    them and a linear model cannot solve it by construction. An earlier version of this
    function projected the exposed features linearly instead, which produced tasks a ridge
    regression solved at Spearman 0.74-0.98 -- all easy, and exactly the "prior of only easy
    targets" §42 warns teaches the wrong thing.

    Difficulty is spanned explicitly by a noise multiplier drawn over two orders of
    magnitude, for the same reason `tests/test_prior.py` pins a difficulty range for
    classification.

    The returned task carries ``y_continuous``; ``prior/base.collate`` bins it on the
    **context rows'** quantiles, so the discretisation the model trains against is the same
    one inference can reproduce from context alone.

    Args:
        rng: NumPy random generator.
        n_rows: Number of rows.
        max_features: Upper bound on exposed features.
        min_features: Lower bound on exposed features.
        n_bins: Bins the continuous target is discretised into.

    Returns:
        A :class:`Task` with ``y_continuous`` set and ``source="scm-regression"``.
    """
    latent_out: list[np.ndarray] = []
    base = sample_scm_task(
        rng, n_rows, max_features=max_features, min_features=min_features,
        max_classes=2, _latent_out=latent_out,
    )
    latent = latent_out[0].astype(np.float64)
    # Log-uniform over [0.02, 3.0]: most tasks learnable, some dominated by noise. A linear
    # sweep would concentrate draws at the easy end and never produce a hard task.
    noise = float(np.exp(rng.uniform(np.log(0.02), np.log(3.0))))
    latent = latent + rng.normal(0, noise * (latent.std() + 1e-9), size=n_rows)
    shape = _TARGET_SHAPES[int(rng.integers(len(_TARGET_SHAPES)))]
    y_cont = _shape_target(latent, shape, rng)
    return Task(
        X=base.X,
        y=np.zeros(n_rows, dtype=np.int64),  # overwritten by collate's context-only binning
        n_classes=int(n_bins),
        is_categorical=base.is_categorical,
        source="scm-regression",
        y_continuous=y_cont,
    )
