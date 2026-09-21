"""Generic structural-causal prior (TabPFN-style random MLP graphs).

Mixed into training so the model learns general tabular structure (arbitrary
nonlinear boundaries, multi-class targets, categorical inputs) rather than
only the financial story. Independent implementation of the well-known idea:
sample a random layered MLP with random activations, feed Gaussian noise,
read features and target from random hidden nodes.
"""

from __future__ import annotations

import numpy as np

from fintfm.prior.base import Task

_ACTS = (np.tanh, np.sin, lambda v: np.maximum(v, 0.0), lambda v: v, lambda v: np.sign(v) * np.sqrt(np.abs(v)))


def sample_scm_task(
    rng: np.random.Generator,
    n_rows: int,
    max_features: int = 24,
    min_features: int = 2,
    max_classes: int = 10,
    _latent_out: list[np.ndarray] | None = None,
) -> Task:
    """Sample one classification task from a random structural causal model.

    Args:
        rng: NumPy random generator.
        n_rows: Number of rows.
        max_features: Upper bound on the number of exposed features.
        min_features: Lower bound on the number of exposed features.
        max_classes: Upper bound on the number of classes (>= 2).
        _latent_out: Internal. When a list is passed, the continuous pre-threshold latent is
            appended to it, so :func:`sample_scm_regression_task` can reuse this exact
            generative process instead of approximating it.
    """
    n_features = int(rng.integers(min_features, max_features + 1))
    n_layers = int(rng.integers(1, 5))
    width = int(rng.integers(max(4, n_features + 1), 3 * max_features + 8))
    n_noise = int(rng.integers(2, 8))
    h = rng.normal(0, 1, size=(n_rows, n_noise)) * np.exp(rng.normal(0, 0.5, n_noise))
    nodes: list[np.ndarray] = []
    for _ in range(n_layers):
        w = rng.normal(0, 1, size=(h.shape[1], width)) / np.sqrt(h.shape[1])
        w *= rng.random((h.shape[1], width)) < rng.uniform(0.3, 1.0)  # sparse edges
        act = _ACTS[int(rng.integers(len(_ACTS)))]
        h = act(h @ w + rng.normal(0, 0.3, size=width)) + rng.normal(0, rng.uniform(0, 0.2), size=(n_rows, width))
        nodes.append(h)
    pool = np.concatenate(nodes, axis=1)
    idx = rng.choice(pool.shape[1], size=n_features + 1, replace=False)
    X = pool[:, idx[:-1]].astype(np.float64)
    t = pool[:, idx[-1]]
    if _latent_out is not None:
        # The caller wants the pre-threshold latent. It is a node of the same random graph as
        # the features -- genuinely nonlinear in them and not recoverable by a linear fit --
        # which is the property that makes a regression task drawn from it as hard as the
        # classification task drawn from the same node.
        _latent_out.append(t.copy())
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
    n_classes = len(present)
    # categorical columns: quantise a few features into integer codes
    is_cat = rng.random(n_features) < rng.uniform(0.0, 0.4)
    for j in np.flatnonzero(is_cat):
        k = int(rng.integers(2, 8))
        X[:, j] = np.searchsorted(np.quantile(X[:, j], np.linspace(0, 1, k + 1)[1:-1]), X[:, j])
    if rng.random() < 0.5:
        X[rng.random(X.shape) < rng.uniform(0, 0.2)] = np.nan
    return Task(X=X.astype(np.float32), y=y, n_classes=n_classes, is_categorical=is_cat, source="scm")


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
