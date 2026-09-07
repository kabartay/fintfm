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
) -> Task:
    """Sample one classification task from a random structural causal model.

    Args:
        rng: NumPy random generator.
        n_rows: Number of rows.
        max_features: Upper bound on the number of exposed features.
        min_features: Lower bound on the number of exposed features.
        max_classes: Upper bound on the number of classes (>= 2).
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
