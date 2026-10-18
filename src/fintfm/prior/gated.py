"""Gated-label prior: a threshold on one or two columns, then a score inside the gate.

Why this prior exists (design review DR2.2)
-------------------------------------------

V4FinBench's label is a gate followed by a ranking: 3,586 of its 3,587 positives sit in the
10.6% of rows with negative equity, at 3.39% inside the gate and 1 in 894,400 outside it
(``docs/design/DESIGN_REVIEW_2026-10-16.md`` §1.1, MEASURED). No general prior here produced
that shape. The SCM and financial labels are smooth functions of every row, and only the tree
prior has thresholds, which fits §1.2's reading of the tree-prior result. A model that has
never seen a label concentrated behind a threshold has no reason to look for one.

The generative story, per task:

1. Correlated features with heterogeneous scales, as in the tree prior.
2. **The gate**: one or two columns, each thresholded at a random tail on a random side, so
   the gate holds a LogU(0.03, 0.4) share of the rows (two columns combine by AND).
3. **The score inside the gate**: a small random MLP (an SCM-style node) over a random subset
   of the columns, at a LogU(0.3, 12) sharpness, cut to a LogU(0.05, 0.6) positive rate
   inside the gate.
4. **Leakage**: a LogU(0.002, 0.05) share of the positives falls outside the gate, at random,
   so the gate is near-necessary rather than a hard rule.

**No global rule** (``CLAUDE.md``, "check the prior admits no global rule"; §47). The gate's
columns, sides and thresholds are drawn per task, and the column order is random, so no column
position points the same way across tasks; ``scripts/prior_stats.py`` checks that none does in
more than 75% of them. Reading the context is the only way to find a task's gate.

Reimplemented from the design review's description; no code from any tabular-foundation-model
project enters this repository.
"""

from __future__ import annotations

import numpy as np

from fintfm.prior.base import Task
from fintfm.prior.knobs import TaskKnobs, log_uniform
from fintfm.prior.scm import _ACTS

#: Share of rows inside the gate, drawn log-uniformly. V4FinBench's flagged share is 10.6%.
GATE_SHARE_RANGE: tuple[float, float] = (0.03, 0.4)

#: Positive rate inside the gate, drawn log-uniformly.
IN_GATE_RATE_RANGE: tuple[float, float] = (0.05, 0.6)

#: Share of all positives that fall outside the gate, drawn log-uniformly. V4FinBench's is 1 in
#: 3,587; the range sits above it so the gate is near-necessary, not a rule a model may assume.
LEAK_SHARE_RANGE: tuple[float, float] = (0.002, 0.05)

#: Sharpness of the in-gate score, drawn log-uniformly, matching the financial prior's span
#: from near-noise to near-deterministic (``docs/results/FINDINGS.md`` §42).
SHARPNESS_RANGE: tuple[float, float] = (0.3, 12.0)

#: Fewest positives a gated task carries; the strongest in-gate rows are promoted to reach it.
MIN_POSITIVES = 2


def _zscore(v: np.ndarray) -> np.ndarray:
    """Standardise columns, guarding a constant one.

    Args:
        v: ``(n,)`` or ``(n, k)`` values.

    Returns:
        Values of the same shape with zero mean and unit spread per column.
    """
    sd = v.std(axis=0)
    return (v - v.mean(axis=0)) / np.where(sd > 1e-12, sd, 1.0)


def sample_gated_task(
    rng: np.random.Generator,
    n_rows: int,
    max_features: int = 24,
    min_features: int = 2,
    knobs: TaskKnobs | None = None,
    n_features: int | None = None,
) -> Task:
    """Sample one binary task whose positives sit behind a threshold gate.

    Args:
        rng: NumPy random generator.
        n_rows: Number of rows.
        max_features: Upper bound on the number of features.
        min_features: Lower bound on the number of features.
        knobs: Design review DR2.3's knobs, or ``None``. Under ``missing_mode="mixed"`` the
            MCAR step here is skipped for the knob layer's mechanisms; the other column-side
            knobs are applied by the mixture.
        n_features: Width from the mixture's band draw; ``None`` draws it here.

    Returns:
        A binary :class:`~fintfm.prior.base.Task` with ``source="gated"``, whose ``meta``
        records the gate (``columns``, ``directions`` with +1 meaning the high tail, the
        ``cuts`` on each column's own scale, the realised ``share``, ``leak`` and
        ``in_gate_rate``). Column indices are before any knob-layer column shuffle.
    """
    if n_features is None:
        n_features = int(rng.integers(min_features, max_features + 1))
    n_features = max(1, n_features)
    n_latent = int(rng.integers(1, max(2, n_features + 1)))
    z = rng.normal(size=(n_rows, n_latent))
    loading = rng.normal(size=(n_latent, n_features)) / np.sqrt(n_latent)
    X = z @ loading + rng.normal(0, rng.uniform(0.1, 0.8), size=(n_rows, n_features))
    X *= np.exp(rng.normal(0, 1.0, size=n_features))  # heterogeneous scales

    # --- the gate: a random tail of one or two columns, on a random side ---------------------
    k = 1 if n_features == 1 or rng.random() < 0.5 else 2
    cols = rng.choice(n_features, size=k, replace=False)
    directions = rng.choice([-1.0, 1.0], size=k)
    target = log_uniform(rng, *GATE_SHARE_RANGE)
    per_col = target ** (1.0 / k)  # AND of k tails; correlated columns overshoot, which is fine
    gate = np.ones(n_rows, dtype=bool)
    cuts: list[float] = []
    for c, d in zip(cols, directions):
        v = d * X[:, c]
        cuts.append(float(np.quantile(v, 1.0 - per_col)))
        gate &= v > cuts[-1]
    if gate.sum() < MIN_POSITIVES:
        # two anti-correlated tails can AND to nothing; fall back to the first column alone,
        # at the drawn share and never fewer rows than the positives a task must carry
        cols, directions = cols[:1], directions[:1]
        v = directions[0] * X[:, cols[0]]
        cuts = [float(np.sort(v)[-max(MIN_POSITIVES + 1, round(target * n_rows))])]
        gate = v > cuts[0]

    # --- the score inside the gate: a small random MLP node -----------------------------------
    m = int(rng.integers(1, min(n_features, 8) + 1))
    feats = rng.choice(n_features, size=m, replace=False)
    width = int(rng.integers(2, 9))
    act = _ACTS[int(rng.integers(5))]  # the elementwise activations
    hidden = act(
        _zscore(X[:, feats]) @ (rng.normal(0, 1, size=(m, width)) / np.sqrt(m))
        + rng.normal(0, 0.5, size=width)
    )
    score = hidden @ rng.normal(0, 1, size=width) + rng.normal(0, rng.uniform(0, 0.5), n_rows)
    sharpness = log_uniform(rng, *SHARPNESS_RANGE)
    s_in = score[gate]
    sd = s_in.std()
    logit = sharpness * (score - s_in.mean()) / (sd if sd > 1e-12 else 1.0)
    in_rate = log_uniform(rng, *IN_GATE_RATE_RANGE)
    lo, hi = -40.0, 40.0
    for _ in range(50):  # intercept so the in-gate rate is met in expectation
        mid = 0.5 * (lo + hi)
        if (1.0 / (1.0 + np.exp(-(logit[gate] + mid)))).mean() < in_rate:
            lo = mid
        else:
            hi = mid
    p = 1.0 / (1.0 + np.exp(-(logit + 0.5 * (lo + hi))))

    # --- leakage: a small share of positives outside the gate, at random ---------------------
    leak = log_uniform(rng, *LEAK_SHARE_RANGE)
    share = float(gate.mean())
    out_rate = in_rate * share * leak / ((1.0 - leak) * max(1.0 - share, 1e-12))
    p = np.where(gate, p, min(out_rate, 1.0))
    y = (rng.random(n_rows) < p).astype(np.int64)
    if y.sum() < MIN_POSITIVES:  # promote the strongest in-gate rows, never an outside one
        order = np.flatnonzero(gate)[np.argsort(-p[gate])]
        y[order[:MIN_POSITIVES]] = 1
    if y.min() == 1:
        y[int(np.argmin(p))] = 0

    if not (knobs is not None and knobs.mixed_missing) and rng.random() < 0.5:
        X[rng.random(X.shape) < rng.uniform(0, 0.2)] = np.nan
    return Task(
        X=X.astype(np.float32),
        y=y,
        n_classes=2,
        is_categorical=np.zeros(n_features, dtype=bool),
        source="gated",
        meta={
            "gate": {
                "columns": [int(c) for c in cols],
                "directions": [int(d) for d in directions],
                "cuts": [float(d) * c for d, c in zip(directions, cuts)],
                "share": share,
                "leak": leak,
                "in_gate_rate": in_rate,
            }
        },
    )
