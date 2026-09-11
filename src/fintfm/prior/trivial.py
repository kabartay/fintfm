"""A deliberately trivial prior, to separate "our data is wrong" from "our model can't learn".

Why this exists
---------------
``docs/FINDINGS.md`` §53: the model scores 0.67 on its own training distribution, 0.69 on a
partly-seen one and 0.63 on iid Gaussian tasks it has never seen — while logistic regression
scores 0.73, 0.89 and 0.99 on the same three. **It produces roughly the same number whatever
it is shown**, including on tasks with an achievable ceiling of 0.995. It is not tracking task
difficulty at all.

That explains why six independent interventions changed nothing — capacity across 17×, the
pooling design, context size, base rate, label noise and training volume. None of them
addresses a model that has converged to a capped predictor.

So the question underneath all of it is: **can this architecture learn in-context prediction at
all, given a task it should find trivial?** This prior is the control that answers it. Few
features, no missingness, no redundant columns, no label noise, a deterministic linear rule,
and a balanced-ish base rate — everything that could obscure the signal is removed.

- If a model trained here reaches ~0.95 on tasks of this kind, the architecture works and our
  real prior is simply too hard everywhere; the fix is difficulty coverage at the easy end.
- If it cannot, something in the training loop or the architecture prevents sharp in-context
  inference regardless of data, and no amount of prior work will fix it.

**This is a diagnostic, not a candidate prior.** A model trained only on trivial tasks would
learn nothing about abstention or noise, which is the failure mode §42's difficulty span exists
to prevent.
"""

from __future__ import annotations

import numpy as np

from fintfm.prior.base import Task


def sample_trivial_task(
    rng: np.random.Generator,
    n_rows: int,
    max_features: int = 8,
    min_features: int = 3,
    rate: float = 0.3,
) -> Task:
    """Sample a task that any competent in-context learner should solve.

    Standard-normal features, a random linear rule with a **random sign per feature** (so the
    direction still has to be read from the context, per §47), and a deterministic threshold
    label. No missing values, no redundant columns, no noise.

    Args:
        rng: NumPy random generator.
        n_rows: Rows to generate.
        max_features: Upper bound on feature count, kept small deliberately.
        min_features: Lower bound on feature count.
        rate: Positive-class rate.

    Returns:
        A binary :class:`Task` whose achievable AUC is 1.0.

    Raises:
        ValueError: If the feature bounds are inconsistent.
    """
    if not 1 <= min_features <= max_features:
        raise ValueError(f"need 1 <= min_features <= max_features, got {min_features}, {max_features}")
    n_features = int(rng.integers(min_features, max_features + 1))
    X = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    # random magnitudes and random signs: the rule is easy to apply and still has to be
    # inferred from the labelled examples rather than assumed (§47)
    w = rng.normal(size=n_features)
    score = X @ w
    y = (score >= np.quantile(score, 1.0 - rate)).astype(np.int64)
    # a degenerate draw teaches nothing; nudge rather than return a single-class task
    if len(np.unique(y)) < 2:
        y[int(np.argmax(score))] = 1
        y[int(np.argmin(score))] = 0
    return Task(
        X=X, y=y, n_classes=2,
        is_categorical=np.zeros(n_features, dtype=bool), source="trivial",
    )
