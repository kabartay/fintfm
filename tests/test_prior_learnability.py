"""The learnability filter (`prior/learnability.py`) on regression tasks.

Audit AU1.17: a regression task's ``y`` is a placeholder of zeros until ``collate`` bins its
continuous target, and the judge read that placeholder, so with ``p_learnability_filter > 0``
every regression task was rejected as single-class and resampled away. Nothing failed; the
regression share of the mixture simply shrank.
"""

import numpy as np

from fintfm.prior.base import Task
from fintfm.prior.learnability import is_learnable
from fintfm.prior.scm import sample_scm_regression_task


def _regression_task(target: np.ndarray, X: np.ndarray) -> Task:
    """Wrap a continuous target as the SCM regression prior does: placeholder ``y`` of zeros."""
    return Task(
        X=X.astype(np.float32),
        y=np.zeros(len(target), dtype=np.int64),
        n_classes=10,
        is_categorical=np.zeros(X.shape[1], dtype=bool),
        source="scm-regression",
        y_continuous=target,
    )


def test_a_learnable_regression_task_passes_and_noise_does_not():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    assert is_learnable(_regression_task(X[:, 0] + 0.1 * rng.normal(size=300), X))
    # a pure-noise target sits near chance; 0.51 is inside its sampling error at 90 held-out
    # rows, so the noise side is checked against a floor clear of it
    assert not is_learnable(_regression_task(rng.normal(size=300), X), auc_floor=0.65)


def test_regression_prior_tasks_pass_at_a_nonzero_rate():
    # The SCM regression prior spans difficulty from near-noise to near-deterministic, so the
    # judge should pass most of its tasks and reject some. Before the fix it passed none.
    rng = np.random.default_rng(1)
    passed = [
        is_learnable(sample_scm_regression_task(rng, 256, max_features=8)) for _ in range(30)
    ]
    assert 0.3 < np.mean(passed) <= 1.0
