"""The task builder of ``scripts/feature_selection_recovery.py``: labels are exactly the rule."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import feature_selection_recovery as f


@pytest.mark.parametrize("name", list(f.TASK_RULES))
def test_labels_are_exactly_the_rule(name: str) -> None:
    """Labels equal 1[score > 0] on context and queries, with both classes present."""
    t = f.make_task(name, 64, 200, 12, seed=3)
    assert np.array_equal(t.y_ctx, (f.rule_score(name, t.X_ctx) > 0).astype(int))
    assert np.array_equal(t.y_q, (f.rule_score(name, t.X_q) > 0).astype(int))
    assert 0 < t.y_ctx.sum() < 64
    assert t.true_cols == tuple(sorted(f.TASK_RULES[name]))


def test_task_is_deterministic_and_negation_flips_sign() -> None:
    """Same seed, same task; y=-x3 is the strict opposite rule on its own features."""
    a = f.make_task("x3", 32, 50, 10, 1)
    b = f.make_task("x3", 32, 50, 10, 1)
    assert np.array_equal(a.X_ctx, b.X_ctx)
    n = f.make_task("neg_x3", 32, 50, 10, 1)
    assert np.array_equal(n.y_q, (n.X_q[:, 3] < 0).astype(int))


def test_recovery_measure() -> None:
    """Arg-max and exact top-k are read from the importance vector."""
    drop = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.2, 0.01])
    r = f.recovery(drop, (2, 5))
    assert r["top_col"] == 2 and r["exact"] and r["top_hit"]
    assert not f.recovery(drop, (2, 6))["exact"]


def test_too_few_features_rejected() -> None:
    """A column beyond the width is an error, not silent clipping."""
    with pytest.raises(ValueError):
        f.make_task("x7", 32, 50, 7, 0)
