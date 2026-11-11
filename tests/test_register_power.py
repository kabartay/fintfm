"""Unit tests for the register power calculation (openspec task 6.2)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "register_power", Path(__file__).parent.parent / "scripts" / "register_power.py"
)
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)


def test_hanley_mcneil_variance_matches_hand_value() -> None:
    """A=0.8, 50 positives, 50 negatives: hand-computed variance 0.0019804444."""
    a = 0.8
    q1 = 0.8 / 1.2
    q2 = 2 * 0.64 / 1.8
    expected = (0.16 + 49 * (q1 - 0.64) + 49 * (q2 - 0.64)) / 2500
    assert math.isclose(rp.hanley_mcneil_var(a, 50, 50), expected, rel_tol=1e-12)
    assert math.isclose(rp.hanley_mcneil_var(a, 50, 50), 0.0019804444, rel_tol=1e-6)


def test_correlation_shrinks_detectable_difference() -> None:
    """Higher assumed correlation must lower the minimum detectable difference."""
    lo = rp.min_detectable_diff(0.8, 100, 9900, 0.3)
    hi = rp.min_detectable_diff(0.8, 100, 9900, 0.9)
    assert hi < lo
