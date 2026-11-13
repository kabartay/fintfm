"""Tests for scripts/colid_surrogate.py (task 48.16 method test)."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "colid_surrogate.py"
RUNS = Path(os.environ.get("FINTFM_RUNS", Path(__file__).resolve().parents[1] / "runs"))


def _load():
    """Import the script by path (it is not a package module)."""
    spec = importlib.util.spec_from_file_location("colid_surrogate", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["colid_surrogate"] = mod  # dataclasses resolves annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


def test_rows_are_unique_and_in_range() -> None:
    """Every row has a distinct run directory and an AP in a plausible range."""
    m = _load()
    assert len({r.run_dir for r in m.ROWS}) == len(m.ROWS)
    assert all(0.1 < r.ap < 0.3 for r in m.ROWS)


def test_loo_matches_a_hand_computed_mean_error() -> None:
    """The mean surrogate's leave-one-out error is exact arithmetic, not a fit."""
    m = _load()
    errs = m.leave_one_out(m.ROWS)["mean"]
    y = np.array([r.ap for r in m.ROWS])
    expect = (y.sum() - y) / (len(y) - 1) - y
    assert np.allclose(errs, expect)


def test_loo_never_uses_the_held_out_row() -> None:
    """Raising one row's score by 0.05 moves only that row's own error, by -0.05."""
    m = _load()
    base = m.leave_one_out(m.ROWS)["mean"]
    rows = list(m.ROWS)
    rows[0] = type(rows[0])(**{**rows[0].__dict__, "ap": rows[0].ap + 0.05})
    shifted = m.leave_one_out(tuple(rows))["mean"]
    assert shifted[0] == pytest.approx(base[0] - 0.05)


def test_untried_predictions_cover_requested_values_and_are_finite() -> None:
    """One finite prediction per untried value, none of them a recorded point."""
    m = _load()
    recorded = {r.column_id_dim for r in m.ROWS}
    preds = m.predict_untried(m.ROWS)
    assert set(preds) == set(m.UNTRIED)
    assert not recorded & set(m.UNTRIED)
    assert all(np.isfinite(v) for p in preds.values() for v in p.values())


@pytest.mark.skipif(
    not (RUNS.is_dir() and any(p.is_dir() for p in RUNS.iterdir())),
    reason="needs the main checkout's runs/ (or FINTFM_RUNS)",
)
def test_embedded_numbers_match_the_run_files() -> None:
    """The embedded APs equal the per-run v4_protocol.json values, full panel, ensemble 8."""
    assert _load().verify_rows(RUNS) == []
