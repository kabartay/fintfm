"""Task 39.20: the label-condition matrix builds each condition correctly and runs end to end."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fintfm.experiments.label_conditions import (
    CONDITIONS,
    make_context_labels,
    run,
    summarise,
)

Y = np.array([0, 1, 1, 0, 0, 0, 1, 0, 0, 1] * 20, dtype=np.int64)
OTHER = np.roll(Y, 7)[::-1].copy()


def _make(cond: str, seed: int = 0) -> np.ndarray:
    return make_context_labels(cond, Y, np.random.default_rng(seed), OTHER)


def test_each_condition_is_what_it_claims() -> None:
    assert np.array_equal(_make("true"), Y)
    assert np.array_equal(_make("inverted"), 1 - Y)
    assert np.array_equal(_make("none"), np.zeros_like(Y))
    assert np.array_equal(_make("unrelated"), OTHER)
    sh = _make("shuffled")
    assert np.array_equal(np.sort(sh), np.sort(Y))  # marginal preserved exactly
    assert not np.array_equal(sh, Y)


def test_flip_conditions_interpolate() -> None:
    assert np.array_equal(_make("flip_0.0"), Y)
    assert np.array_equal(_make("flip_1.0"), 1 - Y)
    frac = (_make("flip_0.25") != Y).mean()
    assert 0.15 < frac < 0.35


def test_conditions_are_deterministic_under_a_seed() -> None:
    for c in ("shuffled", "flip_0.3"):
        assert np.array_equal(_make(c, 3), _make(c, 3))
    assert not np.array_equal(_make("shuffled", 1), _make("shuffled", 2))


def test_bad_inputs_are_refused() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        make_context_labels("bogus", Y, rng)
    with pytest.raises(ValueError):
        make_context_labels("unrelated", Y, rng)
    with pytest.raises(ValueError):
        make_context_labels("flip_1.5", Y, rng)


def test_matrix_runs_end_to_end_on_a_tiny_untrained_model(tmp_path: Path) -> None:
    """Shape and determinism only; an untrained model's AUCs say nothing about any checkpoint."""
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    model = FinancialTFM(
        ModelConfig(max_features=8, d_model=16, d_cell=8, n_layers=1, n_col_layers=1, max_classes=2)
    ).eval()
    kw: dict = {
        "probes": ("orientation", "linear"),
        "bayes_targets": (0.9,),
        "flip_rates": (0.25,),
        "seeds": (0, 1),
        "n": 200,
        "n_features": 6,
        "n_ensemble": 2,
    }
    rec = run({}, tmp_path, models={"m": model}, **kw)
    assert set(rec["arms"]) == {"m", "untrained_control"}
    for tasks in rec["arms"].values():
        assert set(tasks) == {"orientation", "linear", "bayes_0.9"}
        for cell in tasks.values():
            for c in (*CONDITIONS, "flip_0.25"):
                assert 0.0 <= cell[c]["auc_mean"] <= 1.0
                assert cell[c]["n_seeds"] == 2
            assert 0.0 <= cell["_ceiling"] <= 1.0
        assert tasks["bayes_0.9"]["_ceiling"] == 0.9
        assert tasks["orientation"]["_ceiling"] == 1.0
    saved = json.loads((tmp_path / "label_conditions.json").read_text())
    assert saved["arms"]["m"] == rec["arms"]["m"]
    table = summarise(rec)
    assert "untrained_control" in table and "inverted" in table and "chance" in table
    again = run({}, tmp_path, models={"m": model}, **kw)
    assert again["arms"] == rec["arms"]
