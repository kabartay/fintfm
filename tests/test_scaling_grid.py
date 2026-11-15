"""Tests for the scaling-curve grid (openspec scaling-curve 9.3)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from fintfm.experiments import prior_ablation, scaling_grid
from fintfm.experiments.scaling_grid import (
    cell_command,
    expand_grid,
    load_grid,
    plan_grid,
    run_cell,
)

PROPOSED = Path("configs/scaling/grid-2026-11-15.yaml")


def _write_variant(tmp_path: Path, **changes) -> Path:
    data = yaml.safe_load(PROPOSED.read_text())
    data.update(changes)
    path = tmp_path / "g.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_proposed_grid_expands_to_the_twelve_cells_of_the_draft():
    cells = expand_grid(load_grid(PROPOSED))
    assert len(cells) == 12
    counts: dict[tuple[str, int], int] = {}
    for c in cells:
        counts[(c.size, c.tasks)] = counts.get((c.size, c.tasks), 0) + 1
    assert counts == {(s, t): n for s in "SM" for t, n in ((48000, 3), (480000, 2), (4800000, 1))}
    assert {c.cell_id for c in cells} >= {"S-t48000-s2", "M-t4800000-s0"}


def test_tasks_seen_is_steps_times_batch():
    cells = expand_grid(load_grid(PROPOSED))
    for c in cells:
        assert c.tasks_seen == c.tasks == c.steps * c.batch_size
    assert {c.tasks: c.steps for c in cells} == {48000: 6000, 480000: 60000, 4800000: 600000}


def test_task_count_not_a_multiple_of_batch_is_refused(tmp_path: Path):
    path = _write_variant(tmp_path, tasks=[48001], seeds={48001: [0]})
    with pytest.raises(ValueError, match="multiple"):
        load_grid(path)


def test_misspelled_grid_key_is_an_error(tmp_path: Path):
    path = _write_variant(tmp_path, taks=[1])
    with pytest.raises(ValueError, match="unknown grid key"):
        load_grid(path)


def test_checkpoint_every_must_be_positive(tmp_path: Path):
    with pytest.raises(ValueError, match="checkpoint_every"):
        load_grid(_write_variant(tmp_path, checkpoint_every=0))


def test_larger_model_gets_proportional_ffn():
    cells = {c.size: c for c in expand_grid(load_grid(PROPOSED))}
    assert cells["M"].overrides == {"d_cell": 64, "d_model": 256, "n_layers": 6, "d_ff": 1024}
    assert cells["S"].overrides == {}


def test_plan_never_trains_and_prices_every_cell(monkeypatch):
    monkeypatch.setattr(scaling_grid, "train", lambda *a, **k: pytest.fail("trained"))
    text = plan_grid(load_grid(PROPOSED), Path("runs/scaling"))
    assert text.count("--checkpoint-every 1000") == 12
    assert "[S-t480000-s0] " in text and "= 480,000 tasks seen" in text
    # the draft: 280.2 quota hours, 9.3 weeks (its 560.4 GPU-h is rounded per row)
    assert "ESTIMATED 280.2 wall h, 560.5 GPU-h, 9.3 weeks" in text


def test_main_plan_flag_trains_nothing(monkeypatch, capsys):
    monkeypatch.setattr(scaling_grid, "train", lambda *a, **k: pytest.fail("trained"))
    monkeypatch.setattr(sys, "argv", ["fintfm-ablate", "--grid", str(PROPOSED), "--plan"])
    prior_ablation.main()
    assert "PLAN scaling-grid-2026-11-15" in capsys.readouterr().out


def test_command_carries_checkpoint_every_and_size_flags():
    spec = load_grid(PROPOSED)
    cell = next(c for c in expand_grid(spec) if c.cell_id == "M-t48000-s1")
    cmd = cell_command(spec, cell, Path("runs/x"))
    for part in ("--steps 6000", "--seed 1", "--d-model 256", "--checkpoint-every 1000"):
        assert part in cmd


def test_two_step_tiny_cell_writes_the_required_fields(tmp_path: Path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "max_features": 8,
                    "max_classes": 2,
                    "d_cell": 8,
                    "d_model": 16,
                    "n_heads": 2,
                    "n_col_layers": 1,
                    "n_layers": 1,
                    "d_ff": 16,
                },
                "prior": {"n_rows": 16},
                "train": {"eval_every": 1000, "log_every": 1},
            }
        )
    )
    grid = _write_variant(
        tmp_path,
        name="tiny",
        recipe=str(recipe),
        batch_size=2,
        device="cpu",
        checkpoint_every=1,
        horizons=[],
        tasks=[4],
        seeds={4: [0]},
        models={"T": {"step_seconds": 0.1}},
    )
    spec = load_grid(grid)
    (cell,) = expand_grid(spec)
    record = run_cell(spec, cell, tmp_path / "out")
    written = json.loads((tmp_path / "out" / "T-t4-s0" / "results.json").read_text())
    assert written["git_commit"]
    assert written["recipe"]["path"] == str(recipe) and written["recipe"]["sha256"]
    assert written["steps"] == 2 and written["batch_size"] == 2
    assert written["tasks_seen"] == 4 == written["steps"] * written["batch_size"]
    assert written["n_parameters"] == record["n_parameters"] > 0
    assert "--checkpoint-every 1" in written["command"]
