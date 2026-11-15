"""Holm-adjusted paired bootstrap in the term-structure and context-sweep harnesses.

Covers openspec tasks 11.8 (``fintfm-termstruct``) and 32.3 (``fintfm-ctxsweep``): both reuse
``paired_auc_difference`` and ``holm_adjusted_p`` from ``fintfm.evaluation.metrics`` and record
the adjusted p-value beside every AUC difference.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fintfm.evaluation.metrics import holm_adjusted_p
from fintfm.experiments import context_sweep, term_structure

N_H = 3


def test_holm_adjusted_p_matches_hand_computation():
    # sorted 0.01, 0.03, 0.04 times 3, 2, 1 = 0.03, 0.06, 0.04 -> monotone 0.03, 0.06, 0.06
    adj = holm_adjusted_p([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])


def test_holm_adjusted_p_is_monotone_and_not_below_raw():
    rng = np.random.default_rng(0)
    p = rng.random(25).tolist()
    adj = holm_adjusted_p(p)
    assert all(a >= r - 1e-12 and a <= 1.0 for a, r in zip(adj, p, strict=True))
    by_raw = [a for _, a in sorted(zip(p, adj, strict=True))]
    assert by_raw == sorted(by_raw)


def _panel(n_tasks: int = 40, rows: int = 25, seed: int = 0):
    """Truth and three arms' curves: ``joint`` informative, ``copy`` identical, ``noise`` blind."""
    rng = np.random.default_rng(seed)
    n = n_tasks * rows
    tasks = np.repeat(np.arange(n_tasks), rows)
    truth = np.stack([(rng.random(n) < 0.3 + 0.1 * k).astype(int) for k in range(N_H)], 1)
    good = np.clip(0.2 + 0.5 * truth + 0.3 * rng.normal(size=(n, N_H)), 0, 1)
    noise = rng.random((n, N_H))
    return truth, tasks, {"joint": good, "copy": good.copy(), "noise": noise}


def test_term_structure_comparisons_carry_adjusted_p_values():
    truth, tasks, curves = _panel()
    rows = term_structure.paired_arm_comparisons(curves, truth, tasks, n_boot=200)
    assert len(rows) == 2 * N_H  # two other arms x horizons
    for r in rows:
        assert r["p_holm"] >= r["p_raw"] and 0 <= r["p_holm"] <= 1
        assert r["ci_lo"] <= r["delta"] <= r["ci_hi"] or np.isnan(r["ci_lo"])
        assert r["significant"] == (r["p_holm"] < 0.05)
    by = {(r["other"], r["horizon"]): r for r in rows}
    assert by[("copy", 0)]["delta"] == 0.0 and not by[("copy", 0)]["significant"]
    assert by[("noise", 0)]["delta"] > 0.1 and by[("noise", 0)]["significant"]


def test_term_structure_summary_prints_the_holm_table():
    truth, tasks, curves = _panel()
    cmps = term_structure.paired_arm_comparisons(curves, truth, tasks, n_boot=100)
    arms = [
        dataclasses.asdict(term_structure.summarise_curves(k, v, truth, 1, 10))
        for k, v in curves.items()
    ]
    text = term_structure.summarise({"arms": arms, "comparisons": cmps})
    assert "p Holm" in text and "Holm-adjusted" in text


def test_context_sweep_comparisons_adjust_across_the_whole_family():
    truth, tasks, curves = _panel(seed=1)
    seen = np.ones(truth.shape, dtype=bool)
    cs = {
        (100, "uniform", 0): curves["joint"],
        (100, "balanced", 0): curves["noise"],
        (100, "hybrid", 0): curves["copy"],
    }
    rows = context_sweep.paired_strategy_comparisons(
        cs, truth, seen, tasks, min_rows=10, n_boot=200
    )
    assert len(rows) == 3 * N_H  # three pairs x horizons, one family
    assert all(r["p_holm"] >= r["p_raw"] for r in rows)
    assert all(np.isfinite(r["p_holm"]) for r in rows)
    text = context_sweep.summarise({"cells": [], "comparisons": rows})
    assert "Holm" in text and "balanced - hybrid" in text


def test_term_structure_run_seeds_pools_and_writes_per_seed(tmp_path, monkeypatch):
    """``--seeds`` (11.8): one record per seed, plus a Holm table pooled across seeds."""
    truth, tasks, curves = _panel(n_tasks=20, rows=10, seed=3)
    arm_models = iter(["joint", "mt", "mp"] * 3)
    monkeypatch.setattr(term_structure, "train_joint", lambda *a, **kw: next(arm_models))
    monkeypatch.setattr(term_structure, "train_per_horizon", lambda *a, **kw: [next(arm_models)])
    monkeypatch.setattr(term_structure, "_eval_batches", lambda *a, **kw: [])
    by_model = {"joint": curves["joint"], "mt": curves["copy"], "mp": curves["noise"]}
    monkeypatch.setattr(
        term_structure,
        "collect_curves",
        lambda models, batches, device, n_horizons: (
            by_model[models[0] if isinstance(models, list) else models],
            truth,
            tasks,
        ),
    )
    rec = term_structure.run_seeds(tmp_path, [0, 1, 2], n_boot=100, steps=5, n_horizons=N_H)
    assert rec["seeds"] == [0, 1, 2] and len(rec["per_seed"]) == 3
    assert all((tmp_path / f"seed{s}" / "term_structure.json").exists() for s in range(3))
    assert len(rec["pooled_comparisons"]) == 2 * N_H
    assert all(r["p_holm"] >= r["p_raw"] for r in rec["pooled_comparisons"])
    assert "pooled across seeds" in term_structure.summarise_seeds(rec)
    with pytest.raises(ValueError):
        term_structure.run_seeds(tmp_path, [1, 1])


def test_context_sweep_comparisons_skip_unscorable_horizons():
    truth, _tasks, curves = _panel(seed=2)
    seen = np.ones(truth.shape, dtype=bool)
    seen[:, 1] = False  # horizon 1 never observed
    cs = {(100, "a", 0): curves["joint"], (100, "b", 0): curves["noise"]}
    rows = context_sweep.paired_strategy_comparisons(cs, truth, seen, None, 10, n_boot=100)
    assert sorted(r["horizon"] for r in rows) == [0, 2]
    assert (
        context_sweep.paired_strategy_comparisons(
            {(100, "a", 0): curves["joint"]}, truth, seen, None, 10
        )
        == []
    )
