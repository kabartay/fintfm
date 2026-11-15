"""Paired bootstrap and multi-seed aggregation of ``fintfm-retrgroup`` (openspec 17.7)."""

from __future__ import annotations

import numpy as np

from fintfm.experiments import retrieval_grouping as rg

N_H = 2


def _panel(n: int = 400, seed: int = 0):
    """Truth, an exact reference, a near-copy grouped curve and a degraded one."""
    rng = np.random.default_rng(seed)
    truth = np.stack([(rng.random(n) < 0.3).astype(int) for _ in range(N_H)], 1)
    exact = np.clip(0.2 + 0.5 * truth + 0.3 * rng.normal(size=(n, N_H)), 0, 1)
    near = np.clip(exact + 0.005 * rng.normal(size=exact.shape), 0, 1)
    bad = rng.random((n, N_H))
    return truth, exact, {2: near, 128: bad}


def test_paired_comparisons_detect_the_degraded_arm_only():
    truth, exact, curves = _panel()
    seen = np.ones_like(truth, dtype=bool)
    rows = rg.paired_grouping_comparisons(curves, exact, truth, seen, None, 10, n_boot=200)
    assert len(rows) == 2 * N_H
    by = {(r["group_size"], r["horizon"]): r for r in rows}
    assert abs(by[(2, 0)]["delta"]) < 0.02
    assert by[(128, 0)]["delta"] < -0.1
    assert by[(128, 0)]["p_holm"] < 0.05
    assert all(r["p_holm"] >= r["p_raw"] - 1e-12 for r in rows)


def test_unscorable_horizon_is_skipped():
    truth, exact, curves = _panel()
    truth[:, 1] = 0  # one class only
    seen = np.ones_like(truth, dtype=bool)
    rows = rg.paired_grouping_comparisons(curves, exact, truth, seen, None, 10, n_boot=200)
    assert {r["horizon"] for r in rows} == {0}


def _record(seed: int) -> dict:
    truth, exact, curves = _panel(seed=seed)
    seen = np.ones_like(truth, dtype=bool)
    return {
        "config": {"seed": seed},
        "reference": {"mean_auc": 0.8},
        "grouped": [
            {
                "group_size": s,
                "mean_auc": 0.8 - 0.01 * (seed + 1) * (s / 2),
                "mean_abs_deviation": 0.01,
                "spearman": 0.99,
            }
            for s in curves
        ],
        "paired": rg.paired_grouping_comparisons(curves, exact, truth, seen, None, 10, n_boot=200),
        "note": "n",
    }


def test_aggregate_seeds_pools_holm_across_seeds():
    agg = rg.aggregate_seeds([_record(0), _record(1), _record(2)], n_boot=200)
    assert agg["seeds"] == [0, 1, 2]
    assert len(agg["paired"]) == 3 * 2 * N_H
    first = agg["per_size"][0]
    assert first["group_size"] == 2
    assert np.isclose(first["auc_delta_mean"], -0.02)
    assert first["auc_delta_std"] is not None
    assert all(p["p_holm"] >= p["p_raw"] - 1e-12 for p in agg["paired"])
    assert "seeds [0, 1, 2]" in rg.summarise_seeds(agg)
