"""Binary-path context sweep (openspec 32.3, 32.4, 32.6) on synthetic panels and a tiny model."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fintfm.evaluation.datasets import CreditDataset
from fintfm.experiments import binary_context_sweep as bcs
from fintfm.modeling.model import FinancialTFM, ModelConfig


def _tiny_model() -> FinancialTFM:
    torch.manual_seed(0)
    cfg = ModelConfig(max_features=6, max_classes=3, d_model=16, n_heads=2, n_layers=1, d_ff=32)
    return FinancialTFM(cfg)


def _panel(name: str, rate: float, n: int = 400, n_feat: int = 4, seed: int = 0) -> CreditDataset:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    X = rng.normal(size=(n, n_feat)).astype(np.float32)
    X[:, 0] += 1.5 * y
    return CreditDataset(X=X, y=y, name=name, horizon_years=1, licence="test", attribution="test")


def _scores(delta_ab: float, seed: int = 0):
    """Two arms on the same rows: ``a`` informative, ``b`` is ``a`` plus noise of size delta."""
    rng = np.random.default_rng(seed)
    y = (rng.random(600) < 0.2).astype(int)
    a = np.clip(0.2 + 0.4 * y + 0.2 * rng.normal(size=600), 0, 1)
    b = a if delta_ab == 0 else np.clip(a + delta_ab * rng.normal(size=600), 0, 1)
    return y, a, b


def test_comparisons_holm_per_metric_and_pairs():
    y, a, b = _scores(0.6)
    scores = {
        ("p", 100, 0, "uniform"): (y, a),
        ("p", 100, 0, "balanced"): (y, b),
        ("p", 100, 0, "hybrid"): (y, a.copy()),
    }
    rows = bcs.paired_cell_comparisons(scores, n_boot=200)
    assert len(rows) == 3 * 2  # three pairs x (auc, ap)
    for r in rows:
        assert r["n_family"] == 3
        assert r["p_holm"] >= r["p_raw"] - 1e-12
        assert r["significant"] == (r["p_holm"] < 0.05)
    same = next(
        r for r in rows if (r["a"], r["b"]) == ("uniform", "hybrid") and r["metric"] == "ap"
    )
    assert same["delta"] == 0.0 and not same["significant"]
    worse = next(
        r for r in rows if (r["a"], r["b"]) == ("uniform", "balanced") and r["metric"] == "auc"
    )
    assert worse["delta"] > 0.05 and worse["significant"]


def test_comparisons_refuse_different_rows():
    y, a, b = _scores(0.1)
    y2 = y.copy()
    y2[0] = 1 - y2[0]
    with pytest.raises(ValueError, match="other rows"):
        bcs.paired_cell_comparisons(
            {("p", 1, 0, "a"): (y, a), ("p", 1, 0, "b"): (y2, b)}, n_boot=10
        )


def test_run_sweep_records_cells_rates_and_skips(tmp_path):
    panels = [_panel("low", 0.1, seed=1), _panel("high", 0.3, seed=2)]
    rec = bcs.run_sweep(
        _tiny_model(),
        panels,
        strategies=("uniform", "balanced", "prototype"),
        sizes=(60, 10_000),  # the second covers the whole pool: skipped, never scored as a tie
        seeds=(0, 1),
        n_boot=120,
        n_ensemble=1,
        out_dir=tmp_path,
    )
    assert len(rec["cells"]) == 2 * 2 * 3  # panels x seeds x strategies at size 60
    assert rec["coverage"]["n_attempted"] == 2 * 2 * 3 * 2
    assert rec["coverage"]["n_skipped"] == 12
    assert all("whole pool" in r for r in rec["coverage"]["skipped"].values())
    for c in rec["cells"]:
        assert 0 <= c["auc"] <= 1 and 0 <= c["ap"] <= 1 and c["ece"] >= 0
        assert c["n_train"] > c["max_context"]
    rates = {
        p: np.mean([c["slice_default_rate"] for c in rec["cells"] if c["panel"] == p])
        for p in ("low", "high")
    }
    assert rates["low"] < rates["high"]
    # the same split under every strategy: identical slice rate within (panel, seed)
    for p in ("low", "high"):
        for sd in (0, 1):
            vals = {
                c["slice_default_rate"] for c in rec["cells"] if c["panel"] == p and c["seed"] == sd
            }
            assert len(vals) == 1
    assert len(rec["comparisons"]) == 2 * 2 * 3 * 2  # panels x seeds x pairs x metrics
    assert rec["comparisons_coverage"]["n_scored"] == len(rec["comparisons"])
    assert "Holm" in bcs.summarise(rec)
    saved = json.loads((tmp_path / bcs.OUT_FILE).read_text())
    assert len(saved["cells"]) == len(rec["cells"])


def test_run_sweep_skips_panels_wider_than_the_model():
    wide = _panel("wide", 0.2, n_feat=9)
    rec = bcs.run_sweep(
        _tiny_model(),
        [wide],
        strategies=("uniform",),
        sizes=(50,),
        seeds=(0,),
        n_boot=0,
        n_ensemble=1,
    )
    assert rec["cells"] == [] and rec["comparisons"] == []
    assert "features" in next(iter(rec["coverage"]["skipped"].values()))


def test_load_panels_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown panel"):
        bcs.load_panels(["mars"])


def _gap_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "gap_vs_default_rate.py"
    spec = importlib.util.spec_from_file_location("gap_vs_default_rate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gap_script_reads_the_sweep_record(tmp_path, capsys):
    cells = []
    for panel, rate, gap in (("a", 0.02, 0.05), ("b", 0.05, 0.03), ("c", 0.10, 0.01)):
        for seed in (0, 1):
            for strat, off in (("uniform", gap), ("balanced", 0.0)):
                cells.append(
                    {
                        "panel": panel,
                        "max_context": 500,
                        "seed": seed,
                        "strategy": strat,
                        "slice_default_rate": rate,
                        "ap": 0.3 + off,
                        "auc": 0.7 + off,
                    }
                )
    path = tmp_path / "binary_context_sweep.json"
    path.write_text(json.dumps({"cells": cells}))
    mod = _gap_script()
    rows = mod.binary_table(path)
    assert [r["panel"] for r in rows] == ["a", "b", "c"]
    assert rows[0]["gap_ap"] == pytest.approx(0.05) and rows[0]["seeds"] == 2
    mod.print_binary(path, "uniform", "balanced")
    out = capsys.readouterr().out
    assert "Spearman ap gap vs rate -1.00 (n=3)" in out
    assert mod.binary_table(path, "uniform", "retrieval") == []


def _slice_record(panel: str, seed: int, p_raws: list[float], **cfg_over) -> dict:
    """A synthetic one-(panel, seed) record with stale per-record Holm fields."""
    strategies = ["uniform", "balanced"]
    cells = [
        {
            "panel": panel,
            "max_context": 100,
            "seed": seed,
            "strategy": s,
            "n_train": 300,
            "n_test": 100,
            "slice_default_rate": 0.1,
            "train_default_rate": 0.1,
            "context_rate": 0.1,
            "n_context": 100,
            "n_context_positive": 10,
            "auc": 0.7,
            "ap": 0.3,
            "ece": 0.05,
            "brier": 0.1,
            "mean_predicted": 0.1,
            "seconds": 1.0,
        }
        for s in strategies
    ]
    rows = [
        {
            "panel": panel,
            "max_context": 100,
            "seed": seed,
            "a": "uniform",
            "b": "balanced",
            "metric": m,
            "delta": 0.01,
            "ci_lo": -0.01,
            "ci_hi": 0.03,
            "p_raw": p,
        }
        for m, p in zip(bcs.METRICS, p_raws, strict=True)
    ]
    bcs.apply_holm(rows)  # per-record family, as each Kaggle kernel wrote it
    cfg = {
        "model": "ckpt.pt",
        "panels": [panel],
        "strategies": strategies,
        "sizes": [100],
        "seeds": [seed],
        "n_boot": 200,
        "n_ensemble": 8,
        "test_size": 0.3,
        "max_rows": None,
        "feature_transform": "rank",
        "correct_prior": True,
    }
    cfg.update(cfg_over)
    cov = {"n_attempted": 2, "n_scored": 2, "n_skipped": 0, "coverage": 1.0, "skipped": {}}
    ccov = {"n_attempted": 2, "n_scored": 2, "n_skipped": 0, "coverage": 1.0, "skipped": {}}
    return {
        "config": cfg,
        "cells": cells,
        "coverage": cov,
        "comparisons": rows,
        "comparisons_coverage": ccov,
        "comparison_method": "m",
    }


def test_merge_unions_and_recomputes_holm_over_all_rows():
    recs = [
        _slice_record("p1", 0, [0.001, 0.2]),
        _slice_record("p1", 1, [0.01, 0.04]),
        _slice_record("p3", 0, [0.03, 0.5]),
    ]
    merged = bcs.merge_records(recs, sources=["a.json", "b.json", "c.json"])
    assert merged["config"]["panels"] == ["p1", "p3"]
    assert merged["config"]["seeds"] == [0, 1]
    assert len(merged["cells"]) == 6 and len(merged["comparisons"]) == 6
    for metric in bcs.METRICS:
        rows = [r for r in merged["comparisons"] if r["metric"] == metric]
        expect = bcs.holm_adjusted_p([r["p_raw"] for r in rows])
        assert [r["p_holm"] for r in rows] == pytest.approx(expect)
        assert all(r["n_family"] == 3 for r in rows)
        assert all(r["significant"] == (r["p_holm"] < 0.05) for r in rows)
    # a per-record family of 1 would have left p_holm == p_raw; the union is stricter
    first = merged["comparisons"][0]
    assert first["p_holm"] == pytest.approx(3 * 0.001)
    assert merged["coverage"]["n_attempted"] == 6
    assert merged["comparisons_coverage"]["n_scored"] == 6
    assert [p["source"] for p in merged["provenance"]] == ["a.json", "b.json", "c.json"]
    assert "Holm" in bcs.summarise(merged)


def test_merge_refuses_config_mismatch_naming_the_field():
    a = _slice_record("p1", 0, [0.01, 0.02])
    b = _slice_record("p1", 1, [0.01, 0.02], model="other.pt")
    with pytest.raises(ValueError, match="'model'"):
        bcs.merge_records([a, b])
    c = _slice_record("p1", 1, [0.01, 0.02], n_ensemble=4)
    with pytest.raises(ValueError, match="'n_ensemble'"):
        bcs.merge_records([a, c])


def test_merge_refuses_duplicate_cells():
    a = _slice_record("p1", 0, [0.01, 0.02])
    with pytest.raises(ValueError, match="duplicate cell"):
        bcs.merge_records([a, _slice_record("p1", 0, [0.01, 0.02])])


def test_merge_refuses_empty_input_and_source_mismatch():
    with pytest.raises(ValueError, match="no records"):
        bcs.merge_records([])
    with pytest.raises(ValueError, match="sources"):
        bcs.merge_records([_slice_record("p1", 0, [0.1, 0.2])], sources=["a", "b"])


def test_merge_of_one_record_keeps_its_holm():
    rec = _slice_record("p1", 0, [0.01, 0.2])
    merged = bcs.merge_records([rec])
    assert merged["comparisons"] == rec["comparisons"]
    assert merged["cells"] == rec["cells"]
    assert merged["coverage"] == rec["coverage"]
    assert bcs.summarise(merged) == bcs.summarise(rec)


def test_merge_merges_skip_coverage_and_rejects_double_skips():
    a = _slice_record("p1", 0, [0.01, 0.02])
    b = _slice_record("p1", 1, [0.01, 0.02])
    a["coverage"] = {**a["coverage"], "n_attempted": 3, "n_skipped": 1, "skipped": {"u": "r"}}
    merged = bcs.merge_records([a, b])
    assert merged["coverage"]["skipped"] == {"u": "r"}
    assert merged["coverage"]["coverage"] == pytest.approx(4 / 5)
    b["coverage"] = {**b["coverage"], "n_attempted": 3, "n_skipped": 1, "skipped": {"u": "r"}}
    with pytest.raises(ValueError, match="duplicate cell unit"):
        bcs.merge_records([a, b])


def test_merge_cli_writes_record_and_prints_summary(tmp_path, capsys, monkeypatch):
    paths = []
    for i, seed in enumerate((0, 1)):
        f = tmp_path / f"r{i}.json"
        f.write_text(json.dumps(_slice_record("p1", seed, [0.01, 0.02])))
        paths.append(str(f))
    out = tmp_path / "merged"
    monkeypatch.setattr("sys.argv", ["fintfm-binctx", "--merge", *paths, "--out", str(out)])
    bcs.main()
    merged = json.loads((out / bcs.OUT_FILE).read_text())
    assert merged["config"]["seeds"] == [0, 1]
    assert all(r["n_family"] == 2 for r in merged["comparisons"])
    assert [p["source"] for p in merged["provenance"]] == paths
    assert "paired AP comparisons" in capsys.readouterr().out


def test_holm_floor_is_family_over_resolution():
    """The floor is ``n_family / (n_boot + 1)``, capped at 1, and 0 when ``n_boot`` is unknown."""
    assert bcs.holm_floor(2000, 360) == pytest.approx(360 / 2001)
    assert bcs.holm_floor(10, 360) == 1.0
    assert bcs.holm_floor(None, 360) == 0.0
    assert bcs.holm_floor(2000, 0) == 0.0


def test_summary_warns_when_the_holm_floor_blocks_significance():
    """A family too large for the resamples gets a warning; a small one does not (FINDINGS §288)."""
    row = {
        "panel": "p",
        "max_context": 500,
        "seed": 0,
        "a": "uniform",
        "b": "retrieval",
        "metric": "ap",
        "delta": -0.01,
        "ci_lo": -0.02,
        "ci_hi": -0.001,
        "p_raw": 0.0005,
        "p_holm": 0.18,
    }
    rec = {"config": {"n_boot": 2000}, "cells": [], "comparisons": [dict(row) for _ in range(360)]}
    assert "WARNING: n_boot=2000" in bcs.summarise(rec)
    rec["comparisons"] = [dict(row) for _ in range(10)]
    assert "WARNING" not in bcs.summarise(rec)
