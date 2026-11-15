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
