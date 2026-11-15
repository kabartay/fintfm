"""Task 4.3: the fitted-calibration comparison harness runs end to end on synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fintfm.evaluation.datasets import CreditDataset
from fintfm.experiments.fitted_calibration import (
    ARMS,
    PAIRS,
    _split,
    paired_metric_delta,
    run,
    summarise,
)


def _panel(name: str, horizon: int, seed: int) -> CreditDataset:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(300, 5)).astype(np.float32)
    y = (X[:, 0] + rng.normal(size=300) > 1.4).astype(np.int64)
    return CreditDataset(X, y, name, horizon, "test", "synthetic")


def test_split_is_stratified_disjoint_and_seeded() -> None:
    y = (np.arange(200) % 10 == 0).astype(int)
    tr, te = _split(y, 0.3, 1)
    assert not set(tr) & set(te) and len(tr) + len(te) == 200
    assert y[te].sum() == round(0.3 * y.sum())
    assert np.array_equal(te, _split(y, 0.3, 1)[1])
    assert not np.array_equal(te, _split(y, 0.3, 2)[1])


def test_paired_delta_sign_and_validation() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.2, 2000)
    y = (rng.uniform(size=2000) < p).astype(int)
    d, (lo, hi) = paired_metric_delta(y, p, np.full_like(p, 0.9), "brier", n_boot=200)
    assert d < 0 and hi < 0 and lo <= d <= hi
    d0, (lo0, _) = paired_metric_delta(y, p, p, "ece", n_boot=50)
    assert d0 == 0.0 and lo0 == 0.0
    assert np.isnan(paired_metric_delta(y, p, p, "ece", n_boot=0)[1][0])
    with pytest.raises(ValueError):
        paired_metric_delta(y, p, p, "auc")
    with pytest.raises(ValueError):
        paired_metric_delta(y, p, p[:10], "ece")


def test_run_reports_every_arm_level_and_pair(tmp_path: Path) -> None:
    """Shape and bookkeeping only; an untrained model's numbers say nothing about a checkpoint."""
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    model = FinancialTFM(
        ModelConfig(max_features=8, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_classes=2)
    ).eval()
    panels = [_panel("p1", 1, 0), _panel("p3", 3, 1)]
    rec = run(
        model,
        tmp_path,
        strategies=("balanced",),
        seeds=(0, 1, 2),
        max_context=60,
        n_boot=20,
        panels=panels,
    )
    assert len(rec["cells"]) == 2 * 1 * 3 * len(ARMS)
    for c in rec["cells"]:
        for k in ("mean_predicted", "observed_rate", "ece", "brier", "pred_sd", "is_near_constant"):
            assert k in c
        if ARMS[c["arm"]]["fitted"]:
            assert c["n_train_rows"] < 210  # the calibration slice is held out
        else:
            assert c["n_train_rows"] == 210 and c["temperature"] == 1.0
    # one split per seed shared by every arm: identical observed rate within a seed
    for seed in (0, 1, 2):
        got = {c["observed_rate"] for c in rec["cells"] if c["panel"] == "p1" and c["seed"] == seed}
        assert len(got) == 1
    assert all(r["n_seeds"] == 3 for r in rec["seed_means"])
    assert len(rec["paired"]) == 2 * len(PAIRS) * 2
    assert all(len(r["delta_by_seed"]) == 3 for r in rec["paired"])
    saved = json.loads((tmp_path / "fitted_calibration.json").read_text())
    assert saved["config"]["seeds"] == [0, 1, 2]
    table = summarise(rec)
    assert "analytic_fitted" in table and "obs" in table and "p99/p50" in table
