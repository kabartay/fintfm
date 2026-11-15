"""Tests for the tail-weight inversion harness (task 38.14). SMOKE-TEST only."""

import json

import numpy as np
import pytest
import torch

from fintfm.experiments import capability as cap
from fintfm.experiments import tail_inversion as ti
from fintfm.modeling.model import FinancialTFM, ModelConfig


def _tiny() -> FinancialTFM:
    torch.manual_seed(1)
    return FinancialTFM(
        ModelConfig(max_features=8, max_classes=2, d_model=16, n_heads=2, n_layers=1, d_ff=32)
    )


def test_tail_weight_increases_along_the_family_order():
    rng = np.random.default_rng(0)
    stats = [ti.tail_stats(ti.sample_features(f, 200_000, 5, rng)) for f in ti.FAMILIES]
    ratios = [s["tail_ratio"] for s in stats]
    assert ratios == sorted(ratios) and len(set(ratios)) == len(ratios)
    kurt = [s["excess_kurtosis"] for s in stats]
    assert kurt[0] < 0.2 < kurt[1] and kurt == sorted(kurt)


def test_bulk_is_matched_across_families():
    rng = np.random.default_rng(1)
    for f in ti.FAMILIES:
        x = ti.sample_features(f, 200_000, 3, rng)
        iqr = np.quantile(x, 0.75) - np.quantile(x, 0.25)
        assert iqr == pytest.approx(1.349, abs=0.03)


@pytest.mark.parametrize("kind", ti.SYMMETRIC_PROBES)
def test_signal_matched_label_rate_and_oracle_auc(kind):
    from sklearn.metrics import roc_auc_score

    for f in ti.FAMILIES:
        _, _, Xte, yte = ti.make_tail_probe(kind, f, n=4000, seed=2)
        # count(x > 0) is discrete, so ties at the cut inflate the rate (as in make_probe)
        tol = 0.02 if kind == "symmetric_sum" else 0.09
        assert abs(yte.mean() - 0.05) < tol
        score = Xte.sum(axis=1) if kind == "symmetric_sum" else (Xte > 0).sum(axis=1)
        assert roc_auc_score(yte, score) > 0.99


@pytest.mark.parametrize("kind", ti.SYMMETRIC_PROBES)
def test_gaussian_family_reproduces_make_probe(kind):
    a = ti.make_tail_probe(kind, "gaussian", n=600, n_features=6, seed=4)
    b = cap.make_probe(kind, n=600, n_features=6, seed=4)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)


def test_deterministic_and_seed_sensitive():
    a = ti.make_tail_probe("symmetric_sum", "t3", n=300, seed=3)
    b = ti.make_tail_probe("symmetric_sum", "t3", n=300, seed=3)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)
    assert not np.array_equal(a[0], ti.make_tail_probe("symmetric_sum", "t3", n=300, seed=4)[0])


def test_rejects_unknown_inputs():
    with pytest.raises(ValueError):
        ti.make_tail_probe("antisymmetric", "gaussian")
    with pytest.raises(ValueError):
        ti.make_tail_probe("symmetric_sum", "cauchy")


def test_tail_trend():
    t = ti.tail_trend(["t3", "gaussian", "t5"], [0.3, 0.9, 0.6])
    assert t["spearman"] == pytest.approx(-1.0)
    assert t["slope_per_step"] < 0 and t["inverted"] is True
    assert t["total_change"] == pytest.approx(-0.6)
    flat = ti.tail_trend(["gaussian", "t5"], [0.9, 0.9])
    assert np.isnan(flat["spearman"]) and flat["inverted"] is False
    with pytest.raises(ValueError):
        ti.tail_trend(["gaussian"], [0.9])


def test_end_to_end_tiny_run(tmp_path):
    rec = ti.run(
        {},
        tmp_path,
        families=("gaussian", "t3"),
        probes=("symmetric_sum",),
        seeds=(0,),
        n=300,
        n_features=6,
        max_context=100,
        models={"tiny": _tiny()},
    )
    arms = {r["arm"] for r in rec["results"]}
    assert {"tiny", "untrained_control", "logistic_regression", "gradient_boosting"} <= arms
    assert len(rec["results"]) == 4 * 2
    assert all(0.0 <= r["auc_mean"] <= 1.0 for r in rec["results"])
    assert set(rec["tail_stats"]) == {"gaussian", "t3"}
    assert "tiny" in rec["trends"]
    assert json.loads((tmp_path / "tail_inversion.json").read_text())["config"]["seeds"] == [0]
    text = ti.summarise(rec)
    assert "symmetric_sum" in text and "untrained_control" in text
