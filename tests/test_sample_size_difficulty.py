"""Tests for the sample-size x difficulty grid (task 39.16). SMOKE-TEST only."""

import json

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from fintfm.experiments import sample_size_difficulty as ssd
from fintfm.modeling.model import FinancialTFM, ModelConfig


def _tiny(max_features: int = 8) -> FinancialTFM:
    torch.manual_seed(1)
    return FinancialTFM(
        ModelConfig(
            max_features=max_features, max_classes=2, d_model=16, n_heads=2, n_layers=1, d_ff=32
        )
    )


def test_grid_shape_and_balanced_labels():
    assert len(ssd.SIZES) == 5 and set(ssd.TIERS) == {"easy", "medium", "hard"}
    Xtr, ytr, Xte, yte, pb = ssd.make_split(0, "hard", 16, n_query=100)
    assert Xtr.shape == (16, ssd.WIDTH) and Xte.shape == (100, ssd.WIDTH) and pb.shape == (100,)
    assert ytr.sum() == 8 and yte.sum() == 50


def test_bayes_auc_orders_the_tiers_and_matches_targets():
    auc = {}
    for t, target in ssd.TIERS.items():
        *_, yte, pb = ssd.make_split(0, t, 16, n_query=20000)
        auc[t] = roc_auc_score(yte, pb)
        assert auc[t] == pytest.approx(target, abs=0.01)
    assert auc["easy"] > auc["medium"] > auc["hard"]


def test_deterministic_and_query_shared_across_sizes():
    a = ssd.make_split(3, "medium", 64, n_query=50)
    b = ssd.make_split(3, "medium", 64, n_query=50)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)
    c = ssd.make_split(3, "medium", 256, n_query=50)
    np.testing.assert_array_equal(a[2], c[2])
    assert not np.array_equal(a[0], ssd.make_split(4, "medium", 64, n_query=50)[0])


def test_curve_shape():
    s = ssd.curve_shape([10, 100, 1000], [0.5, 0.6, 0.7])
    assert s["slope_per_decade"] == pytest.approx(0.1)
    assert s["step_gains"] == pytest.approx([0.1, 0.1])
    with pytest.raises(ValueError):
        ssd.curve_shape([5, 5], [0.5, 0.6])


def test_end_to_end_2x2(tmp_path):
    rec = ssd.run(
        {},
        tmp_path,
        sizes=(16, 64),
        tiers=("easy", "hard"),
        seeds=(0, 1),
        n_query=200,
        models={"tiny": _tiny()},
    )
    arms = {r["arm"] for r in rec["results"]}
    assert arms == {"tiny", "untrained_control", "logistic_regression", "gradient_boosting"}
    assert len(rec["results"]) == 4 * 2 * 2
    assert all(r["n_seeds"] == 2 for r in rec["results"])
    on_disk = json.loads((tmp_path / "sample_size_difficulty.json").read_text())
    assert on_disk["config"]["sizes"] == [16, 64]
    lr = {(r["tier"], r["n"]): r for r in rec["results"] if r["arm"] == "logistic_regression"}
    assert lr[("easy", 64)]["auc_mean"] > 0.9
    assert lr[("easy", 64)]["regret_mean"] < 0.05
    assert "easy" in ssd.summarise(rec)


def test_unknown_tier_and_narrow_model_refused():
    with pytest.raises(ValueError, match="tier"):
        ssd.run({}, None, tiers=("trivial",), models={"tiny": _tiny()})
    with pytest.raises(ValueError, match="max_features"):
        ssd.run({}, None, sizes=(16,), tiers=("easy",), models={"narrow": _tiny(4)})


def test_plot_writes_one_svg_per_tier(tmp_path):
    import xml.etree.ElementTree as ET

    rec = ssd.run(
        {},
        tmp_path,
        sizes=(16, 64),
        tiers=("easy", "hard"),
        seeds=(0,),
        n_query=100,
        models={"tiny": _tiny()},
    )
    paths = ssd.plot_curves(rec, tmp_path)
    assert [p.name for p in paths] == [
        "sample_size_difficulty_easy.svg",
        "sample_size_difficulty_hard.svg",
    ]
    for p in paths:
        root = ET.fromstring(p.read_text())
        assert len(root.findall("{http://www.w3.org/2000/svg}polyline")) == 4
