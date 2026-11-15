"""Tests for the DGP/task-representation probe (task 39.7). SMOKE-TEST only."""

import json

import numpy as np
import pytest
import torch

from fintfm.experiments import task_probe as tp
from fintfm.modeling.model import FinancialTFM, ModelConfig


def _tiny() -> FinancialTFM:
    torch.manual_seed(1)
    return FinancialTFM(
        ModelConfig(max_features=8, max_classes=2, d_model=16, n_heads=2, n_layers=1, d_ff=32)
    )


def test_twenty_plus_families_balanced_and_deterministic():
    assert len(tp.FAMILIES) >= 20
    for f in tp.FAMILIES:
        X, y = tp.make_task(f, 3, 200, 8)
        assert X.shape == (200, 8) and y.shape == (200,)
        assert abs(y.mean() - 0.5) <= 0.01, f  # median-thresholded or exactly balanced
        X2, y2 = tp.make_task(f, 3, 200, 8)
        np.testing.assert_array_equal(X, X2)
        np.testing.assert_array_equal(y, y2)
    assert not np.array_equal(tp.make_task("xor", 1, 50, 8)[0], tp.make_task("xor", 2, 50, 8)[0])
    assert not np.array_equal(tp.make_task("xor", 1, 50, 8)[0], tp.make_task("and", 1, 50, 8)[0])


def test_family_structure_spot_checks():
    X, y = tp.make_task("threshold", 0, 2000, 8)
    c = np.abs([np.corrcoef(X[:, j], y)[0, 1] for j in range(8)])
    assert c.max() > 0.7 and np.sort(c)[-2] < 0.1  # exactly one informative column
    X, y = tp.make_task("xor", 0, 4000, 8)
    marg = np.abs([np.corrcoef(X[:, j], y)[0, 1] for j in range(8)])
    assert marg.max() < 0.08  # no single column predicts an XOR
    pair = max(
        abs(np.corrcoef(X[:, i] * X[:, j], y)[0, 1]) for i in range(8) for j in range(i + 1, 8)
    )
    assert pair > 0.5  # the product of the pair does
    X, y = tp.make_task("dense_linear", 0, 4000, 8)
    assert (np.abs([np.corrcoef(X[:, j], y)[0, 1] for j in range(8)]) > 0.1).all()
    X, y = tp.make_task("latent_factor", 0, 3000, 8)
    assert np.abs(np.corrcoef(X, rowvar=False)[np.triu_indices(8, 1)]).mean() > 0.2
    X, y = tp.make_task("mixture", 0, 400, 8)
    assert y.sum() == 200


def test_validation():
    with pytest.raises(ValueError):
        tp.make_task("nope", 0, 50, 8)
    with pytest.raises(ValueError):
        tp.make_task("xor", 0, 50, 3)
    with pytest.raises(ValueError):
        tp.make_task("xor", 0, 4, 8)


def test_raw_statistics_fixed_length():
    a = tp.raw_statistics(*tp.make_task("xor", 0, 100, 8))
    b = tp.raw_statistics(*tp.make_task("linear", 1, 150, 8))
    assert a.shape == b.shape and np.isfinite(a).all()


def test_representation_fixed_size_and_deterministic():
    m = _tiny()
    probe = tp.probe_rows(10, 8)
    np.testing.assert_array_equal(probe, tp.probe_rows(10, 8))
    v1 = tp.task_representation(m, *tp.make_task("xor", 0, 64, 8), probe)
    v2 = tp.task_representation(m, *tp.make_task("linear", 1, 100, 8), probe)
    assert v1.shape == v2.shape == (32,)  # 2 * d_model, whatever the context size
    again = tp.task_representation(m, *tp.make_task("xor", 0, 64, 8), probe)
    np.testing.assert_allclose(v1, again, atol=1e-5)
    assert not np.allclose(v1, v2)  # the context moves the representation


def test_end_to_end_untrained(tmp_path):
    fams = ("linear", "xor", "radial", "mixture")
    rec = tp.run(
        {},
        tmp_path,
        families=fams,
        n_tasks=4,
        n_rows=48,
        n_probe=8,
        folds=2,
        n_permutations=2,
        models={"tiny": _tiny()},
    )
    assert set(rec["results"]) == {"raw_statistics", "tiny", "untrained_control"}
    assert rec["chance"] == 0.25
    assert rec["feature_dim"]["tiny"] == 32
    for r in rec["results"].values():
        assert 0.0 <= r["accuracy"] <= 1.0
        assert np.array(r["confusion"]).sum() == 16
    assert json.loads((tmp_path / "task_probe.json").read_text())["chance"] == 0.25
    assert "untrained_control" in tp.summarise(rec)
    with pytest.raises(ValueError):
        tp.run({}, families=fams, n_tasks=1, folds=2, models={"tiny": _tiny()})
    with pytest.raises(ValueError):
        tp.run({}, families=fams, d=9, n_tasks=4, folds=2, models={"tiny": _tiny()})
