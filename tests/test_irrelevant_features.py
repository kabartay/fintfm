"""Tests for the irrelevant-feature robustness harness (tasks 42.1 / 39.13). SMOKE-TEST only."""

import json
import time

import numpy as np
import pytest
import torch

from fintfm.experiments import irrelevant_features as irr
from fintfm.modeling.model import FinancialTFM, ModelConfig


def _tiny() -> FinancialTFM:
    torch.manual_seed(1)
    return FinancialTFM(
        ModelConfig(max_features=8, max_classes=2, d_model=16, n_heads=2, n_layers=1, d_ff=32)
    )


def test_shapes_and_label_rate():
    X_rel, y, noise = irr.make_task(0, n=400, max_irrelevant=10, rate=0.25)
    assert X_rel.shape == (400, 2) and y.shape == (400,) and noise.shape == (400, 10)
    assert abs(y.mean() - 0.25) < 0.01
    assert irr.build_table(X_rel, noise, 0).shape == (400, 2)
    assert irr.build_table(X_rel, noise, 7).shape == (400, 9)
    with pytest.raises(ValueError):
        irr.build_table(X_rel, noise, 11)


def test_deterministic_under_a_seed_and_nested_across_widths():
    a = irr.make_task(3, n=300, max_irrelevant=6)
    b = irr.make_task(3, n=300, max_irrelevant=6)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)
    c = irr.make_task(3, n=300, max_irrelevant=12)  # wider pool: same task, same first columns
    np.testing.assert_array_equal(a[0], c[0])
    np.testing.assert_array_equal(a[1], c[1])
    np.testing.assert_array_equal(a[2], c[2][:, :6])
    assert not np.array_equal(a[0], irr.make_task(4, n=300, max_irrelevant=6)[0])


def test_irrelevant_columns_are_independent_of_y():
    _, y, noise = irr.make_task(0, n=20000, max_irrelevant=8)
    corr = [abs(np.corrcoef(noise[:, j], y)[0, 1]) for j in range(8)]
    assert max(corr) < 0.03  # ~4 sigma at n=20000
    assert abs(np.corrcoef(noise[:, 0], noise[:, 1])[0, 1]) < 0.03


def test_label_depends_on_the_relevant_columns():
    X_rel, y, _ = irr.make_task(0, n=5000, max_irrelevant=1)
    assert abs(np.corrcoef(X_rel[:, 0], y)[0, 1]) > 0.2


def test_curve_slope():
    s = irr.curve_slope([2, 12, 22], [0.9, 0.8, 0.7])
    assert s["slope_per_10_columns"] == pytest.approx(-0.1)
    assert s["total_change"] == pytest.approx(-0.2)
    assert s["slope_per_decade"] < 0
    with pytest.raises(ValueError):
        irr.curve_slope([5, 5], [0.5, 0.6])


def test_end_to_end_untrained_two_widths(tmp_path):
    t0 = time.perf_counter()
    rec = irr.run(
        {},
        tmp_path,
        widths=(2, 6),
        seeds=(0, 1),
        n=200,
        max_context=100,
        models={"tiny": _tiny()},
    )
    print(f"tiny end-to-end: {time.perf_counter() - t0:.1f} s")  # input to the CPU-time estimate
    arms = {r["arm"] for r in rec["results"]}
    assert arms == {"tiny", "untrained_control", "logistic_regression", "gradient_boosting"}
    assert len(rec["results"]) == 4 * 2
    assert all(0.0 <= r["auc_mean"] <= 1.0 for r in rec["results"])
    assert all(r["n_seeds"] == 2 for r in rec["results"])
    assert set(rec["slopes"]) == arms
    on_disk = json.loads((tmp_path / "irrelevant_features.json").read_text())
    assert on_disk["config"]["widths"] == [2, 6]
    assert "d=6" in irr.summarise(rec)
    # the learned baselines must beat chance on the clean two-column task
    gb = next(r for r in rec["results"] if r["arm"] == "gradient_boosting" and r["d"] == 2)
    assert gb["auc_mean"] > 0.8


def test_width_beyond_model_capacity_is_refused():
    with pytest.raises(ValueError, match="max_features"):
        irr.run({}, None, widths=(2, 20), seeds=(0,), n=100, models={"tiny": _tiny()})


def test_plot_writes_svg(tmp_path):
    import xml.etree.ElementTree as ET

    rec = irr.run(
        {}, tmp_path, widths=(2, 6), seeds=(0, 1), n=200, max_context=100, models={"tiny": _tiny()}
    )
    path = irr.plot_curves(rec, tmp_path)
    assert path == tmp_path / "irrelevant_features.svg"
    root = ET.fromstring(path.read_text())
    assert len(root.findall("{http://www.w3.org/2000/svg}polyline")) == 4
