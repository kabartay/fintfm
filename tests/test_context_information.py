"""Tests for the context-information sufficiency harness (task 39.8). SMOKE-TEST only."""

import json

import numpy as np
import pytest
import torch

from fintfm.experiments import context_information as ci
from fintfm.modeling.model import FinancialTFM, ModelConfig


def _tiny() -> FinancialTFM:
    torch.manual_seed(1)
    return FinancialTFM(
        ModelConfig(max_features=8, max_classes=2, d_model=16, n_heads=2, n_layers=1, d_ff=32)
    )


def test_information_hand_examples():
    bal = ci.information(np.array([0, 1] * 8))
    assert bal["label_bits"] == pytest.approx(16.0)  # 16 rows at 1 bit each
    assert bal["n_eff"] == pytest.approx(8.0) and bal["n_minority"] == 8
    quarter = ci.information(np.array([1] + [0] * 3))  # H(0.25) = 0.8112781 bits
    assert quarter["label_bits"] == pytest.approx(4 * 0.8112781, abs=1e-5)
    assert quarter["n_eff"] == pytest.approx(1.5) and quarter["n_pos"] == 1
    one = ci.information(np.zeros(10))
    assert one["label_bits"] == 0.0 and one["n_eff"] == 0.0


def test_split_counts_and_balanced_query():
    Xtr, ytr, _, yte = ci.make_split(0, "and2", 100, 0.1, n_query=60)
    assert Xtr.shape == (100, ci.WIDTH) and ytr.sum() == 10 and yte.sum() == 30
    assert ci.make_split(0, "linear1", 64, 0.03, n_query=20)[1].sum() == 2


def test_labels_follow_the_task_rule():
    X, y, *_ = ci.make_split(1, "interaction4", 200, 0.25, n_query=20)
    t = ci._threshold("interaction4", 0.25)
    np.testing.assert_array_equal((X[:, 0] * X[:, 1] * X[:, 2] * X[:, 3] > t).astype(int), y)


def test_rows_and_information_are_decoupled():
    d = ci.matched_cells()
    assert d["same_rows"] and d["same_bits"]
    a, b = d["same_rows"][0]
    assert a[0] == b[0] and a[1] != b[1]
    c, e = d["same_bits"][0]
    assert c[0] != e[0]
    # equal rows, different information
    assert ci._bits(1024, 0.5) > 3 * ci._bits(1024, 0.03)
    # different rows, (nearly) equal information
    assert ci._bits(1024, 0.25) == pytest.approx(ci._bits(4096, 0.03), rel=0.15)


def test_deterministic_and_query_shared():
    a = ci.make_split(3, "and2", 64, 0.25, n_query=40)
    b = ci.make_split(3, "and2", 64, 0.25, n_query=40)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)
    c = ci.make_split(3, "and2", 256, 0.25, n_query=40)
    np.testing.assert_array_equal(a[2], c[2])
    assert not np.array_equal(a[0], ci.make_split(4, "and2", 64, 0.25, n_query=40)[0])


def test_axis_fit():
    assert ci.axis_fit([10, 100, 1000], [0.5, 0.6, 0.7]) == pytest.approx(1.0)
    assert ci.axis_fit([10, 10, 10], [0.5, 0.6, 0.7]) is None
    assert ci.axis_fit([10, 100], [0.5, 0.6]) is None


def test_end_to_end(tmp_path):
    kw = {
        "rows": (64, 256),
        "rates": (0.5, 0.1),
        "tasks": ("linear1", "and2"),
        "seeds": (0, 1),
        "n_query": 200,
    }
    rec = ci.run({}, tmp_path, models={"tiny": _tiny()}, **kw)
    arms = {r["arm"] for r in rec["results"]}
    assert arms == {"tiny", "untrained_control", "logistic_regression", "gradient_boosting"}
    assert len(rec["results"]) == 4 * 2 * 2 * 2
    on_disk = json.loads((tmp_path / "context_information.json").read_text())
    assert on_disk["config"]["rows"] == [64, 256]
    lr = {
        (r["task"], r["rows"], r["rate"]): r
        for r in rec["results"]
        if r["arm"] == "logistic_regression"
    }
    assert lr[("linear1", 256, 0.5)]["auc_mean"] > 0.9
    assert "bits" in ci.summarise(rec)
    rec2 = ci.run({}, None, models={"tiny": _tiny()}, **kw)
    assert [r["auc_mean"] for r in rec["results"]] == [r["auc_mean"] for r in rec2["results"]]


def test_unknown_task_refused():
    with pytest.raises(ValueError, match="task"):
        ci.make_split(0, "nope", 64, 0.5)
    with pytest.raises(ValueError, match="task"):
        ci.run({}, None, tasks=("nope",), models={"t": _tiny()})
