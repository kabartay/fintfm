"""Tests for the sign/direction inference harness (task 39.10). SMOKE-TEST only."""

import json

import numpy as np
import pytest
import torch

from fintfm.experiments import sign_inference as si
from fintfm.modeling.model import FinancialTFM, ModelConfig


def _tiny() -> FinancialTFM:
    torch.manual_seed(1)
    return FinancialTFM(
        ModelConfig(max_features=8, max_classes=2, d_model=16, n_heads=2, n_layers=1, d_ff=32)
    )


def test_pair_shares_x_and_labels_are_mirrors():
    beta = si.draw_beta("beta", 5, 4)
    X, yp, yn = si.make_pair(beta, 500, 8, 5)
    assert X.shape == (500, 8)
    np.testing.assert_array_equal(yp + yn, np.ones(500))  # exact mirror
    assert abs(yp.mean() - 0.5) < 0.1
    X2, _, _ = si.make_pair(beta, 500, 8, 5)
    np.testing.assert_array_equal(X, X2)  # one table serves both tasks
    # the X distribution does not depend on beta: same seed, opposite beta, identical X
    Xm, ypm, _ = si.make_pair(-beta, 500, 8, 5)
    np.testing.assert_array_equal(X, Xm)
    np.testing.assert_array_equal(ypm, yn)  # flipping beta flips the labels
    # label direction follows beta: sign(corr(x_j, y_pos)) == sign(beta_j)
    Xl, ypl, _ = si.make_pair(beta, 20000, 8, 6)
    c = [np.corrcoef(Xl[:, j], ypl)[0, 1] for j in range(4)]
    np.testing.assert_array_equal(np.sign(c), np.sign(beta))
    assert abs(np.corrcoef(Xl[:, 6], ypl)[0, 1]) < 0.03  # noise column


def test_sign_family_is_x0_threshold():
    X, yp, yn = si.make_pair(si.draw_beta("sign", 0, 4), 300, 8, 0)
    np.testing.assert_array_equal(yp, (X[:, 0] > 0).astype(int))
    np.testing.assert_array_equal(yn, (X[:, 0] < 0).astype(int))


def test_deterministic_and_validation():
    np.testing.assert_array_equal(si.draw_beta("beta", 3, 4), si.draw_beta("beta", 3, 4))
    assert np.all(np.abs(si.draw_beta("beta", 3, 6)) >= 0.3)
    with pytest.raises(ValueError):
        si.draw_beta("nope", 0, 4)
    with pytest.raises(ValueError):
        si.make_pair(np.zeros(2), 10, 4, 0)
    with pytest.raises(ValueError):
        si.make_pair(np.ones(5), 10, 4, 0)


def test_recovered_signs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    pred = 2 * X[:, 0] - X[:, 1]
    np.testing.assert_array_equal(si.recovered_signs(pred, X, 2), [1, -1])
    np.testing.assert_array_equal(si.recovered_signs(np.ones(500), X, 2), [0, 0])


def test_end_to_end_untrained_and_deterministic(tmp_path):
    kw = {"seeds": (0,), "tasks_per_seed": 1, "n": 200, "d": 6, "n_relevant": 3, "max_context": 100}
    m = _tiny()
    rec = si.run({}, tmp_path, models={"tiny": m}, **kw)
    arms = {r["arm"] for r in rec["results"]}
    assert arms == {"tiny", "untrained_control", "logistic_regression"}
    assert {r["family"] for r in rec["results"]} == {"sign", "beta"}
    lr = next(
        r for r in rec["results"] if r["arm"] == "logistic_regression" and r["family"] == "sign"
    )
    assert lr["auc_pos"] > 0.95 and lr["auc_neg"] > 0.95 and lr["auc_flipped"] < 0.05
    assert lr["flip_corr"] < -0.9 and lr["sign_correct"] == 1.0
    on_disk = json.loads((tmp_path / "sign_inference.json").read_text())
    assert on_disk["results"] == rec["results"]
    assert "flip_corr" in si.summarise(rec)
    rec2 = si.run({}, None, models={"tiny": m}, **kw)
    assert rec2["results"] == rec["results"]


def test_width_beyond_model_capacity_is_refused():
    with pytest.raises(ValueError):
        si.run({}, None, models={"tiny": _tiny()}, d=9, n_relevant=2, n=100)
