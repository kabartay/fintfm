"""Tests for the correlated-feature and confounding harness (task 39.14). SMOKE-TEST only."""

import json

import numpy as np
import pytest
import torch

from fintfm.experiments import correlated_features as cf
from fintfm.modeling.model import FinancialTFM, ModelConfig


def _tiny() -> FinancialTFM:
    torch.manual_seed(1)
    return FinancialTFM(
        ModelConfig(max_features=8, max_classes=2, d_model=16, n_heads=2, n_layers=1, d_ff=32)
    )


@pytest.mark.parametrize("design", cf.DESIGNS)
@pytest.mark.parametrize("r", [0.0, 0.5, 0.9])
def test_realised_correlation_matches_request(design, r):
    X, y, z = cf.make_design(design, r, seed=0, n=40000)
    assert X.shape == (40000, 2) and y.shape == (40000,)
    assert abs(cf.realised_correlation(design, X, z) - r) < 0.02
    assert abs(y.mean() - 0.25) < 0.01


@pytest.mark.parametrize("design", ["redundant", "proxy"])
def test_y_depends_only_on_z(design):
    outs = [cf.make_design(design, r, seed=2, n=500) for r in (0.0, 0.6, 0.99)]
    for _, y, z in outs:
        np.testing.assert_array_equal(y, outs[0][1])
        np.testing.assert_array_equal(z, outs[0][2])
        np.testing.assert_array_equal(y, (z >= np.quantile(z, 0.75)).astype(np.int64))
    # In the proxy design y is independent of the secondary column given z.
    X, y, z = cf.make_design("proxy", 0.8, seed=0, n=40000)
    resid = X[:, 1] - 0.8 * z
    assert abs(np.corrcoef(resid, y)[0, 1]) < 0.02


def test_deterministic_and_validated():
    a = cf.make_design("collider", 0.7, seed=5, n=300)
    b = cf.make_design("collider", 0.7, seed=5, n=300)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)
    assert not np.array_equal(a[0], cf.make_design("collider", 0.7, seed=6, n=300)[0])
    with pytest.raises(ValueError):
        cf.make_design("nope", 0.5, 0)
    with pytest.raises(ValueError):
        cf.make_design("proxy", 1.5, 0)


def test_intervention_breaks_only_that_column():
    X, y, _ = cf.make_design("proxy", 0.9, seed=0, n=5000)
    Xi = cf.intervene(X, 1, seed=3)
    np.testing.assert_array_equal(Xi[:, 0], X[:, 0])
    np.testing.assert_array_equal(np.sort(Xi[:, 1]), np.sort(X[:, 1]))  # marginal kept
    assert abs(np.corrcoef(Xi[:, 0], Xi[:, 1])[0, 1]) < 0.05  # link to the cause broken
    assert abs(np.corrcoef(Xi[:, 1], y)[0, 1]) < 0.05
    assert not np.array_equal(X[:, 1], Xi[:, 1])


def test_reliance_share():
    assert cf.reliance_share(0.3, 0.1) == pytest.approx(0.25)
    assert cf.reliance_share(0.2, -0.1) == 0.0
    assert cf.reliance_share(-0.1, -0.2) is None


def test_classical_arms_use_cause_not_proxy_at_high_correlation():
    rec = cf.run({}, designs=("proxy",), levels=(0.9,), seeds=(0,), n=2000, models={})
    gb = next(c for c in rec["results"] if c["arm"] == "gradient_boosting")
    oracle = next(c for c in rec["results"] if c["arm"] == "z_oracle")
    assert oracle["auc_obs"] == pytest.approx(1.0)
    assert gb["drop_primary"] > gb["drop_secondary"]


def test_tiny_end_to_end(tmp_path):
    rec = cf.run(
        {},
        tmp_path,
        designs=("redundant", "collider"),
        levels=(0.0, 0.9),
        seeds=(0,),
        n=200,
        max_context=100,
        models={"tiny": _tiny()},
    )
    arms = {c["arm"] for c in rec["results"]}
    assert arms == {
        "tiny",
        "untrained_control",
        "logistic_regression",
        "gradient_boosting",
        "z_oracle",
    }
    assert len(rec["results"]) == 5 * 2 * 2
    for c in rec["results"]:
        assert 0.0 <= c["auc_obs"] <= 1.0
    assert max(rec["max_correlation_deviation"].values()) < 0.2
    saved = json.loads((tmp_path / "correlated_features.json").read_text())
    assert saved["config"]["levels"] == [0.0, 0.9]
    text = cf.summarise(rec)
    assert "[redundant]" in text and "secondary share" in text and "untrained_control" in text
