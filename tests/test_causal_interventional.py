"""Tests for the observational vs interventional SCM harness (task 39.15). SMOKE-TEST only."""

import json

import numpy as np
import pytest
import torch

from fintfm.experiments import causal_interventional as ci
from fintfm.modeling.model import FinancialTFM, ModelConfig


def _tiny() -> FinancialTFM:
    torch.manual_seed(1)
    return FinancialTFM(
        ModelConfig(max_features=8, max_classes=2, d_model=16, n_heads=2, n_layers=1, d_ff=32)
    )


def test_closed_form_hand_examples():
    # g = 0: Y depends on X1 only, so every closed form is sigmoid(w x1 + c) exactly.
    scm = ci.SCM(a=0.6, b=0.5, w=2.0, g=0.0, c=0.5)
    x1, x2 = np.array([0.0, 1.0, -1.5]), np.array([0.3, -2.0, 1.0])
    want = 1 / (1 + np.exp(-(2.0 * x1 + 0.5)))
    np.testing.assert_allclose(ci.p_y_observational(scm, x1, x2), want, atol=1e-9)
    np.testing.assert_allclose(ci.p_y_interventional(scm, x1, x2), want, atol=1e-9)
    # do(x1) with X2 unobserved and g = 0 is the same again; at x1 = 0, c = 0 it is 1/2.
    flat = ci.SCM(g=0.0, c=0.0)
    assert ci.p_y_do_marginal(flat, np.array([0.0]))[0] == pytest.approx(0.5)


def test_closed_form_confounded_hand_example():
    # w = 0: Y = sigmoid(g Z). do(X1) cannot matter, and with x2 = 0 Z|x2 is centred at 0, so
    # the interventional probability is 1/2 whatever x1 is.
    scm = ci.SCM(a=0.7, b=0.7, w=0.0, g=1.5, c=0.0)
    assert ci.p_y_interventional(scm, np.array([-3.0, 0.0, 4.0]), np.zeros(3)) == pytest.approx(0.5)
    # Observationally x1 = 2 is evidence of a high Z, so P(Y=1 | x1, x2) is above 1/2.
    assert ci.p_y_observational(scm, np.array([2.0]), np.array([0.0]))[0] > 0.7
    # Posterior mean by hand: s = 1 - 0.49 = 0.51, precision = 1 + 2 * 0.49 / 0.51.
    prec = 1 + 2 * 0.49 / 0.51
    mean = (0.7 * 2.0 / 0.51) / prec
    nodes = mean + np.sqrt(1 / prec) * np.random.default_rng(0).normal(size=400000)
    mc = (1 / (1 + np.exp(-1.5 * nodes))).mean()
    assert ci.p_y_observational(scm, np.array([2.0]), np.array([0.0]))[0] == pytest.approx(
        mc, abs=0.005
    )


def test_closed_forms_match_simulation():
    scm = ci.SCM(a=0.8, b=0.6, w=1.0, g=1.0, c=0.0)
    X, y, z = ci.sample_observational(scm, 400000, seed=0)
    near = (np.abs(X[:, 0] - 0.5) < 0.05) & (np.abs(X[:, 1] - 0.5) < 0.05)
    assert near.sum() > 300
    p = ci.p_y_observational(scm, np.array([0.5]), np.array([0.5]))[0]
    assert y[near].mean() == pytest.approx(p, abs=0.06)
    # Interventional: set x1 = 0.5 for rows with x2 near 0.5 and redraw y.
    sel = np.abs(X[:, 1] - 0.5) < 0.05
    pdo = ci.p_y_interventional(scm, np.array([0.5]), np.array([0.5]))[0]
    rng = np.random.default_rng(1)
    ydo = rng.uniform(size=sel.sum()) < 1 / (1 + np.exp(-(1.0 * 0.5 + 1.0 * z[sel])))
    assert ydo.mean() == pytest.approx(pdo, abs=0.02)


def test_observational_and_interventional_differ_only_when_confounded():
    x1, x2 = np.linspace(-2, 2, 9), np.linspace(2, -2, 9)
    null = ci.SCM(g=0.0)
    np.testing.assert_allclose(
        ci.p_y_observational(null, x1, x2), ci.p_y_interventional(null, x1, x2), atol=1e-9
    )
    conf = ci.SCM(g=1.5)
    assert (
        np.abs(ci.p_y_observational(conf, x1, x2) - ci.p_y_interventional(conf, x1, x2)).max() > 0.1
    )


def test_intervention_changes_only_the_targeted_mechanism():
    scm = ci.SCM(g=1.0)
    X, _, z = ci.sample_observational(scm, 5000, seed=0)
    Xd, yd = ci.sample_interventional(scm, X, z, seed=3)
    np.testing.assert_array_equal(Xd[:, 1], X[:, 1])  # X2 untouched
    np.testing.assert_array_equal(np.sort(Xd[:, 0]), np.sort(X[:, 0]))  # marginal kept
    assert abs(np.corrcoef(Xd[:, 0], z)[0, 1]) < 0.05  # Z->X1 cut
    assert abs(np.corrcoef(Xd[:, 0], X[:, 1])[0, 1]) < 0.05
    # The outcome mechanism is unchanged: do-labels follow the closed-form interventional rate.
    p = ci.p_y_interventional(scm, Xd[:, 0].astype(float), Xd[:, 1].astype(float))
    assert yd.mean() == pytest.approx(p.mean(), abs=0.03)


def test_deterministic_and_validated():
    scm = ci.SCM()
    a, b = ci.sample_observational(scm, 300, 5), ci.sample_observational(scm, 300, 5)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)
    assert not np.array_equal(a[0], ci.sample_observational(scm, 300, 6)[0])
    with pytest.raises(ValueError):
        ci.SCM(a=1.0)


def test_classical_arms_and_oracles_end_to_end(tmp_path):
    rec = ci.run({}, tmp_path, gs=(0.0, 2.0), seeds=(0,), n=4000, models={})
    by = {(c["arm"], c["g"]): c for c in rec["results"]}
    # The oracle rows are exact against their own truth and reach the Bayes AUC ordering.
    assert by[("oracle_do", 2.0)]["mae_do"] == pytest.approx(0.0, abs=1e-9)
    assert by[("oracle_obs", 2.0)]["leans_to"] == "observational"
    assert by[("oracle_do", 2.0)]["leans_to"] == "interventional"
    # Under confounding the truths separate; a learner fitted on observational rows is nearer
    # the associational one.
    assert by[("oracle_do", 2.0)]["mae_obs"] > 0.02
    assert by[("logistic_regression", 2.0)]["mae_obs"] < by[("logistic_regression", 2.0)]["mae_do"]
    assert json.loads((tmp_path / "causal_interventional.json").read_text())["config"]["gs"] == [
        0.0,
        2.0,
    ]


def test_tiny_end_to_end(tmp_path):
    rec = ci.run(
        {}, tmp_path, gs=(0.0, 1.0), seeds=(0,), n=200, max_context=100, models={"tiny": _tiny()}
    )
    arms = {c["arm"] for c in rec["results"]}
    assert arms == {
        "tiny",
        "untrained_control",
        "logistic_regression",
        "gradient_boosting",
        "oracle_do",
        "oracle_obs",
    }
    assert len(rec["results"]) == 6 * 2
    for c in rec["results"]:
        assert 0.0 <= c["auc_obs"] <= 1.0 and 0.0 <= c["auc_do"] <= 1.0
    text = ci.summarise(rec)
    assert "[g=0]" in text and "untrained_control" in text and "leans to" in text
