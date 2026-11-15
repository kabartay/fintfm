"""Tests for the posterior-task-uncertainty harness (task 39.19). SMOKE-TEST only."""

import itertools
import json

import numpy as np
import pytest
import torch

from fintfm.experiments import task_posterior as tp
from fintfm.modeling.model import FinancialTFM, ModelConfig


def _tiny() -> FinancialTFM:
    torch.manual_seed(1)
    return FinancialTFM(
        ModelConfig(max_features=8, max_classes=2, d_model=16, n_heads=2, n_layers=1, d_ff=32)
    )


def test_posterior_hand_examples():
    # rows: (+,+) label 1 and (-,-) label 0 fit both; (+,-) label 1 fits only h0
    X = np.array([[1.0, 2.0], [-1.0, -3.0]])
    np.testing.assert_allclose(tp.bayes_posterior(X, np.array([1, 0]), 2), [0.5, 0.5])
    X2 = np.vstack([X, [[1.0, -1.0]]])
    np.testing.assert_allclose(tp.bayes_posterior(X2, np.array([1, 0, 1]), 2), [1.0, 0.0])
    np.testing.assert_allclose(tp.bayes_posterior(X2, np.array([1, 0, 0]), 2), [0.0, 1.0])
    # three hypotheses, one mixed row eliminating one of them
    X3 = np.array([[1.0, 1.0, -1.0]])
    np.testing.assert_allclose(tp.bayes_posterior(X3, np.array([1]), 3), [0.5, 0.5, 0.0])
    with pytest.raises(ValueError):
        tp.bayes_posterior(X2, np.array([0, 1, 0]), 2)  # no hypothesis explains it
    with pytest.raises(ValueError):
        tp.bayes_posterior(X, np.array([1, 0]), 2, eps=0.7)


def test_posterior_with_noise_concentrates_as_rows_are_added():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))
    X[:, 1] = -X[:, 0]  # hypotheses always disagree
    y = (X[:, 0] > 0).astype(int)  # h0 is true
    ent = [1.0]
    for n in (1, 3, 10, 40):
        ent.append(tp._dist_entropy_bits(tp.bayes_posterior(X[:n], y[:n], 2, eps=0.1)))
    assert all(a > b for a, b in itertools.pairwise(ent))
    # exact value for one row: (0.9, 0.1)
    np.testing.assert_allclose(tp.bayes_posterior(X[:1], y[:1], 2, eps=0.1), [0.9, 0.1])


def test_predictive_and_entropy():
    post = np.array([0.5, 0.5])
    Xq = np.array([[1.0, -1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(tp.bayes_predictive(post, Xq), [0.5, 0.5])
    np.testing.assert_allclose(tp.bayes_predictive(np.array([1.0, 0.0]), Xq), [1.0, 0.0])
    np.testing.assert_allclose(tp.entropy_bits(np.array([0.0, 0.5, 1.0])), [0.0, 1.0, 0.0])
    np.testing.assert_allclose(tp.implied_posterior(np.array([0.5, 0.5]), Xq, 2), [0.5, 0.5])


@pytest.mark.parametrize("k_hyp", [2, 3])
def test_contexts_exactly_consistent(k_hyp):
    ctxs, Xq, yq = tp.make_contexts(4, 1, k_hyp, 60, 40, 6, (0, 1, 5, 60))
    for nd, (X, y) in ctxs.items():
        assert X.shape == (60, 6)
        keep = tp.consistent_hypotheses(X, y, k_hyp)
        assert 1 in keep  # the true hypothesis always fits exactly
        if nd == 0:
            np.testing.assert_array_equal(keep, np.arange(k_hyp))  # fully ambiguous
        else:
            assert len(keep) < k_hyp  # a disambiguating row removed some hypothesis
        lab = tp.hypothesis_labels(X, k_hyp)
        assert (lab.min(axis=0) != lab.max(axis=0)).sum() == nd
    # queries are all discriminating and labelled by the true hypothesis
    lab = tp.hypothesis_labels(Xq, k_hyp)
    assert np.all(lab.min(axis=0) != lab.max(axis=0))
    np.testing.assert_array_equal(yq, lab[1])
    post_h = [tp._dist_entropy_bits(tp.bayes_posterior(*ctxs[nd], k_hyp)) for nd in (0, 1, 5, 60)]
    assert post_h[0] == pytest.approx(np.log2(k_hyp))
    assert post_h[-1] < post_h[0]


def test_deterministic_and_validation():
    a = tp.make_contexts(3, 0, 2, 30, 20, 5, (0, 4))
    b = tp.make_contexts(3, 0, 2, 30, 20, 5, (0, 4))
    for nd in (0, 4):
        np.testing.assert_array_equal(a[0][nd][0], b[0][nd][0])
    np.testing.assert_array_equal(a[1], b[1])
    assert all(len(X) == 30 for X, _ in a[0].values())  # context size constant along the curve
    with pytest.raises(ValueError):
        tp.make_contexts(0, 2, 2, 10, 10, 4)
    with pytest.raises(ValueError):
        tp.make_contexts(0, 0, 1, 10, 10, 4)


def test_end_to_end_untrained_and_deterministic(tmp_path):
    kw = {
        "seeds": (0,),
        "k_hyp": 2,
        "n_ctx": 60,
        "n_query": 40,
        "d": 6,
        "n_dis_grid": (0, 2, 16),
        "max_context": 60,
    }
    m = _tiny()
    rec = tp.run({}, tmp_path, models={"tiny": m}, **kw)
    arms = {r["arm"] for r in rec["results"]}
    assert arms == {"tiny", "untrained_control", "logistic_regression", "bayes"}
    bayes = {r["n_dis"]: r for r in rec["results"] if r["arm"] == "bayes"}
    assert bayes[0]["entropy"] == pytest.approx(1.0)
    assert bayes[0]["posterior_entropy"] == pytest.approx(1.0)
    assert bayes[2]["entropy"] == pytest.approx(0.0) and bayes[2]["accuracy"] == 1.0
    assert rec["comparison"]["bayes"]["hedging_gain"] == pytest.approx(1.0)
    on_disk = json.loads((tmp_path / "task_posterior.json").read_text())
    assert on_disk["results"] == rec["results"]
    assert "Reading it" in tp.summarise(rec)
    rec2 = tp.run({}, None, models={"tiny": m}, **kw)
    assert rec2["results"] == rec["results"]
    with pytest.raises(ValueError):
        tp.run({}, None, models={"tiny": m}, **{**kw, "n_dis_grid": (1, 2)})
