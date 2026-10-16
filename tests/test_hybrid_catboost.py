"""The AU4.4 hybrid harness on a tiny synthetic panel: no real data, no scoring.

What must hold before any number from it is believed: no outer validation or test row, nor
any row of their companies, ever enters an inner FinTFM context; every training row gets
exactly one out-of-fold logit; CatBoost's baseline really is the starting score; and the
output pairs with ``scripts/compare_v4_scores.py`` the way a ``fintfm-v4protocol`` run does.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from fintfm.config import load_config
from fintfm.experiments.hybrid_catboost import (
    LeakageError,
    assert_no_leak,
    cap_train_rows,
    fit_predict_catboost,
    inner_oof_logits,
    run_arrays,
    to_logit,
)
from fintfm.experiments.v4_protocol import (
    build_fold_assignments,
    fintfm_classifier,
    split_indices_for_fold,
)
from fintfm.modeling.model import FinancialTFM, ModelConfig

needs_catboost = pytest.mark.skipif(
    importlib.util.find_spec("catboost") is None, reason="catboost not installed (bench extra)"
)

#: Small and quick: these tests check plumbing, never a score.
_CATBOOST = {"iterations": 15, "depth": 2, "thread_count": 1, "random_seed": 0}


def _model() -> FinancialTFM:
    torch.manual_seed(0)
    cfg = ModelConfig(max_features=6, max_classes=2, d_model=16, n_heads=2, n_layers=1, d_ff=32)
    return FinancialTFM(cfg)


def _panel(n_companies=30, n_countries=3, seed=0):
    """Companies with 2-5 yearly rows each, a company-level risk, and ~20% positives."""
    rng = np.random.RandomState(seed)
    country, company, rows = [], [], []
    for c in range(n_countries):
        for k in range(n_companies):
            reps = rng.randint(2, 6)
            risk = rng.normal()
            country += [f"C{c}"] * reps
            company += [f"c{c}_f{k}"] * reps
            rows += [risk] * reps
    risk = np.array(rows)
    X = np.column_stack([risk + rng.normal(scale=0.5, size=len(risk)) for _ in range(4)])
    X[rng.rand(*X.shape) < 0.05] = np.nan  # the protocol imputes; so must this
    y = (risk + rng.normal(scale=0.5, size=len(risk)) > 0.9).astype(np.int64)
    return X.astype(np.float32), y, np.array(country, dtype=object), np.array(company, dtype=object)


def _cfg():
    cfg = load_config()
    # a small context keeps the forward passes cheap; nothing else differs from the default
    from dataclasses import replace

    return replace(cfg, inference=replace(cfg.inference, max_context=60))


def test_to_logit_is_finite_at_the_extremes():
    z = to_logit(np.array([0.0, 0.5, 1.0]))
    assert np.isfinite(z).all()
    assert z[1] == pytest.approx(0.0)
    assert z[0] < -10 and z[2] > 10


def test_assert_no_leak_catches_a_shared_row_and_a_shared_company():
    company = np.array(["a", "a", "b", "c"], dtype=object)
    assert_no_leak(np.array([2]), np.array([3]), company, "clean")
    with pytest.raises(LeakageError, match="row"):
        assert_no_leak(np.array([0, 2]), np.array([2, 3]), company, "row")
    with pytest.raises(LeakageError, match="company"):
        assert_no_leak(np.array([0]), np.array([1]), company, "company")


def test_inner_contexts_never_hold_an_outer_val_or_test_row():
    X, y, country, company = _panel()
    tr, va, te = split_indices_for_fold(build_fold_assignments(country, company), 0)
    held = np.concatenate([va, te])
    cfg = _cfg()
    logits, plan = inner_oof_logits(
        np.nan_to_num(X), y, tr, country, company, held,
        lambda k: fintfm_classifier(_model(), cfg, random_state=k), n_inner=3, seed=7,
    )
    assert logits.shape == (len(tr),) and np.isfinite(logits).all()
    assert len(plan) == 3
    queried = np.concatenate([q for _, q in plan])
    # every training row is scored exactly once, by a fold that did not condition on it
    np.testing.assert_array_equal(np.sort(queried), np.sort(tr))
    for ctx, qry in plan:
        assert np.isin(ctx, tr).all()
        assert not np.isin(ctx, held).any()
        assert not np.isin(company[ctx], company[held]).any()
        assert not np.isin(company[ctx], company[qry]).any()


def test_a_leaking_split_is_refused_before_any_fit():
    X, y, country, company = _panel()
    tr, va, te = split_indices_for_fold(build_fold_assignments(country, company), 0)

    def must_not_fit(k):
        raise AssertionError("an inner fit ran before the leak check")

    # outer test rows wrongly included in the training rows handed to the inner folds
    with pytest.raises(LeakageError, match="outer val/test"):
        inner_oof_logits(X, y, np.concatenate([tr, te]), country, company,
                         np.concatenate([va, te]), must_not_fit, n_inner=3, seed=7)


def test_cap_keeps_whole_companies_within_the_cap():
    _, _, country, company = _panel()
    tr, _, _ = split_indices_for_fold(build_fold_assignments(country, company), 1)
    capped = cap_train_rows(tr, company, max_rows=60, seed=1)
    assert 0 < len(capped) <= 60
    assert np.isin(capped, tr).all()
    for name in np.unique(company[capped]):
        assert np.isin(np.flatnonzero(company == name), capped).all()  # whole, never partial
    np.testing.assert_array_equal(capped, cap_train_rows(tr, company, 60, seed=1))
    assert cap_train_rows(tr, company, max_rows=10**6, seed=1) is tr


@needs_catboost
def test_catboost_baseline_is_the_starting_score():
    """With one tiny tree on top, the prediction is the baseline plus almost nothing."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3)).astype(np.float32)
    y = (rng.random(200) < 0.3).astype(np.int64)
    b_tr, b_te = rng.normal(scale=3, size=200), rng.normal(scale=3, size=50)
    p = fit_predict_catboost(
        X, y, X[:50], {"iterations": 1, "learning_rate": 1e-4, "thread_count": 1},
        baseline_train=b_tr, baseline_test=b_te,
    )
    np.testing.assert_allclose(p, 1 / (1 + np.exp(-b_te)), atol=1e-3)
    with pytest.raises(ValueError, match="together"):
        fit_predict_catboost(X, y, X, baseline_train=b_tr)


@needs_catboost
def test_run_writes_the_v4protocol_layout_and_pairs_with_compare(tmp_path):
    X, y, country, company = _panel()
    out = tmp_path / "hybrid"
    record = run_arrays(
        X, y, country, company, _model(), out, folds=(0,), cfg=_cfg(), n_inner=3,
        max_train_rows=200, catboost_params=_CATBOOST,
        protocol={"horizon": 0, "max_rows": None},
    )
    fold = record["folds"][0]
    assert fold["n_train"] <= 200 and fold["inner_contexts_checked"] == 3
    assert fold["fintfm_query_rows"] == fold["n_train"] + fold["n_val"] + fold["n_test"]
    d = np.load(out / "predictions_fold0.npz")
    for key in ("y_true", "y_val", "company_test", "company_val", "pred_fintfm", "val_fintfm",
                "pred_catboost", "pred_catboost_fintfm_baseline", "pred_catboost_fintfm_feature"):
        assert key in d.files, key
    assert d["pred_catboost"].shape == d["y_true"].shape
    for a in ("catboost", "catboost_fintfm_baseline", "catboost_fintfm_feature"):
        arm = np.load(out / "arms" / a / "predictions_fold0.npz")
        np.testing.assert_array_equal(arm["pred_fintfm"], d[f"pred_{a}"])
        np.testing.assert_array_equal(arm["company_test"], d["company_test"])
        assert ((arm["pred_fintfm"] >= 0) & (arm["pred_fintfm"] <= 1)).all()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from compare_v4_scores import compare

    rows = compare(out / "arms" / "catboost", out / "arms" / "catboost_fintfm_baseline",
                   n_boot=50, cluster=True)
    assert [r["fold"] for r in rows] == [0]
    assert np.isfinite(rows[0]["arm"]) and np.isfinite(rows[0]["control"])
