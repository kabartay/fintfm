"""Openspec 36.4, second batch: term structure, representation transfer, context sweep, hybrid.

Each harness is driven on synthetic inputs, with the expensive scorer replaced where needed,
and must report ``n_attempted == n_scored + n_skipped`` with a reason per skip in its JSON.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch

import fintfm.experiments.hybrid_catboost as hc
import fintfm.experiments.representation_transfer as rt
from fintfm.config import load_config
from fintfm.evaluation.coverage import Coverage
from fintfm.experiments import context_sweep, term_structure
from fintfm.modeling.model import FinancialTFM, ModelConfig


def _balanced(rec: dict) -> None:
    assert rec["n_attempted"] == rec["n_scored"] + rec["n_skipped"]
    assert len(rec["skipped"]) == rec["n_skipped"]
    assert all(reason for reason in rec["skipped"].values())


# ---------------------------------------------------------------- term structure


def test_term_structure_counts_a_one_class_horizon_as_skipped():
    rng = np.random.default_rng(0)
    n, k = 200, 3
    T = np.zeros((n, k), dtype=int)
    T[:, 0] = rng.random(n) < 0.3
    T[:, 1] = rng.random(n) < 0.4
    # horizon 2: nobody defaulted in the held-out tasks, so AUC is undefined
    C = rng.random((n, k))
    cov = Coverage()
    term_structure.summarise_curves("joint", C, T, 1, 10, cov)
    term_structure.summarise_curves("other", C, T, 1, 10, cov)
    rec = cov.to_dict()
    _balanced(rec)
    assert (rec["n_attempted"], rec["n_scored"], rec["n_skipped"]) == (6, 4, 2)
    assert set(rec["skipped"]) == {"joint/h2", "other/h2"}


def test_term_structure_run_writes_coverage(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(1)
    n, k = 120, 3
    T = (rng.random((n, k)) < 0.3).astype(int)
    T[:, 2] = 0
    monkeypatch.setattr(term_structure, "train_joint", lambda *a, **kw: "joint")
    monkeypatch.setattr(term_structure, "train_per_horizon", lambda *a, **kw: ["m"] * k)
    monkeypatch.setattr(term_structure, "_eval_batches", lambda *a, **kw: [])
    monkeypatch.setattr(
        term_structure,
        "collect_curves",
        lambda models, batches, device, n_horizons: (
            np.sort(rng.random((n, k)), axis=1),
            T,
            np.arange(n) // 10,
        ),
    )
    rec = term_structure.run(tmp_path, steps=3, n_horizons=k, n_boot=0)
    cov = rec["coverage"]
    _balanced(cov)
    assert (cov["n_attempted"], cov["n_scored"], cov["n_skipped"]) == (9, 6, 3)
    assert json.loads((tmp_path / "term_structure.json").read_text())["coverage"] == cov
    assert "coverage: 6/9 arm-horizons scored, 3 skipped" in capsys.readouterr().out


# ---------------------------------------------------------------- representation transfer


def test_representation_transfer_fold_coverage_counts_a_gap_as_skipped():
    row = {"own_head_auc": 0.8, "representation_auc": float("nan"), "raw_features_auc": 0.7}
    cov = Coverage()
    rt.record_fold_coverage(cov, 3, row, ["own_head", "representation", "raw_features"])
    rec = cov.to_dict()
    _balanced(rec)
    assert (rec["n_attempted"], rec["n_scored"], rec["n_skipped"]) == (3, 2, 1)
    assert "one class" in rec["skipped"]["fold3/representation"]


def _rt_panel(n: int = 1500):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = (X[:, 0] + rng.normal(size=n) > 1.9).astype(np.int64)
    country = np.where(np.arange(n) % 2 == 0, "CZ", "PL")
    company = np.array([f"c{i % 300}" for i in range(n)])
    return X, y, country, company, [f"f{j}" for j in range(4)]


def _tiny_model() -> FinancialTFM:
    torch.manual_seed(0)
    return FinancialTFM(
        ModelConfig(max_features=6, max_classes=2, d_model=16, n_heads=2, n_layers=1, d_ff=32)
    )


@pytest.mark.parametrize("gap", [False, True])
def test_representation_transfer_run_writes_coverage(tmp_path, monkeypatch, capsys, gap):
    data = _rt_panel()
    monkeypatch.setattr(rt, "_load_horizon", lambda root, horizon, max_rows: data)
    if gap:  # a probe set of one class: every downstream arm has no score
        monkeypatch.setattr(
            rt, "_downstream_arms", lambda clf, X, y, Xt, stage, seed: rt._gap(len(Xt))
        )
    rec = rt.run(
        _tiny_model(),
        tmp_path / "rt.json",
        folds=(0,),
        root=tmp_path,
        train_sample_size=400,
        max_context=120,
    )
    cov = rec["coverage"]
    _balanced(cov)
    assert cov["n_attempted"] == 4
    assert cov["n_skipped"] == (3 if gap else 0)
    assert json.loads((tmp_path / "rt.json").read_text())["coverage"] == cov
    assert f"coverage: {4 - cov['n_skipped']}/4 fold-arms scored" in capsys.readouterr().out


# ---------------------------------------------------------------- context sweep


def test_horizon_skip_reason_names_each_cause():
    seen = np.array([True] * 6 + [False] * 4)
    mixed = np.array([0, 1] * 5)
    assert context_sweep.horizon_skip_reason(seen, mixed, 5) is None
    assert "fewer than the 8" in context_sweep.horizon_skip_reason(seen, mixed, 8)
    assert "one class" in context_sweep.horizon_skip_reason(seen, np.zeros(10, dtype=int), 5)


def test_context_sweep_run_counts_unobserved_horizons_as_skipped(monkeypatch, tmp_path, capsys):
    from test_v4_split_and_sweep import N_H, _ds, _FakeClf

    ds = _ds(scale=10)
    monkeypatch.setattr(context_sweep, "load_v4finbench", lambda max_rows=None: ds)
    monkeypatch.setattr(
        context_sweep.FinancialTFMClassifier,
        "from_config",
        classmethod(lambda cls, *a, **k: _FakeClf(0.25)),
    )
    base = load_config()
    cfg = dataclasses.replace(
        base,
        context_sweep=dataclasses.replace(
            base.context_sweep, context_sizes=(100, 200), strategies=("uniform",)
        ),
        evaluation=dataclasses.replace(base.evaluation, min_rows_per_horizon=10),
    )
    rec = context_sweep.run(
        "m.pt", tmp_path, train_until=2008, test_from=2017, seeds=(0,), cfg=cfg, n_boot=0
    )
    cov = rec["coverage"]
    _balanced(cov)
    assert cov["n_attempted"] == 2 * N_H  # two context sizes x six horizons
    assert cov["n_scored"] == sum(c["n_horizons_scored"] for c in rec["cells"])
    assert cov["n_skipped"] > 0, "test years 2017-2020 leave the later horizons unobserved"
    assert "ctx100/uniform/seed0/h5" in cov["skipped"]
    assert json.loads((tmp_path / "context_sweep.json").read_text())["coverage"] == cov
    assert "cell-horizons scored" in capsys.readouterr().out


# ---------------------------------------------------------------- hybrid


def _score(ap: float) -> dict:
    return {"test": {"avg_precision": ap}, "val": {"avg_precision": ap}}


def test_hybrid_arm_coverage_separates_a_failed_fit_from_a_one_class_split():
    cov = Coverage()
    scores = {
        "fintfm": _score(0.3),
        "catboost": _score(float("nan")),
        "catboost_fintfm_baseline": _score(0.4),
    }
    hc.record_arm_coverage(cov, 2, hc.ARMS, scores)
    rec = cov.to_dict()
    _balanced(rec)
    assert (rec["n_attempted"], rec["n_scored"], rec["n_skipped"]) == (4, 2, 2)
    assert "one class" in rec["skipped"]["fold2/catboost"]
    assert "fit failed" in rec["skipped"]["fold2/catboost_fintfm_feature"]


def test_hybrid_run_arrays_writes_coverage_when_a_booster_fails(tmp_path, monkeypatch, capsys):
    from test_hybrid_catboost import _cfg, _model, _panel

    X, y, country, company = _panel()
    calls = {"n": 0}

    def flaky(X_tr, y_tr, X_te, params=None, baseline_train=None, baseline_test=None):
        calls["n"] += 1
        if calls["n"] == 2:  # the second booster fit, the baseline hybrid, fails
            return None
        return np.full(len(X_te), float(y_tr.mean()))

    monkeypatch.setattr(hc, "fit_predict_catboost", flaky)
    rec = hc.run_arrays(
        X,
        y,
        country,
        company,
        _model(),
        tmp_path,
        folds=(0,),
        cfg=_cfg(),
        n_inner=3,
        max_train_rows=200,
        protocol={"horizon": 0, "max_rows": None},
    )
    cov = rec["coverage"]
    _balanced(cov)
    assert cov["n_attempted"] == len(hc.ARMS)
    assert "fold0/catboost_fintfm_baseline" in cov["skipped"]
    assert json.loads((tmp_path / "hybrid_catboost.json").read_text())["coverage"] == cov
    assert "fold-arms scored" in capsys.readouterr().out
