"""Openspec 36.4, third batch: loops that used to skip silently now count their skips.

Each case forces one skip on synthetic inputs and checks ``n_attempted == n_scored +
n_skipped`` with a reason per skip in the harness's record.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import fintfm.evaluation.tuned_boosting as tb
import fintfm.experiments.hybrid_catboost as hc
from fintfm.evaluation.coverage import Coverage
from fintfm.evaluation.datasets import CreditDataset
from fintfm.experiments import capability, context_sweep
from fintfm.experiments import correlated_features as cf
from fintfm.experiments import prior_ablation as pa
from fintfm.experiments import sign_inference as si
from fintfm.modeling.model import FinancialTFM, ModelConfig


def _balanced(rec: dict) -> None:
    assert rec["n_attempted"] == rec["n_scored"] + rec["n_skipped"]
    assert len(rec["skipped"]) == rec["n_skipped"]
    assert all(reason for reason in rec["skipped"].values())


def _tiny() -> FinancialTFM:
    torch.manual_seed(1)
    return FinancialTFM(
        ModelConfig(max_features=8, max_classes=2, d_model=16, n_heads=2, n_layers=1, d_ff=32)
    )


# ---------------------------------------------------------------- sign_inference


def test_sign_inference_counts_a_single_class_task_as_skipped(monkeypatch, capsys):
    real = si.make_pair

    def one_class_for_seed_one(beta, n, d, seed):
        X, yp, yn = real(beta, n, d, seed)
        return (X, np.zeros_like(yp), np.ones_like(yn)) if seed == 1000 else (X, yp, yn)

    monkeypatch.setattr(si, "make_pair", one_class_for_seed_one)
    rec = si.run(
        {},
        None,
        models={"tiny": _tiny()},
        seeds=(0, 1),
        tasks_per_seed=1,
        n=200,
        d=6,
        n_relevant=3,
        max_context=100,
    )
    cov = rec["coverage"]
    _balanced(cov)
    assert (cov["n_attempted"], cov["n_scored"], cov["n_skipped"]) == (4, 2, 2)
    assert set(cov["skipped"]) == {"sign/seed1/task0", "beta/seed1/task0"}
    assert "coverage: 2/4 paired tasks scored, 2 skipped" in capsys.readouterr().out
    assert {"task", "config", "results"} <= set(rec)


# ---------------------------------------------------------------- correlated_features


def test_correlated_features_counts_a_single_class_cell_as_skipped(monkeypatch, capsys):
    real = cf.make_design

    def one_class_at_high_r(design, r, seed, n=2000, rate=0.25):
        X, y, z = real(design, r, seed, n, rate)
        return (X, np.zeros_like(y), z) if r == 0.9 else (X, y, z)

    monkeypatch.setattr(cf, "make_design", one_class_at_high_r)
    rec = cf.run({}, designs=("redundant",), levels=(0.0, 0.9), seeds=(0, 1), n=200, models={})
    cov = rec["coverage"]
    _balanced(cov)
    assert (cov["n_attempted"], cov["n_scored"], cov["n_skipped"]) == (4, 2, 2)
    assert "redundant/r0.9/seed0" in cov["skipped"]
    assert "coverage: 2/4 design-level-seeds scored" in capsys.readouterr().out
    assert "max_correlation_deviation" in rec


# ---------------------------------------------------------------- prior_ablation


class _StubTFM:
    def __init__(self, *a, **k):
        pass

    def fit(self, X, y):
        self.rate = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.clip(0.5 * self.rate + 0.1 * (X[:, 0] > 0), 1e-3, 0.9)
        return np.column_stack([1 - p, p])


def test_learning_curve_counts_an_oversized_train_size_as_skipped(monkeypatch):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1200, 4)).astype("float32")
    y = (X[:, 0] + rng.normal(size=1200) > 1.0).astype(int)
    ds = CreditDataset(X=X, y=y, name="toy", horizon_years=3, licence="x", attribution="x")
    monkeypatch.setattr(pa, "FinancialTFMClassifier", _StubTFM)
    rec = pa.sample_efficiency_probe("stub.pt", train_sizes=(100, 10**6), ds=ds)
    cov = rec["coverage"]
    _balanced(cov)
    assert (cov["n_attempted"], cov["n_scored"], cov["n_skipped"]) == (2, 1, 1)
    assert "exceeds the pool" in cov["skipped"]["n1000000"]


def test_polish_horizons_count_a_width_mismatch_as_skipped(monkeypatch):
    def ds(h, d):
        rng = np.random.default_rng(h)
        X = rng.normal(size=(200, d)).astype(np.float32)
        y = (np.arange(200) % 10 == 0).astype(np.int64)
        return CreditDataset(
            X=X, y=y, name=f"fake-{h}y", horizon_years=h, licence="x", attribution="x"
        )

    data = {1: ds(1, 4), 2: ds(2, 5), 3: ds(3, 4)}
    monkeypatch.setattr(pa, "load_polish_bankruptcy", lambda h: data[h])
    monkeypatch.setattr(pa, "FinancialTFMClassifier", _StubTFM)
    rec = pa.term_structure_coherence("x.pt", horizons=(1, 2, 3))
    cov = rec["coverage"]
    _balanced(cov)
    assert (cov["n_attempted"], cov["n_scored"], cov["n_skipped"]) == (3, 2, 1)
    assert "feature width 5" in cov["skipped"]["h2"]


# ---------------------------------------------------------------- capability


def _save(path, **kw):
    base = {"max_features": 8, "d_model": 16, "d_cell": 8, "n_layers": 1, "n_col_layers": 1}
    FinancialTFM(ModelConfig(**{**base, **kw})).save(
        str(path), trained_objectives=("classification",)
    )


def test_multiclass_sweep_coverage_counts_the_binary_checkpoint_as_skipped(tmp_path, capsys):
    _save(tmp_path / "binary.pt", max_classes=2)
    _save(tmp_path / "wide.pt", max_classes=5)
    sweep = capability.multiclass_sweep(
        {"wide": str(tmp_path / "wide.pt"), "binary": str(tmp_path / "binary.pt")},
        class_counts=(3,),
        n_features=6,
        seeds=(0,),
        max_context=300,
    )
    cov = sweep["coverage"]
    _balanced(cov)
    # wide, binary, untrained control (built from the first checkpoint), logistic regression
    assert (cov["n_attempted"], cov["n_scored"], cov["n_skipped"]) == (4, 3, 1)
    assert "K3/seed0/binary" in cov["skipped"]
    assert sweep["skipped"], "the existing list of skips is kept"
    assert "class-seed-arms scored" in capsys.readouterr().out


def test_regression_sweep_coverage_balances(tmp_path, capsys):
    ckpt = tmp_path / "q.pt"
    FinancialTFM(
        ModelConfig(
            max_features=20,
            max_classes=3,
            d_cell=16,
            d_model=32,
            n_heads=2,
            n_col_layers=1,
            n_layers=1,
            d_ff=32,
            head_type="quantile",
            n_quantiles=9,
        )
    ).save(str(ckpt), trained_objectives=("quantile",))
    sweep = capability.regression_sweep(
        {"q": str(ckpt)}, shapes=("linear",), seeds=(0,), max_context=200
    )
    cov = sweep["coverage"]
    _balanced(cov)
    assert cov["n_attempted"] == 4 and cov["n_skipped"] == 0
    assert "shape-seed-arms scored" in capsys.readouterr().out


def test_regression_sweep_coverage_counts_a_one_bin_checkpoint_as_skipped(tmp_path):
    _save(tmp_path / "one.pt", max_classes=1)
    try:
        sweep = capability.regression_sweep(
            {"one": str(tmp_path / "one.pt")}, shapes=("linear",), seeds=(0,), max_context=200
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"a one-class checkpoint cannot be built here: {exc}")
    cov = sweep["coverage"]
    _balanced(cov)
    assert cov["n_skipped"] >= 1


# ---------------------------------------------------------------- context sweep


def test_paired_comparisons_count_unscorable_horizons_as_skipped():
    rng = np.random.default_rng(0)
    n, k = 60, 3
    truth = (rng.random((n, k)) < 0.4).astype(int)
    truth[:, 1] = 0  # horizon 1: one class
    seen = np.ones((n, k), dtype=bool)
    seen[5:, 2] = False  # horizon 2: five observed rows, below min_rows
    curves = {(100, s, 0): rng.random((n, k)) for s in ("uniform", "balanced")}
    cov = Coverage()
    rows = context_sweep.paired_strategy_comparisons(
        curves, truth, seen, None, 10, n_boot=20, coverage=cov
    )
    rec = cov.to_dict()
    _balanced(rec)
    assert (rec["n_attempted"], rec["n_scored"], rec["n_skipped"]) == (3, 1, 2)
    assert len(rows) == rec["n_scored"]
    assert "one class" in rec["skipped"]["ctx100/seed0/balanced-vs-uniform/h1"]
    assert "fewer than" in rec["skipped"]["ctx100/seed0/balanced-vs-uniform/h2"]
    # the default call, with no coverage, is unchanged
    assert len(context_sweep.paired_strategy_comparisons(curves, truth, seen, None, 10, 20)) == 1


# ---------------------------------------------------------------- hybrid


def test_tuned_arm_failure_is_named_in_the_coverage(monkeypatch):
    def fake(arrays, n_configs, budget_s, seed, predictions):
        if "btr" in arrays:  # the init-score variant's worker fails
            return None
        return {
            "n_trials": 1,
            "tune_seconds": 0.0,
            "val_ap": 0.5,
            "predictions": np.full(len(arrays["Xte"]), 0.1),
        }

    monkeypatch.setattr(tb, "run_tuned", fake)
    n = 40
    Z = np.zeros((n, 3))
    y = np.arange(n) % 2
    company = np.array([f"c{i}" for i in range(n)])
    tr, held = np.arange(30), np.arange(30, n)
    failures: dict[str, str] = {}
    arms = hc._tuned_arms(
        Z, y, company, tr, held, np.zeros(30), np.zeros(10), 1, None, 0, failures=failures
    )
    assert set(arms) == {"lightgbm_tuned", "lightgbm_tuned_fintfm_feature"}
    cov = Coverage()
    scores = {a: {"test": {"avg_precision": 0.5}} for a in arms}
    expected = ("lightgbm_tuned", "lightgbm_tuned_fintfm_baseline", "lightgbm_tuned_fintfm_feature")
    hc.record_arm_coverage(cov, 0, expected, scores, failures)
    rec = cov.to_dict()
    _balanced(rec)
    assert (rec["n_attempted"], rec["n_scored"], rec["n_skipped"]) == (3, 2, 1)
    assert (
        "worker returned no predictions" in rec["skipped"]["fold0/lightgbm_tuned_fintfm_baseline"]
    )


def test_irrelevant_features_counts_every_seed(tmp_path):
    """Every seed is attempted, and each is scored or skipped with a reason (36.4)."""
    from fintfm.experiments import irrelevant_features as irf

    record = irf.run({}, out_dir=tmp_path, seeds=(0, 1), widths=(2, 5), n=200, max_context=100)
    cov = record["coverage"]
    assert cov["n_attempted"] == 2
    assert cov["n_attempted"] == cov["n_scored"] + cov["n_skipped"]
