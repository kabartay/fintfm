"""DR8.4's hybrid arms in the credit suite: FinTFM's out-of-fold logit fed to LightGBM.

Synthetic panels only. What must hold before a hybrid number is believed: no inner FinTFM
context holds a test row or a row it scores; every training row gets exactly one logit, from
a context without it; LightGBM's starting-score path in the out-of-process worker equals a
direct LightGBM fit; and the hybrid rows reach the CSV in the suite's own columns.
"""

import csv
import dataclasses
import importlib.util
import json
import subprocess
import sys

import numpy as np
import pytest

from fintfm.evaluation import credit_suite as cs
from fintfm.evaluation.boosting import fit_predict_boosting
from fintfm.experiments.hybrid_catboost import LeakageError, assert_no_leak, to_logit

needs_lightgbm = pytest.mark.skipif(
    importlib.util.find_spec("lightgbm") is None, reason="lightgbm not installed (bench extra)"
)


class _RowIdClassifier:
    """Reads a panel row id from column 0, so a test can see which rows a context held.

    A query whose row was in the context scores 0.999; any other query scores a value that
    encodes its own id, so misaligned logits are caught too.
    """

    def __init__(self, n_rows: int, fits: list[np.ndarray]) -> None:
        self.n_rows, self.fits = n_rows, fits

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> object:
        self.context = X[:, 0].astype(int)
        self.fits.append(self.context)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ids = X[:, 0].astype(int)
        p = np.where(np.isin(ids, self.context), 0.999, 0.1 + 0.8 * ids / self.n_rows)
        return np.column_stack([1 - p, p])


def _rows(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.arange(n, dtype=float), rng.normal(size=(n, 2))])
    return X, (rng.random(n) < 0.2).astype(np.int64)


def test_assert_no_leak_without_groups_checks_rows_only():
    assert_no_leak(np.array([0, 1]), np.array([2, 3]), None, "clean")
    with pytest.raises(LeakageError, match="row"):
        assert_no_leak(np.array([0, 2]), np.array([2, 3]), None, "row")


def test_out_of_fold_logits_never_use_their_own_row():
    X, y = _rows()
    train = np.arange(40, 300)
    test = np.arange(40)
    fits: list[np.ndarray] = []
    logits, plan = cs.oof_fintfm_logits(
        X[train], y[train], train, test, None, lambda k: _RowIdClassifier(300, fits),
        n_inner=3, seed=1,
    )
    # no query saw itself, and each logit is its own row's
    np.testing.assert_allclose(logits, to_logit(0.1 + 0.8 * train / 300))
    assert len(fits) == len(plan) == 3
    queried = np.concatenate([q for _, q in plan])
    np.testing.assert_array_equal(np.sort(queried), train)  # each training row exactly once
    for (ctx, qry), seen in zip(plan, fits, strict=True):
        np.testing.assert_array_equal(np.sort(seen), np.sort(ctx))
        assert not np.isin(ctx, qry).any() and not np.isin(ctx, test).any()
        assert np.isin(ctx, train).all()


def test_inner_folds_keep_borrowers_whole_and_stratify():
    X, y = _rows(n=600)
    groups = np.arange(600) // 3  # three rows per borrower
    train, test = np.arange(600), np.array([], dtype=int)
    logits, plan = cs.oof_fintfm_logits(
        X, y, train, test, groups, lambda k: _RowIdClassifier(600, []), n_inner=3, seed=0,
    )
    assert np.isfinite(logits).all()
    for ctx, qry in plan:
        assert not np.isin(groups[ctx], groups[qry]).any()
        assert abs(y[qry].mean() - y.mean()) < 0.05
    plain = cs.inner_folds(y, None, 3, seed=0)
    assert sorted(np.unique(plain)) == [0, 1, 2]
    for k in range(3):
        assert abs(y[plain == k].mean() - y.mean()) < 0.01


def test_a_constructed_leak_is_refused_before_any_fit():
    X, y = _rows()

    def must_not_fit(k: int) -> _RowIdClassifier:
        raise AssertionError("an inner fit ran before the leak check")

    # test rows wrongly handed over as training rows
    train, test = np.arange(0, 300), np.arange(250, 300)
    with pytest.raises(LeakageError, match="test rows"):
        cs.oof_fintfm_logits(X[train], y[train], train, test, None, must_not_fit, 3, 0)
    # disjoint rows, but a test borrower also has training rows
    groups = np.arange(300) // 2
    train, test = np.arange(0, 251), np.arange(251, 300)  # rows 250 and 251 share a borrower
    with pytest.raises(LeakageError, match="company"):
        cs.oof_fintfm_logits(X[train], y[train], train, test, groups, must_not_fit, 3, 0)


@pytest.mark.filterwarnings("ignore:The least populated class")
def test_a_single_class_inner_context_is_an_error_not_a_score():
    X, _ = _rows(n=60)
    y = np.zeros(60, dtype=np.int64)
    y[0] = 1  # one positive: the context of the fold that scores it holds none
    with pytest.raises(ValueError, match="one class"):
        cs.oof_fintfm_logits(X, y, np.arange(60), np.array([], dtype=int), None,
                             lambda k: _RowIdClassifier(60, []), 3, 0)
    with pytest.raises(ValueError, match="at least 2"):
        cs.oof_fintfm_logits(X, y, np.arange(60), np.array([], dtype=int), None,
                             lambda k: _RowIdClassifier(60, []), 1, 0)


_REFERENCE = '''
import json, sys
import numpy as np
import lightgbm as lgb
d = np.load(sys.argv[1])
params = json.loads(sys.argv[3])
booster = lgb.train(
    {"objective": "binary", "verbosity": -1, "learning_rate": params["learning_rate"]},
    lgb.Dataset(d["Xtr"], d["ytr"], init_score=d["btr"]),
    num_boost_round=params["n_estimators"],
)
raw = booster.predict(d["Xte"], raw_score=True) + d["bte"]
np.save(sys.argv[2], 1.0 / (1.0 + np.exp(-raw)))
'''


@needs_lightgbm
def test_init_score_worker_equals_a_direct_lightgbm_fit(tmp_path):
    """The worker's starting-score path against ``lightgbm.train``, both out of process."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = (rng.random(300) < 1 / (1 + np.exp(-2 * X[:, 0]))).astype(np.int64)
    btr, bte = rng.normal(size=300), rng.normal(size=80)
    params = {"n_estimators": 20, "learning_rate": 0.1}
    p = fit_predict_boosting("lightgbm", X, y, X[:80], params=params,
                             init_score_train=btr, init_score_test=bte)
    assert p is not None
    data, out = tmp_path / "d.npz", tmp_path / "p.npy"
    np.savez(data, Xtr=X, ytr=y, Xte=X[:80], btr=btr, bte=bte)
    # LightGBM never runs in the pytest process, where torch is loaded (FINDINGS §25, §179)
    subprocess.run([sys.executable, "-c", _REFERENCE, str(data), str(out), json.dumps(params)],
                   check=True, capture_output=True, text=True, timeout=300)
    np.testing.assert_allclose(p, np.load(out), rtol=1e-6, atol=1e-9)
    # and the starting score matters: without it the prediction differs
    plain = fit_predict_boosting("lightgbm", X, y, X[:80], params=params)
    assert not np.allclose(p, plain, atol=1e-3)


def test_init_score_arguments_are_checked_before_the_subprocess():
    X, y = np.zeros((4, 2)), np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="together"):
        fit_predict_boosting("lightgbm", X, y, X, init_score_train=np.zeros(4))
    with pytest.raises(ValueError, match="takes no starting score"):
        fit_predict_boosting("xgboost", X, y, X, init_score_train=np.zeros(4),
                             init_score_test=np.zeros(4))
    with pytest.raises(ValueError, match="do not match"):
        fit_predict_boosting("lightgbm", X, y, X, init_score_train=np.zeros(3),
                             init_score_test=np.zeros(4))


def _tiny_checkpoint(tmp_path, max_features: int = 8) -> str:
    import torch

    from fintfm.modeling.model import FinancialTFM, ModelConfig

    torch.manual_seed(0)
    ckpt = tmp_path / "m.pt"
    FinancialTFM(ModelConfig(max_features=max_features, max_classes=2, d_cell=8, d_model=16,
                             n_layers=1, n_col_layers=1, n_heads=2)).save(str(ckpt))
    return str(ckpt)


def _admit(monkeypatch, **loaders):
    admitted = dataclasses.replace(cs.DATASETS["ulb_fraud"].licence, include=True)
    monkeypatch.setattr(cs, "DATASETS", {
        key: cs.SuiteDataset(key, "https://example.invalid", loader, admitted)
        for key, loader in loaders.items()
    })


def _panel(n: int, dated: bool, grouped: bool, seed: int, name: str) -> cs.CreditPanel:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    X[rng.random(X.shape) < 0.05] = np.nan
    y = (rng.random(n) < 1 / (1 + np.exp(-(2.0 * np.nan_to_num(X[:, 0]) - 1.5)))).astype(int)
    return cs.CreditPanel(
        name=name, X=X, feature_names=tuple(f"f{i}" for i in range(5)), y=y.astype(np.int64),
        groups=rng.integers(0, n // 3, size=n) if grouped else None,
        time=np.sort(rng.uniform(0, 1000, size=n)) if dated else None,
        embargo=10.0 if dated else 0.0,
    )


@needs_lightgbm
def test_hybrid_rows_reach_the_csv_with_the_suite_columns(tmp_path, monkeypatch):
    _admit(
        monkeypatch,
        dated=lambda root: _panel(900, dated=True, grouped=True, seed=0, name="dated"),
        undated=lambda root: _panel(600, dated=False, grouped=False, seed=1, name="undated"),
    )
    monkeypatch.setattr(cs, "LIGHTGBM_PARAMS", {"n_estimators": 5})
    fintfm_fits = []
    real_predict = cs._predict

    def counting_predict(name, *args, **kwargs):
        fintfm_fits.append(name == "fintfm")
        return real_predict(name, *args, **kwargs)

    monkeypatch.setattr(cs, "_predict", counting_predict)
    ckpt = _tiny_checkpoint(tmp_path)
    out = tmp_path / "results.csv"
    assert cs.main(["--model", ckpt, "--out", str(out), "--datasets", "dated,undated",
                    "--root", str(tmp_path), "--hybrid", "--inner-folds", "2"]) == 0
    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert tuple(rows[0]) == cs.COLUMNS
    models = ("fintfm", "logistic_regression", "lightgbm_1000", *cs.HYBRID_MODELS)
    assert [(r["dataset"], r["split"], r["model"]) for r in rows] == [
        (d, str(k), m) for d in ("dated", "undated") for k in range(3) for m in models
    ]
    # the test side's FinTFM is the fintfm row's, fitted once per split and reused
    assert sum(fintfm_fits) == 6
    sha = {r["checkpoint_sha256"] for r in rows if r["model"] == "fintfm"}
    for r in rows:
        assert r["skipped"] == ""
        assert 0.0 <= float(r["average_precision"]) <= 1.0
        assert 0.0 < float(r["mean_predicted"]) < 1.0
        if r["model"] in cs.HYBRID_MODELS:
            assert {r["checkpoint_sha256"]} == sha
            assert float(r["seconds"]) > 0


def test_a_skipped_fintfm_row_skips_the_hybrid_rows(tmp_path, monkeypatch):
    _admit(monkeypatch, undated=lambda root: _panel(600, False, False, 0, "undated"))
    monkeypatch.setattr(cs, "_predict_hybrid", lambda *a: pytest.fail("hybrid fitted"))
    monkeypatch.setattr(cs, "_hybrid_oof", lambda *a: pytest.fail("inner fits ran"))
    ckpt = _tiny_checkpoint(tmp_path, max_features=4)  # narrower than the panel's 5 features
    real_predict = cs._predict
    monkeypatch.setattr(cs, "_predict", lambda name, *a, **k: (
        np.full(len(a[2]), 0.2) if name == "lightgbm_1000" else real_predict(name, *a, **k)))
    rows = cs.run(tmp_path / "r.csv", ckpt, datasets=("undated",), root=tmp_path, hybrid=True)
    hybrid = [r for r in rows if r["model"] in cs.HYBRID_MODELS]
    assert len(hybrid) == 6 and all("fintfm row" in r["skipped"] for r in hybrid)


def test_hybrid_needs_a_checkpoint_and_lightgbm(tmp_path):
    with pytest.raises(ValueError, match="hybrid arms need"):
        cs.run(tmp_path / "r.csv", None, datasets=("sba",), hybrid=True)
    with pytest.raises(ValueError, match="hybrid arms need"):
        cs.run(tmp_path / "r.csv", "m.pt", datasets=("sba",), hybrid=True, with_lightgbm=False)
    with pytest.raises(SystemExit):
        cs.main(["--out", str(tmp_path / "r.csv"), "--hybrid"])
    assert not (tmp_path / "r.csv").exists()
