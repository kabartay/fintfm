"""The §130 representation-transfer harness and its equal-label-count control (DR3.4).

A tiny random-initialisation model on a synthetic panel, so nothing here says anything about
the published checkpoint: the tests pin down which rows each arm is fitted on, that the
default reproduces the harness as it stood before the control, and that the control keeps
the original arms beside its own.
"""

import json
import warnings

import numpy as np
import pytest
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import fintfm.experiments.representation_transfer as rt
from fintfm.experiments.v4_protocol import build_fold_assignments, split_indices_for_fold
from fintfm.inference.classifier import FinancialTFMClassifier
from fintfm.modeling.model import FinancialTFM, ModelConfig

_CTX = 120
_SAMPLE = 400


def _panel(n=1500, seed=0):
    """A V4FinBench-shaped table: features, a ~8% target, two countries, whole companies."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = (X[:, 0] + rng.normal(size=n) > 1.9).astype(np.int64)
    country = np.where(np.arange(n) % 2 == 0, "CZ", "PL")
    company = np.array([f"c{i % 300}" for i in range(n)])
    return X, y, country, company, [f"f{j}" for j in range(4)]


def _model() -> FinancialTFM:
    """A seeded untrained network, so every call builds the same weights."""
    torch.manual_seed(0)
    return FinancialTFM(ModelConfig(max_features=6, max_classes=2, d_model=16, n_heads=2,
                                    n_layers=1, d_ff=32))


@pytest.fixture
def panel(monkeypatch):
    """The synthetic panel, served where the harness reads a horizon file."""
    data = _panel()
    monkeypatch.setattr(rt, "_load_horizon", lambda root, horizon, max_rows: data)
    return data


def _run(tmp_path, name, **kw):
    """One fold of the harness on the synthetic panel, with its per-row predictions."""
    out = tmp_path / name
    record = rt.run(
        _model(), out / "rt.json", folds=(0,), root=tmp_path, train_sample_size=_SAMPLE,
        max_context=_CTX, predictions_dir=out, **kw,
    )
    return record, dict(np.load(out / "predictions_fold0.npz")), (out / "rt.json").read_bytes()


def _pre_dr34_arms(X, y, country, company, fold=0):
    """The harness's per-fold arms as they stood before DR3.4, kept verbatim (inline fits on
    the stratified sample) as the byte-identity reference for the default path."""
    assignments = build_fold_assignments(country, company)
    rng = np.random.default_rng(fold)
    tr, _va, te = split_indices_for_fold(assignments, fold)
    imputer = SimpleImputer(strategy="median").fit(X[tr])
    scaler = StandardScaler().fit(imputer.transform(X[tr]))
    Xtr_full, Xte = (scaler.transform(imputer.transform(X[i])) for i in (tr, te))
    Xtr_full, Xte = Xtr_full.astype(np.float32), Xte.astype(np.float32)
    ytr_full = y[tr]
    clf = FinancialTFMClassifier(_model(), random_state=fold, max_context=_CTX)
    clf.fit(Xtr_full, ytr_full)
    sub = rt._stratified_subsample(rng, ytr_full, _SAMPLE)
    Xtr, ytr = Xtr_full[sub], ytr_full[sub]
    p_own_head = clf.predict_proba(Xte)[:, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep_tr = clf.transform_representation(Xtr)
        rep_te = clf.transform_representation(Xte)
        p_rep = LogisticRegression(max_iter=1000).fit(rep_tr, ytr).predict_proba(rep_te)[:, 1]
        p_raw = LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        mlp = MLPClassifier(hidden_layer_sizes=(64,), early_stopping=True, n_iter_no_change=15,
                            max_iter=500, random_state=fold).fit(rep_tr, ytr)
        p_mlp = mlp.predict_proba(rep_te)[:, 1]
    return (
        {"own_head": p_own_head, "representation": p_rep, "raw_features": p_raw,
         "mlp_head": p_mlp},
        clf, Xtr_full, ytr_full, Xte,
    )


def test_default_probe_rows_is_the_harness_before_the_control(tmp_path, panel):
    """The default, and ``probe_rows="stratified"`` spelled out, give every arm bit for bit
    as the pre-DR3.4 code computed it, and write the same JSON."""
    ref, *_ = _pre_dr34_arms(*panel[:4])
    rec, preds, raw_json = _run(tmp_path, "default")
    _rec_s, preds_s, raw_json_s = _run(tmp_path, "stratified", probe_rows="stratified")
    assert raw_json == raw_json_s and "probe_rows" not in rec
    assert set(preds) == {"y_true"} | {f"pred_{a}" for a in ref}
    for arm, p in ref.items():
        assert preds[f"pred_{arm}"].tobytes() == p.tobytes() == preds_s[f"pred_{arm}"].tobytes()
    assert list(rec["folds"][0]) == [
        "fold", "n_train_full", "n_train_sample", "n_test",
        *(f"{a}_{m}" for a in ref for m in ("auc", "ap")),
    ]


def test_context_probe_is_fitted_on_the_heads_own_context_rows(tmp_path, panel):
    """``probe_rows="context"`` fits the probe on exactly the rows the head conditions on,
    reports how many labels that is, and keeps the stratified arms under a suffix."""
    ref, clf, Xtr_full, ytr_full, Xte = _pre_dr34_arms(*panel[:4])
    rec, preds, _ = _run(tmp_path, "context", probe_rows="context")
    row = rec["folds"][0]
    assert rec["probe_rows"] == "context"
    ctx = clf.context_indices_
    assert ctx.size == _CTX == row["n_probe_rows"]
    assert row["n_probe_positive"] == clf.context_summary_.n_positive == int(ytr_full[ctx].sum())
    assert 0 < row["n_probe_positive"] < row["n_train_sample"]
    # the context rows, conditioned, are the stored context: the probe reads the head's rows
    np.testing.assert_array_equal(clf._conditioner.transform(Xtr_full[ctx]), clf._ctx_X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probe = LogisticRegression(max_iter=1000).fit(
            clf.transform_representation(Xtr_full[ctx]), ytr_full[ctx]
        )
        want = probe.predict_proba(clf.transform_representation(Xte))[:, 1]
    assert preds["pred_representation"].tobytes() == want.tobytes()
    assert preds["pred_own_head"].tobytes() == ref["own_head"].tobytes()
    for arm in ("representation", "raw_features", "mlp_head"):
        assert preds[f"pred_{arm}_stratified"].tobytes() == ref[arm].tobytes()
        assert np.isfinite(row[f"{arm}_ap"]) and np.isfinite(row[f"{arm}_stratified_ap"])
    json.loads((tmp_path / "context" / "rt.json").read_text())


def test_a_single_class_probe_set_records_a_gap_not_a_crash():
    X, y, *_ = _panel(n=300)
    clf = FinancialTFMClassifier(_model(), max_context=50).fit(X, y)
    arms = rt._downstream_arms(clf, X[:40], np.zeros(40, dtype=np.int64), X[40:60], "encode_rows", 0)
    assert set(arms) == {"representation", "raw_features", "mlp_head"}
    for p in arms.values():
        assert p.shape == (20,) and np.isnan(p).all()
        assert all(np.isnan(v) for v in rt._scores(y[40:60], p))


def test_an_unknown_probe_rows_is_refused(tmp_path):
    with pytest.raises(ValueError, match="probe_rows"):
        rt.run(_model(), tmp_path / "x.json", probe_rows="context_rows")  # type: ignore[arg-type]
