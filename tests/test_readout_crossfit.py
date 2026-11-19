# ruff: noqa: F811
from __future__ import annotations

import numpy as np
import pytest
from readout_support import (
    N_TEST,
    PARTITION,
    Spy,
    checkpoint,  # noqa: F401
    data,
    make,
)

from fintfm.readout.entry import PROBE_CROSSFIT_FOLDS
from fintfm.readout.estimator import LinearReadoutModel


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("partitioned", [False, True])
def test_no_readout_row_is_encoded_against_a_context_holding_it(
    problem_type, partitioned, tmp_path, checkpoint, monkeypatch
):
    X, y, _ = data(problem_type)
    hp = {"probe_crossfit_folds": 5}
    hp[f"partition_{problem_type}"] = dict(PARTITION) if partitioned else None
    model = make(problem_type, tmp_path, checkpoint, **hp)
    spy = Spy(monkeypatch)
    model.fit(X=X, y=y)
    assert spy.calls, "the readout encoded nothing"
    assert spy.rows_seen_in_own_context() == 0
    # Every restriction leaves some context rows out and keeps some.
    assert spy.restricted
    stored = [np.asarray(c[2]) for c in spy.calls]
    for rows in spy.restricted:
        assert rows.size >= 1
    assert all(len(np.unique(rows)) == len(rows) for rows in spy.restricted)
    assert model._train_matrix is None
    assert model._crossfit_stats["restricted"] >= 5
    if partitioned:
        # Leaves with their own probe were cross-fitted too, not only the global estimator.
        leaf_probes = (
            model._ridge_leaf_probes if problem_type == "regression" else model._leaf_probes
        )
        n_leaf = sum(p is not None for leaves in leaf_probes for p in leaves.values())
        assert n_leaf >= 1
        assert len({tuple(c[2]) for c in spy.calls}) > 5
    assert len(stored) == len(spy.calls)


@pytest.mark.parametrize("problem_type", ["binary", "regression"])
def test_without_crossfit_the_spy_finds_the_leak(problem_type, tmp_path, checkpoint, monkeypatch):
    """The positive control: the spy does detect a row encoded against its own labelled copy."""
    X, y, _ = data(problem_type)
    model = make(problem_type, tmp_path, checkpoint, probe_crossfit_folds=0)
    spy = Spy(monkeypatch)
    model.fit(X=X, y=y)
    assert spy.rows_seen_in_own_context() > 0
    assert not spy.restricted


@pytest.mark.parametrize("folds", [0, 1])
def test_disabled_crossfit_reproduces_the_earlier_readout_inputs(
    folds, tmp_path, checkpoint, monkeypatch
):
    """With 0 or 1 fold the readout input is the plain representation of the replayed matrix."""
    X, y, _ = data("binary", categorical=True)
    model = make("binary", tmp_path, checkpoint, probe_crossfit_folds=folds, probe_binary={})
    seen = []
    original = LinearReadoutModel._probe_input

    def record(self, est, Xq, fit_idx=None):
        out = original(self, est, Xq, fit_idx)
        seen.append((est, np.asarray(Xq).copy(), fit_idx, out))
        return out

    monkeypatch.setattr(LinearReadoutModel, "_probe_input", record)
    spy = Spy(monkeypatch)
    model.fit(X=X, y=y)
    assert len(seen) == 1
    est, Xq, fit_idx, out = seen[0]
    assert fit_idx is None
    assert not spy.restricted
    expected = np.asarray(est.transform_representation(Xq, stage="pre_head"), dtype=np.float64)
    assert np.array_equal(out, expected)
    # The rows are the replayed full-statistics encoding.
    replay = np.asarray(model.preprocess(X, is_train=False), dtype=np.float32)
    assert np.array_equal(Xq.astype(np.float32), replay[: len(Xq)])
    assert model._crossfit_on is False
    assert model._train_matrix is None


def test_crossfit_rows_use_out_of_fold_categorical_encoding(tmp_path, checkpoint, monkeypatch):
    X, y, _ = data("binary", categorical=True)
    model = make("binary", tmp_path, checkpoint, probe_crossfit_folds=3, probe_binary={})
    captured = {}
    original = LinearReadoutModel._probe_input

    def record(self, est, Xq, fit_idx=None):
        captured["Xq"] = np.asarray(Xq).copy()
        captured["fit_idx"] = fit_idx
        return original(self, est, Xq, fit_idx)

    monkeypatch.setattr(LinearReadoutModel, "_probe_input", record)
    model.fit(X=X, y=y)
    replay = np.asarray(model.preprocess(X, is_train=False), dtype=np.float32)
    cat_col = list(X.columns).index("cat")
    # Out-of-fold encoding differs from the full-statistics encoding for at least some rows.
    assert not np.array_equal(captured["Xq"][:, cat_col], replay[:, cat_col])
    # The numeric columns are the same either way.
    numeric = [i for i in range(replay.shape[1]) if i != cat_col]
    assert np.array_equal(captured["Xq"][:, numeric], replay[:, numeric])
    assert captured["fit_idx"] is not None


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.parametrize("folds", [PROBE_CROSSFIT_FOLDS, 0])
def test_fit_and_predict_end_to_end(problem_type, partitioned, folds, tmp_path, checkpoint):
    X, y, X_test = data(problem_type, categorical=True)
    hp = {"probe_crossfit_folds": folds}
    hp[f"partition_{problem_type}"] = dict(PARTITION) if partitioned else None
    model = make(problem_type, tmp_path, checkpoint, **hp)
    model.fit(X=X, y=y)
    if problem_type == "regression":
        pred = model.predict(X_test)
        assert pred.shape == (N_TEST,)
        assert np.isfinite(pred).all()
    else:
        proba = model.predict_proba(X_test)
        n_classes = 2 if problem_type == "binary" else 3
        if problem_type == "multiclass":
            assert proba.shape == (N_TEST, n_classes)
            assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        else:
            assert proba.shape == (N_TEST,)
        assert np.isfinite(proba).all()
    if partitioned:
        assert model._leaf_probes is not None or model._ridge_leaf_probes is not None


def test_default_folds_and_validation(tmp_path, checkpoint):
    assert PROBE_CROSSFIT_FOLDS == 5
    model = make("binary", tmp_path, checkpoint)
    params = model._get_model_params()
    assert params["probe_crossfit_folds"] == 5
    model._resolve_entry(params)
    assert "probe_crossfit_folds" not in params
    assert model._crossfit_folds == 5
    for bad in (-1, 2.5, True, "5"):
        m = make("binary", tmp_path, checkpoint, probe_crossfit_folds=bad)
        with pytest.raises(ValueError, match="probe_crossfit_folds"):
            m._resolve_entry(m._get_model_params())


def test_head_option_is_refused_with_crossfit_and_allowed_without(tmp_path, checkpoint):
    m = make("binary", tmp_path, checkpoint, probe_binary={"head": True}, probe_crossfit_folds=5)
    with pytest.raises(ValueError, match="head"):
        m._resolve_entry(m._get_model_params())
    m = make("binary", tmp_path, checkpoint, probe_binary={"head": True}, probe_crossfit_folds=0)
    m._resolve_entry(m._get_model_params())
