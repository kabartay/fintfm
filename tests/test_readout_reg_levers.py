# ruff: noqa: F811
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fintfm")
pytest.importorskip("torch")

from readout_support import N_TEST, N_TRAIN, PARTITION, checkpoint  # noqa: F401
from readout_support import make as make_estimator

from fintfm.readout.estimator import LinearReadoutModel
from fintfm.readout.target_transform import TargetTransform, smearing_factor


def skewed(seed: int = 0, shift: float = 0.0):
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_TEST
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.exp(0.8 * X["f0"] + 0.4 * X["f1"] + rng.normal(scale=0.3, size=n)) + shift)
    return (
        X.iloc[:N_TRAIN].reset_index(drop=True),
        y.iloc[:N_TRAIN].reset_index(drop=True),
        X.iloc[N_TRAIN:],
    )


def make(tmp_path, checkpoint, partitioned=False, **hp) -> LinearReadoutModel:
    """A regression estimator at the entry with the tiny network; the partition is off unless asked for."""
    hp.setdefault("partition", dict(PARTITION) if partitioned else None)
    return make_estimator("regression", tmp_path, checkpoint, **hp)


def test_off_is_byte_identical(tmp_path, checkpoint):
    X, y, X_test = skewed()
    base = make(tmp_path / "a", checkpoint)
    base.fit(X=X, y=y)
    explicit = make(
        tmp_path / "b",
        checkpoint,
        reg_target_transform="none",
        reg_readout_blend=None,
        reg_ridge_alphas=None,
    )
    explicit.fit(X=X, y=y)
    assert np.array_equal(base.predict(X_test), explicit.predict(X_test))
    assert base._target_tf is None
    assert base._reg_blend_weight is None
    assert base._reg_smear == 1.0


def test_default_alpha_grid_is_unchanged(tmp_path, checkpoint):
    m = make(tmp_path, checkpoint)
    m._resolve_entry(m._get_model_params())
    assert np.array_equal(m._alpha_grid(), np.logspace(-3, 5, 17))
    m = make(tmp_path, checkpoint, probe_regression={})
    m._resolve_entry(m._get_model_params())
    assert np.array_equal(m._alpha_grid(), np.logspace(-3, 3, 13))


def test_wide_and_explicit_alpha_grids(tmp_path, checkpoint):
    m = make(tmp_path, checkpoint, reg_ridge_alphas="wide")
    m._resolve_entry(m._get_model_params())
    assert np.array_equal(m._alpha_grid(), np.logspace(-3, 7, 21))
    m = make(tmp_path, checkpoint, reg_ridge_alphas=[0.1, 10.0])
    m._resolve_entry(m._get_model_params())
    assert list(m._alpha_grid()) == [0.1, 10.0]


@pytest.mark.parametrize("name", ["log1p", "quantile"])
def test_transform_inverts_exactly_on_a_fitted_identity_case(name):
    rng = np.random.default_rng(1)
    y = np.exp(rng.normal(size=300))
    tf, reason = TargetTransform.fit(name, y)
    assert reason is None
    assert np.allclose(tf.inverse(tf.forward(y)), y, rtol=1e-6, atol=1e-6)


def test_log1p_smearing_scales_the_back_transform():
    tf, _ = TargetTransform.fit("log1p", np.array([0.0, 1.0, 2.0]))
    z = np.array([0.5, 1.0])
    assert np.allclose(tf.inverse(z, 1.0), np.expm1(z))
    assert np.allclose(tf.inverse(z, 1.2), 1.2 * np.exp(z) - 1.0)
    assert smearing_factor(np.zeros(5)) == 1.0
    assert smearing_factor(np.array([0.3, -0.3])) > 1.0


@pytest.mark.parametrize("name", ["log1p", "quantile"])
def test_model_inverts_the_readout_output(name, tmp_path, checkpoint, monkeypatch):
    X, y, X_test = skewed()
    model = make(tmp_path, checkpoint, reg_target_transform=name)
    model.fit(X=X, y=y)
    assert model._reg_transform_info["effective"] == name
    z = np.linspace(y.min(), y.max(), N_TEST)
    zt = model._target_tf.forward(z)
    monkeypatch.setattr(LinearReadoutModel, "_predict_ridge", lambda self, Xp: zt)
    model._reg_smear = 1.0
    assert np.allclose(model.predict(X_test), z, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.parametrize(
    "hp",
    [
        {"reg_target_transform": "log1p"},
        {"reg_target_transform": "quantile"},
        {"reg_readout_blend": 0.5},
        {"reg_readout_blend": "cv"},
        {"reg_ridge_alphas": "wide"},
        {"reg_target_transform": "log1p", "reg_readout_blend": "cv", "reg_ridge_alphas": "wide"},
    ],
)
def test_each_lever_runs_end_to_end(hp, partitioned, tmp_path, checkpoint):
    X, y, X_test = skewed()
    # The regression entry does not cross-fit; the cv blend needs the out-of-fold head predictions.
    model = make(tmp_path, checkpoint, partitioned=partitioned, probe_crossfit_folds=5, **hp)
    model.fit(X=X, y=y)
    pred = model.predict(X_test)
    assert pred.shape == (N_TEST,)
    assert np.isfinite(pred).all()
    if hp.get("reg_target_transform") == "log1p":
        assert (pred >= 0).all()
        assert model._reg_smear > 0


def test_cv_blend_weight_is_in_unit_interval_and_chosen_on_training_rows(
    tmp_path, checkpoint, monkeypatch
):
    X, y, X_test = skewed()
    seen = []
    original = LinearReadoutModel._blend_weight

    def spy(yy, ridge, head):
        seen.append((np.asarray(yy).copy(), np.asarray(ridge).copy(), np.asarray(head).copy()))
        return original(yy, ridge, head)

    monkeypatch.setattr(LinearReadoutModel, "_blend_weight", staticmethod(spy))
    predicted = []
    orig_predict = LinearReadoutModel._predict_ridge
    monkeypatch.setattr(
        LinearReadoutModel,
        "_predict_ridge",
        lambda self, Xp: predicted.append(len(Xp)) or orig_predict(self, Xp),
    )
    model = make(tmp_path, checkpoint, reg_readout_blend="cv", probe_crossfit_folds=5)
    model.fit(X=X, y=y)
    assert not predicted, "the weight was chosen with predict-time code running"
    assert len(seen) == 1
    ys, ridge, head = seen[0]
    assert ys.size == ridge.size == head.size
    assert 0 < ys.size <= N_TRAIN
    # Only training targets: every value is one of the training y values.
    assert set(np.round(ys, 9)) <= set(np.round(y.to_numpy(), 9))
    assert 0.0 <= model._reg_blend_weight <= 1.0
    model.predict(X_test)


def test_blend_weight_formula():
    f = LinearReadoutModel._blend_weight
    y = np.array([1.0, 2.0, 3.0])
    assert f(y, y, y + 5) == 1.0
    assert f(y, y + 5, y) == 0.0
    assert f(y, y + 1, y - 1) == pytest.approx(0.5)
    assert f(y, y, y) == 1.0


def test_cv_blend_needs_crossfit(tmp_path, checkpoint):
    m = make(tmp_path, checkpoint, reg_readout_blend="cv", probe_crossfit_folds=0)
    with pytest.raises(ValueError, match="crossfit"):
        m._resolve_entry(m._get_model_params())


def test_log1p_with_negative_target_falls_back_to_none(tmp_path, checkpoint):
    X, y, X_test = skewed(shift=-3.0)
    assert (y < 0).any()
    model = make(tmp_path, checkpoint, reg_target_transform="log1p")
    model.fit(X=X, y=y)
    assert model._target_tf is None
    info = model._reg_transform_info
    assert (
        info["requested"] == "log1p" and info["effective"] == "none" and "below 0" in info["note"]
    )
    plain = make(tmp_path / "p", checkpoint)
    plain.fit(X=X, y=y)
    assert np.array_equal(model.predict(X_test), plain.predict(X_test))


@pytest.mark.parametrize(
    ("hp", "match"),
    [
        ({"reg_target_transform": "sqrt"}, "reg_target_transform"),
        ({"reg_readout_blend": 1.5}, "reg_readout_blend"),
        ({"reg_readout_blend": "mean"}, "reg_readout_blend"),
        ({"reg_ridge_alphas": [0.1, -1.0]}, "reg_ridge_alphas"),
        ({"reg_ridge_alphas": []}, "reg_ridge_alphas"),
    ],
)
def test_malformed_levers_raise(hp, match, tmp_path, checkpoint):
    m = make(tmp_path, checkpoint, **hp)
    with pytest.raises(ValueError, match=match):
        m._resolve_entry(m._get_model_params())


def test_levers_are_not_passed_to_the_classifier(tmp_path, checkpoint):
    m = make(
        tmp_path,
        checkpoint,
        reg_target_transform="log1p",
        reg_readout_blend=0.5,
        reg_ridge_alphas="wide",
    )
    params = m._get_model_params()
    m._resolve_entry(params)
    assert not [k for k in params if k.startswith("reg_")]
