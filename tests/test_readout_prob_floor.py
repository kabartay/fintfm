# ruff: noqa: F811
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fintfm")
pytest.importorskip("torch")

from readout_support import (
    PARTITION,
    checkpoint,  # noqa: F401
    data,
    make,
)
from test_readout_entry import (
    make as make_entry,
)
from test_readout_entry import (
    resolve,
)

from fintfm.readout.entry import PROB_FLOOR_GRID, check_prob_floor
from fintfm.readout.estimator import PROB_FLOOR_GRID

PROBE = {"scale": True, "cv": True, "raw": True}


def fitted(problem_type, tmp_path, checkpoint, partitioned, **hp):
    X, y, X_test = data(problem_type)
    extra = {f"partition_{problem_type}": dict(PARTITION)} if partitioned else {}
    m = make(problem_type, tmp_path, checkpoint, **{f"probe_{problem_type}": PROBE, **extra, **hp})
    m.fit(X=X, y=y)
    return m, X_test


@pytest.mark.parametrize("partitioned", [False, True])
def test_binary_default_is_bit_identical(partitioned, tmp_path, checkpoint):
    base, X_test = fitted("binary", tmp_path / "a", checkpoint, partitioned)
    explicit, _ = fitted("binary", tmp_path / "b", checkpoint, partitioned, probe_prob_floor=None)
    assert base._get_model_params()["probe_prob_floor"] is None
    assert base.prob_floor_ == 0.0
    assert np.array_equal(base.predict_proba(X_test), explicit.predict_proba(X_test))


@pytest.mark.parametrize("partitioned", [False, True])
def test_cv_picks_a_grid_floor_and_rows_sum_to_one(partitioned, tmp_path, checkpoint):
    base, X_test = fitted(
        "multiclass", tmp_path / "a", checkpoint, partitioned, probe_prob_floor=None
    )
    m, _ = fitted("multiclass", tmp_path / "b", checkpoint, partitioned, probe_prob_floor="cv")
    assert m.prob_floor_ in PROB_FLOOR_GRID
    assert set(m.prob_floor_losses_) == set(PROB_FLOOR_GRID)
    assert m.prob_floor_ == min(PROB_FLOOR_GRID, key=lambda e: (m.prob_floor_losses_[e], e))
    proba = m.predict_proba(X_test)
    assert np.allclose(proba.sum(axis=1), 1.0)
    if m.prob_floor_ > 0:
        ref = base.predict_proba(X_test)
        assert np.allclose(
            proba, (1 - m.prob_floor_) * ref + m.prob_floor_ / ref.shape[1], atol=1e-6
        )


@pytest.mark.parametrize("partitioned", [False, True])
def test_float_floor(partitioned, tmp_path, checkpoint):
    base, X_test = fitted(
        "multiclass", tmp_path / "a", checkpoint, partitioned, probe_prob_floor=None
    )
    m, _ = fitted("multiclass", tmp_path / "b", checkpoint, partitioned, probe_prob_floor=0.01)
    assert m.prob_floor_ == 0.01
    ref, proba = base.predict_proba(X_test), m.predict_proba(X_test)
    assert np.allclose(proba, 0.99 * ref + 0.01 / 3, atol=1e-6)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert proba.min() >= 0.01 / 3 - 1e-12


def test_binary_floor(tmp_path, checkpoint):
    base, X_test = fitted("binary", tmp_path / "a", checkpoint, False)
    m, _ = fitted("binary", tmp_path / "b", checkpoint, False, probe_prob_floor=0.02)
    ref, proba = np.asarray(base.predict_proba(X_test)), np.asarray(m.predict_proba(X_test))
    assert np.allclose(proba, 0.98 * ref + 0.02 / (2 if ref.ndim == 1 else ref.shape[1]), atol=1e-6)


@pytest.mark.parametrize("bad", [-0.1, 1.0, True, "x", [0.1]])
def test_rejects_bad_value(bad):
    with pytest.raises(ValueError, match="probe_prob_floor"):
        check_prob_floor(bad, source="probe_prob_floor")


@pytest.mark.parametrize(
    ("problem_type", "expected"), [("binary", None), ("multiclass", "cv"), ("regression", None)]
)
def test_entry_default_floor_is_multiclass_only(problem_type, expected, tmp_path):
    """The entry's multiclass configuration carries the floor (fintfm FINDINGS §269); the others do not."""
    model = make_entry(problem_type, tmp_path)
    assert model._get_model_params()["probe_prob_floor"] == expected
    _, rest = resolve(model)
    assert model._prob_floor_setting == expected
    assert "probe_prob_floor" not in rest  # never reaches the classifier


@pytest.mark.parametrize("problem_type", ["binary", "regression"])
def test_floor_off_for_binary_and_regression_after_resolve(problem_type, tmp_path):
    model = make_entry(problem_type, tmp_path)
    resolve(model)
    assert model._prob_floor_setting is None


def test_multiclass_default_resolves_to_the_floor(tmp_path):
    model = make_entry("multiclass", tmp_path)
    entry, _ = resolve(model)
    assert model._prob_floor_setting == "cv"
    # The rest of the entry is unchanged by the floor.
    assert entry["partition"] == {"n_bags": 8}
    assert entry["probe"] == {"scale": True, "cv": True, "raw": True}


def test_multiclass_default_floor_is_fitted_on_held_out_rows(tmp_path, checkpoint):
    X, y, X_test = data("multiclass")
    m = make("multiclass", tmp_path, checkpoint, probe_multiclass=PROBE)
    m.fit(X=X, y=y)
    assert m._get_model_params()["probe_prob_floor"] == "cv"
    assert m.prob_floor_ in PROB_FLOOR_GRID
    assert m.prob_floor_losses_ is not None
    assert np.allclose(m.predict_proba(X_test).sum(axis=1), 1.0)
