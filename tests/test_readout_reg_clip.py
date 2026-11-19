# ruff: noqa: F811
"""``reg_clip_margin``: regression predictions clipped to the widened training target range (FINDINGS §295)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fintfm")
pytest.importorskip("torch")

from readout_support import checkpoint  # noqa: F401
from readout_support import make as make_estimator

from fintfm.readout.entry import check_reg_clip_margin, entry_defaults
from fintfm.readout.estimator import LinearReadoutModel

N_TRAIN = 240


def data(seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """A linear target in [roughly] 0 to 10 on five features."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(N_TRAIN, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(5 + 1.5 * X["f0"] + rng.normal(scale=0.3, size=N_TRAIN))
    return X, y


def make(tmp_path, checkpoint, **hp) -> LinearReadoutModel:
    """A regression model with the entry's defaults and the given overrides."""
    return make_estimator(
        "regression", tmp_path, checkpoint, **{"partition": None, "probe_spline": None, **hp}
    )


@pytest.mark.parametrize(
    ("problem_type", "expected"), [("binary", None), ("multiclass", None), ("regression", 0.1)]
)
def test_entry_default_is_on_for_regression_only(problem_type: str, expected: float | None) -> None:
    """The clip is a regression lever; classification never sets it."""
    assert entry_defaults(problem_type)["reg_clip_margin"] == expected


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), "0.1", True])
def test_malformed_margin_is_refused(bad) -> None:
    """A negative, non-finite or non-numeric margin raises instead of silently clipping wrongly."""
    with pytest.raises(ValueError, match="reg_clip_margin"):
        check_reg_clip_margin(bad, source="reg_clip_margin")


def test_predictions_stay_inside_the_widened_training_range(tmp_path, checkpoint) -> None:
    """Rows far off the training data are clipped to min/max widened by the margin; others are untouched."""
    X, y = data()
    model = make(tmp_path / "on", checkpoint)
    model.fit(X=X, y=y)
    lo, hi = float(y.min()), float(y.max())
    pad = 0.1 * (hi - lo)
    assert model._reg_clip == pytest.approx((lo - pad, hi + pad))
    far = pd.DataFrame(np.full((4, 5), 0.0), columns=X.columns)
    far["f0"] = [-1e6, -50.0, 50.0, 1e6]
    pred = model.predict(far)
    assert np.all(pred >= lo - pad - 1e-9)
    assert np.all(pred <= hi + pad + 1e-9)
    off = make(tmp_path / "off", checkpoint, reg_clip_margin=None)
    off.fit(X=X, y=y)
    assert off._reg_clip is None
    inside = X.iloc[:20]
    inside_on, inside_off = model.predict(inside), off.predict(inside)
    unclipped = (inside_off >= lo - pad) & (inside_off <= hi + pad)
    assert np.array_equal(inside_on[unclipped], inside_off[unclipped])


def test_zero_margin_clips_to_the_observed_range(tmp_path, checkpoint) -> None:
    """``reg_clip_margin=0`` is allowed and bounds predictions by the training targets themselves."""
    X, y = data(1)
    model = make(tmp_path, checkpoint, reg_clip_margin=0)
    model.fit(X=X, y=y)
    assert model._reg_clip == pytest.approx((float(y.min()), float(y.max())))
