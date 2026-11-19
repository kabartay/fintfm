"""``minority_positive`` codes the minority class as 1 and maps the head's probabilities back."""

from __future__ import annotations

import numpy as np
import pytest

from fintfm.readout import LinearReadoutModel


class _StubClassifier:
    """Returns fixed two-column probabilities, so the column handling is all that varies."""

    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.proba


def _model(problem_type: str = "binary", **settings) -> LinearReadoutModel:
    model = LinearReadoutModel(problem_type, "/unused.pt", **settings)
    model._minority_positive = settings.get("minority_positive", False)
    model.model = _StubClassifier()
    return model


@pytest.mark.parametrize(
    ("flag", "share_of_ones", "expected"),
    [(False, 0.9, False), (True, 0.9, True), (True, 0.5, False), (True, 0.1, False)],
)
def test_recodes_only_when_opted_in_and_class_one_is_the_majority(
    flag: bool, share_of_ones: float, expected: bool
) -> None:
    """Off by default; on, only a strict class-1 majority is recoded (an exact half is not)."""
    y = np.zeros(100, dtype=np.int64)
    y[: round(100 * share_of_ones)] = 1
    assert _model(minority_positive=flag)._flip_to_minority(y) is expected


def test_regression_is_never_recoded() -> None:
    assert _model("regression", minority_positive=True)._flip_to_minority(np.ones(10)) is False


def test_the_head_fallback_maps_the_columns_back() -> None:
    """A recoded fit's column 0 is the caller's class 1, so that is what the head fallback returns;
    without a recoding the columns are returned as they are.
    """
    X = np.zeros((3, 1), dtype=np.float32)
    plain, flipped = _model(), _model()
    flipped._minority_flipped = True
    for model in (plain, flipped):
        model._linear_probe = None
        model._categorical_features = ()
    np.testing.assert_array_equal(plain.predict_proba(X), _StubClassifier.proba)
    np.testing.assert_array_equal(flipped.predict_proba(X), _StubClassifier.proba[:, ::-1])
