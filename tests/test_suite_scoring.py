"""Tests for the suite runner (tasks 37.3 and 37.4), on synthetic data with stub models.

SMOKE-TEST only: these prove the plumbing (folds, cap, coverage, resume, spread), not any
model's quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fintfm.experiments import suite_scoring as ss


@dataclass
class Fake:
    """Minimal dataset: a frame and labels."""

    X: pd.DataFrame
    y: np.ndarray


def make(n: int, d: int, seed: int) -> Fake:
    """Synthetic binary task whose first column carries signal."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int)
    return Fake(pd.DataFrame(X), y)


def signal(Xs: np.ndarray, ys: np.ndarray, Xt: np.ndarray, seed: int) -> np.ndarray:
    """Stub that scores by the first column."""
    return Xt[:, 0]


def noise(Xs: np.ndarray, ys: np.ndarray, Xt: np.ndarray, seed: int) -> np.ndarray:
    """Stub that scores randomly (the floor)."""
    return np.random.default_rng(seed).random(len(Xt))


def narrow(Xs: np.ndarray, ys: np.ndarray, Xt: np.ndarray, seed: int) -> np.ndarray:
    """Stub that refuses wide tables, like a checkpoint's max_features."""
    if Xs.shape[1] > 4:
        raise ss.ArmSkip("6 features exceed the checkpoint's max_features 4")
    return Xt[:, 0]


def test_cap_rows_stratified_and_noop() -> None:
    """The cap keeps class shares and leaves small datasets alone."""
    d = make(1000, 3, 0)
    X = d.X.to_numpy()
    _, y2 = ss.cap_rows(X, d.y, 200)
    assert len(y2) == 200 and abs(y2.mean() - d.y.mean()) < 0.02
    X3, y3 = ss.cap_rows(X, d.y, 5000)
    assert X3 is X and y3 is d.y


def test_to_float_matrix_maps_categories_to_codes() -> None:
    """Category columns become codes, missing ones NaN."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": pd.Categorical(["x", None])})
    m = ss.to_float_matrix(df)
    assert m.dtype == np.float32 and m[0, 1] == 0 and np.isnan(m[1, 1])


def test_run_suite_table_spread_coverage_and_resume(tmp_path: Path) -> None:
    """Per-dataset cells, spread with range, skips with reasons, resume without re-fitting."""
    loaders = {"a/one": lambda: make(300, 3, 1), "a/two": lambda: make(300, 6, 2)}
    arms = {"signal": signal, "noise": noise, "narrow": narrow}
    recs = ss.run_suite(loaders, arms, tmp_path, folds=3, max_rows=240, log=lambda s: None)
    assert [r["n_rows"] for r in recs] == [240, 240]
    assert recs[0]["n_rows_before_cap"] == 300
    one, two = recs
    assert len(one["cells"]["signal"]["average_precision"]) == 3
    assert np.isfinite(one["cells"]["narrow"]["average_precision"]).all()
    assert np.isnan(two["cells"]["narrow"]["average_precision"]).all()
    assert two["coverage"]["n_skipped"] == 3 and "max_features" in str(two["coverage"]["skipped"])
    agg = ss.aggregate(recs)
    assert agg["spread"]["average_precision"]["narrow"]["n"] == 1
    assert agg["common_datasets"] == ["a/one"]
    sig = agg["spread"]["average_precision"]["signal"]
    assert sig["n"] == 2 and sig["min"] <= sig["mean"] <= sig["max"] and np.isfinite(sig["std"])
    assert agg["spread"]["average_precision"]["noise"]["mean"] < sig["mean"]
    text = ss.format_report(recs, agg)
    assert "a/one" in text and "std (ddof=1)" in text and "skipped 3 units" in text

    def boom() -> Fake:
        """Loader that must not be called on resume."""
        raise AssertionError("re-scored a finished dataset")

    again = ss.run_suite(
        {"a/one": boom, "a/two": boom}, arms, tmp_path, folds=3, max_rows=240, log=lambda s: None
    )
    assert again[0]["cells"] == recs[0]["cells"]


def test_changed_configuration_refuses_to_mix(tmp_path: Path) -> None:
    """A different fold count against the same output directory is an error."""
    ld = {"a/one": lambda: make(200, 3, 1)}
    ss.run_suite(ld, {"signal": signal}, tmp_path, folds=2, log=lambda s: None)
    with pytest.raises(RuntimeError, match="different configuration"):
        ss.run_suite(ld, {"signal": signal}, tmp_path, folds=3, log=lambda s: None)


def test_none_return_is_a_skip_with_reason() -> None:
    """A predictor returning None (a failed booster) is recorded as skipped."""
    d = make(100, 2, 0)
    rec = ss.score_dataset("x", d.X.to_numpy(), d.y, {"b": lambda *a: None}, 2, 0, 100)
    assert rec["coverage"]["n_skipped"] == 2 and rec["coverage"]["n_scored"] == 0


def test_non_binary_labels_rejected() -> None:
    """Multi-class labels are an error, not a silent AP."""
    with pytest.raises(ValueError, match="binary"):
        ss.score_dataset("x", np.zeros((9, 2)), np.arange(9) % 3, {"s": signal}, 3, 0, 9)
