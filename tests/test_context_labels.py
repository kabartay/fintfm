"""Task 31.2 (survival-context-labels): ``fit`` carries period and n_observed intact.

Column 0 of every table is the row index and ``feature_transform="none"`` keeps it exact, so
a stored context row names its source row and the tests can check that the stored labels
belong to the stored rows, under every context strategy.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fintfm.inference.classifier import FinancialTFMClassifier
from fintfm.modeling.hazard import CENSORED
from fintfm.modeling.model import FinancialTFM, ModelConfig

_CFG = {"max_features": 6, "max_classes": 2, "d_model": 32, "d_cell": 16, "n_heads": 2,
        "n_col_layers": 1, "n_layers": 2, "d_ff": 64, "n_horizons": 4}
_STRATEGIES = ["uniform", "balanced", "hybrid", "prototype", "retrieval"]


def _model(period_context: bool) -> FinancialTFM:
    """A tiny randomly initialised model, the flag-on one with a trained-looking table."""
    torch.manual_seed(0)
    model = FinancialTFM(ModelConfig(**_CFG, hazard_period_context=period_context)).eval()
    if period_context:
        with torch.no_grad():
            model.period_embed.weight.normal_()
    return model


def _table(n: int = 80, seed: int = 2) -> tuple[np.ndarray, ...]:
    """Features (column 0 = row index), labels, periods and observed-horizon counts."""
    g = np.random.default_rng(seed)
    X = g.normal(size=(n, 6)).astype(np.float32)
    X[:, 0] = np.arange(n)
    y = (g.random(n) < 0.35).astype(np.int64)
    period = np.where(y == 1, g.integers(0, 4, n), CENSORED).astype(np.int64)
    n_obs = g.integers(1, 5, n).astype(np.int64)
    return X, y, period, n_obs


def _kw(strategy: str) -> dict:
    return {"max_context": 30, "context_strategy": strategy, "feature_transform": "none",
            "retrieval_groups": 2, "retrieval_min_positive": 2}


@pytest.mark.parametrize("strategy", _STRATEGIES)
def test_selected_context_labels_equal_the_selected_rows_labels(strategy):
    X, y, period, n_obs = _table()
    clf = FinancialTFMClassifier(_model(True), **_kw(strategy)).fit(
        X, y, periods=period, n_observed=n_obs
    )
    idx = clf.context_indices_
    assert np.array_equal(clf._ctx_X[:, 0].astype(np.int64), idx)
    assert np.array_equal(clf.context_period_, period[idx])
    assert np.array_equal(clf.context_n_observed_, n_obs[idx])
    if strategy != "retrieval":
        assert len(idx) <= 30  # a genuine subselection, not the whole table
        assert not np.array_equal(clf.context_period_, period)


@pytest.mark.parametrize("extra", [{"context_positive_floor": 3}, {"context_positive_cap": 0.2}])
def test_positive_floor_and_cap_contexts_keep_their_own_labels(extra):
    X, y, period, n_obs = _table()
    kw = {**_kw("uniform"), **extra}
    clf = FinancialTFMClassifier(_model(True), **kw).fit(X, y, periods=period, n_observed=n_obs)
    idx = clf.context_indices_
    assert np.array_equal(clf.context_period_, period[idx])
    assert np.array_equal(clf.context_n_observed_, n_obs[idx])


def test_calibration_split_and_restriction_keep_labels_aligned():
    X, y, period, n_obs = _table()
    kw = {**_kw("uniform"), "calibration_val_frac": 0.25}
    clf = FinancialTFMClassifier(_model(True), **kw).fit(X, y, periods=period, n_observed=n_obs)
    rows = clf.train_rows_[clf.context_indices_]
    assert np.array_equal(clf.context_n_observed_, n_obs[rows])
    twin = FinancialTFMClassifier(_model(True), **_kw("uniform")).fit(
        X, y, periods=period, n_observed=n_obs
    ).restrict_context(np.array([3, 9, 20]))
    assert np.array_equal(twin.context_period_, period[[3, 9, 20]])
    assert np.array_equal(twin.context_n_observed_, n_obs[[3, 9, 20]])


def test_ensemble_members_keep_their_own_rows_labels():
    X, y, period, n_obs = _table()
    clf = FinancialTFMClassifier(
        _model(True), max_context=30, n_ensemble=3, feature_transform="none"
    ).fit(X, y, periods=period, n_observed=n_obs)
    twin = clf._clone_with_seed(1)
    idx = twin.context_indices_
    assert np.array_equal(twin.context_period_, period[idx])
    assert np.array_equal(twin.context_n_observed_, n_obs[idx])


def test_accessors_are_copies_and_none_when_omitted():
    X, y, period, n_obs = _table()
    clf = FinancialTFMClassifier(_model(False), **_kw("uniform")).fit(X, y)
    assert clf.context_period_ is None and clf.context_n_observed_ is None
    given = FinancialTFMClassifier(_model(False), **_kw("uniform")).fit(
        X, y, periods=period, n_observed=n_obs
    )
    assert given.context_period_ is None  # a flag-off checkpoint stores no periods
    kept = given.context_n_observed_.copy()
    given.context_n_observed_[:] = 99
    assert np.array_equal(given.context_n_observed_, kept)


def test_n_observed_is_validated():
    X, y, _period, n_obs = _table()
    clf = FinancialTFMClassifier(_model(False), **_kw("uniform"))
    with pytest.raises(ValueError, match="n_observed has shape"):
        clf.fit(X, y, n_observed=n_obs[:-1])
    with pytest.raises(ValueError, match="integers"):
        clf.fit(X, y, n_observed=n_obs.astype(np.float32))
    with pytest.raises(ValueError, match="non-negative"):
        clf.fit(X, y, n_observed=n_obs - 5)


@pytest.mark.parametrize("strategy", _STRATEGIES)
@pytest.mark.parametrize("n_ensemble", [1, 3])
def test_omitting_or_adding_n_observed_leaves_the_binary_path_byte_identical(strategy, n_ensemble):
    """No model reads n_observed, so predictions match with and without it, bit for bit."""
    X, y, period, n_obs = _table()
    kw = {**_kw(strategy), "n_ensemble": n_ensemble, "random_state": 7}
    plain = FinancialTFMClassifier(_model(False), **kw).fit(X, y)
    given = FinancialTFMClassifier(_model(False), **kw).fit(
        X, y, periods=period, n_observed=n_obs
    )
    assert np.array_equal(plain.predict_proba(X), given.predict_proba(X))
    assert np.array_equal(plain.context_indices_, given.context_indices_)
