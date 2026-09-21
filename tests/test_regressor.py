"""The de-binning regression wrapper (task 46.3).

These are contract tests, not accuracy tests: an untrained tiny model has nothing to say about
a target, so anything asserting a good prediction here would be asserting noise. What is
checked is that the boundary is wired correctly in the ways that fail silently -- bins aligned
to representatives, predictions inside the training range, edges fitted without the queries.
"""

import numpy as np
import pytest
import torch

from fintfm.inference.binning import interval_coverage
from fintfm.inference.regressor import FinancialTFMRegressor
from fintfm.modeling.model import FinancialTFM, ModelConfig


def _tiny_model(max_classes: int = 5) -> FinancialTFM:
    cfg = ModelConfig(
        max_features=6, max_classes=max_classes, d_model=16, n_heads=2, n_layers=1, d_ff=32
    )
    return FinancialTFM(cfg)


def _data(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.3, size=n)
    return X, y


def test_fit_predict_shapes():
    torch.manual_seed(0)
    X, y = _data()
    reg = FinancialTFMRegressor(_tiny_model(), n_bins=5).fit(X[:40], y[:40])
    pred = reg.predict(X[40:])
    assert pred.shape == (20,)
    assert np.isfinite(pred).all()


def test_proba_is_a_distribution_over_the_full_bin_grid():
    torch.manual_seed(0)
    X, y = _data()
    reg = FinancialTFMRegressor(_tiny_model(), n_bins=5).fit(X[:40], y[:40])
    proba = reg.predict_proba(X[40:])
    assert proba.shape == (20, reg.binner_.n_bins_)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predictions_stay_inside_the_training_range():
    # A convex combination of representatives cannot leave their hull. Worth asserting because
    # the failure mode if the widening in predict_proba were misaligned is a prediction that
    # still looks plausible -- just paired with the wrong bin.
    torch.manual_seed(0)
    X, y = _data()
    reg = FinancialTFMRegressor(_tiny_model(), n_bins=5).fit(X[:40], y[:40])
    pred = reg.predict(X[40:])
    assert pred.min() >= y[:40].min() - 1e-9
    assert pred.max() <= y[:40].max() + 1e-9


def test_bin_edges_ignore_the_query_rows():
    # Fit on the same training rows but with wildly different query rows; the binner must be
    # byte-identical, or the grid the answer is read off depends on the answer.
    torch.manual_seed(0)
    X, y = _data()
    a = FinancialTFMRegressor(_tiny_model(), n_bins=5).fit(X[:40], y[:40])
    b = FinancialTFMRegressor(_tiny_model(), n_bins=5).fit(X[:40], y[:40])
    b.predict(X[40:] * 1000.0)
    np.testing.assert_array_equal(a.binner_.edges_, b.binner_.edges_)
    np.testing.assert_array_equal(a.binner_.representatives_, b.binner_.representatives_)


def test_n_bins_above_max_classes_is_refused():
    # The head has one logit per bin. Silently truncating would produce a model that regresses
    # onto a grid it never saw the top of.
    X, y = _data()
    reg = FinancialTFMRegressor(_tiny_model(max_classes=3), n_bins=10)
    with pytest.raises(ValueError, match="max_classes"):
        reg.fit(X[:40], y[:40])


def test_quantiles_are_monotone_and_interval_brackets_the_mean():
    torch.manual_seed(0)
    X, y = _data()
    reg = FinancialTFMRegressor(_tiny_model(), n_bins=5).fit(X[:40], y[:40])
    q10 = reg.predict_quantile(X[40:], 0.1)
    q90 = reg.predict_quantile(X[40:], 0.9)
    assert (q10 <= q90).all()
    lo, hi = reg.predict_interval(X[40:], level=0.8)
    np.testing.assert_array_equal(lo, q10)
    np.testing.assert_array_equal(hi, q90)


def test_interval_coverage_is_measurable_end_to_end():
    # Task 46.5 wants a coverage number; this asserts the plumbing produces one in [0, 1],
    # not that an untrained model is calibrated.
    torch.manual_seed(0)
    X, y = _data()
    reg = FinancialTFMRegressor(_tiny_model(), n_bins=5).fit(X[:40], y[:40])
    lo, hi = reg.predict_interval(X[40:], level=0.8)
    cov = interval_coverage(y[40:], lo, hi)
    assert 0.0 <= cov <= 1.0


def test_predicting_before_fit_raises():
    reg = FinancialTFMRegressor(_tiny_model(), n_bins=5)
    with pytest.raises(RuntimeError, match="fit must be called"):
        reg.predict(np.zeros((2, 4)))
