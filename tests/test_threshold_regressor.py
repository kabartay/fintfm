"""The threshold regressor: K binary classifiers on ``y > t_k`` (FINDINGS §235).

Contract tests on an untrained tiny model, as in ``test_regressor.py``: what is checked is the
wiring that fails silently (thresholds from the training targets only, a valid step
distribution, predictions inside the training range), not accuracy.
"""

import numpy as np
import pytest
import torch
from sklearn.base import clone

from fintfm.inference.threshold_regressor import ThresholdTFMRegressor
from fintfm.modeling.model import FinancialTFM, ModelConfig


def _tiny_model(max_classes: int = 2) -> FinancialTFM:
    cfg = ModelConfig(
        max_features=6, max_classes=max_classes, d_model=16, n_heads=2, n_layers=1, d_ff=32
    )
    return FinancialTFM(cfg)


def _data(n: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.3, size=n)
    return X, y


def test_fit_predict_shapes_and_a_valid_step_distribution():
    torch.manual_seed(0)
    X, y = _data()
    reg = ThresholdTFMRegressor(_tiny_model(), n_thresholds=5).fit(X[:60], y[:60])
    assert reg.thresholds_.shape == (5,) and np.all(np.diff(reg.thresholds_) > 0)
    assert reg.interval_means_.shape == (6,) and np.all(np.diff(reg.interval_means_) > 0)
    s = reg.predict_survival(X[60:])
    assert s.shape == (20, 5) and np.all(np.diff(s, axis=1) <= 0)
    p = reg.predict_proba(X[60:])
    assert p.shape == (20, 6) and np.all(p >= 0) and np.allclose(p.sum(axis=1), 1.0)
    pred = reg.predict(X[60:])
    assert pred.shape == (20,)
    assert np.all(pred >= y[:60].min()) and np.all(pred <= y[:60].max())
    np.testing.assert_allclose(pred, p @ reg.interval_means_)
    feats = reg.readout_features(X[60:])
    assert feats.shape == (20, 5) and np.isfinite(feats).all()


def test_thresholds_come_from_the_training_targets_only():
    torch.manual_seed(0)
    X, y = _data()
    a = ThresholdTFMRegressor(_tiny_model(), n_thresholds=4).fit(X[:60], y[:60])
    assert set(a.thresholds_) <= set(y[:60])
    y2 = y.copy()
    y2[60:] = 1e6  # the queries' targets never reach fit
    b = ThresholdTFMRegressor(_tiny_model(), n_thresholds=4).fit(X[:60], y2[:60])
    np.testing.assert_array_equal(a.thresholds_, b.thresholds_)


def test_ties_drop_duplicate_and_top_thresholds():
    torch.manual_seed(0)
    X, _ = _data()
    y = np.repeat([0.0, 1.0, 2.0], [40, 15, 25])
    reg = ThresholdTFMRegressor(_tiny_model(), n_thresholds=9).fit(X, y)
    np.testing.assert_array_equal(reg.thresholds_, [0.0, 1.0])
    np.testing.assert_allclose(reg.interval_means_, [0.0, 1.0, 2.0])


def test_a_constant_target_is_refused():
    X, _ = _data()
    with pytest.raises(ValueError, match="single distinct value"):
        ThresholdTFMRegressor(_tiny_model()).fit(X, np.ones(len(X)))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_targets_are_refused(bad):
    X, y = _data()
    y[3] = bad
    with pytest.raises(ValueError, match="finite"):
        ThresholdTFMRegressor(_tiny_model()).fit(X, y)


def test_predicting_before_fit_raises():
    X, _ = _data()
    with pytest.raises(RuntimeError, match="not fitted"):
        ThresholdTFMRegressor(_tiny_model()).predict(X)


def test_clone_keeps_the_settings():
    reg = ThresholdTFMRegressor(_tiny_model(), n_thresholds=7, classifier_params={"device": "cpu"})
    c = clone(reg)
    assert c.n_thresholds == 7 and c.classifier_params == {"device": "cpu"}
