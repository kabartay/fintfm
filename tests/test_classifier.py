import numpy as np
import torch

from fintfm.classifier import FinancialTFMClassifier
from fintfm.model import FinancialTFM, ModelConfig


def _tiny_model() -> FinancialTFM:
    cfg = ModelConfig(max_features=6, max_classes=3, d_model=16, n_heads=2, n_layers=1, d_ff=32)
    return FinancialTFM(cfg)


def test_fit_predict_shapes():
    torch.manual_seed(0)
    clf = FinancialTFMClassifier(_tiny_model())
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(40, 4))
    y_train = rng.integers(0, 2, size=40)
    clf.fit(X_train, y_train)
    X_test = rng.normal(size=(10, 4))
    proba = clf.predict_proba(X_test)
    assert proba.shape == (10, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    pred = clf.predict(X_test)
    assert set(pred) <= set(clf.classes_)


def test_rejects_too_many_classes():
    clf = FinancialTFMClassifier(_tiny_model())
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    y = rng.integers(0, 5, size=20)
    try:
        clf.fit(X, y)
        assert False, "expected ValueError"
    except ValueError:
        pass
