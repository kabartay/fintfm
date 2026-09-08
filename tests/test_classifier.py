import numpy as np
import torch

from fintfm.inference.classifier import FinancialTFMClassifier, _select_context
from fintfm.modeling.model import FinancialTFM, ModelConfig


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


def test_balanced_context_lifts_minority_representation():
    """Balanced selection must beat uniform on a realistically imbalanced credit table."""
    rng = np.random.default_rng(0)
    y = np.zeros(10_000, dtype=np.int64)
    y[rng.choice(10_000, size=470, replace=False)] = 1  # ~4.7%, the Polish-bankruptcy rate

    uniform = _select_context(y, 2000, "uniform", np.random.default_rng(1))
    balanced = _select_context(y, 2000, "balanced", np.random.default_rng(1))
    hybrid = _select_context(y, 2000, "hybrid", np.random.default_rng(1))

    for idx in (uniform, balanced, hybrid):
        assert idx.shape[0] == 2000
        assert len(np.unique(idx)) == 2000  # no row selected twice

    # balanced takes every positive available; uniform gets only the base rate
    assert y[balanced].sum() == 470
    assert y[uniform].sum() < 150
    # hybrid sits between: all positives it can afford, but keeps a uniform tail
    assert y[uniform].sum() < y[hybrid].sum() <= y[balanced].sum()


def test_select_context_returns_everything_when_under_budget():
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    idx = _select_context(y, 100, "balanced", np.random.default_rng(0))
    np.testing.assert_array_equal(idx, np.arange(4))


def test_context_strategy_reaches_stored_context():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4))
    y = np.zeros(500, dtype=np.int64)
    y[rng.choice(500, size=25, replace=False)] = 1

    balanced = FinancialTFMClassifier(_tiny_model(), max_context=100, context_strategy="balanced")
    balanced.fit(X, y)
    uniform = FinancialTFMClassifier(_tiny_model(), max_context=100, context_strategy="uniform")
    uniform.fit(X, y)

    assert balanced._ctx_y.sum() == 25  # every default kept
    assert uniform._ctx_y.sum() < 15
    assert balanced._ctx_X.shape[0] == uniform._ctx_X.shape[0] == 100


def test_prior_correction_preserves_ranking_but_shifts_probabilities():
    """The correction must fix calibration without touching AUC.

    Adding a constant per-class shift to the logits is a monotone transform of the
    binary score, so the ordering of predictions cannot change.
    """
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 4))
    y = np.zeros(600, dtype=np.int64)
    y[rng.choice(600, size=30, replace=False)] = 1  # 5% base rate

    model = _tiny_model()
    corrected = FinancialTFMClassifier(
        model, max_context=100, context_strategy="balanced", correct_prior=True
    ).fit(X, y)
    raw = FinancialTFMClassifier(
        model, max_context=100, context_strategy="balanced", correct_prior=False
    ).fit(X, y)

    X_test = rng.normal(size=(50, 4))
    p_corrected = corrected.predict_proba(X_test)[:, 1]
    p_raw = raw.predict_proba(X_test)[:, 1]

    # identical ordering
    np.testing.assert_array_equal(np.argsort(p_corrected), np.argsort(p_raw))
    # but the corrected probabilities are pulled down towards the true 5% base rate,
    # because the balanced context claimed defaults were far more common than they are
    assert p_corrected.mean() < p_raw.mean()


def test_prior_correction_is_inert_when_context_is_not_resampled():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = rng.integers(0, 2, size=40)
    clf = FinancialTFMClassifier(_tiny_model(), max_context=1000, correct_prior=True).fit(X, y)
    np.testing.assert_allclose(clf._log_prior_shift, 0.0, atol=1e-12)
