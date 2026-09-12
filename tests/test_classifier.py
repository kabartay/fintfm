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


def test_chunked_prediction_is_exact_not_approximate():
    """Queries cannot attend to each other, so chunking must change nothing at all.

    This is what makes scoring 48,000 rows feasible without a 50k x 50k attention matrix,
    and it is a direct consequence of the row mask rather than a tolerated approximation.
    """
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = rng.integers(0, 2, size=200)
    X_test = rng.normal(size=(97, 4))

    model = _tiny_model()
    whole = FinancialTFMClassifier(model, max_context=50, query_chunk=10_000).fit(X, y)
    chunked = FinancialTFMClassifier(model, max_context=50, query_chunk=8).fit(X, y)
    np.testing.assert_allclose(
        whole.predict_proba(X_test), chunked.predict_proba(X_test), rtol=1e-5, atol=1e-6
    )


# --- ensembling over context draws (openspec adopt-published-methods 36.2) ---------


def _toy(n=600, n_feat=6, rate=0.15, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat)).astype(np.float32)
    s = X @ rng.normal(size=n_feat)
    y = (s >= np.quantile(s, 1 - rate)).astype(np.int64)
    k = n // 2
    return X[:k], y[:k], X[k:], y[k:]


def _model(n_feat=6):
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    torch.manual_seed(0)
    return FinancialTFM(
        ModelConfig(max_features=n_feat, d_model=32, d_cell=16, n_layers=1,
                    n_col_layers=1, max_classes=2)
    )


def test_ensemble_averages_distinct_members_rather_than_repeating_one():
    """Members must differ, or the ensemble is one prediction computed n times."""
    import numpy as np

    from fintfm.inference.classifier import FinancialTFMClassifier

    Xtr, ytr, Xte, _ = _toy()
    m = _model()
    single = FinancialTFMClassifier(m, max_context=100, context_strategy="uniform",
                                    random_state=0).fit(Xtr, ytr).predict_proba(Xte)
    other = FinancialTFMClassifier(m, max_context=100, context_strategy="uniform",
                                   random_state=1).fit(Xtr, ytr).predict_proba(Xte)
    assert not np.allclose(single, other, atol=1e-4), "context draw does not vary with seed"

    ens = FinancialTFMClassifier(m, max_context=100, context_strategy="uniform",
                                 random_state=0, n_ensemble=2).fit(Xtr, ytr).predict_proba(Xte)
    np.testing.assert_allclose(ens, (single + other) / 2, atol=1e-5)


def test_ensemble_of_one_is_exactly_the_single_model():
    import numpy as np

    from fintfm.inference.classifier import FinancialTFMClassifier

    Xtr, ytr, Xte, _ = _toy()
    m = _model()
    a = FinancialTFMClassifier(m, max_context=100, random_state=0,
                               n_ensemble=1).fit(Xtr, ytr).predict_proba(Xte)
    b = FinancialTFMClassifier(m, max_context=100, random_state=0).fit(Xtr, ytr).predict_proba(Xte)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_ensemble_output_is_a_valid_probability_distribution():
    import numpy as np

    from fintfm.inference.classifier import FinancialTFMClassifier

    Xtr, ytr, Xte, _ = _toy()
    p = FinancialTFMClassifier(_model(), max_context=100, random_state=0,
                               n_ensemble=3).fit(Xtr, ytr).predict_proba(Xte)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert (p >= 0).all()


def test_ensemble_members_refit_the_conditioner_not_inherit_it():
    """Each member draws its own context *and* fits its own feature conditioner.

    Sharing the parent's conditioner would make members correlated through preprocessing
    rather than independent draws, which is most of the point of ensembling.
    """
    import numpy as np

    from fintfm.inference.classifier import FinancialTFMClassifier

    Xtr, ytr, _, _ = _toy()
    parent = FinancialTFMClassifier(_model(), max_context=100, feature_transform="rank",
                                    random_state=0, n_ensemble=2).fit(Xtr, ytr)
    twin = parent._clone_with_seed(7)
    assert twin.random_state == 7
    assert twin.n_ensemble == 1  # no recursion
    assert twin._conditioner is not parent._conditioner
    np.testing.assert_array_equal(twin._raw_X, parent._raw_X)


def test_label_swap_averaging_cancels_the_asymmetry_it_targets():
    """§45: relabelling the context 0<->1 and inverting should be a no-op and is not.

    Averaging the two orientations removes the component of the prediction that depends on
    which class occupies the "1" slot. The averaged predictor must therefore be (near-)
    invariant to relabelling, even though a single member is not.
    """
    import numpy as np

    from fintfm.inference.classifier import FinancialTFMClassifier

    Xtr, ytr, Xte, _ = _toy()
    m = _model()
    kw = {"max_context": 100, "context_strategy": "uniform", "random_state": 0}

    plain = FinancialTFMClassifier(m, **kw).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    plain_swapped = 1.0 - FinancialTFMClassifier(m, **kw).fit(Xtr, 1 - ytr).predict_proba(Xte)[:, 1]
    single_gap = float(np.abs(plain - plain_swapped).max())

    ekw = {**kw, "ensemble_label_swap": True}
    ens = FinancialTFMClassifier(m, **ekw).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    ens_swapped = 1.0 - FinancialTFMClassifier(m, **ekw).fit(Xtr, 1 - ytr).predict_proba(Xte)[:, 1]
    ens_gap = float(np.abs(ens - ens_swapped).max())

    assert ens_gap < single_gap, (ens_gap, single_gap)
    assert ens_gap < 1e-5, f"averaging should make relabelling a no-op, gap {ens_gap}"


def test_feature_subsetting_produces_genuinely_different_members():
    import numpy as np

    from fintfm.inference.classifier import FinancialTFMClassifier

    Xtr, ytr, Xte, _ = _toy()
    m = _model()
    full = FinancialTFMClassifier(m, max_context=100, random_state=0).fit(Xtr, ytr)
    sub = FinancialTFMClassifier(m, max_context=100, random_state=0,
                                 ensemble_feature_frac=0.5).fit(Xtr, ytr)
    assert not np.allclose(full.predict_proba(Xte), sub.predict_proba(Xte), atol=1e-3)


def test_feature_subset_ensemble_stays_a_valid_distribution():
    import numpy as np

    from fintfm.inference.classifier import FinancialTFMClassifier

    Xtr, ytr, Xte, _ = _toy()
    p = FinancialTFMClassifier(_model(), max_context=100, random_state=0, n_ensemble=3,
                               ensemble_feature_frac=0.6, ensemble_label_swap=True
                               ).fit(Xtr, ytr).predict_proba(Xte)
    assert np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_ensembling_recovers_column_order_invariance():
    """Decision D12's stated remedy, asserted rather than promised.

    Random column identities make the model expressive enough to learn column-specific rules
    (``docs/FINDINGS.md`` §54, §56) at the cost of exact column-order invariance, which
    becomes distributional. D12 says callers recover it by ensembling, and each ensemble
    member draws its own identities -- so a wider ensemble must agree more closely across a
    column permutation than a single member does. If it does not, D12's mitigation is a
    claim with nothing behind it.

    This is an **end-to-end** check, not an isolation: an ensemble member varies its context
    draw as well as its column identities, so the gap it closes includes context resampling.
    Measured when written, the mean gap in predicted probability fell 0.0098 (k=1) -> 0.0014
    (k=32), non-monotonically -- k=2 was worse than k=1. Hence the wide k and the margin
    rather than a strict ordering at small k, which would flake.
    """
    import numpy as np

    from fintfm.inference.classifier import FinancialTFMClassifier
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    torch.manual_seed(0)
    model = FinancialTFM(
        ModelConfig(max_features=8, max_classes=2, d_cell=16, d_model=32, n_heads=2,
                    n_col_layers=1, n_layers=2, d_ff=64)
    ).eval()
    model.trained_objectives = ("classification",)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(160, 6)).astype(np.float32)
    y = (X[:, 0] - X[:, 1] > 0).astype(np.int64)
    perm = rng.permutation(6)
    Xq = rng.normal(size=(40, 6)).astype(np.float32)

    def gap(n_ensemble):
        kw = {
            "max_context": 120, "context_strategy": "uniform", "feature_transform": "none",
            "n_ensemble": n_ensemble, "random_state": 0,
        }
        a = FinancialTFMClassifier(model, **kw).fit(X, y).predict_proba(Xq)[:, 1]
        b = FinancialTFMClassifier(model, **kw).fit(X[:, perm], y).predict_proba(Xq[:, perm])[:, 1]
        return float(np.abs(a - b).mean())

    one, many = gap(1), gap(32)
    assert many < 0.6 * one, (
        f"ensembling did not reduce the column-permutation gap ({one:.4f} -> {many:.4f}); "
        "decision D12 relies on it doing so"
    )
