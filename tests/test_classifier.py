import numpy as np
import pytest
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
    (``docs/results/FINDINGS.md`` §54, §56) at the cost of exact column-order invariance, which
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


ACCELERATORS = [
    pytest.param(
        "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
    ),
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
    ),
]


@pytest.mark.parametrize("device", ACCELERATORS)
def test_set_params_device_moves_the_model_at_fit(device):
    """AU1.11: __init__ placed the model, so set_params(device=...) left it where it was and
    the first forward pass mixed devices."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    clf = FinancialTFMClassifier(_tiny_model(), device="cpu", max_context=40)
    clf.set_params(device=device).fit(X, y)
    assert next(clf.model_.parameters()).device.type == device
    assert clf.predict_proba(X[:5]).shape == (5, 2)


def test_fit_records_the_feature_count():
    """AU1.11: no n_features_in_, which sklearn tooling reads."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4)).astype(np.float32)
    clf = FinancialTFMClassifier(_tiny_model(), max_context=40).fit(X, (X[:, 0] > 0).astype(int))
    assert clf.n_features_in_ == 4


def test_predicting_before_fit_raises_not_fitted():
    """AU1.11: predict before fit raised AttributeError, which sklearn does not recognise."""
    from sklearn.exceptions import NotFittedError

    clf = FinancialTFMClassifier(_tiny_model())
    X = np.zeros((3, 4), dtype=np.float32)
    for method in (clf.predict_proba, clf.predict, clf.transform_representation):
        with pytest.raises(NotFittedError):
            method(X)


def test_pooled_prior_shift_uses_each_class_share_in_multiclass_retrieval():
    """AU1.11: the pooled correction gave every non-positive class an equal share of the
    context, which is wrong whenever the retrieved classes are unequal."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4)).astype(np.float32)
    y = rng.choice(3, size=300, p=[0.7, 0.2, 0.1])
    torch.manual_seed(0)
    clf = FinancialTFMClassifier(
        _tiny_model(), max_context=60, context_strategy="retrieval", retrieval_groups=4,
    ).fit(X, y)
    Xq = clf._conditioner.transform(rng.normal(size=(40, 4)).astype(np.float32))
    plan = clf._retrieval_plan(Xq)
    rate = clf._pooled_context_rate(plan)
    shares = sum(q.size * np.bincount(clf._pool_y[c], minlength=3) / c.size for q, c in plan)
    shares = shares / sum(q.size for q, _ in plan)
    expected = np.log(clf._full_prior) - np.log(shares)
    np.testing.assert_allclose(clf._pooled_prior_shift(rate), expected, rtol=1e-10)


def test_label_swap_leaves_a_multiclass_task_unchanged():
    """AU1.4: the swap is defined for two classes only. On three, reversing the columns paired
    each class with another, so label swap on scrambled the probabilities it averaged."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(90, 4)).astype(np.float32)
    y = np.digitize(X[:, 0], [-0.4, 0.4])  # three classes
    Xq = rng.normal(size=(12, 4)).astype(np.float32)
    kw = {"max_context": 60, "n_ensemble": 2, "random_state": 0}
    torch.manual_seed(0)
    model = _tiny_model()
    off = FinancialTFMClassifier(model, **kw).fit(X, y).predict_proba(Xq)
    on = FinancialTFMClassifier(model, ensemble_label_swap=True, **kw).fit(X, y).predict_proba(Xq)
    assert off.shape == (12, 3)
    np.testing.assert_array_equal(on, off)


# --- AU5.7 / AU5.12: member-diversity and wide-table levers, all off by default -------------

_LEVERS_OFF = {
    "ensemble_transforms": None,
    "ensemble_svd_components": 0,
    "ensemble_class_shift": False,
    "max_features_strategy": "raise",
}


@pytest.mark.parametrize(
    "kw",
    [{}, {"n_ensemble": 3}, {"ensemble_label_swap": True},
     {"n_ensemble": 2, "ensemble_feature_frac": 0.5}],
)
def test_levers_default_off_and_the_default_member_is_the_plain_model(kw):
    """The defaults reproduce the classifier as it was, bit for bit.

    Two checks: the defaults equal every lever passed explicitly off, and the default
    ensemble is exactly the mean of plain single models at consecutive seeds -- the member
    construction the levers were added around. (Checked once more by hand when the levers
    landed: every default path, retrieval, prototype, multiclass, the regressor and
    ``transform_representation`` included, returned the same bytes before and after.)
    """
    Xtr, ytr, Xte, _ = _toy(n=240)
    m = _model()
    base = {"max_context": 60, "random_state": 2, **kw}
    default = FinancialTFMClassifier(m, **base).fit(Xtr, ytr).predict_proba(Xte)
    off = FinancialTFMClassifier(m, **base, **_LEVERS_OFF).fit(Xtr, ytr).predict_proba(Xte)
    np.testing.assert_array_equal(default, off)
    if kw == {"n_ensemble": 3}:
        singles = [
            FinancialTFMClassifier(m, max_context=60, random_state=s).fit(Xtr, ytr).predict_proba(Xte)
            for s in (2, 3, 4)
        ]
        np.testing.assert_array_equal(default, np.mean(singles, axis=0))


def test_ensemble_transforms_cycle_the_conditioning_over_members():
    Xtr, ytr, Xte, _ = _toy(n=240)
    m = _model()
    kw = {"max_context": 60, "random_state": 0}
    ens = FinancialTFMClassifier(
        m, n_ensemble=3, ensemble_transforms=("rank", "squash"), **kw
    ).fit(Xtr, ytr).predict_proba(Xte)
    by_hand = [
        FinancialTFMClassifier(m, max_context=60, random_state=s, feature_transform=t)
        .fit(Xtr, ytr).predict_proba(Xte)
        for s, t in ((0, "rank"), (1, "squash"), (2, "rank"))
    ]
    np.testing.assert_array_equal(ens, np.mean(by_hand, axis=0))
    plain = FinancialTFMClassifier(m, n_ensemble=3, **kw).fit(Xtr, ytr).predict_proba(Xte)
    assert not np.allclose(ens, plain, atol=1e-4), "the squash member changed nothing"


def test_ensemble_transforms_rejects_a_misspelling_rather_than_running_rank():
    Xtr, ytr, _, _ = _toy(n=120)
    for bad in (("rank", "powr"), ()):
        with pytest.raises(ValueError, match="ensemble_transforms"):
            FinancialTFMClassifier(_model(), ensemble_transforms=bad).fit(Xtr, ytr)


def test_svd_components_go_to_odd_members_only_fitted_on_their_context():
    from fintfm.inference.preprocess import SVDAugmenter

    Xtr, ytr, Xte, _ = _toy(n=240, n_feat=6)
    m = _model(n_feat=9)  # room for up to 3 appended columns
    parent = FinancialTFMClassifier(
        m, max_context=60, random_state=0, n_ensemble=2, ensemble_svd_components=2
    ).fit(Xtr, ytr)
    plain_member = parent._clone_with_seed(0, svd_components=0)
    svd_member = parent._clone_with_seed(1, svd_components=2)
    assert plain_member.svd_components_ == 0 and plain_member._ctx_X.shape[1] == 6
    assert svd_member.svd_components_ == 2 and svd_member._ctx_X.shape[1] == 8
    # the projections come from the member's own conditioned context rows
    ctx_raw = svd_member._ctx_X[:, :6]
    np.testing.assert_array_equal(
        svd_member._ctx_X, SVDAugmenter(2).fit(ctx_raw).transform(ctx_raw)
    )
    ens = parent.predict_proba(Xte)
    np.testing.assert_array_equal(
        ens, np.mean([plain_member.predict_proba(Xte), svd_member.predict_proba(Xte)], axis=0)
    )
    no_svd = FinancialTFMClassifier(m, max_context=60, random_state=0, n_ensemble=2)
    assert not np.allclose(ens, no_svd.fit(Xtr, ytr).predict_proba(Xte), atol=1e-5)
    # capped at the model's spare width: 9 - 6 leaves room for 3, not 5
    assert parent._clone_with_seed(1, svd_components=5).svd_components_ == 3


def test_svd_components_refuse_settings_where_no_member_would_carry_them():
    Xtr, ytr, _, _ = _toy(n=120)
    with pytest.raises(ValueError, match="n_ensemble >= 2"):
        FinancialTFMClassifier(_model(8), ensemble_svd_components=2).fit(Xtr, ytr)
    with pytest.raises(ValueError, match="retrieval"):
        FinancialTFMClassifier(
            _model(8), n_ensemble=2, ensemble_svd_components=2, context_strategy="retrieval"
        ).fit(Xtr, ytr)


def test_class_shift_on_a_binary_task_swaps_labels_on_odd_members():
    Xtr, ytr, Xte, _ = _toy(n=240)
    m = _model()
    ens = FinancialTFMClassifier(
        m, max_context=60, random_state=0, n_ensemble=2, ensemble_class_shift=True
    ).fit(Xtr, ytr).predict_proba(Xte)
    member0 = FinancialTFMClassifier(m, max_context=60, random_state=0).fit(Xtr, ytr)
    member1 = FinancialTFMClassifier(m, max_context=60, random_state=1).fit(Xtr, 1 - ytr)
    np.testing.assert_array_equal(
        ens, np.mean([member0.predict_proba(Xte), member1.predict_proba(Xte)[:, ::-1]], axis=0)
    )


def test_class_shift_maps_multiclass_probabilities_back_to_the_original_classes():
    """A member trained on labels shifted by s reports class c in column c + s; undo that."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 4)).astype(np.float32)
    names = np.array(["a", "b", "c"])
    y = names[np.digitize(X[:, 0], [-0.4, 0.4])]  # string labels, three classes
    Xq = rng.normal(size=(12, 4)).astype(np.float32)
    torch.manual_seed(0)
    m = _tiny_model()
    ens = FinancialTFMClassifier(
        m, max_context=60, random_state=0, n_ensemble=3, ensemble_class_shift=True
    ).fit(X, y).predict_proba(Xq)
    by_hand = []
    for s in range(3):
        shifted = names[(np.searchsorted(names, y) + s) % 3]
        p = FinancialTFMClassifier(m, max_context=60, random_state=s).fit(X, shifted).predict_proba(Xq)
        by_hand.append(np.stack([p[:, (c + s) % 3] for c in range(3)], axis=1))
    np.testing.assert_array_equal(ens, np.mean(by_hand, axis=0))
    np.testing.assert_allclose(ens.sum(axis=1), 1.0, atol=1e-5)
    unshifted = FinancialTFMClassifier(m, max_context=60, random_state=0, n_ensemble=3)
    assert not np.allclose(ens, unshifted.fit(X, y).predict_proba(Xq), atol=1e-5)


def _wide(n=160, n_feat=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat)).astype(np.float32)
    y = (X[:, 0] + X[:, 7] > 0).astype(np.int64)
    return X[: n // 2], y[: n // 2], X[n // 2 :]


def test_wide_table_raises_by_default():
    Xtr, ytr, _ = _wide()
    with pytest.raises(ValueError, match="at most 4 features"):
        FinancialTFMClassifier(_model(4), max_context=40).fit(Xtr, ytr)


@pytest.mark.parametrize(
    ("n_ensemble", "frac", "n_members", "size"),
    [(1, 1.0, 3, 4), (5, 1.0, 5, 4), (1, 0.3, 4, 3)],
)
def test_subsample_covers_every_feature_with_balanced_round_robin_members(
    n_ensemble, frac, n_members, size
):
    """Ten features through a four-feature model: every column seen, coverage within one."""
    Xtr, ytr, Xq = _wide()
    clf = FinancialTFMClassifier(
        _model(4), max_context=40, n_ensemble=n_ensemble, ensemble_feature_frac=frac,
        max_features_strategy="subsample",
    ).fit(Xtr, ytr)
    assert clf.n_members_ == n_members  # raised above n_ensemble when 10 / size needs it
    cols = clf._member_columns
    assert all(len(c) == size == len(set(c.tolist())) for c in cols)
    counts = np.bincount(np.concatenate(cols), minlength=10)
    assert counts.min() >= 1 and counts.max() - counts.min() <= 1
    p = clf.predict_proba(Xq)
    assert p.shape == (80, 2) and np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
    by_hand = [
        FinancialTFMClassifier(clf.model_, max_context=40, random_state=i)
        .fit(Xtr[:, c], ytr).predict_proba(Xq[:, c])
        for i, c in enumerate(cols)
    ]
    np.testing.assert_array_equal(p, np.mean(by_hand, axis=0))


def test_subsample_refuses_paths_that_need_the_whole_row():
    Xtr, ytr, Xq = _wide()
    clf = FinancialTFMClassifier(
        _model(4), max_context=40, max_features_strategy="subsample"
    ).fit(Xtr, ytr)
    with pytest.raises(ValueError, match="one forward pass"):
        clf.transform_representation(Xq)
    with pytest.raises(ValueError, match="feature width"):
        clf.predict_proba(Xq[:, :9])
    with pytest.raises(ValueError, match="fine-tuning"):
        FinancialTFMClassifier(
            _model(4), max_context=40, max_features_strategy="subsample", finetune_steps=1
        ).fit(Xtr, ytr)
    with pytest.raises(ValueError, match="max_features_strategy"):
        FinancialTFMClassifier(_model(4), max_features_strategy="drop").fit(Xtr, ytr)


# --- DR1.4: one inference path and a public context summary --------------------------------


def test_every_inference_caller_goes_through_predict_context(monkeypatch):
    """predict_proba, the term structure, the representation, the quantile regressor and the
    capability probes all score through ``FinancialTFM.predict_context``, so a change there
    reaches all of them (``CLAUDE.md``, "a correction that lives in one method")."""
    from fintfm.experiments.capability import bayes_ceiling_probe
    from fintfm.inference.quantile_regressor import FinancialTFMQuantileRegressor

    calls: list[str] = []
    real = FinancialTFM.predict_context

    def spy(self, *args, **kwargs):
        calls.append(kwargs.get("output", "logits"))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(FinancialTFM, "predict_context", spy)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4)).astype(np.float32)
    y = (rng.random(60) < 0.4).astype(np.int64)
    torch.manual_seed(0)
    hz = FinancialTFM(ModelConfig(max_features=6, max_classes=2, d_model=16, n_heads=2,
                                  n_layers=1, d_ff=32, n_horizons=3))
    clf = FinancialTFMClassifier(hz).fit(X[:50], y[:50])
    for method, expected in (
        (lambda: clf.predict_proba(X[50:]), "logits"),
        (lambda: clf.predict_term_structure(X[50:]), "term_structure"),
        (lambda: clf.transform_representation(X[50:], stage="pre_head"), "representation"),
        (lambda: bayes_ceiling_probe(hz, targets=(0.8,), seeds=1, n=40, n_ensemble=1),
         "logits"),
    ):
        calls.clear()
        method()
        assert calls == [expected]
    q = FinancialTFM(ModelConfig(max_features=6, max_classes=2, d_model=16, n_heads=2,
                                 n_layers=1, d_ff=32, head_type="quantile", n_quantiles=5))
    calls.clear()
    FinancialTFMQuantileRegressor(q).fit(X[:50], X[:50, 0]).predict_quantile_grid(X[50:])
    assert calls == ["logits"]


def test_context_summary_states_the_context_beside_the_population():
    from dataclasses import FrozenInstanceError

    from sklearn.exceptions import NotFittedError

    clf = FinancialTFMClassifier(_tiny_model(), max_context=100, context_strategy="balanced")
    with pytest.raises(NotFittedError):
        _ = clf.context_summary_
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4))
    y = np.zeros(500, dtype=np.int64)
    y[rng.choice(500, size=25, replace=False)] = 1
    torch.manual_seed(0)
    s = clf.fit(X, y).context_summary_
    assert (s.n_context, s.n_positive) == (100, 25)
    assert s.context_rate == pytest.approx(0.25) and s.population_rate == pytest.approx(0.05)
    assert s.correct_prior and s.pooled_context_rate is None
    np.testing.assert_allclose(
        s.log_prior_shift, [np.log(0.95 / 0.75), np.log(0.05 / 0.25)], rtol=1e-12
    )
    with pytest.raises(FrozenInstanceError):
        s.n_positive = 0  # type: ignore[misc]
    with pytest.raises(AttributeError):
        clf.context_summary_ = s  # type: ignore[misc]


def test_context_summary_reports_what_retrieval_retrieved():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 4))
    y = (rng.random(200) < 0.2).astype(np.int64)
    torch.manual_seed(0)
    clf = FinancialTFMClassifier(_tiny_model(), max_context=40, context_strategy="retrieval")
    clf.fit(X[:150], y[:150])
    assert clf.context_summary_.pooled_context_rate is None
    clf.predict_proba(X[150:])
    assert clf.context_summary_.pooled_context_rate == clf.pooled_context_rate_


# --- a fixed positive count in the uniform context (design review DR3.3) ---------------------


def _rare(n=10_000, n_pos=47, seed=0):
    """Labels at a V4FinBench-like rate: ``n_pos`` positives among ``n`` rows."""
    y = np.zeros(n, dtype=np.int64)
    y[np.random.default_rng(seed).choice(n, size=n_pos, replace=False)] = 1
    return y


@pytest.mark.parametrize(("floor", "expected"), [(0, 9), (5, 9), (16, 16), (100, 47)])
def test_context_positive_floor_holds_exactly_the_requested_positives(floor, expected):
    """0.47% of 2,000 is 9.4: the exact arm takes 9, a floor of 16 lifts it, a floor above the
    pool's 47 positives is capped at them, and every draw is a distinct-row uniform context
    whose count no longer depends on the seed."""
    y = _rare()
    counts = set()
    for seed in range(6):
        idx = _select_context(y, 2000, "uniform", np.random.default_rng(seed), positive_floor=floor)
        assert idx.shape == (2000,) and len(np.unique(idx)) == 2000
        assert np.all(np.diff(idx) > 0)
        counts.add(int(y[idx].sum()))
    assert counts == {expected}
    # the uniform draw it replaces leaves the count to chance
    plain = {int(y[_select_context(y, 2000, "uniform", np.random.default_rng(s))].sum())
             for s in range(6)}
    assert len(plain) > 1


def test_context_positive_floor_off_is_the_uniform_draw_byte_for_byte():
    y = _rare()
    for seed in range(3):
        legacy = np.sort(np.random.default_rng(seed).choice(len(y), size=2000, replace=False))
        got = _select_context(y, 2000, "uniform", np.random.default_rng(seed))
        off = _select_context(
            y, 2000, "uniform", np.random.default_rng(seed), positive_floor=None
        )
        assert got.tobytes() == legacy.tobytes() == off.tobytes()
    torch.manual_seed(0)
    model = _tiny_model()
    rng = np.random.default_rng(0)
    X, Xq = rng.normal(size=(400, 4)), rng.normal(size=(30, 4))
    y = _rare(400, 12)
    plain = FinancialTFMClassifier(model, max_context=100).fit(X, y)
    off = FinancialTFMClassifier(model, max_context=100, context_positive_floor=None).fit(X, y)
    assert plain._ctx_X.tobytes() == off._ctx_X.tobytes()
    assert plain.predict_proba(Xq).tobytes() == off.predict_proba(Xq).tobytes()


def test_context_positive_floor_reaches_the_stored_context_and_every_member():
    """Through fit, the summary reports the fixed count; ensemble members, label-swapped ones
    included, fix the count of the same rows under their own labels."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 4))
    y = _rare(2000, 10)  # 0.5%: round(200 x 0.005) = 1, the floor lifts it to 6
    clf = FinancialTFMClassifier(
        _tiny_model(), max_context=200, context_positive_floor=6, n_ensemble=3,
        ensemble_label_swap=True,
    ).fit(X, y)
    s = clf.context_summary_
    assert (s.n_context, s.n_positive) == (200, 6)
    assert s.context_rate == pytest.approx(0.03) and s.population_rate == pytest.approx(0.005)
    for seed in (1, 2):
        twin = clf._clone_with_seed(seed, clf._raw_X, clf._raw_y)
        swapped = clf._clone_with_seed(seed, clf._raw_X, clf._flip(clf._raw_y))
        assert twin.context_positive_floor == 6
        assert (twin._ctx_y == 1).sum() == 6
        # the swapped twin calls the defaults class 0; it still holds exactly 6 of them
        assert (swapped._ctx_y == 0).sum() == 6
    p = clf.predict_proba(rng.normal(size=(20, 4)))
    assert p.shape == (20, 2) and np.isfinite(p).all()


class _BaseRateReader(FinancialTFM):
    """An ideal in-context reader of the base rate: every query gets the context's class
    frequencies as its probabilities, whatever its features. What a calibrated in-context
    model does on a featureless task, and exactly what the D5 correction is meant to undo."""

    def predict_context(self, Xc, yc, Xq, *, n_classes=None, **kw):
        """Log context class frequencies, one identical row per query."""
        freq = np.bincount(np.asarray(yc), minlength=n_classes) / len(yc)
        logits = np.log(np.maximum(freq, 1e-12))
        return torch.tensor(np.tile(logits, (len(Xq), 1)), dtype=torch.float32)


def test_context_positive_floor_stays_calibrated_with_correct_prior():
    """A floor of 16 in a 400-row context over a 1% pool gives the context a 4% rate. The
    reader predicts 4% before the correction, and the correction, which reads the context's
    actual rate, brings the mean back to the pool's 1%."""
    cfg = ModelConfig(max_features=6, max_classes=2, d_model=16, n_heads=2, n_layers=1, d_ff=32)
    rng = np.random.default_rng(0)
    X, Xq = rng.normal(size=(5000, 4)), rng.normal(size=(200, 4))
    y = _rare(5000, 50)
    kw = {"max_context": 400, "context_positive_floor": 16}
    torch.manual_seed(0)
    fixed = FinancialTFMClassifier(_BaseRateReader(cfg), **kw).fit(X, y)
    raw = FinancialTFMClassifier(_BaseRateReader(cfg), correct_prior=False, **kw).fit(X, y)
    assert fixed.context_summary_.n_positive == 16
    assert raw.predict_proba(Xq)[:, 1].mean() == pytest.approx(16 / 400, rel=1e-5)
    assert fixed.predict_proba(Xq)[:, 1].mean() == pytest.approx(50 / 5000, rel=1e-5)
    # and on a real network the shift is the same log-ratio, leaving the ranking alone
    torch.manual_seed(0)
    model = _tiny_model()
    net = FinancialTFMClassifier(model, **kw).fit(X, y)
    net_raw = FinancialTFMClassifier(model, correct_prior=False, **kw).fit(X, y)
    np.testing.assert_allclose(
        net._log_prior_shift, [np.log(0.99 / 0.96), np.log(0.01 / 0.04)], rtol=1e-12
    )
    pc, pr = net.predict_proba(Xq)[:, 1], net_raw.predict_proba(Xq)[:, 1]
    np.testing.assert_array_equal(np.argsort(pc, kind="stable"), np.argsort(pr, kind="stable"))
    assert pc.mean() < pr.mean()


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"context_strategy": "balanced", "context_positive_floor": 0}, "uniform"),
        ({"context_strategy": "retrieval", "context_positive_floor": 4}, "uniform"),
        ({"context_positive_floor": -1}, ">= 0"),
    ],
)
def test_context_positive_floor_refuses_what_it_cannot_mean(kw, match):
    rng = np.random.default_rng(0)
    clf = FinancialTFMClassifier(_tiny_model(), max_context=40, **kw)
    with pytest.raises(ValueError, match=match):
        clf.fit(rng.normal(size=(100, 4)), _rare(100, 10))


def test_restrict_context_to_the_fitted_rows_is_the_fitted_classifier():
    """Restricting to the stored context's own rows changes nothing, bit for bit; a smaller
    restriction stores exactly those rows, corrects against the same population rate, and
    leaves the original untouched (DR3.4's cross-fitting builds on this)."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    X, Xq = rng.normal(size=(500, 4)), rng.normal(size=(25, 4))
    y = _rare(500, 40)
    clf = FinancialTFMClassifier(_tiny_model(), max_context=100).fit(X, y)
    before = clf.predict_proba(Xq)
    same = clf.restrict_context(clf.context_indices_)
    assert same.predict_proba(Xq).tobytes() == before.tobytes()
    assert same._ctx_X.tobytes() == clf._ctx_X.tobytes()
    assert same._log_prior_shift.tobytes() == clf._log_prior_shift.tobytes()
    rows = clf.context_indices_[::2]
    half = clf.restrict_context(rows[::-1])  # order and duplicates do not matter
    np.testing.assert_array_equal(half.context_indices_, rows)
    assert half._ctx_X.shape[0] == rows.size
    rate = y[rows].mean()
    assert half.context_summary_.context_rate == pytest.approx(rate)
    assert half.context_summary_.population_rate == pytest.approx(0.08)
    np.testing.assert_allclose(half._log_prior_shift[1], np.log(0.08 / rate), rtol=1e-12)
    assert not np.array_equal(half.predict_proba(Xq), before)
    assert clf.predict_proba(Xq).tobytes() == before.tobytes()
    assert half.transform_representation(Xq).shape == (25, 16)


@pytest.mark.parametrize(
    ("kw", "rows", "match"),
    [
        ({"context_strategy": "retrieval"}, [0, 1, 2], "retrieval"),
        ({"n_ensemble": 2}, [0, 1, 2], "ensemble"),
        ({}, [], "index"),
        ({}, [0, 10_000], "index"),
    ],
)
def test_restrict_context_refuses_what_it_cannot_restrict(kw, rows, match):
    rng = np.random.default_rng(0)
    clf = FinancialTFMClassifier(_tiny_model(), max_context=40, **kw)
    clf.fit(rng.normal(size=(100, 4)), _rare(100, 10))
    with pytest.raises(ValueError, match=match):
        clf.restrict_context(np.array(rows, dtype=np.int64))


def test_context_positive_floor_is_inert_when_the_pool_fits():
    y = _rare(300, 3)
    idx = _select_context(y, 2000, "uniform", np.random.default_rng(0), positive_floor=16)
    np.testing.assert_array_equal(idx, np.arange(300))
