"""Retrieved contexts, and the invariant retrieval knowingly gives up.

`docs/FINDINGS.md` §31 established that 84% of the out-of-time gap is horizon-independent and
cannot be closed by supplying more context rows, which leaves *which* rows as the only lever.
These tests guard the mechanism and, as importantly, pin down the batch-independence property
that retrieval trades away — an invariant that stops holding silently is worse than one that
never held.
"""

import numpy as np
import pytest
import torch

from fintfm.inference.classifier import FinancialTFMClassifier
from fintfm.inference.retrieval import (
    distance_stats,
    group_queries,
    normalise_for_distance,
    prototype_context,
    retrieve,
)
from fintfm.modeling.model import FinancialTFM, ModelConfig


def _clustered(n_per=200, seed=0):
    """Three well-separated clusters, so 'nearest' has an unambiguous meaning."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [30.0, 30.0], [-30.0, 25.0]], dtype=np.float32)
    X = np.concatenate([c + rng.normal(scale=1.0, size=(n_per, 2)) for c in centres])
    label = np.repeat(np.arange(3), n_per)
    return X.astype(np.float32), label


def test_normalisation_puts_wildly_scaled_features_on_one_footing():
    X = np.array([[1.0, 1e6], [2.0, 2e6], [3.0, 3e6]], dtype=np.float32)
    c, s = distance_stats(X)
    Z = normalise_for_distance(X, c, s)
    assert Z.std(axis=0)[0] == pytest.approx(Z.std(axis=0)[1], rel=0.05)


def test_missing_features_become_distance_neutral_rather_than_fatal():
    X = np.array([[1.0, np.nan], [2.0, 5.0], [3.0, 7.0]], dtype=np.float32)
    c, s = distance_stats(X)
    Z = normalise_for_distance(X, c, s)
    assert np.isfinite(Z).all()
    assert Z[0, 1] == pytest.approx(0.0)


def test_retrieved_context_is_nearer_than_a_uniform_sample():
    """Task 17.1's stated verification, and the whole premise of the change."""
    X, _ = _clustered()
    y = np.zeros(len(X), dtype=np.int64)
    c, s = distance_stats(X)
    Z = normalise_for_distance(X, c, s)
    rng = np.random.default_rng(0)
    target = Z[0]
    got = retrieve(Z, y, target, max_context=50)
    unif = rng.choice(len(Z), size=50, replace=False)
    assert np.linalg.norm(Z[got] - target, axis=1).mean() < np.linalg.norm(
        Z[unif] - target, axis=1
    ).mean()


def test_retrieval_prefers_the_query_own_cluster():
    X, label = _clustered()
    y = np.zeros(len(X), dtype=np.int64)
    c, s = distance_stats(X)
    Z = normalise_for_distance(X, c, s)
    got = retrieve(Z, y, Z[0], max_context=100)
    assert (label[got] == label[0]).mean() > 0.95


def test_min_positive_rescues_a_context_with_no_defaults():
    """A nearest draw in a low-default book can return zero positives, which is useless."""
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(2000, 4)).astype(np.float32)
    y = np.zeros(2000, dtype=np.int64)
    y[:3] = 1
    Z[:3] += 50.0  # positives are far from the target, so a plain draw misses them
    target = np.zeros(4, dtype=np.float32)
    assert (y[retrieve(Z, y, target, 100, min_positive=0)] == 1).sum() == 0
    got = retrieve(Z, y, target, 100, min_positive=2)
    assert (y[got] == 1).sum() >= 2
    assert got.size == 100  # the floor must not silently change the context size


def test_grouping_covers_every_query_exactly_once():
    X, _ = _clustered(n_per=150)
    c, s = distance_stats(X)
    Z = normalise_for_distance(X, c, s)
    groups = group_queries(Z, n_groups=8, max_group=64, rng=np.random.default_rng(0))
    covered = np.sort(np.concatenate(groups))
    np.testing.assert_array_equal(covered, np.arange(len(Z)))
    assert max(g.size for g in groups) <= 64


def test_grouping_respects_the_size_cap_even_with_one_dense_cluster():
    rng = np.random.default_rng(2)
    Z = np.concatenate(
        [rng.normal(scale=0.01, size=(500, 3)), rng.normal(loc=50, size=(5, 3))]
    ).astype(np.float32)
    groups = group_queries(Z, n_groups=2, max_group=32, rng=np.random.default_rng(0))
    assert max(g.size for g in groups) <= 32
    np.testing.assert_array_equal(np.sort(np.concatenate(groups)), np.arange(len(Z)))


def _assert_same_ranking(a: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> None:
    """Assert two score vectors rank rows identically, tolerating exact ties.

    A constant log-odds shift is rank-preserving in exact arithmetic, but in float32 it can
    push two nearby values onto the same representable number near saturation — and once two
    values tie, ``argsort`` picks between them arbitrarily. Comparing ``argsort`` directly
    therefore fails intermittently on tied pairs while the property under test still holds.
    So compare the *orderings of pairs that are actually distinguishable*.
    """
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if abs(a[i] - a[j]) > tol and abs(b[i] - b[j]) > tol:
                assert (a[i] < a[j]) == (b[i] < b[j]), (
                    f"rows {i} and {j} swapped order: {a[i]},{a[j]} vs {b[i]},{b[j]}"
                )


def _small_model(n_features=4, n_horizons=None):
    # seeded, because these weights are random and every test below compares two
    # predictions from the *same* model: without a seed the suite's outcome depends on
    # which other test files ran first, which is how a passing file failed in the suite
    torch.manual_seed(0)
    return FinancialTFM(
        ModelConfig(
            max_features=n_features, d_model=32, d_cell=16, n_layers=1, n_col_layers=1,
            max_classes=2, **({"n_horizons": n_horizons} if n_horizons else {}),
        )
    )


def test_retrieval_returns_a_prediction_for_every_row_in_order():
    X, label = _clustered(n_per=60)
    X = np.concatenate([X, X[:, :1] * 0.5, X[:, 1:] * -1], axis=1).astype(np.float32)
    y = (label == 2).astype(np.int64)
    clf = FinancialTFMClassifier(
        _small_model(), max_context=40, context_strategy="retrieval",
        retrieval_groups=4, retrieval_min_positive=2,
    ).fit(X, y)
    p = clf.predict_proba(X)
    assert p.shape == (len(X), 2)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_retrieval_term_structure_stays_coherent():
    X, label = _clustered(n_per=60)
    X = np.concatenate([X, X[:, :1] * 0.5, X[:, 1:] * -1], axis=1).astype(np.float32)
    y = (label == 2).astype(np.int64)
    clf = FinancialTFMClassifier(
        _small_model(n_horizons=4), max_context=40, context_strategy="retrieval",
        retrieval_groups=4, retrieval_min_positive=2,
    ).fit(X, y)
    curve = clf.predict_term_structure(X)
    assert curve.shape == (len(X), 4)
    assert (np.diff(curve, axis=1) >= -1e-6).all()


def test_exact_per_query_retrieval_is_available_and_differs_from_grouped():
    """Both modes exist so the approximation can be measured rather than assumed small."""
    X, label = _clustered(n_per=25)
    X = np.concatenate([X, X[:, :1] * 0.5, X[:, 1:] * -1], axis=1).astype(np.float32)
    y = (label == 2).astype(np.int64)
    kw = {"max_context": 30, "context_strategy": "retrieval", "retrieval_min_positive": 2}
    model = _small_model()
    exact = FinancialTFMClassifier(model, retrieval_groups=0, **kw).fit(X, y)
    grouped = FinancialTFMClassifier(model, retrieval_groups=3, **kw).fit(X, y)
    pe, pg = exact.predict_proba(X), grouped.predict_proba(X)
    assert pe.shape == pg.shape
    assert np.isfinite(pe).all()
    # they must not be identical: if they were, grouping would be doing nothing
    assert not np.allclose(pe, pg, atol=1e-4)


def test_blind_strategies_keep_batch_independence_and_retrieval_does_not():
    """The invariant retrieval trades away, pinned in both directions.

    Chunked scoring is exact for the blind strategies because the row mask forbids
    query-to-query attention. Retrieval breaks that deliberately — a query's context depends
    on its group-mates — and a test asserting so is what keeps the trade visible.
    """
    X, label = _clustered(n_per=40)
    X = np.concatenate([X, X[:, :1] * 0.5, X[:, 1:] * -1], axis=1).astype(np.float32)
    y = (label == 2).astype(np.int64)
    model = _small_model()

    blind_a = FinancialTFMClassifier(model, max_context=30, context_strategy="uniform",
                                     query_chunk=4096).fit(X, y)
    blind_b = FinancialTFMClassifier(model, max_context=30, context_strategy="uniform",
                                     query_chunk=7).fit(X, y)
    np.testing.assert_allclose(blind_a.predict_proba(X), blind_b.predict_proba(X), atol=1e-5)

    kw = {"max_context": 30, "context_strategy": "retrieval", "retrieval_min_positive": 2}
    all_rows = FinancialTFMClassifier(model, retrieval_groups=4, **kw).fit(X, y)
    subset = FinancialTFMClassifier(model, retrieval_groups=4, **kw).fit(X, y)
    full = all_rows.predict_proba(X)[:20]
    alone = subset.predict_proba(X[:20])
    assert not np.allclose(full, alone, atol=1e-4), (
        "retrieval is expected to be batch-dependent; if this passes, grouping changed"
    )


def test_retrieval_correction_is_one_global_shift_not_per_group():
    """§32: a per-group correction inverted the ranking and drove AUC below chance.

    The correction is exact under label shift, which holds by construction for the blind
    strategies because they select on ``y`` alone. Retrieval selects on ``x``, so applying a
    per-group shift erases the between-group risk differences retrieval exists to find. A
    single pooled shift is constant across queries and therefore cannot reorder anything —
    which is the property this test pins.
    """
    X, label = _clustered(n_per=50)
    X = np.concatenate([X, X[:, :1] * 0.5, X[:, 1:] * -1], axis=1).astype(np.float32)
    y = (label == 2).astype(np.int64)
    kw = {"max_context": 40, "context_strategy": "retrieval", "retrieval_groups": 3,
          "retrieval_min_positive": 2}
    model = _small_model()
    on = FinancialTFMClassifier(model, correct_prior=True, **kw).fit(X, y)
    off = FinancialTFMClassifier(model, correct_prior=False, **kw).fit(X, y)
    p_on, p_off = on.predict_proba(X)[:, 1], off.predict_proba(X)[:, 1]
    # levels may move; the ordering must not
    _assert_same_ranking(p_on, p_off)


def test_retrieval_term_structure_correction_preserves_ranking_too():
    X, label = _clustered(n_per=50)
    X = np.concatenate([X, X[:, :1] * 0.5, X[:, 1:] * -1], axis=1).astype(np.float32)
    y = (label == 2).astype(np.int64)
    kw = {"max_context": 40, "context_strategy": "retrieval", "retrieval_groups": 3,
          "retrieval_min_positive": 2}
    model = _small_model(n_horizons=4)
    on = FinancialTFMClassifier(model, correct_prior=True, **kw).fit(X, y)
    off = FinancialTFMClassifier(model, correct_prior=False, **kw).fit(X, y)
    c_on, c_off = on.predict_term_structure(X), off.predict_term_structure(X)
    for k in range(c_on.shape[1]):
        _assert_same_ranking(c_on[:, k], c_off[:, k])
    assert (np.diff(c_on, axis=1) >= -1e-6).all()


def test_pooled_context_rate_weights_by_queries_served():
    X, label = _clustered(n_per=40)
    X = np.concatenate([X, X[:, :1] * 0.5, X[:, 1:] * -1], axis=1).astype(np.float32)
    y = (label == 2).astype(np.int64)
    clf = FinancialTFMClassifier(
        _small_model(), max_context=30, context_strategy="retrieval",
        retrieval_groups=4, retrieval_min_positive=1,
    ).fit(X, y)
    plan = clf._retrieval_plan(X)
    rate = clf._pooled_context_rate(plan)
    assert 0.0 <= rate <= 1.0
    manual = sum(q.size * (y[c] == 1).mean() for q, c in plan) / sum(q.size for q, _ in plan)
    assert rate == pytest.approx(manual)


# --- prototype undersampling, the published comparison (§36) -----------------------


def test_prototype_context_respects_the_budget_and_keeps_the_ratio():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(5000, 6)).astype(np.float32)
    y = np.zeros(5000, dtype=np.int64)
    y[:200] = 1
    got = prototype_context(Z, y, max_context=520, minority_ratio=0.3, rng=rng)
    assert got.size <= 520
    n_pos = int((y[got] == 1).sum())
    n_neg = got.size - n_pos
    assert 0.2 < n_pos / n_neg < 0.45, (n_pos, n_neg)


def test_prototype_context_picks_real_rows_not_centroids():
    """A synthetic mean row is not a company; the context must be actual observations."""
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(800, 4)).astype(np.float32)
    y = np.zeros(800, dtype=np.int64)
    y[:40] = 1
    got = prototype_context(Z, y, max_context=200, minority_ratio=0.3, rng=rng)
    assert got.dtype.kind == "i"
    assert got.max() < 800 and got.min() >= 0
    assert len(set(got.tolist())) == got.size  # no duplicated rows


def test_prototype_context_spans_the_majority_clusters():
    """The point of clustering is coverage: prototypes must not all come from one blob."""
    rng = np.random.default_rng(2)
    blobs = [np.full((300, 2), c, dtype=np.float32) + rng.normal(scale=0.2, size=(300, 2))
             for c in (-20.0, 0.0, 20.0)]
    Z = np.concatenate(blobs).astype(np.float32)
    y = np.zeros(len(Z), dtype=np.int64)
    y[:30] = 1  # minority inside the first blob only
    got = prototype_context(Z, y, max_context=60, minority_ratio=0.3, rng=rng)
    majority = got[y[got] == 0]
    which = np.digitize(Z[majority, 0], [-10.0, 10.0])
    assert len(np.unique(which)) == 3, "prototypes drawn from fewer than all three clusters"


def test_prototype_context_handles_a_single_class():
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(200, 3)).astype(np.float32)
    y = np.zeros(200, dtype=np.int64)
    got = prototype_context(Z, y, max_context=50, minority_ratio=0.3, rng=rng)
    assert got.size == 50


def test_prototype_strategy_keeps_batch_independence():
    """It is blind, so unlike retrieval it must give identical predictions when chunked."""
    X, label = _clustered(n_per=60)
    X = np.concatenate([X, X[:, :1] * 0.5, X[:, 1:] * -1], axis=1).astype(np.float32)
    y = (label == 2).astype(np.int64)
    model = _small_model()
    kw = {"max_context": 60, "context_strategy": "prototype"}
    whole = FinancialTFMClassifier(model, query_chunk=4096, **kw).fit(X, y)
    split = FinancialTFMClassifier(model, query_chunk=13, **kw).fit(X, y)
    np.testing.assert_allclose(whole.predict_proba(X), split.predict_proba(X), atol=1e-5)
