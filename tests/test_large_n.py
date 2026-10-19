"""Large-table wrappers (audit AU5.8): a tree partition and boosting around the classifier.

These are experiment levers with no measured effect yet, so the tests pin down mechanism
only: each wrapper returns valid probabilities, is reproducible at a fixed seed, routes or
samples the way its docstring says, and the configuration's default wraps nothing at all.
"""

import numpy as np
import pytest
import torch

from fintfm.config import LargeNConfig
from fintfm.inference.classifier import FinancialTFMClassifier
from fintfm.inference.large_n import (
    BoostedTFMClassifier,
    PartitionedTFMClassifier,
    wrap_large_n,
)
from fintfm.modeling.model import FinancialTFM, ModelConfig


def _model() -> FinancialTFM:
    # seeded: every test compares predictions of one model, and the weights are random
    torch.manual_seed(0)
    cfg = ModelConfig(max_features=6, max_classes=3, d_model=16, n_heads=2, n_layers=1, d_ff=32)
    return FinancialTFM(cfg)


def _data(n=600, seed=0, noise=1.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = (X[:, 0] + noise * rng.normal(size=n) > 1.0).astype(np.int64)
    return X, y


def _base(model=None, **kw) -> FinancialTFMClassifier:
    return FinancialTFMClassifier(model or _model(), max_context=80, **kw)


def _valid(p, n, k=2):
    assert p.shape == (n, k)
    assert np.isfinite(p).all()
    assert (p >= 0).all() and (p <= 1).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)


# --- partition ---------------------------------------------------------------------------


def test_partition_fits_predicts_and_is_deterministic():
    X, y = _data()
    model = _model()
    kw = {"max_depth": 2, "min_samples_leaf": 0.1, "min_leaf_rows": 20, "random_state": 3}
    a = PartitionedTFMClassifier(_base(model), **kw).fit(X, y)
    b = PartitionedTFMClassifier(_base(model), **kw).fit(X, y)
    pa = a.predict_proba(X[:50])
    _valid(pa, 50)
    assert pa.tobytes() == b.predict_proba(X[:50]).tobytes()
    assert set(a.predict(X[:50])) <= set(a.classes_)
    # at least one leaf really got its own context, or the test measures only the fallback
    assert any(e is not None for e in a.leaf_estimators_[0].values())


def test_partition_routes_each_query_to_its_own_leaf_context():
    X, y = _data()
    clf = PartitionedTFMClassifier(
        _base(), max_depth=2, min_samples_leaf=0.1, min_leaf_rows=20
    ).fit(X, y)
    Xq = X[:60]
    routes = clf.route(Xq)
    np.testing.assert_array_equal(routes[0], clf.trees_[0].apply(Xq))
    p = clf.predict_proba(Xq)
    for leaf in np.unique(routes[0]):
        rows = np.flatnonzero(routes[0] == leaf)
        est = clf.leaf_estimators_[0][int(leaf)] or clf.global_estimator_
        np.testing.assert_allclose(p[rows], est.predict_proba(Xq[rows]), atol=1e-6)
        if clf.leaf_estimators_[0][int(leaf)] is not None:
            # the leaf's context is drawn from that leaf's training rows and no other
            members = np.flatnonzero(clf.trees_[0].apply(X) == leaf)
            assert len(est._raw_y) == members.size


def test_partition_small_or_single_class_leaves_fall_back_to_the_global_context():
    # noise 0: the tree's first split on x0 separates the classes, so leaves come out pure
    X, y = _data(noise=0.0)
    clf = PartitionedTFMClassifier(
        _base(), max_depth=2, min_samples_leaf=0.05, min_leaf_rows=20
    ).fit(X, y)
    leaf_of = clf.trees_[0].apply(X)
    for leaf, est in clf.leaf_estimators_[0].items():
        if len(np.unique(y[leaf_of == leaf])) < 2:
            assert est is None
    assert any(e is None for e in clf.leaf_estimators_[0].values())
    tiny = PartitionedTFMClassifier(_base(), min_samples_leaf=0.1, min_leaf_rows=10_000).fit(X, y)
    assert all(e is None for e in tiny.leaf_estimators_[0].values())
    np.testing.assert_allclose(
        tiny.predict_proba(X[:30]), tiny.global_estimator_.predict_proba(X[:30]), atol=1e-6
    )


def test_partition_bags_draw_distinct_subsets_and_average_them():
    X, y = _data()
    clf = PartitionedTFMClassifier(
        _base(), n_bags=3, bag_frac=0.6, min_samples_leaf=0.1, min_leaf_rows=20
    ).fit(X, y)
    assert [r.size for r in clf.bag_rows_] == [360, 360, 360]
    assert not np.array_equal(clf.bag_rows_[0], clf.bag_rows_[1])
    Xq = X[:40]
    routes = clf.route(Xq)
    assert routes.shape == (3, 40)
    per_bag = []
    for b, leaves in enumerate(clf.leaf_estimators_):
        out = np.zeros((40, 2))
        for leaf in np.unique(routes[b]):
            rows = np.flatnonzero(routes[b] == leaf)
            out[rows] = (leaves[int(leaf)] or clf.global_estimator_).predict_proba(Xq[rows])
        per_bag.append(out)
    np.testing.assert_allclose(clf.predict_proba(Xq), np.mean(per_bag, axis=0), atol=1e-6)


def test_partition_leaf_contexts_carry_the_fixed_positive_count():
    """DR3.3 through the partition: every leaf larger than ``max_context``, and the global
    fallback, holds exactly ``max(floor, round(max_context x leaf rate))`` positives of its
    own rows, and its base-rate correction reads that context against the leaf's own rate."""
    X, y = _data(n=2000)
    clf = PartitionedTFMClassifier(
        _base(context_positive_floor=10), max_depth=2, min_samples_leaf=0.1, min_leaf_rows=20
    ).fit(X, y)
    inner = [clf.global_estimator_] + [e for e in clf.leaf_estimators_[0].values() if e]
    checked = 0
    for est in inner:
        assert est.context_positive_floor == 10
        n, pos = est._raw_y.size, int(est._raw_y.sum())
        if n <= est.max_context:
            continue
        want = min(pos, max(10, round(est.max_context * pos / n)))
        s = est.context_summary_
        assert (s.n_context, s.n_positive) == (est.max_context, want)
        assert s.population_rate == pytest.approx(pos / n)
        checked += 1
    assert checked >= 2  # the global context and at least one leaf
    _valid(clf.predict_proba(X[:30]), 30)


@pytest.mark.parametrize("kw", [{"max_depth": 4}, {"n_bags": 0}, {"bag_frac": 0.0}])
def test_partition_refuses_out_of_range_settings(kw):
    X, y = _data(n=100)
    with pytest.raises(ValueError):
        PartitionedTFMClassifier(_base(), **kw).fit(X, y)


# --- boosting ----------------------------------------------------------------------------


def test_boosting_fits_predicts_and_is_deterministic():
    X, y = _data()
    model = _model()
    kw = {"n_rounds": 3, "context_rows": 64, "random_state": 5}
    a = BoostedTFMClassifier(_base(model), **kw).fit(X, y)
    b = BoostedTFMClassifier(_base(model), **kw).fit(X, y)
    pa = a.predict_proba(X[:50])
    _valid(pa, 50)
    assert pa.tobytes() == b.predict_proba(X[:50]).tobytes()
    assert len(a.estimators_) == len(a.gammas_) == len(a.contexts_) == 3
    assert all(c.size == 64 for c in a.contexts_)
    assert all(0.0 <= g <= a.max_step for g in a.gammas_)
    c = BoostedTFMClassifier(_base(model), **{**kw, "random_state": 6}).fit(X, y)
    assert not all(np.array_equal(u, v) for u, v in zip(a.contexts_, c.contexts_, strict=True))


def test_boosting_score_is_the_prior_plus_the_stepped_evidence():
    X, y = _data()
    clf = BoostedTFMClassifier(_base(), n_rounds=2, context_rows=64, learning_rate=0.3).fit(X, y)
    assert clf.init_logit_ == pytest.approx(np.log(y.mean() / (1 - y.mean())))
    Xq = X[:20]
    want = np.full(20, clf.init_logit_)
    for est, g, ctx in zip(clf.estimators_, clf.gammas_, clf.contexts_, strict=True):
        p = np.clip(est.predict_proba(Xq)[:, 1].astype(np.float64), 1e-6, 1 - 1e-6)
        r = y[ctx].mean()
        want += 0.3 * g * (np.log(p / (1 - p)) - np.log(r / (1 - r)))
    np.testing.assert_allclose(clf.decision_function(Xq), want, atol=1e-6)


def test_sampling_weights_follow_the_residual():
    clf = BoostedTFMClassifier(_base(), weighting="abs")
    r = np.array([0.9, -0.1, 0.0, -0.5])
    w = clf.sampling_weights(r)
    assert w.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(w[[0, 1, 3]] / w[0], [1.0, 0.1 / 0.9, 0.5 / 0.9], rtol=1e-6)
    assert 0 < w[2] < 1e-9  # a fitted row stays drawable, barely
    e = BoostedTFMClassifier(_base(), weighting="exp").sampling_weights(r)
    np.testing.assert_allclose(e / e[0], np.exp(np.abs(r)) / np.exp(0.9), rtol=1e-6)
    with pytest.raises(ValueError, match="weighting"):
        BoostedTFMClassifier(_base(), weighting="square").sampling_weights(r)


def test_heavy_rows_dominate_the_draw_and_both_classes_are_forced_in():
    clf = BoostedTFMClassifier(_base(), context_rows=10)
    y = np.zeros(200, dtype=np.int64)
    y[:5] = 1
    prob = np.full(200, 1e-9)
    prob[10:30] = 1.0  # twenty heavy negatives, every positive nearly weightless
    prob /= prob.sum()
    picked = clf._draw_context(y, prob, np.random.default_rng(0))
    assert picked.size == 10 and len(np.unique(picked)) == 10
    assert y[picked].sum() == 1  # one positive forced in for a missing class
    assert np.isin(picked[y[picked] == 0], np.arange(10, 30)).all()


def test_line_search_takes_a_step_only_where_the_evidence_helps():
    clf = BoostedTFMClassifier(_base())
    y = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    f = np.zeros(6)
    good = np.array([2.0, 2.0, -2.0, -2.0, -2.0, -2.0])
    assert clf._line_search(f, good, y) > 0.0
    assert clf._line_search(f, -good, y) == 0.0
    assert clf._line_search(f, np.zeros(6), y) == 0.0


def test_boosting_pool_caps_the_residual_rows_and_contexts_come_from_it():
    X, y = _data()
    clf = BoostedTFMClassifier(_base(), n_rounds=2, context_rows=32, pool_rows=200).fit(X, y)
    assert clf.pool_.size == 200
    for ctx in clf.contexts_:
        assert np.isin(ctx, clf.pool_).all()


def test_boosting_refuses_a_multiclass_task():
    X, _ = _data(n=90)
    y = np.repeat([0, 1, 2], 30)
    with pytest.raises(ValueError, match="binary"):
        BoostedTFMClassifier(_base()).fit(X, y)


# --- configuration -----------------------------------------------------------------------


def test_the_default_configuration_wraps_nothing():
    """``large_n.method: none`` must hand back the very classifier object, so the
    fintfm-v4protocol arm is the plain classifier, byte for byte, unless asked otherwise."""
    base = _base()
    assert wrap_large_n(base, LargeNConfig()) is base


def test_configuration_reaches_the_wrapper_arguments():
    base = _base(random_state=4)
    part = wrap_large_n(base, LargeNConfig(method="partition", partition_n_bags=3))
    assert isinstance(part, PartitionedTFMClassifier)
    assert part.n_bags == 3 and part.estimator is base and part.random_state == 4
    boost = wrap_large_n(base, LargeNConfig(method="boost", boost_pool_rows=500))
    assert isinstance(boost, BoostedTFMClassifier)
    assert boost.pool_rows == 500 and boost.n_rounds == LargeNConfig().boost_n_rounds
    with pytest.raises(ValueError, match="large_n.method"):
        wrap_large_n(base, LargeNConfig(method="forest"))
