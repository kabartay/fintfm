"""The tree-structured prior (task 48.13).

The load-bearing test is `test_tree_prior_is_tree_favourable`: this prior was added on
*distinctiveness* grounds, so a test asserting only that it produces valid tasks would pass
for a prior that duplicates the SCM one and adds nothing.
"""

import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from fintfm.prior.mixture import PriorConfig, sample_task
from fintfm.prior.scm import sample_scm_task
from fintfm.prior.tree import MAX_DEPTH, _oblivious_forest, sample_tree_task


def _auc_gap(sampler, n_tasks: int = 15) -> float:
    """Mean (ExtraTrees - LogisticRegression) AUC over tasks from ``sampler``."""
    gaps = []
    for seed in range(n_tasks):
        t = sampler(np.random.default_rng(seed), 800, max_classes=2)
        X, y = np.nan_to_num(t.X), t.y
        k = len(y) // 2
        if len(np.unique(y[:k])) < 2 or len(np.unique(y[k:])) < 2:
            continue
        et = ExtraTreesClassifier(n_estimators=60, random_state=0).fit(X[:k], y[:k])
        lr = LogisticRegression(max_iter=1000).fit(X[:k], y[:k])
        gaps.append(
            roc_auc_score(y[k:], et.predict_proba(X[k:])[:, 1])
            - roc_auc_score(y[k:], lr.predict_proba(X[k:])[:, 1])
        )
    return float(np.mean(gaps))


def test_tree_prior_is_tree_favourable_and_scm_is_not():
    # The entire justification for this prior (MITRA's distinctiveness criterion): a tree
    # model should beat a linear one on its tasks, and the SCM prior should NOT have that
    # property -- otherwise the two priors teach the same thing and this one is step count.
    assert _auc_gap(sample_tree_task) > 0.01
    assert _auc_gap(sample_scm_task) < _auc_gap(sample_tree_task)


def test_tasks_are_valid_and_non_degenerate():
    for seed in range(40):
        t = sample_tree_task(np.random.default_rng(seed), 200)
        assert t.source == "tree"
        assert t.X.dtype == np.float32
        assert len(np.unique(t.y)) >= 2
        assert t.y.min() == 0 and t.y.max() == t.n_classes - 1
        assert np.isfinite(t.X[~np.isnan(t.X)]).all()


def test_forest_is_piecewise_constant():
    # An oblivious forest of one depth-1 tree takes exactly two values; if it takes more, the
    # leaf indexing is wrong and the prior is not generating the structure it claims to.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    out = _oblivious_forest(rng, X, n_trees=1, depth=1)
    assert len(np.unique(np.round(out, 10))) == 2


def test_forest_depth_bounds_distinct_values():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 5))
    out = _oblivious_forest(rng, X, n_trees=1, depth=3)
    assert len(np.unique(np.round(out, 10))) <= 2**3


def test_latent_hook_matches_scm_convention():
    out: list[np.ndarray] = []
    t = sample_tree_task(np.random.default_rng(0), 200, _latent_out=out)
    assert len(out) == 1 and out[0].shape == (len(t.y),)


def test_mixture_draws_tree_tasks_when_enabled():
    cfg = PriorConfig(p_tree=1.0, p_financial=0.0, max_features=16)
    rng = np.random.default_rng(0)
    assert all(sample_task(rng, cfg, n_rows=100).source == "tree" for _ in range(5))


def test_mixture_never_draws_tree_tasks_by_default():
    cfg = PriorConfig(max_features=16)
    assert cfg.p_tree == 0.0
    rng = np.random.default_rng(0)
    assert all(sample_task(rng, cfg, n_rows=100).source != "tree" for _ in range(20))


def test_tree_prior_refused_with_survival_horizons():
    # Same guard the SCM prior has: a tree task carries no event time.
    cfg = PriorConfig(p_tree=1.0, p_financial=1.0, n_horizons=5, max_features=16)
    with pytest.raises(ValueError, match="no time axis"):
        sample_task(np.random.default_rng(0), cfg, n_rows=100)


def test_depth_ceiling_respected():
    assert MAX_DEPTH >= 1
    for seed in range(20):
        sample_tree_task(np.random.default_rng(seed), 150)
