"""The prior-scoring harness (task 48.14).

The tests that matter here are the calibration ones: a harness whose numbers nobody has
checked against a prior of *known* shape cannot be trusted on a prior of unknown shape. §112
is the record of what happens when this instrument is wrong in a plausible-looking way.
"""

import numpy as np

from fintfm.experiments.prior_score import PRIORS, _load, score_prior, summarise
from fintfm.prior.scm import sample_scm_task
from fintfm.prior.tree import sample_tree_task
from fintfm.prior.trivial import sample_trivial_task


def test_trivial_prior_scores_as_easy_and_undiverse():
    # The calibration check. A prior built to be trivial must come back near-perfect and
    # near-zero-variance, or every other number this harness produces is unreadable.
    s = score_prior(sample_trivial_task, n_tasks=8, n_rows=400)
    assert s["performance"] > 0.95
    assert s["diversity"] < 0.05


def test_tree_prior_is_the_tree_favourable_one():
    # §112's corrected claim, asserted directly: tree is positive, scm is not meaningfully so.
    tree = score_prior(sample_tree_task, n_tasks=12, n_rows=600)
    scm = score_prior(sample_scm_task, n_tasks=12, n_rows=600)
    assert tree["distinctiveness"] > scm["distinctiveness"]
    assert tree["distinctiveness"] > 0.0


def test_linear_baseline_is_conditioned():
    # §112: an unconditioned linear arm inflates distinctiveness on heavy-tailed features. A
    # feature scaled by 1e6 must not change the verdict, and would if the baseline were raw.
    def wild(rng, n_rows, **kw):
        t = sample_tree_task(rng, n_rows, **kw)
        t.X[:, 0] = t.X[:, 0] * 1e6
        return t

    plain = score_prior(sample_tree_task, n_tasks=8, n_rows=500)
    scaled = score_prior(wild, n_tasks=8, n_rows=500)
    assert abs(plain["linear_auc"] - scaled["linear_auc"]) < 0.05


def test_unscorable_tasks_are_skipped_not_counted_as_zero():
    # A single-class split is unscorable. Recording it as a low AUC would make an imbalanced
    # prior look like a weak one, which is a distinction §43 exists to preserve.
    def degenerate(rng, n_rows, **kw):
        t = sample_tree_task(rng, n_rows, **kw)
        t.y[:] = 0
        return t

    s = score_prior(degenerate, n_tasks=5, n_rows=300)
    assert s["n_scored"] == 0
    assert np.isnan(s["performance"])


def test_every_registered_prior_loads_and_scores():
    for name, spec in PRIORS.items():
        s = score_prior(_load(spec), n_tasks=3, n_rows=300)
        assert s["n_scored"] >= 0, name


def test_summarise_renders_all_priors():
    s = {"tree": score_prior(sample_tree_task, n_tasks=3, n_rows=300)}
    text = summarise(s)
    assert "tree" in text and "distinct" in text
