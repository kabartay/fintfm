"""The labelled task-family generators (task 40.2, `mechanism-diverse-prior`).

The load-bearing test is `test_realised_difficulty_matches_target`: task 40.2 requires each
family's *realised* Bayes AUC to be measured against its requested target, not assumed correct
because the construction looks right on paper -- exactly the discipline `CLAUDE.md`'s "Claims:
label every number by how it was produced" names.

Two measurement routes, matched to what each group of families actually admits:

- The five closed-form families (`linear`, `sparse`, `dense`, `threshold`, `latent_factor`) are
  linearly (or, for `threshold`, monotonically-then-linearly) separable by construction, so a
  plain `LogisticRegression` on the observed features recovers their Bayes AUC almost exactly --
  a flexible learner is not needed and would only add its own approximation error.
- The four families with no closed form (`xor`, `interaction`, `max_min`, `piecewise`) are
  genuinely hard to recover from raw features by design -- that difficulty is the whole point
  of including them -- so a generic learner cannot be used as the yardstick (a gradient-boosted
  tree recovers barely half the requested separation on `interaction` at n=15,000, not because
  the generator is wrong but because that is what "no linear or shallow-tree shortcut" means).
  These are measured instead against their own true, by-construction decision statistic,
  exposed via each sampler's `_score_out` hook (mirrors `scm.sample_scm_task`'s `_latent_out`
  convention) -- computed on a large, freshly-drawn sample, independent of whatever internal
  calibration sample the generator's own bisection used.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from fintfm.prior.task_families import (
    FAMILIES,
    sample_family_task,
    sample_interaction_order_task,
    sample_interaction_task,
    sample_max_min_task,
    sample_piecewise_task,
    sample_xor_task,
)

_TARGETS = (0.6, 0.75, 0.9, 0.98)
_TOL = 0.03
_N_MEASURE = 20_000

_CLOSED_FORM = ("linear", "sparse", "dense", "threshold", "latent_factor")
_NO_CLOSED_FORM = {
    "xor": sample_xor_task,
    "interaction": sample_interaction_task,
    "max_min": sample_max_min_task,
    "piecewise": sample_piecewise_task,
}
assert set(_CLOSED_FORM) | set(_NO_CLOSED_FORM) == set(FAMILIES)


def _measure_closed_form_auc(family: str, target_auc: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    t = sample_family_task(rng, family, _N_MEASURE, n_features=12, target_auc=target_auc)
    X, y = np.nan_to_num(t.X), t.y
    cut = len(y) // 2
    lr = LogisticRegression(max_iter=2000).fit(X[:cut], y[:cut])
    return float(roc_auc_score(y[cut:], lr.predict_proba(X[cut:])[:, 1]))


@pytest.mark.parametrize("family", _CLOSED_FORM)
@pytest.mark.parametrize("target_auc", _TARGETS)
def test_closed_form_families_hit_target(family: str, target_auc: float) -> None:
    measured = _measure_closed_form_auc(
        family, target_auc, seed=hash((family, target_auc)) % 2**31
    )
    assert abs(measured - target_auc) < _TOL, (
        f"{family} at target {target_auc}: measured {measured:.3f}, outside tolerance {_TOL}"
    )


@pytest.mark.parametrize("family", sorted(_NO_CLOSED_FORM))
@pytest.mark.parametrize("target_auc", _TARGETS)
def test_calibrated_families_hit_target(family: str, target_auc: float) -> None:
    rng = np.random.default_rng(hash((family, target_auc)) % 2**31)
    score_out: list[np.ndarray] = []
    t = _NO_CLOSED_FORM[family](
        rng, _N_MEASURE, n_features=12, target_auc=target_auc, _score_out=score_out
    )
    r = score_out[0]
    measured = float(roc_auc_score(t.y, r))
    assert abs(measured - target_auc) < _TOL, (
        f"{family} at target {target_auc}: measured {measured:.3f}, outside tolerance {_TOL}"
    )


def test_interaction_order_matches_xor_at_k2() -> None:
    # sample_xor_task is sample_interaction_order_task(k=2) under a different tag -- verify
    # the two calibrate to the same achieved AUC given the same rng state, not just the same
    # source code path.
    target = 0.8
    r_xor: list[np.ndarray] = []
    t_xor = sample_xor_task(
        np.random.default_rng(0), 20_000, n_features=12, target_auc=target, _score_out=r_xor
    )
    r_k2: list[np.ndarray] = []
    t_k2 = sample_interaction_order_task(
        np.random.default_rng(0), 20_000, n_features=12, k=2, target_auc=target, _score_out=r_k2
    )
    assert t_k2.source == "family:interaction_order_k2"
    auc_xor = roc_auc_score(t_xor.y, r_xor[0])
    auc_k2 = roc_auc_score(t_k2.y, r_k2[0])
    assert abs(auc_xor - auc_k2) < 1e-9


def test_interaction_order_realised_difficulty_matches_target() -> None:
    # k=1..5, the sweep task 40.3's curve is built from. Confirms the generalisation calibrates
    # correctly at every order, not only k=2 (already covered by the xor family's own test).
    for k in range(1, 6):
        rng = np.random.default_rng(k)
        score_out: list[np.ndarray] = []
        t = sample_interaction_order_task(
            rng, 20_000, n_features=12, k=k, target_auc=0.85, _score_out=score_out
        )
        measured = roc_auc_score(t.y, score_out[0])
        assert abs(measured - 0.85) < _TOL, f"k={k}: measured {measured:.3f}"


def test_interaction_order_rejects_k_out_of_range() -> None:
    with pytest.raises(ValueError, match="k must be in"):
        sample_interaction_order_task(np.random.default_rng(0), 100, n_features=5, k=6)
    with pytest.raises(ValueError, match="k must be in"):
        sample_interaction_order_task(np.random.default_rng(0), 100, n_features=5, k=0)


def test_every_family_is_tagged_and_valid() -> None:
    for family in FAMILIES:
        rng = np.random.default_rng(0)
        t = sample_family_task(rng, family, 200, n_features=10)
        assert t.source == f"family:{family}"
        assert t.X.dtype == np.float32
        assert t.X.shape == (200, 10)
        assert len(np.unique(t.y)) == 2
        assert t.n_classes == 2
        assert not t.is_categorical.any()


def test_unknown_family_rejected() -> None:
    with pytest.raises(ValueError, match="unknown task family"):
        sample_family_task(np.random.default_rng(0), "not-a-family", 100)


def test_families_are_not_the_same_generator_in_disguise() -> None:
    # A cheap distinctness check in the spirit of test_tree_prior.py's justification test: xor
    # (a genuine interaction) should not be solvable by a linear model at high target AUC, while
    # linear should be -- otherwise "xor" is not testing what its name claims.
    rng = np.random.default_rng(0)
    t_lin = sample_family_task(rng, "linear", 4000, n_features=8, target_auc=0.9)
    t_xor = sample_family_task(rng, "xor", 4000, n_features=8, target_auc=0.9)

    def _linear_auc(t) -> float:
        X, y = np.nan_to_num(t.X), t.y
        cut = len(y) // 2
        lr = LogisticRegression(max_iter=1000).fit(X[:cut], y[:cut])
        return float(roc_auc_score(y[cut:], lr.predict_proba(X[cut:])[:, 1]))

    assert _linear_auc(t_lin) > 0.8
    assert _linear_auc(t_xor) < 0.65


def test_composition_components_hit_target_and_share_features() -> None:
    from fintfm.prior.task_families import sample_composition_task

    rng_a = np.random.default_rng(0)
    t_a = sample_composition_task(rng_a, 20_000, n_features=8, target_component_auc=0.9, mode="component_a")
    rng_b = np.random.default_rng(0)
    t_b = sample_composition_task(rng_b, 20_000, n_features=8, target_component_auc=0.9, mode="component_b")
    rng_c = np.random.default_rng(0)
    t_c = sample_composition_task(rng_c, 20_000, n_features=8, target_component_auc=0.9, mode="composed")

    # Same rng seed -> same X draw regardless of mode (only the exposed label differs).
    assert np.allclose(t_a.X, t_b.X)
    assert np.allclose(t_a.X, t_c.X)

    lr = LogisticRegression(max_iter=1000)
    cut = 10_000
    auc_a = roc_auc_score(t_a.y[cut:], lr.fit(t_a.X[:cut], t_a.y[:cut]).predict_proba(t_a.X[cut:])[:, 1])
    auc_b = roc_auc_score(t_b.y[cut:], lr.fit(t_b.X[:cut], t_b.y[:cut]).predict_proba(t_b.X[cut:])[:, 1])
    assert abs(auc_a - 0.9) < 0.03
    assert abs(auc_b - 0.9) < 0.03

    # composed is y_a AND y_b: strictly rarer than either component alone.
    assert t_c.y.mean() < t_a.y.mean()
    assert t_c.source == "family:composition_composed"


def test_composition_rejects_bad_input() -> None:
    from fintfm.prior.task_families import sample_composition_task

    with pytest.raises(ValueError, match="n_features must be"):
        sample_composition_task(np.random.default_rng(0), 100, n_features=1)
    with pytest.raises(ValueError, match="unknown mode"):
        sample_composition_task(np.random.default_rng(0), 100, n_features=8, mode="bogus")


def test_confound_collider_pair_shares_context_and_labels() -> None:
    from fintfm.prior.task_families import sample_confound_collider_pair

    rng = np.random.default_rng(0)
    X_obs, X_int, _y, active = sample_confound_collider_pair(rng, n_ctx=500, n_query=500, n_features=8)
    assert X_obs.shape == X_int.shape == (1000, 8)
    assert np.array_equal(X_obs[:500], X_int[:500])  # context untouched
    assert not np.array_equal(X_obs[500:], X_int[500:])  # query proxy differs
    proxy, cause = active
    assert np.array_equal(X_obs[500:, cause], X_int[500:, cause])  # cause column untouched
    assert not np.array_equal(X_obs[500:, proxy], X_int[500:, proxy])


def test_confound_collider_cause_alone_is_untouched_by_intervention() -> None:
    # A classifier trained only on the cause column should score identically on both
    # conditions -- the intervention severs only the proxy's link to the confound.
    from fintfm.prior.task_families import sample_confound_collider_pair

    rng = np.random.default_rng(1)
    X_obs, X_int, y, active = sample_confound_collider_pair(
        rng, n_ctx=8000, n_query=8000, n_features=8, target_auc=0.9
    )
    _proxy, cause = active
    lr = LogisticRegression(max_iter=1000).fit(X_obs[:8000, [cause]], y[:8000])
    auc_obs = roc_auc_score(y[8000:], lr.predict_proba(X_obs[8000:, [cause]])[:, 1])
    auc_int = roc_auc_score(y[8000:], lr.predict_proba(X_int[8000:, [cause]])[:, 1])
    assert abs(auc_obs - auc_int) < 0.02


def test_confound_collider_proxy_reliance_costs_accuracy_under_intervention() -> None:
    # A classifier trained on BOTH observed columns (proxy + cause) -- exploiting the confound
    # exactly as a naive observational fit would -- must lose accuracy once the proxy is severed
    # from the confound, even though the cause's own contribution is untouched.
    from fintfm.prior.task_families import sample_confound_collider_pair

    rng = np.random.default_rng(2)
    X_obs, X_int, y, active = sample_confound_collider_pair(
        rng, n_ctx=8000, n_query=8000, n_features=8, target_auc=0.9
    )
    cols = list(active)
    lr = LogisticRegression(max_iter=1000).fit(X_obs[:8000][:, cols], y[:8000])
    auc_obs = roc_auc_score(y[8000:], lr.predict_proba(X_obs[8000:][:, cols])[:, 1])
    auc_int = roc_auc_score(y[8000:], lr.predict_proba(X_int[8000:][:, cols])[:, 1])
    assert auc_obs - auc_int > 0.03


def test_confound_collider_rejects_bad_input() -> None:
    from fintfm.prior.task_families import sample_confound_collider_pair

    with pytest.raises(ValueError, match="n_features must be"):
        sample_confound_collider_pair(np.random.default_rng(0), 100, 100, n_features=1)
    with pytest.raises(ValueError, match="cause_weight must be"):
        sample_confound_collider_pair(np.random.default_rng(0), 100, 100, n_features=8, cause_weight=1.5)
