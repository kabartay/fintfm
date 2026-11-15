"""Persistent firm effects and autocorrelated ratios in the financial prior
(``temporal-financial-prior`` 2.3).

Off is invisible (the recorded stream is untouched, and the other digest tests pin it); on,
margin and leverage histories carry the configured AR(1) coefficient and a firm effect that
persists, and the last year is the snapshot the task shows.
"""

from __future__ import annotations

import numpy as np
import pytest

from fintfm.prior.financial import (
    _firm_effect_ar1_paths,
    _macro_cycle_path,
    sample_financial_task,
)
from fintfm.prior.mixture import PriorConfig, _check_levers, sample_task


def _lag1_corr(x: np.ndarray) -> float:
    """Pooled correlation between successive columns of an ``(n, T)`` array."""
    return float(np.corrcoef(x[:, :-1].ravel(), x[:, 1:].ravel())[0, 1])


def _draw(seed: int, **kw):
    return sample_financial_task(np.random.default_rng(seed), 300, max_features=12, **kw)


def test_off_is_the_recorded_task():
    a, b = _draw(5), _draw(5, temporal_years=None)
    assert np.array_equal(a.X, b.X, equal_nan=True) and np.array_equal(a.y, b.y)
    assert a.meta is None and b.meta is None


def test_estimated_ar_coefficient_matches_the_configured_one():
    """With no firm effect the OLS slope of x_t on x_{t-1} is the configured coefficient."""
    rng = np.random.default_rng(0)
    for ar in (0.3, 0.7, 0.9):
        x = _firm_effect_ar1_paths(rng, 20_000, 6, ar, 0.0)
        slope = float((x[:, :-1] * x[:, 1:]).sum() / (x[:, :-1] ** 2).sum())
        assert slope == pytest.approx(ar, abs=0.02)
        assert x.var(axis=0) == pytest.approx(np.ones(6), abs=0.05)  # stationary


def test_successive_years_correlate_and_the_firm_effect_persists():
    rng = np.random.default_rng(1)
    ar, share, years = 0.6, 0.4, 6
    x = _firm_effect_ar1_paths(rng, 20_000, years, ar, share)
    assert _lag1_corr(x) == pytest.approx(share + (1 - share) * ar, abs=0.02)
    # persistence: the first and last year, five years apart, still share the firm effect
    far = float(np.corrcoef(x[:, 0], x[:, -1])[0, 1])
    assert far == pytest.approx(share + (1 - share) * ar ** (years - 1), abs=0.02)
    assert far > 0.3
    # and with no firm effect and no autocorrelation, nothing persists
    flat = _firm_effect_ar1_paths(rng, 20_000, years, 0.0, 0.0)
    assert abs(_lag1_corr(flat)) < 0.02


def test_macro_cycle_ends_at_the_sampled_cycle_and_evolves():
    rng = np.random.default_rng(2)
    path = _macro_cycle_path(rng, 1.3, 5, 0.6)
    assert path.shape == (5,) and path[-1] == 1.3
    assert len(set(path.tolist())) == 5
    # a regime of unit variance, reversible: lag-1 correlation over many histories is ar
    finals = rng.normal(size=4_000)
    paths = np.stack([_macro_cycle_path(rng, f, 4, 0.6) for f in finals])
    assert _lag1_corr(paths) == pytest.approx(0.6, abs=0.04)


def test_task_carries_histories_whose_last_year_is_the_snapshot():
    t = _draw(7, temporal_years=5, temporal_ar=0.8, temporal_firm_share=0.3)
    tmp = t.meta["temporal"]
    assert tmp["margin_dev"].shape == (300, 5) and tmp["leverage_dev"].shape == (300, 5)
    assert tmp["cycle"].shape == (5,)
    assert np.isfinite(t.X[~np.isnan(t.X)]).all() and t.y.min() == 0 and t.y.max() == 1
    # the exposed history has the configured dynamics, across many tasks
    paths = np.concatenate(
        [
            _draw(s, temporal_years=5, temporal_ar=0.8, temporal_firm_share=0.3).meta["temporal"][
                "margin_dev"
            ]
            for s in range(30)
        ]
    )
    assert _lag1_corr(paths) == pytest.approx(0.3 + 0.7 * 0.8, abs=0.03)
    assert paths.std() == pytest.approx(0.08, rel=0.1)


def test_on_changes_the_task_and_is_reproducible():
    off, on, again = _draw(9), _draw(9, temporal_years=3), _draw(9, temporal_years=3)
    assert not np.array_equal(off.X, on.X, equal_nan=True)
    assert np.array_equal(on.X, again.X, equal_nan=True)


def test_survives_the_other_options():
    t = _draw(4, temporal_years=3, n_horizons=6, sample_horizon=True, sign_flip_prob=0.5)
    assert {"temporal", "horizon", "flipped"} <= set(t.meta)
    g = _draw(4, temporal_years=3, gate_prob=1.0, max_lr_auc=0.99)
    assert "temporal" in g.meta


def test_refuses_nonsense():
    for kw in (
        {"temporal_years": 1},
        {"temporal_years": 3, "temporal_ar": 1.0},
        {"temporal_years": 3, "temporal_firm_share": 1.5},
    ):
        with pytest.raises(ValueError, match="temporal_years"):
            _draw(0, **kw)
    with pytest.raises(ValueError, match="financial_temporal_years"):
        _check_levers(PriorConfig(financial_temporal_years=1))


def test_reaches_the_task_through_the_mixture():
    cfg = PriorConfig(p_financial=1.0, financial_temporal_years=4, max_features=8, n_rows=64)
    t = sample_task(np.random.default_rng(1), cfg)
    assert t.meta["temporal"]["years"] == 4
