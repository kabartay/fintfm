"""Per-task prediction horizon of the financial prior (``temporal-financial-prior`` 2.4).

Off is invisible (the recorded stream is untouched); on, each task carries a horizon, its
label is default within that horizon, and shorter horizons give lower base rates.
"""

from __future__ import annotations

import numpy as np
import pytest

from fintfm.prior.financial import sample_financial_task
from fintfm.prior.mixture import PriorConfig, _check_levers, sample_task

K = 6


def _draw(seed: int, **kw):
    return sample_financial_task(np.random.default_rng(seed), 400, max_features=8, **kw)


def test_off_is_the_recorded_task():
    a, b = _draw(3, n_horizons=K), _draw(3, n_horizons=K, sample_horizon=False)
    assert np.array_equal(a.X, b.X, equal_nan=True)
    assert np.array_equal(a.y, b.y) and np.array_equal(a.period, b.period)
    assert a.meta is None and b.meta is None


def test_horizon_is_exposed_and_defines_the_label():
    for seed in range(20):
        t = _draw(seed, n_horizons=K, sample_horizon=True)
        h = t.meta["horizon"]
        assert 1 <= h <= K
        assert np.array_equal(t.y, ((t.period >= 0) & (t.period < h)).astype(int))
        assert t.y.min() == 0 and t.y.max() == 1


def test_longer_horizons_have_higher_base_rates():
    rates: dict[int, list[float]] = {}
    for seed in range(300):
        t = _draw(seed, n_horizons=K, sample_horizon=True)
        rates.setdefault(t.meta["horizon"], []).append(float(t.y.mean()))
    means = {h: np.mean(v) for h, v in rates.items() if len(v) >= 5}
    assert len(means) >= 3
    hs = sorted(means)
    assert means[hs[0]] < means[hs[-1]]
    assert np.corrcoef(hs, [means[h] for h in hs])[0, 1] > 0.8


def test_requires_a_horizon_grid():
    with pytest.raises(ValueError, match="n_horizons"):
        _draw(0, sample_horizon=True)
    with pytest.raises(ValueError, match="financial_sample_horizon"):
        _check_levers(PriorConfig(financial_sample_horizon=True))


def test_reaches_the_task_through_the_prior_config():
    cfg = PriorConfig(
        p_financial=1.0, n_horizons=K, max_features=8, n_rows=64, financial_sample_horizon=True
    )
    assert 1 <= sample_task(np.random.default_rng(1), cfg).meta["horizon"] <= K
