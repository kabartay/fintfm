"""Survival structure of the financial prior (``pd-term-structure`` 11.3).

The prior samples per-period hazards inside ``_sample_survival`` and emits only the first
event period. These tests capture the sampled hazard matrix and check that the cumulative
hazard and cumulative default probability are non-decreasing and in [0, 1] by construction,
for every hazard shape the sampler draws, and that the emitted period matches the hazards.
"""

from __future__ import annotations

import numpy as np
import pytest

from fintfm.modeling.hazard import CENSORED
from fintfm.prior import financial
from fintfm.prior.financial import _sample_survival, sample_financial_task


def _hazards_and_period(seed: int, n: int, k: int, rate: float, distress: np.ndarray | None = None):
    """Run ``_sample_survival`` and return the hazard matrix it sampled from.

    Args:
        seed: Generator seed.
        n: Number of rows.
        k: Number of periods.
        rate: Target cumulative default rate.
        distress: Optional ``(n,)`` latent; standard normal when ``None``.

    Returns:
        ``(h, period, y)`` with ``h`` the ``(n, k)`` per-period hazards of the final draw.
    """
    rng = np.random.default_rng(seed)
    if distress is None:
        distress = rng.standard_normal(n)
    seen: list[np.ndarray] = []
    real = financial._sigmoid

    def spy(z):
        out = real(z)
        seen.append(out)
        return out

    financial._sigmoid = spy
    try:
        period, y = _sample_survival(rng, distress, rate, k)
    finally:
        financial._sigmoid = real
    h = seen[-1].clip(1e-7, 1 - 1e-7)
    assert h.shape == (n, k)
    return h, period, y


def _profile(h: np.ndarray) -> str:
    """Classify the mean hazard profile as rising, falling, hump or flat."""
    m = h.mean(axis=0)
    d = np.diff(np.log(m / (1 - m)))
    if np.abs(d).max() < 1e-9:
        return "flat"
    if (d >= -1e-9).all():
        return "rising"
    if (d <= 1e-9).all():
        return "falling"
    return "hump"


def test_cumulative_hazard_and_pd_are_non_decreasing_for_every_shape():
    shapes = set()
    for seed in range(120):
        k = 3 + seed % 6
        rate = (0.01, 0.05, 0.15, 0.30)[seed % 4]
        h, _, _ = _hazards_and_period(seed, 300, k, rate)
        assert ((h > 0) & (h < 1)).all()
        cum_h = np.cumsum(-np.log1p(-h), axis=1)
        assert (np.diff(cum_h, axis=1) > 0).all()
        pd_k = 1.0 - np.cumprod(1.0 - h, axis=1)
        assert (np.diff(pd_k, axis=1) >= 0).all()
        assert (pd_k >= 0).all() and (pd_k <= 1).all()
        shapes.add(_profile(h))
    assert shapes == {"rising", "falling", "hump", "flat"}


def test_cumulative_pd_hits_the_target_rate_at_the_end_of_the_grid():
    for seed, rate in [(0, 0.02), (1, 0.1), (2, 0.25)]:
        h, _, _ = _hazards_and_period(seed, 2000, 5, rate)
        final = (1.0 - np.cumprod(1.0 - h, axis=1)[:, -1]).mean()
        assert final == pytest.approx(rate, abs=2e-3)


def test_emitted_periods_follow_the_sampled_hazards():
    """With identical rows the period pmf is h_k * prod_{j<k}(1 - h_j), censored remainder."""
    n = 60000
    k = 5
    for seed in range(8):
        h, period, _ = _hazards_and_period(seed, n, k, 0.2, distress=np.zeros(n))
        h0 = h[0]
        assert np.allclose(h, h0)
        surv = np.concatenate([[1.0], np.cumprod(1.0 - h0)])
        pmf = np.append(h0 * surv[:-1], surv[-1])  # last entry: censored
        counts = np.append(
            np.bincount(period[period >= 0], minlength=k), (period == CENSORED).sum()
        )
        emp = counts / n
        assert np.allclose(emp, pmf, atol=5 * np.sqrt(pmf * (1 - pmf) / n).max())
        assert np.isclose(pmf.sum(), 1.0)
        # empirical cumulative default fraction per period is non-decreasing and in [0, 1]
        cdf = np.cumsum(emp[:k])
        assert (np.diff(cdf) >= 0).all() and cdf[0] >= 0 and cdf[-1] <= 1


def test_label_is_one_exactly_when_the_period_is_not_censored():
    for seed in range(30):
        k = 2 + seed % 7
        _, period, y = _hazards_and_period(seed, 500, k, 0.1)
        assert ((y == 1) == (period != CENSORED)).all()
        assert period.min() >= CENSORED and period.max() < k


@pytest.mark.parametrize("seed", range(6))
def test_sampled_task_period_is_consistent_with_the_label(seed):
    t = sample_financial_task(np.random.default_rng(seed), 600, max_features=16, n_horizons=4)
    assert t.period is not None
    assert ((t.y == 1) == (t.period != CENSORED)).all()
