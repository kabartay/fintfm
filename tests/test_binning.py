"""Tests for quantile binning of continuous targets (tasks 46.3, 46.5, 46.6).

The bimodal test is the one that matters. LGD piles up at both ends of [0, 1], so a point
estimate or a Gaussian head reports a middle value that rarely occurs and is confidently
wrong. A distributional head must be shown to place mass at both modes rather than averaging
them, and only a test constructed that way can show it.
"""

from __future__ import annotations

import numpy as np
import pytest

from fintfm.inference.binning import QuantileBinner, interval_coverage


def test_bins_are_equally_populated_by_construction():
    """The property that makes cross-entropy trainable: no bin is starved."""
    rng = np.random.default_rng(0)
    y = rng.lognormal(size=5000)  # heavy-tailed on purpose; fixed-width bins would fail here
    b = QuantileBinner(n_bins=10).fit(y)
    counts = np.bincount(b.transform(y), minlength=b.n_bins_)

    assert b.n_bins_ == 10
    assert counts.min() > 0.08 * len(y), f"a bin is starved: {counts}"
    assert counts.max() < 0.12 * len(y), f"a bin dominates: {counts}"


def test_the_point_estimate_recovers_a_known_linear_target():
    """Task 46.3's stated verification: a head that cannot recover y = 2x + noise is not one.

    Scored against the binning *ceiling* rather than against zero error. Discretisation
    destroys within-bin resolution by construction, so requiring exact recovery would be
    requiring the bins not to be bins. The honest bar is that a perfect classifier over these
    bins does well, which is what the oracle distribution measures.
    """
    rng = np.random.default_rng(1)
    x = rng.normal(size=4000)
    y = 2.0 * x + rng.normal(0, 0.1, size=4000)
    b = QuantileBinner(n_bins=20).fit(y)

    # The oracle: all mass on the true bin. This is the best any model over these bins can do.
    oracle = np.zeros((len(y), b.n_bins_))
    oracle[np.arange(len(y)), b.transform(y)] = 1.0
    pred = b.expected_value(oracle)

    assert np.corrcoef(pred, y)[0, 1] > 0.99, "binning must preserve the ordering"
    assert np.sqrt(np.mean((pred - y) ** 2)) < 0.2 * y.std(), "discretisation error too large"


def test_the_distribution_integrates_to_one_and_the_estimate_stays_in_range():
    """A predicted value outside the data's range would mean the representatives are wrong."""
    rng = np.random.default_rng(2)
    y = rng.uniform(-5, 5, size=2000)
    b = QuantileBinner(n_bins=8).fit(y)

    proba = rng.dirichlet(np.ones(b.n_bins_), size=100)
    assert np.allclose(proba.sum(axis=1), 1.0)
    est = b.expected_value(proba)
    assert est.min() >= y.min() and est.max() <= y.max()


def test_a_bimodal_bounded_target_keeps_mass_at_both_modes():
    """Task 46.6, the LGD case, and the specific failure a Gaussian head would exhibit.

    Half the mass near 0 and half near 1.

    **Where bimodality shows up is not bin occupancy.** Quantile bins are equally populated by
    construction, so the aggregate distribution over bins is flat for *every* target and could
    never evidence anything. It shows up in the **bin representatives**: quantile bins are
    narrow where data is dense, so a bimodal target spends most of its bins near the two modes
    and spans the empty middle with a few wide ones. That is exactly the property that lets
    the head express bimodality -- it has resolution where the mass is.
    """
    rng = np.random.default_rng(3)
    y = np.concatenate([rng.beta(0.5, 8, size=2000), rng.beta(8, 0.5, size=2000)])
    b = QuantileBinner(n_bins=10).fit(y)
    reps = b.representatives_

    at_modes = np.sum((reps < 0.2) | (reps > 0.8))
    in_middle = np.sum((reps >= 0.4) & (reps <= 0.6))
    assert at_modes > in_middle, (
        f"resolution must sit at the modes: {at_modes} representatives near the ends, "
        f"{in_middle} in the middle ({np.round(reps, 3)})"
    )

    # The contrast that motivates a distributional head at all: put half the mass on the
    # lowest bin and half on the highest -- the true LGD shape -- and the point estimate is a
    # middle value that almost no observation takes.
    two_point = np.zeros((1, b.n_bins_))
    two_point[0, 0] = two_point[0, -1] = 0.5
    mean_of_modes = b.expected_value(two_point)[0]
    assert 0.3 < mean_of_modes < 0.7, f"the mean of two modes is the middle: {mean_of_modes}"
    assert np.mean(np.abs(y - mean_of_modes) < 0.1) < 0.10, "and that middle is a rare value"


def test_interval_coverage_tracks_its_nominal_level():
    """Task 46.5: an 80% interval must contain the truth about 80% of the time."""
    rng = np.random.default_rng(4)
    y = rng.normal(size=6000)
    b = QuantileBinner(n_bins=20).fit(y)
    oracle = np.zeros((len(y), b.n_bins_))
    oracle[np.arange(len(y)), b.transform(y)] = 1.0

    # A smoothed oracle: the true bin plus its neighbours, so the interval has real width.
    smooth = oracle.copy()
    smooth[:, 1:] += 0.5 * oracle[:, :-1]
    smooth[:, :-1] += 0.5 * oracle[:, 1:]
    smooth /= smooth.sum(axis=1, keepdims=True)

    lo, hi = b.interval(smooth, level=0.8)
    cov = interval_coverage(y, lo, hi)
    assert 0.55 < cov < 0.98, f"coverage {cov:.3f} is not in a plausible band for this oracle"
    assert np.all(hi >= lo)


def test_a_degenerate_target_collapses_bins_instead_of_producing_empty_ones():
    """A target with three distinct values must give three bins, not ten unreachable ones."""
    y = np.array([1.0] * 50 + [5.0] * 50 + [9.0] * 50)
    b = QuantileBinner(n_bins=10).fit(y)

    assert b.n_bins_ <= 3, f"ties must collapse edges, got {b.n_bins_} bins"
    assert np.all(np.isfinite(b.representatives_))
    assert set(np.unique(b.transform(y))) == set(range(b.n_bins_))


def test_rejects_nonsense_input():
    with pytest.raises(ValueError, match="n_bins must be >= 2"):
        QuantileBinner(n_bins=1)
    with pytest.raises(ValueError, match="no finite target values"):
        QuantileBinner().fit(np.array([np.nan, np.inf]))
    with pytest.raises(RuntimeError, match="fit must be called"):
        QuantileBinner().transform(np.array([1.0]))
    b = QuantileBinner(n_bins=4).fit(np.arange(100.0))
    with pytest.raises(ValueError, match="proba must be"):
        b.expected_value(np.ones((3, 2)))
    with pytest.raises(ValueError, match="q must be in"):
        b.quantile(np.ones((1, b.n_bins_)) / b.n_bins_, 1.5)
