"""The V4FinBench out-of-time harnesses split without leakage and report NaN honestly.

Covers openspec tasks 49.7 (company check and embargo), 49.9(c) (context sweep records the
level; NaN-aware ``best``) and 49.9(d) (``score_curve`` NaN horizons).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fintfm.evaluation.datasets import SurvivalDataset
from fintfm.experiments import context_sweep
from fintfm.experiments.v4_out_of_time import (
    score_curve,
    split_dataset,
    time_split,
)

N_H = 6


def _ds(scale: int = 1, seed: int = 0) -> SurvivalDataset:
    """A panel with three kinds of firm: early (2006-2016), shared (2006-2020), late (2013-2020)."""
    rng = np.random.default_rng(seed)
    spans = [(range(2006, 2017), 20), (range(2006, 2021), 10), (range(2013, 2021), 20)]
    firm, year, fid = [], [], 0
    for years, n in spans:
        for _ in range(n * scale):
            firm += [fid] * len(years)
            year += list(years)
            fid += 1
    firm, year = np.array(firm), np.array(year)
    n = len(firm)
    y = (rng.random(n) < 0.1).astype(int)
    return SurvivalDataset(
        X=rng.normal(size=(n, 3)).astype(np.float32), y=y,
        period=np.where(y == 1, rng.integers(0, N_H, n), -1),
        n_observed=np.minimum(N_H, 2020 - year), n_horizons=N_H, year=year,
        name="synthetic", licence="MIT", attribution="test", groups=firm,
    )


def test_a_firm_cannot_be_on_both_sides():
    ds = _ds()
    # the 10 shared firms are on both sides of the legacy split
    assert split_dataset(ds, 2010, 2017, split="legacy").n_shared_ids_after == 10
    s = split_dataset(ds, 2010, 2017)
    assert not set(ds.groups[s.train]) & set(ds.groups[s.test])
    assert s.n_shared_ids_after == 0 and s.n_shared_ids == 10 and s.n_train_dropped > 0
    assert set(ds.groups[s.train]) == set(range(20))  # only the early firms remain
    assert (ds.year[s.test] >= 2017).all()  # the test window is never altered
    assert len(s.test) == int((ds.year >= 2017).sum())


def test_embargo_keeps_every_training_label_window_out_of_the_test_period():
    ds = _ds()
    s = split_dataset(ds, 2016, 2017)  # the configured default window
    assert s.embargo_years == N_H and s.effective_train_until == 2010
    reach = ds.year[s.train] + N_H  # last year any horizon label of a training row covers
    assert (reach < 2017).all()
    # one year later would reach into the test period
    assert ds.year[s.train].max() + N_H + 1 >= 2017
    # a train_until already earlier than the embargo bound is respected
    assert split_dataset(ds, 2008, 2017).effective_train_until == 2008


def test_embargo_can_be_overridden_and_empties_are_refused():
    ds = _ds()
    assert split_dataset(ds, 2016, 2017, embargo_years=2).effective_train_until == 2014
    with pytest.raises(ValueError, match="empty window"):
        split_dataset(ds, 2016, 2017, embargo_years=20)
    with pytest.raises(ValueError, match="windows overlap"):
        split_dataset(ds, 2017, 2017)
    with pytest.raises(ValueError, match="unknown split"):
        split_dataset(ds, 2010, 2017, split="nope")  # type: ignore[arg-type]


def test_legacy_option_reproduces_the_old_split_exactly():
    ds = _ds()
    tr, te = time_split(ds, 2016, 2017, split="legacy")
    assert np.array_equal(tr, np.flatnonzero(ds.year <= 2016))
    assert np.array_equal(te, np.flatnonzero(ds.year >= 2017))
    # the old split leaks companies and label windows; the default does neither
    tr2, te2 = time_split(ds, 2016, 2017)
    assert len(tr2) < len(tr) and np.array_equal(te, te2)
    assert (ds.year[tr] + N_H >= 2017).any() and not (ds.year[tr2] + N_H >= 2017).any()


def test_default_split_needs_company_ids_but_legacy_does_not():
    ds = _ds()
    nogroups = SurvivalDataset(**{**ds.__dict__, "groups": None})
    with pytest.raises(ValueError, match="company identifiers"):
        time_split(nogroups, 2010, 2017)
    tr, _ = time_split(nogroups, 2010, 2017, split="legacy")
    assert len(tr)


def test_score_curve_counts_only_comparable_pairs_and_records_them():
    rng = np.random.default_rng(0)
    n = 400
    truth = np.zeros((n, 3), dtype=int)
    truth[:50, :] = 1
    seen = np.ones((n, 3), dtype=bool)
    base = rng.random(n)
    curve = np.stack([base * 0.1, np.full(n, np.nan), base * 0.3], axis=1)  # h1 not fitted
    arm = score_curve("x", curve, truth, seen, min_rows=10)
    assert np.isnan(arm.horizons[1].auc)
    # every adjacent pair touches the NaN column: nothing comparable, so no rates
    assert arm.n_comparable_pairs == 0 and arm.n_nan_pairs == 2 * n
    assert np.isnan(arm.violation_rate) and np.isnan(arm.fully_monotone)


def test_score_curve_violation_rate_ignores_nan_pairs():
    n = 200
    truth = np.zeros((n, 3), dtype=int)
    truth[:60] = 1
    seen = np.ones((n, 3), dtype=bool)
    curve = np.zeros((n, 3))
    curve[:, 0] = np.linspace(0.1, 0.5, n)
    curve[:, 1] = curve[:, 0] - 0.05  # every row violates h0 -> h1
    curve[:, 2] = np.nan  # h2 not fitted
    arm = score_curve("x", curve, truth, seen, min_rows=10)
    assert arm.n_comparable_pairs == n and arm.n_nan_pairs == n
    assert arm.violation_rate == 1.0  # old code: 0.5, the NaN pairs counted as passes
    assert arm.fully_monotone == 0.0  # old code: 0.0 only by luck of d>=0 on NaN being False
    assert arm.n_rows_compared == n


def test_score_curve_clean_curve_has_no_nan_pairs():
    rng = np.random.default_rng(1)
    n = 300
    truth = (rng.random((n, 3)) < 0.3).astype(int)
    seen = np.ones((n, 3), dtype=bool)
    curve = np.sort(rng.random((n, 3)), axis=1)
    arm = score_curve("x", curve, truth, seen, min_rows=10)
    assert arm.n_nan_pairs == 0 and arm.n_comparable_pairs == 2 * n
    assert arm.violation_rate == 0.0 and arm.fully_monotone == 1.0


class _FakeClf:
    """Stands in for FinancialTFMClassifier: a constant-probability term structure."""

    def __init__(self, level: float) -> None:
        self.level = level
        self.context_summary_ = type(
            "S", (), {"context_rate": 0.1, "population_rate": 0.1, "n_positive": 5,
                      "pooled_context_rate": 0.1},
        )()

    def fit(self, X, y, periods=None):
        """Accept the context and return self."""
        self._n = len(X)
        return self

    def predict_term_structure(self, X):
        """A random curve whose mean is ``level`` at every horizon."""
        rng = np.random.default_rng(0)
        return np.clip(self.level + 0.01 * rng.normal(size=(len(X), N_H)), 0, 1)


def test_context_sweep_records_mean_predicted_beside_observed(monkeypatch, tmp_path):
    from fintfm.config import load_config

    ds = _ds(scale=10)
    monkeypatch.setattr(context_sweep, "load_v4finbench", lambda max_rows=None: ds)
    monkeypatch.setattr(
        context_sweep.FinancialTFMClassifier, "from_config",
        classmethod(lambda cls, *a, **k: _FakeClf(0.25)),
    )
    base = load_config()
    cfg = dataclasses.replace(
        base,
        context_sweep=dataclasses.replace(
            base.context_sweep, context_sizes=(100,), strategies=("uniform",)
        ),
        evaluation=dataclasses.replace(base.evaluation, min_rows_per_horizon=10),
    )
    rec = context_sweep.run("m.pt", tmp_path, train_until=2008, test_from=2017, seeds=(0,), cfg=cfg)
    cell = rec["cells"][0]
    assert cell["mean_predicted"] == pytest.approx(0.25, abs=0.01)
    assert 0 < cell["observed_rate"] < 1
    assert len(cell["mean_predicted_by_horizon"]) == cell["n_horizons_scored"] > 0
    assert rec["config"]["split"] == "embargoed" and rec["split"]["n_shared_ids_after"] == 0
    text = context_sweep.summarise(rec)
    assert "meanPD" in text and "obs" in text


def test_context_sweep_best_skips_nan_cells():
    def cell(auc: float, strategy: str) -> dict:
        return {"max_context": 1000, "strategy": strategy, "seed": 0, "context_rate": 0.1,
                "n_context_positive": 1, "mean_auc": auc, "mean_ece": 0.01, "seconds": 1.0}

    # a NaN first would win max() in Python (nothing compares greater than it afterwards)
    rec = {"cells": [cell(float("nan"), "nanny"), cell(0.7, "ok"), cell(0.9, "best")]}
    out = context_sweep.summarise(rec)
    assert "best mean AUC: best" in out and "1 cell(s)" in out
    assert "no cell had a scorable horizon" in context_sweep.summarise(
        {"cells": [cell(float("nan"), "a")]}
    )
