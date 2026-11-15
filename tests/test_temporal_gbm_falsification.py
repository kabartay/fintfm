"""SMOKE-TEST for the temporal falsification harness (openspec 2.2): synthetic panels only.

Runs in a subprocess that never imports torch (LightGBM and torch segfault together on macOS).
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("lightgbm")

MODULE = (
    Path(__file__).resolve().parents[1] / "src/fintfm/experiments/temporal_gbm_falsification.py"
)

LOAD = textwrap.dedent("""
    import importlib.util, sys
    import numpy as np
    spec = importlib.util.spec_from_file_location("tg", sys.argv[1])
    tg = importlib.util.module_from_spec(spec)
    sys.modules["tg"] = tg
    spec.loader.exec_module(tg)

    def panel(n_comp=60, years=8, d=3, seed=0):
        rng = np.random.default_rng(seed)
        g, yr = np.meshgrid(np.arange(n_comp), np.arange(2010, 2010 + years), indexing="ij")
        g, yr = g.ravel(), yr.ravel()
        X = rng.normal(size=(len(g), d)).astype(np.float32)
        return X, g, yr
""")


def _run(body: str) -> None:
    code = LOAD + textwrap.dedent(body) + "\nassert 'torch' not in sys.modules\n"
    r = subprocess.run(
        [sys.executable, "-c", code, str(MODULE)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_delta_and_trend_values_and_first_year_nan():
    _run("""
        X, g, yr = panel()
        d, t = tg.trajectory_features(X, g, yr)
        first = yr == 2010
        assert np.isnan(d[first]).all() and np.isnan(t[first]).all()
        second = yr == 2011
        assert np.isnan(t[second]).all()  # only two points
        i = np.flatnonzero((g == 5) & (yr == 2014))[0]
        j = np.flatnonzero((g == 5) & (yr == 2013))[0]
        assert np.allclose(d[i], X[i] - X[j])
        # a linear ramp has trend equal to its slope
        Xl = (yr - 2010).astype(np.float32)[:, None] * 2.0 + np.zeros((len(g), 3), np.float32)
        d2, t2 = tg.trajectory_features(Xl, g, yr)
        assert np.allclose(d2[yr >= 2011], 2.0) and np.allclose(t2[yr >= 2012], 2.0, atol=1e-5)
    """)


def test_no_future_leakage_changing_later_rows_leaves_earlier_features_unchanged():
    _run("""
        X, g, yr = panel()
        d, t = tg.trajectory_features(X, g, yr)
        cut = 2013
        X2 = X.copy()
        X2[yr > cut] += 100.0  # corrupt every row strictly after the cut
        d2, t2 = tg.trajectory_features(X2, g, yr)
        early = yr <= cut
        assert np.array_equal(d[early], d2[early], equal_nan=True)
        assert np.array_equal(t[early], t2[early], equal_nan=True)
        assert not np.array_equal(d[~early], d2[~early], equal_nan=True)
    """)


def test_gap_in_history_gives_nan_not_stretched_delta():
    _run("""
        X, g, yr = panel()
        keep = ~((g == 3) & (yr == 2012))
        X, g, yr = X[keep], g[keep], yr[keep]
        d, t = tg.trajectory_features(X, g, yr)
        i = np.flatnonzero((g == 3) & (yr == 2013))[0]
        assert np.isnan(d[i]).all()
        k = np.flatnonzero((g == 3) & (yr == 2014))[0]
        assert not np.isnan(d[k]).any()
    """)


def test_lag_index_refuses_non_positive_lag_and_duplicates():
    _run("""
        X, g, yr = panel(n_comp=3, years=3)
        for bad in (0, -1):
            try:
                tg.lag_index(g, yr, bad)
            except ValueError:
                pass
            else:
                raise AssertionError("lag <= 0 must raise")
        try:
            tg.lag_index(np.r_[g, g[:1]], np.r_[yr, yr[:1]], 1)
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate rows must raise")
    """)


def test_folds_keep_each_company_in_one_fold_and_sample_keeps_whole_companies():
    _run("""
        X, g, yr = panel(n_comp=90)
        rng = np.random.default_rng(1)
        y = (rng.random(len(g)) < 0.1).astype(int)
        folds = tg.grouped_folds(y, g)
        assert len(folds) == tg.N_SPLITS
        seen = []
        for tr, te in folds:
            assert not set(g[tr]) & set(g[te])
            seen.append(set(g[te]))
        assert sum(len(s) for s in seen) == len(set(g))  # each company tests exactly once
        idx = tg.sample_companies(g, 200, seed=0)
        assert len(idx) <= 200
        for c in set(g[idx]):
            assert (g[idx] == c).sum() == (g == c).sum()
    """)


def test_run_end_to_end_reports_both_arms_and_paired_difference():
    _run("""
        X, g, yr = panel(n_comp=300, years=6)
        d, t = tg.trajectory_features(X, g, yr)
        sig = np.nan_to_num(d[:, 0])  # label depends on the trajectory
        rng = np.random.default_rng(2)
        y = (rng.random(len(g)) < 1 / (1 + np.exp(-(2 * sig - 1.5)))).astype(int)
        r = tg.run(X, y, g, yr, n_boot=30, seed=0, params={"n_estimators": 20})
        assert len(r["folds"]) == 3
        for f in r["folds"]:
            assert set(f["paired"]) == {"average_precision", "roc_auc"}
            assert f["levels+traj"]["n_features"] == 3 * f["levels"]["n_features"]
        assert np.mean([f["paired"]["roc_auc"]["diff"] for f in r["folds"]]) > 0
    """)
