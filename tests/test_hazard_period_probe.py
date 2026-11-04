"""AU5.9's Verify generator and scorer: invariances, floors and the pass rule, on tiny data."""

import numpy as np
import pytest
import torch

from fintfm.experiments import hazard_period_probe as hp
from fintfm.modeling.hazard import CENSORED
from fintfm.modeling.model import FinancialTFM, ModelConfig

CFG = hp.ProbeConfig(n_horizons=4, n_features=6, n_ctx=40, n_query=20)


def test_regimes_share_features_and_binary_labels_and_differ_only_in_periods():
    a, b = hp.make_task(CFG, "early", 3), hp.make_task(CFG, "late", 3)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.y, b.y)
    np.testing.assert_array_equal(a.p, b.p)
    assert not np.array_equal(a.period, b.period)
    for t in (a, b):
        np.testing.assert_array_equal(t.period != CENSORED, t.y == 1)


def test_periods_cluster_early_or_late_over_many_defaults():
    big = hp.ProbeConfig(n_horizons=4, n_ctx=4000, n_query=10)
    e, late = hp.make_task(big, "early", 1), hp.make_task(big, "late", 1)
    assert e.period[e.y == 1].mean() < 1.0
    assert late.period[late.y == 1].mean() > 2.0


def test_cdf_is_monotone_ends_at_one_and_regimes_mirror():
    e, late = hp.regime_cdf("early", 5, 0.5), hp.regime_cdf("late", 5, 0.5)
    assert np.all(np.diff(e) >= 0) and e[-1] == pytest.approx(1.0)
    assert e[0] > late[0]
    with pytest.raises(ValueError):
        hp.regime_cdf("sideways", 5, 0.5)
    with pytest.raises(ValueError):
        hp.regime_cdf("early", 5, 1.0)


def test_blind_periods_keep_censoring_and_drop_regime_information():
    big = hp.ProbeConfig(n_horizons=4, n_ctx=4000, n_query=10)
    e = hp.make_task(big, "early", 1)
    bl = hp.blind_periods(e, 4, 0)
    np.testing.assert_array_equal(bl == CENSORED, e.y == 0)
    assert abs(bl[e.y == 1].mean() - 1.5) < 0.15  # uniform on 0..3


def test_oracle_has_zero_error_and_blind_mixture_sits_on_the_floor():
    tasks = hp.make_tasks(CFG, 6)
    fl = hp.floors(CFG, tasks)
    f = {r: hp.regime_cdf(r, 4, CFG.decay) for r in hp.REGIMES}
    mix = 0.5 * (f["early"] + f["late"])
    for t in tasks:
        pq = t.p[t.n_ctx :]
        oracle = hp.task_errors(CFG, t, pq[:, None] * f[t.regime][None, :])
        assert oracle["shape_rmse"] == pytest.approx(0)
        assert oracle["cum_rmse"] == pytest.approx(0)
        blind = hp.task_errors(CFG, t, pq[:, None] * mix[None, :])
        assert blind["shape_rmse"] == pytest.approx(fl["shape_floor"])
    assert fl["shape_floor"] > 0 and fl["cum_floor"] > 0


def test_verdict_rules():
    base = {
        "reads_periods": True,
        "floors": {"shape_floor": 0.2},
        "with_periods": {"shape_rmse": 0.05},
        "paired_with_minus_blind": {"shape_rmse": {"ci95": [-0.2, -0.1]}},
    }
    assert hp.verdict(base) == {
        "pass": True, "strong": True, "below_floor": True, "ci_below_zero": True,
    }
    assert not hp.verdict({**base, "reads_periods": False})["pass"]
    assert not hp.verdict({**base, "with_periods": {"shape_rmse": 0.3}})["pass"]
    wide = {**base, "paired_with_minus_blind": {"shape_rmse": {"ci95": [-0.2, 0.1]}}}
    assert not hp.verdict(wide)["pass"]


def _tiny_ckpt(path, on: bool) -> str:
    torch.manual_seed(0)
    m = FinancialTFM(
        ModelConfig(max_features=6, max_classes=2, d_cell=8, d_model=16, n_heads=2,
                    n_col_layers=1, n_layers=1, d_ff=32, n_horizons=4, hazard_period_context=on)
    )
    torch.save({"config": m.cfg.__dict__, "state_dict": m.state_dict()}, path)
    return str(path)


@pytest.mark.parametrize("on", [True, False])
def test_scorer_runs_on_an_untrained_checkpoint(tmp_path, on):
    tasks = hp.make_tasks(CFG, 4)
    rec = hp.score_checkpoint(_tiny_ckpt(tmp_path / "m.pt", on), CFG, tasks)
    assert rec["reads_periods"] is on and rec["n_tasks"] == 4
    assert "pass" in rec["verdict"] and "shape RMSE" in hp.render(rec)
    if not on:  # periods are ignored, so both arms coincide
        assert rec["with_periods"] == rec["blind"]
        assert not rec["verdict"]["pass"]


def test_scorer_refuses_a_horizon_mismatch(tmp_path):
    other = hp.ProbeConfig(n_horizons=5, n_ctx=40, n_query=20)
    with pytest.raises(ValueError, match="horizons"):
        hp.score_checkpoint(_tiny_ckpt(tmp_path / "m.pt", True), other, hp.make_tasks(other, 2))
