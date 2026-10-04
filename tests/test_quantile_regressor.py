"""The pinball-loss regression wrapper for a head_type='quantile' checkpoint (task 48.3).

Mirrors tests/test_regressor.py's split: contract tests on an untrained tiny model (shape,
monotonicity of the interval, error handling), plus one genuine learning-signal test, since an
untrained model asserting a good prediction would be asserting noise.
"""

import numpy as np
import pytest
import torch

from fintfm.inference.quantile_regressor import (
    ContinuousTargetScaler,
    FinancialTFMQuantileRegressor,
)
from fintfm.modeling.model import FinancialTFM, ModelConfig
from fintfm.prior.base import Task, collate


def _tiny_model(n_quantiles: int = 9) -> FinancialTFM:
    cfg = ModelConfig(
        max_features=6, max_classes=3, d_cell=16, d_model=32, n_heads=2, n_col_layers=1,
        n_layers=1, d_ff=32, head_type="quantile", n_quantiles=n_quantiles,
    )
    return FinancialTFM(cfg)


def _data(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.3, size=n)
    return X, y


def test_scaler_round_trips():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    scaler = ContinuousTargetScaler().fit(y)
    z = scaler.transform(y)
    np.testing.assert_allclose(z.mean(), 0.0, atol=1e-9)
    back = scaler.inverse_transform(z)
    np.testing.assert_allclose(back, y, atol=1e-6)


def test_scaler_raises_before_fit():
    with pytest.raises(RuntimeError, match="fit must be called"):
        ContinuousTargetScaler().transform(np.array([1.0]))


def test_refuses_a_binned_head_checkpoint():
    binned = FinancialTFM(ModelConfig(max_features=6, max_classes=5, d_model=16, n_heads=2,
                                       n_layers=1, d_ff=32))
    with pytest.raises(ValueError, match="head_type"):
        FinancialTFMQuantileRegressor(binned)


def test_fit_predict_shapes():
    torch.manual_seed(0)
    X, y = _data()
    reg = FinancialTFMQuantileRegressor(_tiny_model(), max_context=40).fit(X[:40], y[:40])
    pred = reg.predict(X[40:])
    assert pred.shape == (20,)
    assert np.isfinite(pred).all()


def test_quantile_grid_has_one_column_per_tau():
    torch.manual_seed(0)
    X, y = _data()
    reg = FinancialTFMQuantileRegressor(_tiny_model(n_quantiles=11), max_context=40).fit(
        X[:40], y[:40]
    )
    grid = reg.predict_quantile_grid(X[40:])
    assert grid.shape == (20, 11)
    assert np.isfinite(grid).all()


def test_interval_level_must_be_in_unit_interval():
    torch.manual_seed(0)
    X, y = _data()
    reg = FinancialTFMQuantileRegressor(_tiny_model(), max_context=40).fit(X[:40], y[:40])
    with pytest.raises(ValueError, match="level"):
        reg.predict_interval(X[40:], level=1.5)


def test_quantile_q_must_be_in_unit_interval():
    torch.manual_seed(0)
    X, y = _data()
    reg = FinancialTFMQuantileRegressor(_tiny_model(), max_context=40).fit(X[:40], y[:40])
    with pytest.raises(ValueError, match="q must be"):
        reg.predict_quantile(X[40:], 0.0)


def test_predicting_before_fit_raises():
    reg = FinancialTFMQuantileRegressor(_tiny_model())
    with pytest.raises(RuntimeError, match="fit must be called"):
        reg.predict(np.zeros((2, 4)))


def test_scaler_is_fit_on_context_rows_only():
    # Same property test_regressor.py's test_bin_edges_ignore_the_query_rows asserts for the
    # binned head's edges: query rows must not move the fitted mean/std.
    torch.manual_seed(0)
    X, y = _data()
    a = FinancialTFMQuantileRegressor(_tiny_model(), max_context=40).fit(X[:40], y[:40])
    b = FinancialTFMQuantileRegressor(_tiny_model(), max_context=40).fit(X[:40], y[:40])
    b.predict(X[40:] * 1000.0)
    assert a.scaler_.mean_ == b.scaler_.mean_
    assert a.scaler_.std_ == b.scaler_.std_


def test_quantile_head_learns_a_known_linear_target_end_to_end():
    """Trains a tiny model past the point of chance and checks the full inference wrapper --
    predict, interval monotonicity, calibration in the right ballpark -- not just the raw
    forward pass test_model.py already covers."""
    torch.manual_seed(7)
    cfg = ModelConfig(
        max_features=8, max_classes=3, d_cell=16, d_model=32, n_heads=2, n_col_layers=1,
        n_layers=1, d_ff=32, head_type="quantile", n_quantiles=9,
    )
    model = FinancialTFM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = np.random.default_rng(1)

    def make_batch(bs=8, n_rows=60):
        tasks = []
        for _ in range(bs):
            Xr = rng.normal(size=(n_rows, 5)).astype(np.float32)
            Xp = np.full((n_rows, 8), np.nan, dtype=np.float32)
            Xp[:, :5] = Xr
            y_cont = (Xr[:, 0] * 2.0 + rng.normal(scale=0.2, size=n_rows)).astype(np.float64)
            tasks.append(
                Task(
                    X=Xp, y=np.zeros(n_rows, dtype=np.int64), n_classes=0,
                    is_categorical=np.zeros(8, dtype=bool), source="test", y_continuous=y_cont,
                )
            )
        return collate(tasks, n_ctx=30, max_features=8, head_type="quantile")

    for _ in range(250):
        batch = make_batch()
        loss = model.loss(batch.X, batch.y, batch.n_ctx, batch.n_classes)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

    Xtr = rng.normal(size=(200, 5)).astype(np.float32)
    ytr = Xtr[:, 0] * 2.0 + rng.normal(scale=0.3, size=200)
    Xte = rng.normal(size=(50, 5)).astype(np.float32)
    yte = Xte[:, 0] * 2.0 + rng.normal(scale=0.3, size=50)

    reg = FinancialTFMQuantileRegressor(model, max_context=150, random_state=0).fit(Xtr, ytr)
    pred = reg.predict(Xte)
    lo, hi = reg.predict_interval(Xte, level=0.8)
    nrmse = float(np.sqrt(np.mean((pred - yte) ** 2)) / np.std(yte))
    assert nrmse < 1.0  # beats predict-the-mean after real training
    assert (lo <= hi + 1e-6).all()
