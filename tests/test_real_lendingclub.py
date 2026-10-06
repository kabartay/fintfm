"""Sampling real LendingClub tasks (D17's third real source).

Every test uses a small in-memory fixture panel, never the real downloaded data -- CI has no
network access and the real panel file never enters this repository (`CLAUDE.md`'s convention
for anything that is not original code). Mirrors `tests/test_real_mortgage.py` field for field.
"""

import numpy as np
import pandas as pd
import pytest

from fintfm.prior.base import collate
from fintfm.prior.mixture import PriorConfig, sample_batch, sample_task
from fintfm.prior.real_lendingclub import feature_columns, sample_real_lendingclub_task


def _fixture_panel(n: int = 200, n_features: int = 10, seed: int = 0) -> pd.DataFrame:
    """A small, fully synthetic stand-in for a real `build_panel.py` output."""
    rng = np.random.default_rng(seed)
    data = {f"feat_{i}": rng.normal(size=n) for i in range(n_features)}
    data["loan_id"] = np.arange(n)
    # A moderate-rate label, matching the measured regime (~20% positive) rather than a toy.
    data["label"] = (rng.random(n) < 0.2).astype(np.int64)
    return pd.DataFrame(data)


def test_feature_columns_excludes_bookkeeping():
    panel = _fixture_panel()
    cols = feature_columns(panel)
    assert "label" not in cols
    assert "loan_id" not in cols
    assert len(cols) == 10


def test_sample_task_shapes_and_source():
    panel = _fixture_panel(n=200, n_features=12)
    rng = np.random.default_rng(1)
    task = sample_real_lendingclub_task(rng, n_rows=50, max_features=20, panel=panel)
    assert task.X.shape == (50, 12)
    assert task.y.shape == (50,)
    assert task.n_classes == 2
    assert task.source == "real_lendingclub"
    assert task.period is None
    assert task.is_categorical.shape == (12,)
    assert not task.is_categorical.any()


def test_sample_task_rows_are_drawn_without_replacement():
    panel = _fixture_panel(n=30, n_features=4)
    rng = np.random.default_rng(2)
    task = sample_real_lendingclub_task(rng, n_rows=30, max_features=10, panel=panel)
    expected = np.sort(panel["feat_0"].to_numpy(dtype=np.float32))
    actual = np.sort(task.X[:, 0])
    np.testing.assert_allclose(actual, expected)


def test_sample_task_refuses_more_rows_than_the_panel_has():
    panel = _fixture_panel(n=20, n_features=4)
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="exceeds the panel"):
        sample_real_lendingclub_task(rng, n_rows=50, max_features=10, panel=panel)


def test_sample_task_refuses_a_panel_outside_the_feature_bounds():
    panel = _fixture_panel(n=100, n_features=2)
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="feature columns"):
        sample_real_lendingclub_task(rng, n_rows=10, max_features=136, min_features=4, panel=panel)


def test_degenerate_single_class_draw_is_repaired():
    n = 20
    panel = _fixture_panel(n=n, n_features=4)
    panel["label"] = 0
    rng = np.random.default_rng(5)
    task = sample_real_lendingclub_task(rng, n_rows=n, max_features=10, panel=panel)
    assert len(np.unique(task.y)) == 2


def test_p_real_lendingclub_zero_never_touches_the_panel_path():
    cfg = PriorConfig(
        p_real_lendingclub=0.0, real_lendingclub_panel_path="/does/not/exist.parquet"
    )
    rng = np.random.default_rng(6)
    for _ in range(20):
        sample_task(rng, cfg, n_rows=32)  # must not raise


def test_p_real_lendingclub_one_always_draws_from_the_panel(tmp_path):
    panel = _fixture_panel(n=100, n_features=6)
    path = tmp_path / "panel.parquet"
    panel.to_parquet(path)
    cfg = PriorConfig(
        p_real_lendingclub=1.0, max_features=20, real_lendingclub_panel_path=str(path),
        n_rows=40,
    )
    rng = np.random.default_rng(7)
    task = sample_task(rng, cfg, n_rows=40)
    assert task.source == "real_lendingclub"


def test_batch_with_p_real_lendingclub_collates(tmp_path):
    panel = _fixture_panel(n=100, n_features=6)
    path = tmp_path / "panel.parquet"
    panel.to_parquet(path)
    cfg = PriorConfig(
        p_real_lendingclub=1.0, max_features=20, real_lendingclub_panel_path=str(path),
        n_rows=40,
    )
    rng = np.random.default_rng(8)
    batch = sample_batch(rng, cfg, batch_size=4)
    assert batch.X.shape == (4, 40, 20)
    assert batch.y.shape == (4, 40)


def test_real_lendingclub_task_collates_fine_next_to_synthetic_tasks(tmp_path):
    panel = _fixture_panel(n=200, n_features=6)
    path = tmp_path / "panel.parquet"
    panel.to_parquet(path)
    real_task = sample_real_lendingclub_task(
        np.random.default_rng(9), n_rows=32, max_features=20, panel=panel
    )
    from fintfm.prior.financial import sample_financial_task

    synth_task = sample_financial_task(np.random.default_rng(10), n_rows=32, max_features=20)
    batch = collate([real_task, synth_task], n_ctx=16, max_features=20)
    assert batch.X.shape == (2, 32, 20)


def _learnable_fixture_panel(n: int, n_features: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {f"feat_{i}": rng.normal(size=n) for i in range(n_features)}
    logit = 2.5 * data["feat_0"] - 1.5
    data["label"] = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(np.int64)
    data["loan_id"] = np.arange(n)
    return pd.DataFrame(data)


def test_training_at_p_real_lendingclub_one_decreases_loss(tmp_path, monkeypatch):
    """A few steps at p_real_lendingclub=1.0 train, and the loss actually decreases -- not
    merely "does not crash". Mirrors tests/test_real_mortgage.py's equivalent test."""
    import torch

    from fintfm.modeling.model import FinancialTFM, ModelConfig
    from fintfm.modeling.train import TrainConfig, train

    panel = _learnable_fixture_panel(n=2000, n_features=6, seed=0)
    path = tmp_path / "panel.parquet"
    panel.to_parquet(path)

    model_cfg = ModelConfig(
        max_features=10, max_classes=2, d_cell=16, d_model=32, n_heads=2,
        n_col_layers=1, n_layers=1, d_ff=32,
    )
    prior_cfg = PriorConfig(
        max_features=10, max_classes=2, n_rows=64, p_real_lendingclub=1.0,
        real_lendingclub_panel_path=str(path),
    )
    train_cfg = TrainConfig(
        steps=60, batch_size=4, eval_every=1000, log_every=1000, device="cpu", seed=0,
    )

    losses: list[float] = []
    orig_loss = FinancialTFM.loss

    def _recording_loss(self, X, y, n_ctx, n_classes):
        loss = orig_loss(self, X, y, n_ctx, n_classes)
        losses.append(float(loss.item()))
        return loss

    monkeypatch.setattr(FinancialTFM, "loss", _recording_loss)
    train(model_cfg, prior_cfg, train_cfg, str(tmp_path / "real_lendingclub_smoke.pt"))

    assert all(torch.isfinite(torch.tensor(losses)))
    assert np.mean(losses[-10:]) < np.mean(losses[:10])
