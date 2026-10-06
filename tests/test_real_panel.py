"""The generic real-panel sampler and its `PriorConfig.real_panels` wiring (FinTFM-R).

Fixture panels only -- the real files never enter this repository. Covers what fails silently:
spec parsing, bookkeeping exclusion, source labelling, the empty default never touching disk,
mixed batches collating, and loss actually decreasing when training on a learnable panel.
"""

import numpy as np
import pandas as pd
import pytest

from fintfm.prior.mixture import PriorConfig, sample_batch, sample_task
from fintfm.prior.real_panel import (
    feature_columns,
    parse_real_panels,
    sample_real_panel_task,
    source_name,
)


def _panel(n: int = 200, n_features: int = 8, rate: float = 0.1, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {f"f{i}": rng.normal(size=n) for i in range(n_features)}
    data["_row"] = np.arange(n)
    data["label"] = (rng.random(n) < rate).astype(np.int64)
    return pd.DataFrame(data)


def _write(tmp_path, name: str, panel: pd.DataFrame) -> str:
    d = tmp_path / name
    d.mkdir()
    path = d / "panel.parquet"
    panel.to_parquet(path)
    return str(path)


def test_parse_accepts_well_formed_specs():
    assert parse_real_panels("") == ()
    assert parse_real_panels("a/panel.parquet=0.25, b/panel.parquet=1") == (
        ("a/panel.parquet", 0.25), ("b/panel.parquet", 1.0),
    )


@pytest.mark.parametrize("spec", ["a.parquet", "a.parquet=0", "a.parquet=1.5", "a=0.1,a=0.2"])
def test_parse_rejects_malformed_specs(spec):
    with pytest.raises(ValueError):
        parse_real_panels(spec)


def test_bookkeeping_columns_are_never_features():
    panel = _panel().assign(loan_id=1, cik=2)
    cols = feature_columns(panel)
    assert not {"label", "_row", "loan_id", "cik"} & set(cols)
    assert len(cols) == 8


def test_task_shape_and_source_label(tmp_path):
    path = _write(tmp_path, "sba", _panel())
    task = sample_real_panel_task(np.random.default_rng(1), 50, panel_path=path, max_features=20)
    assert task.X.shape == (50, 8)
    assert task.source == "real:sba" == source_name(path)
    assert len(np.unique(task.y)) == 2


def test_rows_are_drawn_without_replacement(tmp_path):
    panel = _panel(n=30)
    task = sample_real_panel_task(
        np.random.default_rng(2), 30, panel_path="x/panel.parquet", max_features=20, panel=panel
    )
    np.testing.assert_allclose(np.sort(task.X[:, 0]), np.sort(panel["f0"].to_numpy(np.float32)))


def test_refusals():
    with pytest.raises(ValueError, match="exceeds"):
        sample_real_panel_task(np.random.default_rng(3), 500, "x/p.parquet", 20, panel=_panel())
    with pytest.raises(ValueError, match="feature columns"):
        sample_real_panel_task(np.random.default_rng(3), 10, "x/p.parquet", 4, panel=_panel())


def test_min_positives_swaps_in_real_positives_not_flipped_labels():
    # A 0.5%-rate panel drawn at 64 rows usually has zero positives. With the floor, every task
    # carries >= 2 positives, and every positive row is a genuinely positive panel row -- checked
    # through a feature that encodes each row's true label, so a flipped label would show.
    n = 4000
    rng = np.random.default_rng(0)
    label = (rng.random(n) < 0.005).astype(np.int64)
    panel = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(5)})
    panel["true_label_echo"] = label.astype(float)
    panel["label"] = label
    for seed in range(50):
        task = sample_real_panel_task(
            np.random.default_rng(seed), 64, "x/p.parquet", 20, panel=panel, min_positives=2
        )
        assert task.y.sum() >= 2
        np.testing.assert_array_equal(task.X[:, -1].astype(np.int64), task.y)


def test_mixture_applies_the_positive_floor(tmp_path):
    n = 4000
    rng = np.random.default_rng(1)
    panel = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(5)})
    panel["label"] = (rng.random(n) < 0.002).astype(np.int64)
    path = _write(tmp_path, "rare", panel)
    cfg = PriorConfig(real_panels=f"{path}=1.0", max_features=20, n_rows=64)
    draws = [sample_task(np.random.default_rng(s), cfg, n_rows=64) for s in range(30)]
    assert all(t.y.sum() >= 2 for t in draws)


def test_degenerate_draw_is_repaired():
    panel = _panel(n=40, rate=0.0)
    task = sample_real_panel_task(np.random.default_rng(4), 40, "x/p.parquet", 20, panel=panel)
    assert len(np.unique(task.y)) == 2


def test_empty_default_never_opens_a_file():
    cfg = PriorConfig()
    assert cfg.real_panels == ""
    rng = np.random.default_rng(5)
    for _ in range(20):
        assert not sample_task(rng, cfg, n_rows=32).source.startswith("real:")


def test_two_panels_both_fire_and_collate(tmp_path):
    a = _write(tmp_path, "ppdai", _panel(n_features=6, seed=1))
    b = _write(tmp_path, "sba", _panel(n_features=12, seed=2))
    cfg = PriorConfig(real_panels=f"{a}=0.5,{b}=0.5", max_features=20, n_rows=40)
    rng = np.random.default_rng(6)
    sources = {sample_task(rng, cfg, n_rows=40).source for _ in range(200)}
    assert {"real:ppdai", "real:sba"} <= sources
    batch = sample_batch(rng, cfg, batch_size=4)
    assert batch.X.shape == (4, 40, 20)


def test_training_on_a_learnable_panel_decreases_loss(tmp_path, monkeypatch):
    """A few steps with every draw from a real panel train, and loss falls -- not merely
    "does not crash" (same bar as test_real_edgar's equivalent)."""
    import torch

    from fintfm.modeling.model import FinancialTFM, ModelConfig
    from fintfm.modeling.train import TrainConfig, train

    rng = np.random.default_rng(0)
    n = 2000
    data = {f"f{i}": rng.normal(size=n) for i in range(6)}
    data["label"] = (rng.random(n) < 1 / (1 + np.exp(-(2.5 * data["f0"] - 1.5)))).astype(np.int64)
    path = _write(tmp_path, "learnable", pd.DataFrame(data))

    model_cfg = ModelConfig(
        max_features=10, max_classes=2, d_cell=16, d_model=32, n_heads=2,
        n_col_layers=1, n_layers=1, d_ff=32,
    )
    prior_cfg = PriorConfig(max_features=10, max_classes=2, n_rows=64, real_panels=f"{path}=1.0")
    train_cfg = TrainConfig(steps=60, batch_size=4, eval_every=1000, log_every=1000, device="cpu", seed=0)

    losses: list[float] = []
    orig = FinancialTFM.loss

    def _rec(self, X, y, n_ctx, n_classes):
        loss = orig(self, X, y, n_ctx, n_classes)
        losses.append(float(loss.item()))
        return loss

    monkeypatch.setattr(FinancialTFM, "loss", _rec)
    train(model_cfg, prior_cfg, train_cfg, str(tmp_path / "panel_smoke.pt"))
    assert all(torch.isfinite(torch.tensor(losses)))
    assert np.mean(losses[-10:]) < np.mean(losses[:10])
