"""Openspec 34.1: the survival loss plus a weighted classification loss, and what it records."""

import numpy as np
import pytest
import torch

from fintfm.config import ConfigError, load_config
from fintfm.modeling import ModelConfig
from fintfm.modeling.model import FinancialTFM
from fintfm.modeling.train import (
    TrainConfig,
    _build_optimizer,
    _training_step,
    build_configs,
    build_parser,
    train,
)
from fintfm.prior import PriorConfig
from fintfm.prior.mixture import sample_batch

_MODEL = {
    "d_model": 16, "n_heads": 2, "n_layers": 1, "n_col_layers": 1, "d_ff": 32, "d_cell": 8, "max_features": 8,
    "max_classes": 2, "n_cell_blocks": 0, "cell_labels": False, "column_id_dim": None, "pooling": "meanmax",
    "n_horizons": 4,
}
_PRIOR = PriorConfig(max_features=8, max_classes=2, p_financial=1.0, n_rows=48, n_horizons=4)


def _fresh(seed: int = 0) -> tuple[FinancialTFM, object]:
    """A seeded tiny hazard model and one batch carrying periods."""
    torch.manual_seed(seed)
    model = FinancialTFM(ModelConfig(**_MODEL))
    return model, sample_batch(np.random.default_rng(seed), _PRIOR, 2)


def _step(weight: float | None, objectives: set[str]) -> torch.Tensor:
    """One optimiser step from the seeded state; ``None`` uses the pre-34.1 call signature."""
    model, batch = _fresh()
    opt, sched = _build_optimizer(model, TrainConfig(steps=2, batch_size=2))
    if weight is None:
        return _training_step(model, batch, opt, sched, "none", objectives)
    return _training_step(model, batch, opt, sched, "none", objectives, None, None, weight)


def test_default_step_is_the_survival_loss_alone_and_records_only_survival():
    model, batch = _fresh()
    expected = model.survival_loss(batch.X, batch.y, batch.period, batch.n_ctx).item()
    objectives: set[str] = set()
    loss = _step(0.0, objectives)
    assert loss.item() == expected
    assert objectives == {"survival"}
    assert _step(None, set()).item() == expected


def test_weighted_step_sums_both_terms_with_the_given_weight():
    model, batch = _fresh()
    surv = model.survival_loss(batch.X, batch.y, batch.period, batch.n_ctx).item()
    cls = model.loss(batch.X, batch.y, batch.n_ctx, batch.n_classes).item()
    objectives: set[str] = set()
    loss = _step(0.5, objectives)
    assert loss.item() == pytest.approx(surv + 0.5 * cls, rel=1e-5)
    assert objectives == {"survival", "classification"}


def _run(tmp_path, name: str, weight: float) -> FinancialTFM:
    """Train three CPU steps and return the reloaded checkpoint."""
    out = tmp_path / name
    train(
        ModelConfig(**_MODEL), _PRIOR,
        TrainConfig(steps=3, batch_size=2, eval_every=2, log_every=1000, seed=0,
                    device="cpu", joint_classification_weight=weight),
        str(out),
    )
    return FinancialTFM.load(str(out))


def test_joint_checkpoint_passes_the_guard_for_both_paths_and_records_the_weight(tmp_path):
    joint = _run(tmp_path, "joint.pt", 0.3)
    joint.assert_trained_for("survival")
    joint.assert_trained_for("classification")
    ckpt = torch.load(tmp_path / "joint.pt", weights_only=False)
    assert ckpt["provenance"]["train_config"]["joint_classification_weight"] == 0.3

    only = _run(tmp_path, "base.pt", 0.0)
    only.assert_trained_for("survival")
    with pytest.raises(Exception, match="classification"):
        only.assert_trained_for("classification")


def test_nonsense_combinations_are_refused(tmp_path):
    out = str(tmp_path / "x.pt")
    with pytest.raises(ValueError, match="no hazard head"):
        train(
            ModelConfig(**{**_MODEL, "n_horizons": None}),
            PriorConfig(max_features=8, max_classes=2, n_rows=48),
            TrainConfig(steps=1, batch_size=2, joint_classification_weight=1.0), out,
        )
    with pytest.raises(ValueError, match="at least 0"):
        train(
            ModelConfig(**_MODEL), _PRIOR,
            TrainConfig(steps=1, batch_size=2, joint_classification_weight=-1.0), out,
        )


def test_flag_and_config_reach_train_config_and_a_negative_config_is_refused(tmp_path):
    args = build_parser().parse_args(["--joint-classification-weight", "0.25"])
    assert build_configs(args, load_config())[2].joint_classification_weight == 0.25
    default = build_configs(build_parser().parse_args([]), load_config())[2]
    assert default.joint_classification_weight == 0.0
    f = tmp_path / "c.yaml"
    f.write_text("train:\n  joint_classification_weight: -0.1\n")
    with pytest.raises(ConfigError, match="joint_classification_weight"):
        load_config(f)
