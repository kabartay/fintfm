"""Training-loop tests: the properties that a loss curve cannot reveal."""

import pytest
import torch

from fintfm.modeling.model import ModelConfig
from fintfm.prior import PriorConfig


def test_resuming_a_chunked_run_matches_an_uninterrupted_one(tmp_path):
    """Two 6-step chunks must produce the same weights as one 12-step run.

    This is the property that makes long runs splittable across jobs, and every part of it
    is invisible in a loss curve if it breaks: a cold-restarted AdamW, a restarted cosine
    schedule, or a replayed synthetic task sequence all produce plausible-looking training
    output and a different model. `openspec/changes/...` scale work depends on this holding,
    since a 10^7-task run cannot fit in one job's wall-clock.
    """
    import torch
    from fintfm.modeling.train import TrainConfig, train

    model_cfg = ModelConfig(
        max_features=8, max_classes=2, d_cell=8, d_model=16, n_layers=1, n_col_layers=1,
        n_heads=2,
    )
    prior_cfg = PriorConfig(max_features=8, max_classes=2, n_rows=32)
    common = dict(steps=12, batch_size=2, eval_every=1000, log_every=1000, device="cpu", seed=3)

    whole = train(model_cfg, prior_cfg, TrainConfig(**common), str(tmp_path / "whole.pt"))

    part = tmp_path / "part.pt"
    train(model_cfg, prior_cfg, TrainConfig(**common, run_steps=6, checkpoint_every=0), str(part))
    resumed = train(
        model_cfg, prior_cfg,
        TrainConfig(**common, resume=f"{part}.state"),
        str(tmp_path / "resumed.pt"),
    )

    for (name, a), (_, b) in zip(whole.state_dict().items(), resumed.state_dict().items()):
        assert torch.allclose(a, b, atol=1e-6), f"{name} diverged between chunked and whole run"


def test_resume_refuses_a_different_schedule_length(tmp_path):
    """The cosine curve is a function of total steps, so resuming against a different
    ``--steps`` would train the tail under a curve the head never saw. That must be an error,
    not a silently different run."""
    from fintfm.modeling.train import TrainConfig, train

    model_cfg = ModelConfig(
        max_features=8, max_classes=2, d_cell=8, d_model=16, n_layers=1, n_col_layers=1,
        n_heads=2,
    )
    prior_cfg = PriorConfig(max_features=8, max_classes=2, n_rows=32)
    out = tmp_path / "m.pt"
    train(
        model_cfg, prior_cfg,
        TrainConfig(steps=8, run_steps=4, batch_size=2, eval_every=100, log_every=100,
                    device="cpu", seed=0),
        str(out),
    )
    with pytest.raises(ValueError, match="schedule"):
        train(
            model_cfg, prior_cfg,
            TrainConfig(steps=99, resume=f"{out}.state", batch_size=2, eval_every=100,
                        log_every=100, device="cpu", seed=0),
            str(tmp_path / "x.pt"),
        )
