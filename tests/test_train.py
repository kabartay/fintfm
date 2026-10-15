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
    from fintfm.modeling.train import TrainConfig, train

    model_cfg = ModelConfig(
        max_features=8, max_classes=2, d_cell=8, d_model=16, n_layers=1, n_col_layers=1,
        n_heads=2,
    )
    prior_cfg = PriorConfig(max_features=8, max_classes=2, n_rows=32)
    common = {
        "steps": 12, "batch_size": 2, "eval_every": 1000, "log_every": 1000,
        "device": "cpu", "seed": 3,
    }

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


def test_train_config_defaults_to_adamw():
    from fintfm.modeling.train import TrainConfig

    assert TrainConfig().optimizer == "adamw"


def test_schedulefree_run_completes_and_saves_a_usable_checkpoint(tmp_path):
    """Task 48.9: the training loop must run end to end under the alternative optimiser,
    including the train/eval weight-set switching a checkpoint save requires."""
    pytest.importorskip("schedulefree")
    from fintfm.modeling.train import TrainConfig, train

    model_cfg = ModelConfig(
        max_features=8, max_classes=2, d_cell=8, d_model=16, n_layers=1, n_col_layers=1,
        n_heads=2,
    )
    prior_cfg = PriorConfig(max_features=8, max_classes=2, n_rows=32)
    model = train(
        model_cfg, prior_cfg,
        TrainConfig(steps=4, batch_size=2, eval_every=100, log_every=100, device="cpu",
                    seed=0, optimizer="schedulefree"),
        str(tmp_path / "sf.pt"),
    )
    X = torch.randn(1, 8, 8)
    y = torch.randint(0, 2, (1, 8))
    with torch.no_grad():
        logits = model(X, y, 4, torch.tensor([2]))
    assert torch.isfinite(logits).all()


def test_schedulefree_run_extends_past_its_original_steps_without_restarting(tmp_path):
    """Task 48.9's second, more important property: unlike the cosine schedule, a
    schedule-free run can be resumed against a *larger* total step count, because there is no
    external LR curve whose shape depends on the original total."""
    pytest.importorskip("schedulefree")
    from fintfm.modeling.train import TrainConfig, train

    model_cfg = ModelConfig(
        max_features=8, max_classes=2, d_cell=8, d_model=16, n_layers=1, n_col_layers=1,
        n_heads=2,
    )
    prior_cfg = PriorConfig(max_features=8, max_classes=2, n_rows=32)
    out = tmp_path / "sf.pt"
    train(
        model_cfg, prior_cfg,
        TrainConfig(steps=4, batch_size=2, eval_every=100, log_every=100, device="cpu",
                    seed=0, optimizer="schedulefree"),
        str(out),
    )
    # steps=10 is *larger* than the original run's steps=4 -- exactly what AdamW's cosine
    # schedule refuses by name (test_resume_refuses_a_different_schedule_length above).
    extended = train(
        model_cfg, prior_cfg,
        TrainConfig(steps=10, resume=f"{out}.state", batch_size=2, eval_every=100,
                    log_every=100, device="cpu", seed=0, optimizer="schedulefree"),
        str(tmp_path / "sf_extended.pt"),
    )
    X = torch.randn(1, 8, 8)
    y = torch.randint(0, 2, (1, 8))
    with torch.no_grad():
        logits = extended(X, y, 4, torch.tensor([2]))
    assert torch.isfinite(logits).all()


def test_resume_refuses_mismatched_optimizer(tmp_path):
    """Loading schedule-free's optimiser state into AdamW (or back) would corrupt training
    silently rather than raise on its own -- the state dicts carry different keys."""
    pytest.importorskip("schedulefree")
    from fintfm.modeling.train import TrainConfig, train

    model_cfg = ModelConfig(
        max_features=8, max_classes=2, d_cell=8, d_model=16, n_layers=1, n_col_layers=1,
        n_heads=2,
    )
    prior_cfg = PriorConfig(max_features=8, max_classes=2, n_rows=32)
    out = tmp_path / "sf.pt"
    train(
        model_cfg, prior_cfg,
        TrainConfig(steps=4, batch_size=2, eval_every=100, log_every=100, device="cpu",
                    seed=0, optimizer="schedulefree"),
        str(out),
    )
    with pytest.raises(ValueError, match="optimizer"):
        train(
            model_cfg, prior_cfg,
            TrainConfig(steps=8, resume=f"{out}.state", batch_size=2, eval_every=100,
                        log_every=100, device="cpu", seed=0, optimizer="adamw"),
            str(tmp_path / "x.pt"),
        )


def test_evaluation_does_not_change_the_training_batches(tmp_path, monkeypatch):
    """Audit AU1.18: evaluation used to draw its held-out tasks from the training generator,
    so changing ``eval_every`` changed every training batch after the first evaluation. Two
    runs differing only in ``eval_every`` must see identical training batches and end with
    identical weights."""
    import importlib

    from fintfm.modeling.train import TrainConfig, train

    # the package re-exports a function named `train`, which shadows the module attribute
    train_mod = importlib.import_module("fintfm.modeling.train")

    model_cfg = ModelConfig(
        max_features=8, max_classes=2, d_cell=8, d_model=16, n_layers=1, n_col_layers=1,
        n_heads=2,
    )
    prior_cfg = PriorConfig(max_features=8, max_classes=2, n_rows=32)
    real_sample_batch = train_mod.sample_batch

    def run(eval_every: int, name: str):
        calls: list = []

        def recording(rng, cfg, batch_size):
            batch = real_sample_batch(rng, cfg, batch_size)
            calls.append((rng, batch.y.clone()))
            return batch

        monkeypatch.setattr(train_mod, "sample_batch", recording)
        model = train(
            model_cfg, prior_cfg,
            TrainConfig(steps=6, batch_size=2, eval_every=eval_every, log_every=1000,
                        device="cpu", seed=4),
            str(tmp_path / name),
        )
        training_rng = calls[0][0]  # the loop's first call precedes any evaluation
        return [y for rng, y in calls if rng is training_rng], model

    with_eval, a = run(2, "eval.pt")
    without_eval, b = run(1000, "noeval.pt")
    assert len(with_eval) == len(without_eval) == 6
    for ya, yb in zip(with_eval, without_eval):
        assert torch.equal(ya, yb)
    for (name, wa), (_, wb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(wa, wb), name


# --- Audit Phase 5 levers (AU5.2, AU5.10, AU5.17). Every one is behind a flag whose default
# must reproduce pretraining exactly as it ran before the flag existed.

_TINY_RICH = {
    "max_features": 8, "max_classes": 2, "d_cell": 8, "d_model": 16, "n_layers": 1,
    "n_col_layers": 1, "n_heads": 2, "n_cell_blocks": 1, "cell_labels": True,
    "pooling": "attention", "mask_embedding": True,
}


def test_default_training_matches_the_pre_lever_loop(tmp_path):
    """The levers' defaults must change no number. This is the loop as it stood before them
    (AdamW over ``model.parameters()`` at torch's default decay, 500-step warmup then cosine,
    float32), written out here so a refactor of ``train()`` cannot move the reference with it.
    Weights must be *equal*, not close, after five steps."""
    import numpy as np

    from fintfm.modeling.model import FinancialTFM
    from fintfm.modeling.train import TrainConfig, train
    from fintfm.prior.mixture import sample_batch

    cfg = TrainConfig()
    assert (cfg.optimizer, cfg.amp, cfg.decay_norm_bias, cfg.warmup_steps) == (
        "adamw", "none", True, 500,
    )
    model_cfg = ModelConfig(**_TINY_RICH)
    prior_cfg = PriorConfig(max_features=8, max_classes=2, n_rows=32)
    steps, seed = 5, 7

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ref = FinancialTFM(model_cfg)
    opt = torch.optim.AdamW(ref.parameters(), lr=3e-4)

    def multiplier(s: int) -> float:
        if s < 500:
            return (s + 1) / 500
        return 0.5 * (1 + np.cos(np.pi * (s - 500) / max(1, steps - 500)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, multiplier)
    for _ in range(steps):
        batch = sample_batch(rng, prior_cfg, 2)
        loss = ref.loss(batch.X, batch.y, batch.n_ctx, batch.n_classes)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
        opt.step()
        sched.step()

    # The run under test passes every model lever explicitly at the packaged config's value
    # (n_heads aside, which this tiny model sets itself), so the levers' defaults are held to
    # the same byte-for-byte standard as the training ones.
    from fintfm.config import load_config
    from fintfm.modeling.train import _MODEL_CONFIG_KNOBS

    model_section = load_config().model
    levers_at_default = ModelConfig(**_TINY_RICH, **{
        k: getattr(model_section, k) for k in _MODEL_CONFIG_KNOBS if k != "n_heads"
    })
    got = train(
        levers_at_default, prior_cfg,
        TrainConfig(steps=steps, batch_size=2, eval_every=1000, log_every=1000, seed=seed),
        str(tmp_path / "m.pt"),
    )
    assert list(ref.state_dict()) == list(got.state_dict())
    for (name, a), (_, b) in zip(ref.state_dict().items(), got.state_dict().items()):
        assert torch.equal(a, b), name


def test_cli_without_model_lever_flags_builds_the_pre_lever_architecture(tmp_path, monkeypatch):
    """``fintfm-train`` with no lever flag must build exactly the ``ModelConfig`` it built
    before the levers existed: every new field at its old default and ``n_heads`` at 4."""
    from dataclasses import asdict

    model_cfg, _ = _captured_configs(monkeypatch, ["--out", str(tmp_path / "x.pt")])
    before = ModelConfig(
        pooling="meanmax", column_id_dim=None, n_cell_blocks=0, cell_labels=False,
        mask_embedding=False, head_type="binned", n_quantiles=99, d_ff=4 * 192,
        max_features=24, max_classes=10, d_cell=64, d_model=192, n_col_layers=2, n_layers=6,
        n_horizons=None,
    )
    assert asdict(model_cfg) == asdict(before)
    assert model_cfg.n_heads == 4


def test_bf16_autocast_runs_the_forward_in_bf16_and_keeps_loss_and_weights_float32(tmp_path):
    """AU5.2: under ``amp="bf16"`` the forward pass computes in bf16, the cross-entropy and
    the weights stay float32, and the logits agree with float32 within bf16 tolerance on a
    fixed batch. The tolerance is a smoke bound for a tiny model, not an accuracy claim."""
    from fintfm.modeling.model import FinancialTFM
    from fintfm.modeling.train import TrainConfig, _autocast, train

    torch.manual_seed(0)
    model = FinancialTFM(ModelConfig(**_TINY_RICH)).eval()
    X, y = torch.randn(2, 32, 8), torch.randint(0, 2, (2, 32))
    n_classes = torch.tensor([2, 2])
    with torch.no_grad():
        full = model(X, y, 16, n_classes)
        with _autocast(model, "bf16"):
            half = model(X, y, 16, n_classes)
            loss = model.loss(X, y, 16, n_classes)
    assert half.dtype == torch.bfloat16
    assert loss.dtype == torch.float32
    assert torch.allclose(full, half.float(), atol=0.05)

    prior_cfg = PriorConfig(max_features=8, max_classes=2, n_rows=32)
    common = {"steps": 3, "batch_size": 2, "eval_every": 2, "log_every": 1000, "seed": 1}
    fp32 = train(ModelConfig(**_TINY_RICH), prior_cfg, TrainConfig(**common),
                 str(tmp_path / "a.pt"))
    bf16 = train(ModelConfig(**_TINY_RICH), prior_cfg, TrainConfig(**common, amp="bf16"),
                 str(tmp_path / "b.pt"))
    assert all(p.dtype == torch.float32 and torch.isfinite(p).all() for p in bf16.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(fp32.parameters(), bf16.parameters()))


@pytest.mark.parametrize("amp", ["bf16", "fp16"])
@pytest.mark.parametrize(
    "extra",
    [{"head_type": "quantile"}, {"n_horizons": 3}],
    ids=["quantile", "survival"],
)
def test_half_precision_refuses_objectives_whose_loss_would_reduce_in_it(extra, amp):
    """The pinball and survival losses are not on autocast's float32 list; under bf16 or
    fp16 they would be reduced in half precision (the survival NLL takes ``log1p(-h)`` of
    hazards bf16 rounds to one), so the flag must refuse them rather than train on a degraded
    loss. Checked before the device, so this runs on CPU for fp16 too."""
    from fintfm.modeling.model import FinancialTFM
    from fintfm.modeling.train import _autocast

    cfg = {**_TINY_RICH, "cell_labels": False, **extra}
    with pytest.raises(ValueError, match="binned classification"):
        _autocast(FinancialTFM(ModelConfig(**cfg)), amp)


def test_amp_refuses_an_unknown_mode_fp16_off_cuda_and_mps():
    from fintfm.modeling.model import FinancialTFM
    from fintfm.modeling.train import _autocast

    model = FinancialTFM(ModelConfig(**_TINY_RICH))
    with pytest.raises(ValueError, match="amp must be"):
        _autocast(model, "fp8")
    with pytest.raises(ValueError, match="fp16 is for CUDA"):
        _autocast(model, "fp16")  # the model is on CPU
    if not torch.backends.mps.is_available():
        pytest.skip("MPS refusal needs an MPS device to place the model on")
    for amp in ("bf16", "fp16"):
        with pytest.raises(ValueError, match="MPS"):
            _autocast(model.to("mps"), amp)


def _one_step(model, batch, scaler, loss_override=None):
    """One ``_training_step`` of a fresh AdamW on ``model`` with a fixed column-id draw."""
    from fintfm.modeling.train import TrainConfig, _build_optimizer, _training_step

    opt, sched = _build_optimizer(model, TrainConfig())
    if loss_override is not None:
        model.loss = loss_override
    torch.manual_seed(11)  # the column identities are drawn from the global stream
    _training_step(model, batch, opt, sched, "none", set(), scaler)
    return opt


def test_grad_scaler_wiring_matches_the_unscaled_step_on_cpu():
    """The fp16 path's scaler wiring, runnable without CUDA: a CPU ``GradScaler`` on a
    float32 model. Scaling by a power of two and unscaling before clipping is exact, so the
    weights after one scaled step must equal the plain step's. Unscaling *after* the clip
    would clip the scaled gradients and fail this."""
    import copy

    import numpy as np

    from fintfm.modeling.model import FinancialTFM
    from fintfm.prior.mixture import sample_batch

    torch.manual_seed(0)
    plain = FinancialTFM(ModelConfig(**_TINY_RICH))
    scaled = copy.deepcopy(plain)
    batch = sample_batch(np.random.default_rng(0), PriorConfig(max_features=8, max_classes=2,
                                                                n_rows=32), 2)
    _one_step(plain, batch, None)
    _one_step(scaled, batch, torch.amp.GradScaler("cpu", init_scale=2.0**12))
    for (name, a), (_, b) in zip(plain.state_dict().items(), scaled.state_dict().items()):
        assert torch.equal(a, b), name


def test_grad_scaler_skips_a_step_whose_gradients_overflow():
    """The point of the scaler: an inf gradient skips the optimiser step and halves the
    scale, where the plain step would write inf/NaN into the weights."""
    import copy

    import numpy as np

    from fintfm.modeling.model import FinancialTFM
    from fintfm.prior.mixture import sample_batch

    torch.manual_seed(0)
    model = FinancialTFM(ModelConfig(**_TINY_RICH))
    before = copy.deepcopy(model.state_dict())
    batch = sample_batch(np.random.default_rng(0), PriorConfig(max_features=8, max_classes=2,
                                                                n_rows=32), 2)
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0**12)
    _one_step(model, batch, scaler,
              loss_override=lambda *a, **k: (model.head.weight * float("inf")).sum())
    assert scaler.get_scale() == 2.0**11
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="--amp fp16 runs on CUDA only")
def test_fp16_trains_on_cuda_and_its_scaler_survives_a_resume(tmp_path):
    """AU5.2's T4/P100 path end to end: float32 weights after fp16 steps, the scaler's state
    in the ``.state`` file, and a resumed chunk that picks it up."""
    from fintfm.modeling.train import TrainConfig, train

    model_cfg = ModelConfig(**_TINY_RICH)
    prior_cfg = PriorConfig(max_features=8, max_classes=2, n_rows=32)
    common = {"steps": 4, "batch_size": 2, "eval_every": 2, "log_every": 1000, "seed": 0,
              "device": "cuda", "amp": "fp16"}
    part = tmp_path / "part.pt"
    train(model_cfg, prior_cfg, TrainConfig(**common, run_steps=2), str(part))
    state = torch.load(f"{part}.state", weights_only=False)
    assert "grad_scaler" in state and state["grad_scaler"]["scale"] > 0
    model = train(model_cfg, prior_cfg, TrainConfig(**common, resume=f"{part}.state"),
                  str(tmp_path / "whole.pt"))
    assert all(p.dtype == torch.float32 and torch.isfinite(p).all() for p in model.parameters())


def _decay_by_name(opt, model) -> dict[str, float]:
    """Map every parameter name to its group's weight decay, checking each is in one group."""
    names = {id(p): n for n, p in model.named_parameters()}
    out: dict[str, float] = {}
    for group in opt.param_groups:
        for p in group["params"]:
            assert names[id(p)] not in out, f"{names[id(p)]} is in two groups"
            out[names[id(p)]] = group["weight_decay"]
    assert set(out) == set(names.values()), "some parameter is in no group"
    return out


def test_decay_norm_bias_default_decays_every_parameter():
    """Today's behaviour: one AdamW group, every parameter at torch's default decay."""
    from fintfm.modeling.model import FinancialTFM
    from fintfm.modeling.train import TrainConfig, _build_optimizer

    model = FinancialTFM(ModelConfig(**_TINY_RICH))
    opt, _ = _build_optimizer(model, TrainConfig())
    assert isinstance(opt, torch.optim.AdamW) and len(opt.param_groups) == 1
    assert set(_decay_by_name(opt, model).values()) == {0.01}


def test_decay_norm_bias_false_exempts_norms_biases_and_embeddings():
    """AU5.10's Verify: LayerNorm, biases and embeddings excluded from weight decay, every
    weight matrix still decayed. ``pool_query`` is (1, 1, d_cell), so a rule of "fewer than two
    dimensions" alone would miss it; it is a token, not a weight."""
    from fintfm.modeling.model import FinancialTFM
    from fintfm.modeling.train import TrainConfig, _build_optimizer

    model = FinancialTFM(ModelConfig(**_TINY_RICH))
    opt, _ = _build_optimizer(model, TrainConfig(decay_norm_bias=False))
    wd = _decay_by_name(opt, model)
    exempt = [n for n in wd if "norm" in n or n.endswith("bias")] + [
        "mask_embed.weight", "query_token", "cell_mask_token", "pool_query",
    ]
    assert any("norm" in n for n in exempt) and any(n.endswith("bias") for n in exempt)
    for n in exempt:
        assert wd[n] == 0.0, n
    for n in ("head.weight", "row_proj.weight", "y_proj.weight", "col_id_proj.weight",
              "encoder.layers.0.linear1.weight", "encoder.layers.0.self_attn.in_proj_weight",
              "pool_attn.out_proj.weight", "value_embed.0.weight"):
        assert wd[n] == 0.01, n
    assert set(wd) - set(exempt) == {n for n, v in wd.items() if v == 0.01}


def test_decay_norm_bias_false_is_refused_where_it_would_be_inert():
    """Schedule-free applies no weight decay, so the flag would silently do nothing there."""
    from fintfm.modeling.model import FinancialTFM
    from fintfm.modeling.train import TrainConfig, _build_optimizer

    model = FinancialTFM(ModelConfig(**_TINY_RICH))
    with pytest.raises(ValueError, match="no effect"):
        _build_optimizer(model, TrainConfig(optimizer="schedulefree", decay_norm_bias=False))


def test_warmup_steps_sets_the_schedule():
    from fintfm.modeling.train import TrainConfig, _lr_schedule

    cfg = TrainConfig(steps=10, warmup_steps=4)
    assert [_lr_schedule(s, cfg) for s in range(4)] == [0.25, 0.5, 0.75, 1.0]
    assert _lr_schedule(4, cfg) == pytest.approx(1.0)
    assert _lr_schedule(9, cfg) < 0.1


def _captured_configs(monkeypatch, argv):
    """Run ``fintfm-train``'s ``main`` with ``train`` stubbed out; return (ModelConfig,
    TrainConfig) as the CLI built them."""
    import importlib

    train_mod = importlib.import_module("fintfm.modeling.train")
    seen = {}
    monkeypatch.setattr(train_mod, "train", lambda m, p, t, out: seen.update(m=m, t=t))
    monkeypatch.delenv("FINTFM_CONFIG", raising=False)
    train_mod.main(argv)
    return seen["m"], seen["t"]


def _captured_train_cfg(monkeypatch, argv):
    """The TrainConfig half of :func:`_captured_configs`."""
    return _captured_configs(monkeypatch, argv)[1]


_TINY_CLI = [
    "--steps", "2", "--batch-size", "2", "--max-features", "8", "--max-classes", "2",
    "--d-model", "16", "--d-cell", "8", "--n-layers", "1", "--n-col-layers", "1",
    "--n-rows", "32",
]


@pytest.mark.parametrize(
    "flags, expected",
    [
        (["--n-heads", "2"], {"n_heads": 2}),
        (["--n-cell-heads", "2"], {"n_cell_heads": 2}),
        (["--n-summary-tokens", "3"], {"n_summary_tokens": 3}),
        (["--value-embedding", "rbf", "--value-basis-size", "6", "--value-basis-learnable"],
         {"value_embedding": "rbf", "value_basis_size": 6, "value_basis_learnable": True}),
        (["--value-embedding", "fourier", "--value-basis-size", "4"],
         {"value_embedding": "fourier", "value_basis_size": 4}),
        (["--value-ecdf"], {"value_ecdf": True}),
        (["--attn-length-scale", "--attn-length-ref", "64"],
         {"attn_length_scale": True, "attn_length_ref": 64}),
        (["--hazard-period-context", "--n-horizons", "3"],
         {"hazard_period_context": True, "n_horizons": 3}),
    ],
    ids=["n-heads", "n-cell-heads", "summary-tokens", "rbf", "fourier", "ecdf",
         "length-scale", "hazard-period"],
)
def test_each_model_lever_trains_from_the_cli(tmp_path, monkeypatch, flags, expected):
    """Smoke test per Phase 5 model lever: the flag reaches the model, two steps train, and
    the checkpoint reloads with the lever recorded. Pipeline only; says nothing of quality."""
    import importlib

    from fintfm.modeling.model import FinancialTFM

    train_mod = importlib.import_module("fintfm.modeling.train")
    monkeypatch.delenv("FINTFM_CONFIG", raising=False)
    out = tmp_path / "m.pt"
    train_mod.main([*_TINY_CLI, *flags, "--out", str(out)])
    model = FinancialTFM.load(str(out))
    for name, value in expected.items():
        assert getattr(model.cfg, name) == value, name
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_cli_takes_the_new_knobs_from_config_and_flags(tmp_path, monkeypatch):
    """Configuration, not constants: no flag means the config's value (today's behaviour), a
    config file overrides the packaged default, and an explicit flag overrides both."""
    base = ["--steps", "1", "--out", str(tmp_path / "x.pt")]
    cfg = _captured_train_cfg(monkeypatch, base)
    assert (cfg.warmup_steps, cfg.decay_norm_bias, cfg.amp, cfg.muon_lr, cfg.optimizer) == (
        500, True, "none", 0.02, "adamw",
    )
    override = tmp_path / "o.yaml"
    override.write_text("train:\n  warmup_steps: 9\n  decay_norm_bias: false\n")
    cfg = _captured_train_cfg(monkeypatch, [*base, "--config", str(override)])
    assert (cfg.warmup_steps, cfg.decay_norm_bias) == (9, False)
    cfg = _captured_train_cfg(monkeypatch, [
        *base, "--config", str(override), "--warmup-steps", "7", "--decay-norm-bias",
        "--amp", "bf16", "--optimizer", "muon", "--muon-lr", "0.01",
    ])
    assert (cfg.warmup_steps, cfg.decay_norm_bias, cfg.amp, cfg.optimizer, cfg.muon_lr) == (
        7, True, "bf16", "muon", 0.01,
    )


def test_benchmark_reports_throughput_and_saves_nothing(tmp_path, monkeypatch, capsys):
    """AU5.2's measuring instrument: ``--benchmark-steps`` prints one JSON record of tasks per
    second for the exact configuration given, and writes no checkpoint."""
    import importlib
    import json

    train_mod = importlib.import_module("fintfm.modeling.train")
    monkeypatch.delenv("FINTFM_CONFIG", raising=False)
    train_mod.main([
        "--benchmark-steps", "2", "--batch-size", "2", "--max-features", "8",
        "--max-classes", "2", "--d-model", "16", "--d-cell", "8", "--n-layers", "1",
        "--n-col-layers", "1", "--n-rows", "32", "--out", str(tmp_path / "x.pt"),
    ])
    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert record["steps"] == 2 and record["tasks"] == 4
    assert record["tasks_per_second"] > 0
    assert record["tasks_per_hour"] == pytest.approx(3600 * record["tasks_per_second"])
    assert record["seconds"] == pytest.approx(record["sample_seconds"] + record["step_seconds"])
    assert list(tmp_path.iterdir()) == []


def test_newton_schulz_orthogonalises_in_both_orientations():
    """Muon's core: the update's singular values land near one (the write-up's iteration is
    tuned for roughly [0.7, 1.2], not exact convergence) and its direction stays U V^T."""
    from fintfm.modeling.muon import newton_schulz

    torch.manual_seed(0)
    for shape in ((16, 40), (40, 16)):
        g = torch.randn(*shape)
        x = newton_schulz(g).float()
        assert x.shape == g.shape
        s = torch.linalg.svdvals(x)
        assert s.min() > 0.6 and s.max() < 1.25, s
        u, _, vh = torch.linalg.svd(g, full_matrices=False)
        polar = u @ vh
        cos = (x * polar).sum() / (x.norm() * polar.norm())
        assert cos > 0.95


def test_muon_step_applies_an_orthogonalised_update():
    """One Muon step from zero momentum moves a matrix by ``lr`` times a near-orthogonal
    update (aspect-ratio scaled), whatever the gradient's own scale; an AdamW-group parameter
    beside it gets the plain Adam update."""
    from fintfm.modeling.muon import Muon

    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(24, 8))
    b = torch.nn.Parameter(torch.zeros(8))
    opt = Muon([{"params": [w], "use_muon": True, "lr": 0.02},
                {"params": [b], "use_muon": False, "lr": 0.001}])
    w0 = w.detach().clone()
    w.grad = 1000.0 * torch.randn(24, 8)
    b.grad = torch.randn(8)
    opt.step()
    s = torch.linalg.svdvals((w0 - w.detach()) / (0.02 * (24 / 8) ** 0.5))
    assert s.min() > 0.6 and s.max() < 1.25, s
    # Adam's first step moves every coordinate by lr in the gradient's sign
    assert torch.allclose(b.detach(), -0.001 * torch.sign(b.grad), atol=1e-6)
    with pytest.raises(ValueError, match="2-D"):
        Muon([{"params": [b], "use_muon": True}])


def test_muon_takes_the_hidden_matrices_and_adamw_the_rest():
    """The write-up's rule: Muon for hidden-layer matrices; inputs, outputs, gains and biases
    on AdamW. Here that is the 2-D weights of every transformer stack and attention module."""
    from fintfm.modeling.model import FinancialTFM
    from fintfm.modeling.train import TrainConfig, _build_optimizer

    model = FinancialTFM(ModelConfig(**_TINY_RICH))
    opt, _ = _build_optimizer(model, TrainConfig(optimizer="muon", muon_lr=0.05))
    names = {id(p): n for n, p in model.named_parameters()}
    muon = {names[id(p)] for g in opt.param_groups if g["use_muon"] for p in g["params"]}
    # the schedule has already scaled "lr" by the first warmup step; the base is initial_lr
    muon_lrs = {g["initial_lr"] for g in opt.param_groups if g["use_muon"]}
    adam_lrs = {g["initial_lr"] for g in opt.param_groups if not g["use_muon"]}
    assert muon_lrs == {0.05} and adam_lrs == {3e-4}
    assert "encoder.layers.0.self_attn.in_proj_weight" in muon
    assert "column_encoder.layers.0.linear2.weight" in muon
    assert "row_within_feature_encoder.layers.0.linear1.weight" in muon
    assert "pool_attn.out_proj.weight" in muon
    for n in ("head.weight", "row_proj.weight", "y_proj.weight", "value_embed.0.weight",
              "mask_embed.weight", "encoder.layers.0.norm1.weight", "query_token"):
        assert n not in muon, n
    assert all(dict(model.named_parameters())[n].ndim == 2 for n in muon)


def test_muon_run_resumes_to_the_same_weights_as_an_uninterrupted_one(tmp_path):
    """Muon's momentum and its AdamW side's moments must survive a resume, and the cosine
    schedule must stay keyed to the total, exactly as for AdamW."""
    from fintfm.modeling.train import TrainConfig, train

    model_cfg = ModelConfig(**_TINY_RICH)
    prior_cfg = PriorConfig(max_features=8, max_classes=2, n_rows=32)
    common = {"steps": 6, "batch_size": 2, "eval_every": 1000, "log_every": 1000, "seed": 3,
              "optimizer": "muon"}
    whole = train(model_cfg, prior_cfg, TrainConfig(**common), str(tmp_path / "whole.pt"))
    part = tmp_path / "part.pt"
    train(model_cfg, prior_cfg, TrainConfig(**common, run_steps=3), str(part))
    resumed = train(model_cfg, prior_cfg, TrainConfig(**common, resume=f"{part}.state"),
                    str(tmp_path / "resumed.pt"))
    for (name, a), (_, b) in zip(whole.state_dict().items(), resumed.state_dict().items()):
        assert torch.allclose(a, b, atol=1e-6), name
    with pytest.raises(ValueError, match="schedule"):
        train(model_cfg, prior_cfg, TrainConfig(**{**common, "steps": 9},
                                                resume=f"{part}.state"),
              str(tmp_path / "x.pt"))
