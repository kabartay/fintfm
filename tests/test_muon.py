"""Muon at the settings design review DR1.7 asks for, before any run uses it.

The properties pinned here: the ``rms_match`` update has the per-element size it claims, a
fused QKV projection is orthogonalised one block at a time, Muon's matrices take decoupled
weight decay, Newton-Schulz runs in float32 where bf16 is not native, and the learning-rate
default follows the scale rule. Resume exactness is
``tests/test_train.py::test_muon_run_resumes_to_the_same_weights_as_an_uninterrupted_one``,
which runs at these defaults.
"""

import copy

import numpy as np
import pytest
import torch

from fintfm.modeling.model import FinancialTFM, ModelConfig
from fintfm.prior import PriorConfig

_TINY = {
    "max_features": 8, "max_classes": 2, "d_cell": 8, "d_model": 16, "n_layers": 1,
    "n_col_layers": 1, "n_heads": 2, "n_cell_blocks": 1, "cell_labels": True,
    "pooling": "attention", "mask_embedding": True,
}


def _rms(t: torch.Tensor) -> float:
    """Root mean square of a tensor's elements."""
    return float(t.pow(2).mean().sqrt())


def _step_one_updates(optimizer: str, **train_kwargs) -> tuple[dict[str, torch.Tensor], float]:
    """One optimiser step on one fixed batch from a fixed initialisation.

    Returns:
        ``(update per Muon-eligible parameter name, the step's effective base lr)``. The
        effective lr is the first warmup step's, ``lr / 500``, for both optimisers alike.
    """
    from fintfm.modeling.train import TrainConfig, _build_optimizer, _muon_parameters
    from fintfm.prior.mixture import sample_batch

    torch.manual_seed(0)
    model = FinancialTFM(ModelConfig(**_TINY))
    batch = sample_batch(np.random.default_rng(0), PriorConfig(max_features=8, max_classes=2,
                                                               n_rows=32), 2)
    model.loss(batch.X, batch.y, batch.n_ctx, batch.n_classes).backward()
    muon_ids = _muon_parameters(model)
    opt, _ = _build_optimizer(model, TrainConfig(optimizer=optimizer, **train_kwargs))
    before = {n: p.detach().clone() for n, p in model.named_parameters() if id(p) in muon_ids}
    opt.step()
    params = dict(model.named_parameters())
    lr_eff = TrainConfig().lr / 500
    return {n: params[n].detach() - b for n, b in before.items()}, lr_eff


def test_rms_match_update_is_shape_independent_and_sized_to_adamws_steady_state():
    """Moonlight's rule: an orthogonalised update scaled by ``0.2·sqrt(max(n, m))`` has
    per-element RMS about 0.2·lr whatever the shape, which is AdamW's steady-state RMS
    (``sqrt((1 - b1) / (1 + b1))`` = 0.229 at b1 = 0.9 under uncorrelated gradients) and is
    why the two can share one lr. Checked on full-rank gradients across shapes, against an
    AdamW steady state measured here at torch's defaults."""
    from fintfm.modeling.muon import Muon

    torch.manual_seed(1)
    w = torch.nn.Parameter(torch.zeros(64, 64))
    adamw = torch.optim.AdamW([w], lr=1.0, weight_decay=0.0)
    for _ in range(300):
        prev = w.detach().clone()
        w.grad = torch.randn_like(w)
        adamw.step()
    steady = _rms(w.detach() - prev)
    assert 0.18 < steady < 0.3, steady
    for shape in ((16, 16), (48, 16), (16, 512), (512, 16)):
        m = torch.nn.Parameter(torch.zeros(*shape))
        opt = Muon([{"params": [m], "use_muon": True, "lr": 1.0}])
        m.grad = torch.randn(*shape)
        opt.step()
        rms = _rms(m.detach())
        # the quintic leaves singular values in roughly [0.7, 1.2], not exactly 1
        assert 0.14 < rms < 0.24, (shape, rms)
        assert 0.5 < rms / steady < 2.0, (shape, rms, steady)


def test_step_one_update_on_a_tiny_model_against_adamw():
    """At step 1, on one batch, Muon (``rms_match``, lr shared with AdamW) moves the tiny
    model's hidden matrices by 0.04-0.2·lr per element, AdamW by lr.

    DR1.7's literal check, Muon within 2x of AdamW at step 1, does not hold and cannot for a
    correct implementation: Adam's first step is ``lr·sign(g)``, RMS lr exactly, while the
    0.2 factor targets AdamW's steady state (test above). On real gradients Muon also falls
    below 0.2·lr, because five Newton-Schulz steps do not lift an ill-conditioned matrix's
    small singular values to 1: at seed 0 the aggregate is 0.088·lr (SMOKE-TEST scale), with
    the 512 x 16 feed-forward matrices lowest. Pinned here so a change to either is seen."""
    muon, lr_eff = _step_one_updates("muon")
    adam, _ = _step_one_updates("adamw")
    muon_rms = _rms(torch.cat([u.flatten() for u in muon.values()])) / lr_eff
    adam_rms = _rms(torch.cat([u.flatten() for u in adam.values()])) / lr_eff
    assert 0.98 < adam_rms <= 1.0001, adam_rms
    assert 0.06 < muon_rms < 0.12, muon_rms
    for name, u in muon.items():
        assert 0.04 < _rms(u) / lr_eff < 0.22, (name, _rms(u) / lr_eff)


def test_keller_scale_is_the_design_reviews_larger_update():
    """``--muon-scale keller`` at its default lr 0.02 moves the tiny model's matrices by
    more than 20x the ``rms_match`` default's per-element update: the gap DR1.7 removed."""
    keller, _ = _step_one_updates("muon", muon_scale="keller")
    rms, _ = _step_one_updates("muon")
    ratio = _rms(torch.cat([u.flatten() for u in keller.values()])) / _rms(
        torch.cat([u.flatten() for u in rms.values()]))
    assert ratio > 20, ratio


def test_each_in_proj_block_is_orthogonalised_separately():
    """A fused ``in_proj_weight`` stacks Q, K and V by rows. Orthogonalised as one matrix, a
    block whose gradient is small gets almost no update; orthogonalised per block, each of
    Q, K and V gets an update whose singular values are all near one."""
    from fintfm.modeling.muon import Muon, update_scale

    d = 8
    torch.manual_seed(0)
    g = torch.cat([1000.0 * torch.randn(d, d), torch.randn(d, d), 0.001 * torch.randn(d, d)])

    def block_singular_values(row_blocks: int) -> list[torch.Tensor]:
        w = torch.nn.Parameter(torch.zeros(3 * d, d))
        opt = Muon([{"params": [w], "use_muon": True, "lr": 1.0, "row_blocks": row_blocks}])
        w.grad = g.clone()
        opt.step()
        return [torch.linalg.svdvals(-b / update_scale(d, d, "rms_match"))
                for b in w.detach().chunk(3)]

    for s in block_singular_values(3):
        assert s.min() > 0.6 and s.max() < 1.25, s
    joint = block_singular_values(1)
    assert joint[2].max() < 0.1, joint[2]  # the V block starves when orthogonalised jointly
    with pytest.raises(ValueError, match="row_blocks"):
        Muon([{"params": [torch.nn.Parameter(torch.zeros(10, 4))], "use_muon": True,
               "row_blocks": 3}])


def test_build_optimizer_splits_every_fused_in_proj_and_nothing_else():
    """Every attention module's fused projection sits in a three-block Muon group; the other
    hidden matrices are orthogonalised whole."""
    from fintfm.modeling.train import TrainConfig, _build_optimizer

    model = FinancialTFM(ModelConfig(**_TINY))
    opt, _ = _build_optimizer(model, TrainConfig(optimizer="muon"))
    names = {id(p): n for n, p in model.named_parameters()}
    by_blocks: dict[int, set[str]] = {}
    for g in opt.param_groups:
        if g["use_muon"]:
            by_blocks.setdefault(g["row_blocks"], set()).update(names[id(p)] for p in g["params"])
    fused = {n for n in names.values() if n.endswith("in_proj_weight")}
    assert fused and by_blocks[3] == fused
    assert by_blocks[1] and not by_blocks[1] & fused


def test_muon_matrices_take_decoupled_weight_decay():
    """With a zero gradient Muon's update is zero, so one step leaves exactly the decay:
    ``w·(1 - lr·wd)``. ``_build_optimizer`` gives Muon's groups AdamW's 0.01."""
    from fintfm.modeling.muon import Muon
    from fintfm.modeling.train import TrainConfig, _build_optimizer

    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(6, 4))
    w0 = w.detach().clone()
    opt = Muon([{"params": [w], "use_muon": True, "lr": 0.5, "weight_decay": 0.01}])
    w.grad = torch.zeros_like(w)
    opt.step()
    assert torch.allclose(w.detach(), w0 * (1 - 0.5 * 0.01), atol=0, rtol=1e-7)
    model = FinancialTFM(ModelConfig(**_TINY))
    opt, _ = _build_optimizer(model, TrainConfig(optimizer="muon"))
    assert {g["weight_decay"] for g in opt.param_groups if g["use_muon"]} == {0.01}


def test_newton_schulz_runs_in_float32_unless_bf16_is_native(monkeypatch):
    """bf16 only on CUDA with compute capability 8.0+, the line ``--amp bf16`` draws; float32
    on CPU, MPS and T4/P100-class GPUs. On CPU the update is computed in float32."""
    from fintfm.modeling import muon

    assert muon.newton_schulz_dtype(torch.device("cpu")) == torch.float32
    assert muon.newton_schulz_dtype(torch.device("mps")) == torch.float32
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: (7, 5))
    assert muon.newton_schulz_dtype(torch.device("cuda", 0)) == torch.float32
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: (8, 0))
    assert muon.newton_schulz_dtype(torch.device("cuda", 0)) == torch.bfloat16
    g = torch.randn(12, 5, dtype=torch.float64)
    assert muon.newton_schulz(g).dtype == torch.float32
    assert muon.newton_schulz(g, dtype=torch.bfloat16).dtype == torch.bfloat16


def test_muon_lr_follows_the_scale_rule_unless_given():
    """``rms_match`` shares AdamW's lr, ``keller`` keeps the write-up's 0.02, and an explicit
    ``muon_lr`` overrides either. The AdamW side always keeps ``lr``."""
    from fintfm.modeling.train import TrainConfig, _build_optimizer

    def rates(**kw) -> tuple[set[float], set[float]]:
        model = FinancialTFM(ModelConfig(**_TINY))
        opt, _ = _build_optimizer(model, TrainConfig(optimizer="muon", **kw))
        return ({g["initial_lr"] for g in opt.param_groups if g["use_muon"]},
                {g["initial_lr"] for g in opt.param_groups if not g["use_muon"]})

    assert rates() == ({3e-4}, {3e-4})
    assert rates(lr=1e-3) == ({1e-3}, {1e-3})
    assert rates(muon_scale="keller") == ({0.02}, {3e-4})
    assert rates(muon_scale="keller", muon_lr=0.01) == ({0.01}, {3e-4})
    assert rates(muon_lr=8e-4) == ({8e-4}, {3e-4})
    with pytest.raises(ValueError, match="muon_scale"):
        rates(muon_scale="spectral")


def test_cli_muon_scale_comes_from_config_unless_flagged(tmp_path, monkeypatch):
    """No flag gives the config's ``train.muon_scale`` (``rms_match``) and ``muon_lr`` null;
    ``--muon-scale keller`` wins."""
    import importlib

    train_mod = importlib.import_module("fintfm.modeling.train")
    seen = {}
    monkeypatch.setattr(train_mod, "train", lambda m, p, t, out: seen.update(t=t))
    monkeypatch.delenv("FINTFM_CONFIG", raising=False)
    base = ["--steps", "1", "--out", str(tmp_path / "x.pt"), "--optimizer", "muon"]
    train_mod.main(base)
    assert (seen["t"].muon_scale, seen["t"].muon_lr) == ("rms_match", None)
    train_mod.main([*base, "--muon-scale", "keller"])
    assert seen["t"].muon_scale == "keller"


def test_muon_state_carries_its_group_settings_through_a_resume():
    """``row_blocks`` and ``scale`` live in the parameter groups, so ``state_dict`` saves them
    and a reloaded optimiser steps exactly as the original."""
    from fintfm.modeling.train import TrainConfig, _build_optimizer

    torch.manual_seed(0)
    model = FinancialTFM(ModelConfig(**_TINY))
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    opt, _ = _build_optimizer(model, TrainConfig(optimizer="muon"))
    opt.step()
    twin = copy.deepcopy(model)
    for a, b in zip(model.parameters(), twin.parameters()):
        b.grad = a.grad.clone()
    opt2, _ = _build_optimizer(twin, TrainConfig(optimizer="muon"))
    # deep-copied, as a save and load would: load_state_dict keeps the very tensors it is
    # given when no cast is needed, and the two optimisers would share moment buffers
    opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
    assert [(g["row_blocks"], g["scale"]) for g in opt2.param_groups] == [
        (g["row_blocks"], g["scale"]) for g in opt.param_groups]
    opt.step()
    opt2.step()
    for a, b in zip(model.parameters(), twin.parameters()):
        assert torch.equal(a, b)
