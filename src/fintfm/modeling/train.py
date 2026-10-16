"""Pretraining loop: sample a fresh synthetic batch every step, no epochs.

Usage:
    fintfm-train --steps 20000 --batch-size 64 --out runs/v0.pt
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from fintfm.config import load_config
from fintfm.modeling.model import FinancialTFM, ModelConfig
from fintfm.prior import PriorConfig
from fintfm.prior.dump import open_task_dump
from fintfm.prior.levers import add_prior_lever_arguments, prior_lever_values
from fintfm.prior.mixture import sample_batch

#: Second element of the evaluation generator's seed, ``[seed, _EVAL_STREAM]``, so held-out
#: evaluation has a stream of its own that no training draw touches (audit AU1.18).
_EVAL_STREAM = 1


@dataclass
class TrainConfig:
    """Everything the pretraining loop reads, in one place.

    Carried into each checkpoint's sidecar so a resumed chunk reconstructs the same schedule
    and the same prior, and so a finished run records what produced it. Defaults here are the
    library's; the values actually used by an experiment come from
    ``src/fintfm/configs/default.yaml`` layered under ``--config`` and explicit flags.
    """

    steps: int = 20_000
    batch_size: int = 64
    lr: float = 3e-4
    #: Linear learning-rate warmup before the cosine decay. 500 is every checkpoint's value;
    #: ``--warmup-steps`` and ``train.warmup_steps`` expose it (audit AU5.10).
    warmup_steps: int = 500
    log_every: int = 50
    eval_every: int = 500
    #: Save the checkpoint every this many steps, in addition to the end. A long run that
    #: only saves at the end loses everything if it dies -- and jobs do die: three did on
    #: 2026-09-10 alone, to a Python version floor, a CUDA OOM and a crash in the metric.
    #: At 0 the loop saves only on completion, which is the old behaviour.
    checkpoint_every: int = 0
    seed: int = 0
    device: str = "cpu"
    #: Steps to run in *this* invocation, when a job's wall-clock is shorter than the run.
    #: ``None`` runs to :attr:`steps`. The learning-rate schedule always spans :attr:`steps`,
    #: so chunking a run changes nothing about it -- which is the whole point, and the reason
    #: this is a separate field rather than lowering ``steps``.
    run_steps: int | None = None
    #: Training state to resume from, written by :func:`_save_training_state`. Carries the
    #: optimiser moments, the schedule position, both RNG streams and the step counter --
    #: everything a bare model checkpoint lacks. Resuming from a model checkpoint alone would
    #: restart AdamW cold and replay the same synthetic tasks, neither of which is visible in
    #: a loss curve.
    resume: str | None = None
    #: A *model* checkpoint whose weights initialise this run, with a fresh optimiser and
    #: schedule -- continued pretraining, not a resume. Exists for matched-pair ablations that
    #: start every arm from the same trained weights and vary only the prior mixture, which is
    #: what lets a short CPU run test a data change at the real architecture's scale. The
    #: checkpoint's architecture must equal this run's exactly; mutually exclusive with
    #: :attr:`resume`.
    init_from: str | None = None
    #: Features per row-within-feature attention call, for ``n_cell_blocks > 0`` models.
    #: Identity-preserving (``docs/results/FINDINGS.md`` §81), so it changes no number and only
    #: bounds memory. ``None`` keeps the unchunked path, which is what every checkpoint
    #: before §81 trained with; it is also what made §78's run OOM on a T4 and forced that
    #: finding's documented protocol deviation (§79).
    feature_chunk: int | None = None
    #: ``"adamw"`` (default, cosine-annealed) or ``"schedulefree"`` (task 48.9, Defazio et al.,
    #: NeurIPS 2024). The cosine schedule is a function of the full step budget
    #: (:func:`_lr_schedule`), which is what makes AdamW's run length a real operational
    #: constraint rather than a hyperparameter -- §108's matched-task rerun had to start fresh
    #: rather than resume, because a 6,000-step schedule cannot be extended to 12,000 without
    #: annealing the second half toward a rate that has already reached zero. Schedule-free
    #: carries its own internal warmup/averaging and needs no external curve, so a run under
    #: it can be resumed against a *larger* total step count -- the ``requires schedulefree``
    #: extra must be installed. ``"muon"`` (audit AU5.17, :mod:`fintfm.modeling.muon`) runs
    #: Muon on the attention and feed-forward matrices and AdamW on everything else, under
    #: the same cosine schedule as ``"adamw"``.
    optimizer: str = "adamw"
    #: Pre-generated batches to train from instead of sampling (audit AU5.15); see
    #: :mod:`fintfm.prior.dump`. ``None`` samples live, as every run before it did.
    task_dump: str | None = None
    #: Mixed precision for the training forward pass (audit AU5.2). ``"none"`` (default) is
    #: plain float32, what every checkpoint trained with. ``"bf16"`` wraps the forward and the
    #: loss in ``torch.autocast(dtype=torch.bfloat16)`` on CUDA (Ampere or newer) or CPU;
    #: weights, gradients, the loss reduction and the optimiser stay float32, and held-out
    #: evaluation stays float32. ``"fp16"`` does the same in float16 with a loss scaler, on
    #: CUDA only, for GPUs without native bf16 (T4, P100). Both are refused on MPS, bf16 on
    #: GPUs without native bf16, and both for the quantile and survival objectives, whose
    #: losses would reduce in half precision (:func:`_autocast`).
    amp: str = "none"
    #: ``True`` (default) applies AdamW's weight decay to every parameter, which is what every
    #: checkpoint trained with. ``False`` (audit AU5.10) exempts LayerNorm gains, every bias,
    #: embedding tables and the free token parameters, the usual transformer practice.
    #: ``"adamw"`` and ``"muon"`` only; schedule-free applies no weight decay to exempt.
    decay_norm_bias: bool = True
    #: Base learning rate of Muon's matrix groups under ``optimizer="muon"``; the AdamW side
    #: keeps :attr:`lr`. 0.02 is the write-up's default. Ignored by the other optimisers.
    muon_lr: float = 0.02


#: AdamW's weight decay. It is ``torch.optim.AdamW``'s own default, which every checkpoint
#: trained with implicitly; named here only so the ``decay_norm_bias=False`` groups can state
#: it without changing it.
_ADAMW_WEIGHT_DECAY = 0.01

#: Accepted values of :attr:`TrainConfig.amp`.
AMP_MODES = ("none", "bf16", "fp16")

#: ``ModelConfig`` fields ``fintfm-train`` takes from the config's ``model`` section unless a
#: flag of the same name is given. Each is also a field of
#: :class:`fintfm.config.ModelConfigDefaults` (``tests/test_config.py`` checks both ways).
_MODEL_CONFIG_KNOBS = (
    "n_heads", "n_cell_heads", "n_summary_tokens", "value_embedding", "value_basis_size",
    "value_basis_learnable", "value_ecdf", "attn_length_scale", "attn_length_ref",
    "hazard_period_context",
)

#: Accepted values of :attr:`TrainConfig.optimizer`.
OPTIMIZERS = ("adamw", "schedulefree", "muon")


def _no_decay_parameters(model: torch.nn.Module) -> set[int]:
    """Ids of the parameters ``decay_norm_bias=False`` exempts from weight decay.

    LayerNorm gains and biases, every bias, every ``nn.Embedding`` table, and the free token
    parameters registered directly on the model (``query_token``, ``cell_mask_token``,
    ``pool_query``). The last are why this is not simply "parameters with fewer than two
    dimensions": ``pool_query`` is ``(1, 1, d_cell)``, a learned token, not a weight matrix.

    Args:
        model: The model being trained.

    Returns:
        ``id()`` of every exempt parameter.
    """
    exempt: set[int] = set()
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if (
                isinstance(module, (torch.nn.LayerNorm, torch.nn.Embedding))
                or name.endswith("bias")
                or p.ndim < 2
                or module is model  # free tokens, not weight matrices
            ):
                exempt.add(id(p))
    return exempt


def _muon_parameters(model: torch.nn.Module) -> set[int]:
    """Ids of the parameters Muon updates under ``optimizer="muon"``.

    The 2-D weight matrices of every transformer stack and attention module: the hidden
    layers Muon is defined for. Cell and label embeddings, the column-identity and row
    projections and the output heads stay on AdamW, per the write-up's rule that inputs and
    outputs are not hidden layers.

    Args:
        model: The model being trained.

    Returns:
        ``id()`` of every Muon parameter.
    """
    ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, (torch.nn.TransformerEncoder, torch.nn.MultiheadAttention)):
            ids.update(id(p) for p in module.parameters() if p.ndim == 2)
    return ids


def _build_optimizer(
    model: FinancialTFM, train_cfg: TrainConfig
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Construct the optimiser and learning-rate schedule a run trains under.

    At the defaults (``optimizer="adamw"``, ``decay_norm_bias=True``) this is exactly
    ``torch.optim.AdamW(model.parameters(), lr=...)`` with the warmup-then-cosine schedule,
    the construction every checkpoint trained under.

    Args:
        model: The model being trained, already on its device.
        train_cfg: Read for ``optimizer``, ``lr``, ``warmup_steps``, ``decay_norm_bias`` and
            ``muon_lr``.

    Returns:
        ``(optimizer, schedule)``.

    Raises:
        ValueError: For an unknown optimiser, or ``decay_norm_bias=False`` under
            schedule-free, where it would be silently inert.
    """
    if train_cfg.optimizer not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {train_cfg.optimizer!r}")
    if train_cfg.optimizer == "schedulefree":
        if not train_cfg.decay_norm_bias:
            raise ValueError(
                "decay_norm_bias=False has no effect under the schedulefree optimiser, which "
                "applies no weight decay; use --optimizer adamw or muon"
            )
        import schedulefree

        # warmup_steps was missing here until §146's addendum (2026-10-05) found it: the
        # package recommends warmup in place of a schedule, and the measured negative result
        # used none. Reusing train_cfg.warmup_steps rather than inventing a second knob --
        # the two optimisers' warmup serves the same purpose even though the rest of their
        # schedules differ completely.
        opt = schedulefree.AdamWScheduleFree(
            model.parameters(), lr=train_cfg.lr, warmup_steps=train_cfg.warmup_steps
        )
        return opt, _NullSchedule(train_cfg.lr)  # type: ignore[return-value]

    exempt = _no_decay_parameters(model) if not train_cfg.decay_norm_bias else set()
    if train_cfg.optimizer == "muon":
        from fintfm.modeling.muon import Muon

        # Muon's matrices take no weight decay, as in the write-up's reference; the AdamW
        # side follows decay_norm_bias like plain AdamW does.
        muon_ids = _muon_parameters(model)
        params = list(model.parameters())
        groups = [
            {"params": [p for p in params if id(p) in muon_ids], "use_muon": True,
             "lr": train_cfg.muon_lr, "weight_decay": 0.0},
            {"params": [p for p in params if id(p) not in muon_ids and id(p) not in exempt],
             "use_muon": False, "lr": train_cfg.lr, "weight_decay": _ADAMW_WEIGHT_DECAY},
            {"params": [p for p in params if id(p) not in muon_ids and id(p) in exempt],
             "use_muon": False, "lr": train_cfg.lr, "weight_decay": 0.0},
        ]
        opt = Muon([g for g in groups if g["params"]], lr=train_cfg.lr)
    elif train_cfg.decay_norm_bias:
        opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
    else:
        params = list(model.parameters())
        opt = torch.optim.AdamW(
            [
                {"params": [p for p in params if id(p) not in exempt],
                 "weight_decay": _ADAMW_WEIGHT_DECAY},
                {"params": [p for p in params if id(p) in exempt], "weight_decay": 0.0},
            ],
            lr=train_cfg.lr,
        )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr_schedule(s, train_cfg))
    return opt, sched


def _autocast(model: torch.nn.Module, amp: str) -> contextlib.AbstractContextManager:
    """The precision context for one training forward pass.

    ``"none"`` returns a null context, so the default path runs exactly as before this
    option existed. ``"bf16"`` returns ``torch.autocast`` for the model's device (CUDA with
    native bf16, or CPU); ``"fp16"`` returns CUDA float16 autocast, for GPUs without native
    bf16 such as the T4 and P100, and is paired with the loss scaler of :func:`_grad_scaler`.

    Args:
        model: The model being trained; its device decides the autocast backend.
        amp: :attr:`TrainConfig.amp`.

    Returns:
        A context manager to wrap the forward pass and loss in.

    Raises:
        ValueError: For an unknown mode; a quantile or survival model; MPS; ``"bf16"`` on a
            CUDA device without native bf16 (an emulated bf16 would run and report a
            throughput the lever does not deliver); ``"fp16"`` anywhere but CUDA.
    """
    if amp not in AMP_MODES:
        raise ValueError(f"amp must be one of {AMP_MODES}, got {amp!r}")
    if amp == "none":
        return contextlib.nullcontext()
    if getattr(model, "hazard", None) is not None or model.cfg.head_type != "binned":
        # Only the binned head's cross-entropy is on autocast's float32 list. The pinball
        # loss would be reduced in half precision against a rounded target, and the hazard
        # NLL takes log1p(-h) of hazards clamped just below one, which bf16 rounds to one.
        # Both live in model.py/hazard.py; until they cast to float32 themselves, refuse.
        raise ValueError(
            f"--amp {amp} supports the binned classification objective only; this model has "
            f"head_type={model.cfg.head_type!r} and "
            f"{'a' if getattr(model, 'hazard', None) is not None else 'no'} hazard head"
        )
    device_type = next(model.parameters()).device.type
    if device_type == "mps":
        raise ValueError(
            f"--amp {amp} is not supported on MPS here: train in float32 (--amp none) or use "
            "a CUDA device (CLAUDE.md records MPS memory trouble at the reference recipe)"
        )
    if amp == "fp16":
        if device_type != "cuda":
            raise ValueError(
                f"--amp fp16 is for CUDA GPUs without native bf16 (T4, P100), not "
                f"{device_type}; use --amp none, or --amp bf16 on CPU"
            )
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if device_type == "cuda" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError(
            f"--amp bf16 needs native bf16 (compute capability 8.0+, e.g. A10G, L4, A100); "
            f"{torch.cuda.get_device_name()} would emulate it. Use --amp fp16 on this GPU"
        )
    if device_type not in ("cuda", "cpu"):
        raise ValueError(f"--amp bf16 supports cuda and cpu, not {device_type}")
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16)


def _grad_scaler(amp: str) -> torch.amp.GradScaler | None:
    """The loss scaler a precision mode needs: one for ``"fp16"``, none otherwise.

    float16 has five exponent bits, so small gradients underflow to zero in the backward
    pass. The scaler multiplies the loss by a large power of two before ``backward`` and
    divides the gradients by it before clipping and the optimiser step, skipping any step
    whose gradients overflowed and lowering the scale. bf16 keeps float32's exponent range
    and needs none, and ``"none"`` returns ``None`` so the default step is unchanged.

    Args:
        amp: :attr:`TrainConfig.amp`.

    Returns:
        A CUDA ``GradScaler``, or ``None``.
    """
    return torch.amp.GradScaler("cuda") if amp == "fp16" else None


def _training_step(
    model: FinancialTFM,
    batch: Any,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    amp: str,
    objectives: set[str],
    scaler: torch.amp.GradScaler | None = None,
) -> torch.Tensor:
    """One optimiser step on one batch: the loop body :func:`train` and :func:`benchmark` share.

    Under ``amp`` other than ``"none"`` only the forward pass and loss run inside autocast;
    the backward pass, clipping and the optimiser step run on the float32 weights outside it.
    With a ``scaler`` the gradients are unscaled *before* clipping, so the clip threshold of
    1.0 means what it does in float32, and a step whose gradients overflowed is skipped.

    Args:
        model: The model being trained.
        batch: A sampled batch, already on the model's device.
        opt: The optimiser.
        sched: The learning-rate schedule.
        amp: :attr:`TrainConfig.amp`.
        objectives: Updated in place with the objective this step trained.
        scaler: Loss scaler from :func:`_grad_scaler`; ``None`` is the plain step.

    Returns:
        The step's (unscaled) loss tensor; the caller reads ``.item()``.
    """
    with _autocast(model, amp):
        # survival objective when the model has a hazard head and the prior emits periods
        if model.hazard is not None and batch.period is not None:
            loss = model.survival_loss(batch.X, batch.y, batch.period, batch.n_ctx)
            objectives.add("survival")
        else:
            loss = model.loss(batch.X, batch.y, batch.n_ctx, batch.n_classes)
            objectives.add("quantile" if model.cfg.head_type == "quantile" else "classification")
    opt.zero_grad(set_to_none=True)
    if scaler is None:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)  # skipped when the unscaled gradients hold an inf or NaN
        scaler.update()
    sched.step()
    return loss


class _NullSchedule:
    """No-op stand-in for ``torch.optim.lr_scheduler`` when the optimiser needs no schedule.

    Schedule-free optimisers need no *external* LR curve. They are not warmup-free, though:
    ``schedulefree.AdamWScheduleFree`` takes its own ``warmup_steps`` argument (default 0,
    recommended by the package in place of a schedule), which the construction site below
    does not pass -- so a run under ``--optimizer schedulefree`` as currently wired gets zero
    warmup, not an automatic one. §146's addendum (2026-10-05) found this after the fact,
    once a negative result made the configuration worth re-reading rather than assuming.
    Standing a no-op in for the scheduler keeps every call site in the training loop uniform
    across both optimiser choices, rather than branching on the choice at every call.
    """

    def __init__(self, lr: float) -> None:
        self._lr = lr

    def step(self) -> None:
        """No-op: schedule-free optimisers need no external step."""

    def get_last_lr(self) -> list[float]:
        """Return the constant base rate, for the training loop's progress log."""
        return [self._lr]

    def state_dict(self) -> dict:
        """Return an empty state; there is no schedule position to carry across a resume."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        """No-op: nothing to restore."""


def _set_optimizer_mode(opt: torch.optim.Optimizer, optimizer_name: str, mode: str) -> None:
    """Switch a schedule-free optimiser between its fast and averaged weights.

    Schedule-free optimisers maintain two weight sequences: a fast-moving one used while
    training (``opt.train()``) and a Polyak-averaged one that should be active whenever the
    model is evaluated or saved (``opt.eval()``) -- saving or scoring the fast sequence would
    silently report a noisier model than the one actually being trained toward. A no-op for
    plain AdamW, which has no such distinction, so call sites need not branch on the optimiser
    choice themselves.

    Args:
        opt: The optimiser in use.
        optimizer_name: :attr:`TrainConfig.optimizer`.
        mode: ``"train"`` or ``"eval"``.
    """
    if optimizer_name == "schedulefree":
        getattr(opt, mode)()


def _lr_schedule(step: int, cfg: TrainConfig) -> float:
    """Learning-rate multiplier: linear warmup, then cosine decay to zero.

    **The schedule is a function of the full ``cfg.steps`` budget, not of progress so far.**
    That is what makes a chunked run resumable — a chunk starting at step 4,000 of 12,000
    picks up the same multiplier it would have had in one continuous run — and it is also why
    a run cannot simply be extended past its budget: the rate has already annealed to zero
    (`docs/results/FINDINGS.md` §108, §114).

    Args:
        step: Zero-based global step.
        cfg: Training configuration, read for ``warmup_steps`` and ``steps``.

    Returns:
        Multiplier in ``[0, 1]`` applied to the base learning rate.
    """
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return 0.5 * (1 + np.cos(np.pi * progress))


@torch.no_grad()
def _eval_quality(
    model: FinancialTFM,
    prior_cfg: PriorConfig,
    rng: np.random.Generator,
    n_batches: int = 5,
    batch_size: int = 8,
) -> dict[str, float]:
    """Held-out quality on freshly sampled synthetic tasks, against a constant baseline.

    **Not accuracy.** This function used to report query accuracy, and at the prior's mean base
    rate of about 0.047 a constant majority-class predictor scores 0.953 — so three pretraining
    runs logged "held-out accuracy 0.935-0.945", *below the constant predictor*, and it read as
    progress (``docs/results/FINDINGS.md`` §42). ``CLAUDE.md`` already said accuracy is not a proper
    scoring rule and that this is why training optimises cross-entropy; the evaluation inside
    the training loop did not follow the repository's own rule for three runs.

    Reports instead:

    - **AUC**, pooled over every query row of every task. Kept for continuity with earlier
      runs, and **it is the inflated one**: pooling across tasks whose base rates differ lets
      a model score well by predicting each task's base rate without discriminating *within*
      any task. Measured on the financial prior, whose per-task rates span 0.003 to 0.986,
      pooled AUC read 0.9080 where the per-task mean was 0.6426 -- an overstatement of 0.26,
      and the reason a checkpoint could log "held-out AUC 0.943" while scoring 0.63-0.69 on
      every probe (``docs/results/FINDINGS.md`` §57). The trivial prior hides this, because every one
      of its tasks has the same base rate;
    - **AUC per task**, the mean of AUCs computed inside each task. This is the honest
      discrimination number and the one to read;
    - **Brier skill** against a predictor that ignores every feature and returns the context's
      base rate. Zero means "no better than knowing the base rate"; negative means worse.

    The baseline is the point. A metric with no floor underneath it cannot distinguish a model
    that learned something from one that learned the class balance.

    The device is read off the model rather than passed in. Taking it as an argument is how
    this function shipped a crash: the training loop moved its batches and this path did not,
    which nothing caught because the path had only ever run on CPU.

    Args:
        model: The model being trained.
        prior_cfg: Prior configuration, so held-out tasks match training tasks.
        rng: Random generator.
        n_batches: Held-out batches to average over.
        batch_size: Rows per held-out batch. **Must not exceed the training loop's own batch
            size.** This was hardcoded to 16 regardless of the caller's training batch size
            until a two-way-cell-attention run OOMed here specifically -- 16 is double the
            usual training batch of 8, and a memory-heavier architecture that trains fine
            hit CUDA OOM in *this* function alone, mid-run, at step 500 (task 39.4). Eval
            evaluating at a batch size training was never proven to fit at is the bug; the
            fix is to inherit training's own batch size, not to hand-tune a bigger one.

    Returns:
        ``{"auc": ..., "brier_skill": ..., "base_rate": ...}``. AUC is NaN when no held-out
        batch contained both classes, which is itself worth seeing rather than hiding.
    """
    from sklearn.metrics import roc_auc_score as _auc

    device = next(model.parameters()).device
    model.eval()
    probs, targets, per_task = [], [], []
    for _ in range(n_batches):
        batch = sample_batch(rng, prior_cfg, batch_size=batch_size).to(device)
        logits = model(batch.X, batch.y, batch.n_ctx, batch.n_classes)[:, batch.n_ctx :]
        p_all = torch.softmax(logits.float(), dim=-1)
        # Per-task AUC, scored inside each task before anything is pooled. See the docstring:
        # the pooled number is inflated whenever tasks differ in base rate.
        pt = p_all.cpu().numpy()
        yt = batch.y[:, batch.n_ctx :].cpu().numpy()
        for i in range(pt.shape[0]):
            yi = yt[i]
            present_i = np.unique(yi)
            if len(present_i) == 2 and set(present_i.tolist()) <= {0, 1}:
                per_task.append(_auc(yi, pt[i, :, 1]))
        probs.append(p_all.reshape(-1, p_all.shape[-1]).cpu().numpy())
        targets.append(yt.reshape(-1))
    model.train()

    p = np.concatenate(probs)
    y = np.concatenate(targets).astype(np.int64)
    present = np.unique(y)
    out = {
        "auc": float("nan"),
        "auc_per_task": float(np.mean(per_task)) if per_task else float("nan"),
        "brier_skill": float("nan"),
        "base_rate": float("nan"),
    }
    if len(present) < 2:
        return out
    from sklearn.metrics import roc_auc_score

    # The generic SCM prior emits multi-class tasks whenever max_classes > 2, and scoring the
    # class-1 column alone raises "multi_class must be in ('ovo', 'ovr')". That crash killed a
    # pretraining run, which is the right failure mode -- a silently wrong AUC here would have
    # been far worse, since this metric is the instrument everything else is read through
    # (docs/results/FINDINGS.md §42).
    k = p.shape[-1]
    onehot = np.zeros((len(y), k), dtype=np.float64)
    onehot[np.arange(len(y)), y] = 1.0
    if len(present) == 2 and set(present.tolist()) <= {0, 1}:
        out["base_rate"] = float((y == 1).mean())
        out["auc"] = float(roc_auc_score(y, p[:, 1]))
    else:
        # macro one-vs-rest over the classes actually present; columns for absent classes
        # would make the average meaningless rather than merely noisy
        out["base_rate"] = float((y == present[0]).mean())
        cols = p[:, present]
        cols = cols / np.clip(cols.sum(axis=1, keepdims=True), 1e-12, None)
        out["auc"] = float(
            roc_auc_score(y, cols, multi_class="ovr", average="macro", labels=present)
        )
    # Brier skill against the feature-free predictor returning the class frequencies. The
    # multi-class form reduces to the binary one when k == 2, so both paths are comparable.
    freq = onehot.mean(axis=0)
    brier = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
    reference = float(np.mean(np.sum((freq - onehot) ** 2, axis=1)))
    out["brier_skill"] = 1.0 - brier / reference if reference > 0 else float("nan")
    return out


def _eval_quality_quantile(
    model: FinancialTFM,
    prior_cfg: PriorConfig,
    rng: np.random.Generator,
    n_batches: int = 5,
    batch_size: int = 8,
) -> dict[str, float]:
    """Held-out quality for a ``--head-type quantile`` checkpoint (task 48.3).

    :func:`_eval_quality`'s AUC/Brier-skill metrics have no analogue for a continuous,
    pinball-trained head, so this reports the regression equivalents instead:

    - **pinball loss**, the same quantity training minimises, on freshly sampled held-out
      tasks -- the direct read of whether the head is still learning;
    - **nRMSE**, root-mean-squared error of the median-quantile point estimate divided by the
      held-out targets' own standard deviation, so 1.0 is exactly the predict-the-mean
      baseline -- the same convention ``fintfm-capability --regression-sweep`` reports.

    Both are computed in the context-normalised space :func:`fintfm.prior.base.collate`
    already put the targets in, since recovering each task's own mean/std here would need
    them threaded back out of ``collate`` for no benefit to a monitoring signal read for
    trend, not trusted as a benchmark result.

    Args:
        model: The model being trained. Must have ``cfg.head_type == "quantile"``.
        prior_cfg: Prior configuration, so held-out tasks match training tasks.
        rng: Random generator.
        n_batches: Held-out batches to average over.
        batch_size: Rows per held-out batch; must not exceed the training batch size (see
            :func:`_eval_quality`'s docstring for why).

    Returns:
        ``{"pinball": ..., "nrmse": ...}``.
    """
    device = next(model.parameters()).device
    model.eval()
    pinballs: list[float] = []
    sq_errs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    taus = model.quantile_taus
    median_idx = int(torch.argmin(torch.abs(taus - 0.5)).item())
    with torch.no_grad():
        for _ in range(n_batches):
            batch = sample_batch(rng, prior_cfg, batch_size=batch_size).to(device)
            out = model(batch.X, batch.y, batch.n_ctx)[:, batch.n_ctx :]
            target = batch.y[:, batch.n_ctx :].to(out.dtype)
            diff = target.unsqueeze(-1) - out
            pinball = torch.maximum(taus * diff, (taus - 1.0) * diff).mean()
            pinballs.append(float(pinball.item()))
            point = out[..., median_idx]
            sq_errs.append(((point - target) ** 2).reshape(-1).cpu().numpy())
            targets.append(target.reshape(-1).cpu().numpy())
    model.train()
    y = np.concatenate(targets)
    sd = float(np.std(y)) + 1e-12
    rmse = float(np.sqrt(np.mean(np.concatenate(sq_errs))))
    return {"pinball": float(np.mean(pinballs)), "nrmse": rmse / sd}


def _save_training_state(
    path: str,
    model: FinancialTFM,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    rng: np.random.Generator,
    step: int,
    objectives: set[str],
    total_steps: int,
    optimizer_name: str = "adamw",
    eval_rng: np.random.Generator | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    """Write everything needed to continue a run, not just the weights.

    A model checkpoint carries weights and nothing else, so resuming from one restarts AdamW
    with zeroed moments, restarts the cosine schedule, and replays the identical synthetic
    task sequence from the same seed. **None of those three show up in a loss curve**, which
    is why this is a separate artifact rather than a flag on :meth:`FinancialTFM.save`.

    Args:
        path: Destination. By convention ``<out_path>.state``.
        model: The model being trained.
        opt: Optimiser whose moment estimates must survive the restart.
        sched: Learning-rate schedule, whose position is part of the run's identity.
        rng: The numpy generator driving prior sampling; its state is saved so a resumed run
            draws *new* tasks rather than repeating the ones already seen.
        step: Number of steps completed, so the resumed loop starts at the right index.
        objectives: Which losses have been optimised so far, carried forward because a
            resumed run must not claim an objective it never trained (see
            :meth:`FinancialTFM.save`).
        total_steps: The schedule length this run was planned against. Resuming with a
            different value would silently change the learning-rate curve under AdamW, so it
            is recorded and checked; schedule-free has no such curve and the check is skipped.
        optimizer_name: :attr:`TrainConfig.optimizer`. Recorded so a resume cannot silently
            load one optimiser's state into the other, which would corrupt training without
            raising -- AdamW's and schedule-free's state dicts carry different keys.
        eval_rng: The separate generator held-out evaluation draws from (audit AU1.18), saved
            so a resumed run continues the evaluation stream rather than replaying it. Omitted
            from the file when ``None``.
        scaler: The ``--amp fp16`` loss scaler, whose current scale is saved so a resumed run
            does not restart at the initial scale and skip steps re-finding it. Omitted from
            the file when ``None``.
    """
    state = {
        "model_state": model.state_dict(),
        "model_config": asdict(model.cfg),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "numpy_rng": rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "step": step,
        "objectives": sorted(objectives),
        "total_steps": total_steps,
        "optimizer_name": optimizer_name,
    }
    if eval_rng is not None:
        state["eval_numpy_rng"] = eval_rng.bit_generator.state
    if scaler is not None:
        state["grad_scaler"] = scaler.state_dict()
    torch.save(state, path)


def train(
    model_cfg: ModelConfig, prior_cfg: PriorConfig, train_cfg: TrainConfig, out_path: str
) -> FinancialTFM:
    """Run pretraining and save the final checkpoint to ``out_path``."""
    rng = np.random.default_rng(train_cfg.seed)
    # Held-out evaluation draws from its own stream (audit AU1.18). Sharing ``rng`` made every
    # training batch after the first evaluation depend on ``eval_every``, so two runs that
    # differed only in how often they were monitored trained on different tasks. A run with
    # evaluation on therefore draws different training batches from 0.7.1 and earlier at the
    # same seed; one whose evaluation never fires draws the same ones.
    eval_rng = np.random.default_rng([train_cfg.seed, _EVAL_STREAM])
    torch.manual_seed(train_cfg.seed)
    model = FinancialTFM(model_cfg).to(train_cfg.device)
    model.feature_chunk = train_cfg.feature_chunk
    print(f"model parameters: {model.num_parameters():,}")
    opt, sched = _build_optimizer(model, train_cfg)
    _autocast(model, train_cfg.amp)  # refuse an unsupported precision before any step
    scaler = _grad_scaler(train_cfg.amp)
    if train_cfg.amp != "none":
        print(
            f"mixed precision: {train_cfg.amp} autocast (weights and optimiser float32"
            f"{', loss scaling' if scaler is not None else ''})"
        )
    t0 = time.time()
    running = 0.0
    # recorded into the checkpoint: the two objectives are exclusive per step, so a hazard
    # run leaves the classification head untrained and predict_proba would silently serve it
    objectives: set[str] = set()

    start_step = 0
    if train_cfg.init_from is not None:
        if train_cfg.resume is not None:
            raise ValueError("init_from and resume are mutually exclusive")
        init = torch.load(train_cfg.init_from, map_location=train_cfg.device, weights_only=False)
        mismatched = {
            k: (init["config"].get(k), v)
            for k, v in asdict(model_cfg).items()
            if init["config"].get(k, v) != v
        }
        if mismatched:
            # A silent partial load (strict=False) would train a different architecture than
            # the one named on the command line; refuse and name every differing field.
            raise ValueError(
                f"{train_cfg.init_from} has a different architecture "
                f"(checkpoint, this run): {mismatched}"
            )
        model.load_state_dict(init["state_dict"])
        objectives = set(init.get("trained_objectives", []))
        print(f"initialised weights from {train_cfg.init_from} (fresh optimiser and schedule)")
    if train_cfg.resume is not None:
        state = torch.load(train_cfg.resume, map_location=train_cfg.device, weights_only=False)
        saved_optimizer = state.get("optimizer_name", "adamw")
        if saved_optimizer != train_cfg.optimizer:
            # The two optimisers' state dicts carry different keys; loading one into the
            # other would corrupt training silently rather than raise on its own.
            raise ValueError(
                f"{train_cfg.resume} was trained with optimizer={saved_optimizer!r}, but "
                f"--optimizer is {train_cfg.optimizer!r}."
            )
        if train_cfg.optimizer != "schedulefree" and state["total_steps"] != train_cfg.steps:
            # The cosine schedule is a function of total steps, so resuming against a
            # different total silently trains under a different curve than the one the
            # earlier steps used. Refuse rather than produce a run nobody can interpret.
            # Schedule-free carries no such curve, so this is the one check task 48.9 lifts --
            # extending past the original total steps is the property it exists to test.
            raise ValueError(
                f"{train_cfg.resume} was written for a {state['total_steps']}-step schedule, "
                f"but --steps is {train_cfg.steps}. Pass --steps {state['total_steps']} to "
                f"continue that run, or start a fresh one."
            )
        model.load_state_dict(state["model_state"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        rng.bit_generator.state = state["numpy_rng"]
        if "eval_numpy_rng" in state:  # absent from states written before AU1.18
            eval_rng.bit_generator.state = state["eval_numpy_rng"]
        if scaler is not None and "grad_scaler" in state:
            scaler.load_state_dict(state["grad_scaler"])
        torch.set_rng_state(
            state["torch_rng"].cpu() if hasattr(state["torch_rng"], "cpu") else state["torch_rng"]
        )
        start_step = int(state["step"])
        objectives = set(state["objectives"])
        print(f"resumed from {train_cfg.resume} at step {start_step}/{train_cfg.steps}")

    stop_step = train_cfg.steps
    if train_cfg.run_steps is not None:
        stop_step = min(train_cfg.steps, start_step + train_cfg.run_steps)
    _set_optimizer_mode(opt, train_cfg.optimizer, "train")
    dump = (
        None
        if train_cfg.task_dump is None
        else open_task_dump(train_cfg.task_dump, prior_cfg, train_cfg.batch_size, train_cfg.steps)
    )
    for step in range(start_step, stop_step):
        batch = (
            sample_batch(rng, prior_cfg, train_cfg.batch_size) if dump is None else dump.batch(step)
        ).to(train_cfg.device)
        loss = _training_step(model, batch, opt, sched, train_cfg.amp, objectives, scaler)
        running += loss.item()
        if (step + 1) % train_cfg.log_every == 0:
            elapsed = time.time() - t0
            print(
                f"step {step + 1}/{train_cfg.steps}  loss {running / train_cfg.log_every:.4f}  "
                f"lr {sched.get_last_lr()[0]:.2e}  {elapsed:.0f}s"
            )
            running = 0.0
        if train_cfg.checkpoint_every and (step + 1) % train_cfg.checkpoint_every == 0:
            # written to a sibling path, not out_path: a run killed *during* a save would
            # otherwise leave a truncated file where the final checkpoint belongs
            _set_optimizer_mode(opt, train_cfg.optimizer, "eval")
            partial = f"{out_path}.step{step + 1}"
            model.save(partial, trained_objectives=tuple(sorted(objectives)))
            _save_training_state(
                f"{out_path}.state",
                model,
                opt,
                sched,
                rng,
                step + 1,
                objectives,
                train_cfg.steps,
                optimizer_name=train_cfg.optimizer,
                eval_rng=eval_rng,
                scaler=scaler,
            )
            _set_optimizer_mode(opt, train_cfg.optimizer, "train")
            print(f"  checkpoint at step {step + 1}: {partial}", flush=True)
        if (step + 1) % train_cfg.eval_every == 0:
            _set_optimizer_mode(opt, train_cfg.optimizer, "eval")
            if model.cfg.head_type == "quantile":
                qq = _eval_quality_quantile(
                    model, prior_cfg, eval_rng, batch_size=train_cfg.batch_size
                )
                print(f"  held-out: pinball {qq['pinball']:.4f}  nRMSE {qq['nrmse']:.4f}")
            else:
                q = _eval_quality(model, prior_cfg, eval_rng, batch_size=train_cfg.batch_size)
                print(
                    f"  held-out: AUC/task {q['auc_per_task']:.3f}  AUC pooled {q['auc']:.3f}"
                    f"  Brier skill vs base rate {q['brier_skill']:+.3f}"
                    f"  (base rate {q['base_rate']:.3f})"
                )
            _set_optimizer_mode(opt, train_cfg.optimizer, "train")
    _set_optimizer_mode(opt, train_cfg.optimizer, "eval")
    model.save(out_path, trained_objectives=tuple(sorted(objectives)))
    _save_training_state(
        f"{out_path}.state",
        model,
        opt,
        sched,
        rng,
        stop_step,
        objectives,
        train_cfg.steps,
        optimizer_name=train_cfg.optimizer,
        eval_rng=eval_rng,
        scaler=scaler,
    )
    print(f"saved checkpoint to {out_path}")
    if stop_step < train_cfg.steps:
        print(
            f"ran {start_step}->{stop_step} of {train_cfg.steps}; continue with "
            f"--resume {out_path}.state --steps {train_cfg.steps}"
        )
    return model


def _synchronize(device: torch.device) -> None:
    """Wait for queued accelerator work, so a wall-clock reading covers it.

    Args:
        device: The model's device. A no-op on CPU, where work is synchronous.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark(
    model_cfg: ModelConfig,
    prior_cfg: PriorConfig,
    train_cfg: TrainConfig,
    n_steps: int,
    warmup: int = 3,
) -> dict[str, Any]:
    """Measure pretraining throughput: tasks per second, and peak memory where available.

    Builds the model, optimiser and schedule exactly as :func:`train` does and runs the same
    step (:func:`_training_step`), so ``--amp``, ``--optimizer`` and the architecture flags
    are measured as they would train. ``warmup`` untimed steps go first (allocator growth,
    kernel selection); then ``n_steps`` timed ones. Prior sampling and the device step are
    timed separately, because a faster kernel cannot speed up a run whose step is waiting on
    the CPU-side prior. Nothing is saved and no evaluation runs; ``resume``, ``init_from``
    and ``run_steps`` are ignored, since throughput does not depend on the weights.

    One task is one ``(context, query)`` problem, so tasks per step is the batch size --
    the unit ``CLAUDE.md`` asks training volume to be counted in.

    Args:
        model_cfg: Architecture to measure.
        prior_cfg: Prior to sample from; its cost is part of the step.
        train_cfg: Read for ``batch_size``, ``device``, ``seed``, ``amp``, the optimiser and
            ``feature_chunk``.
        n_steps: Timed steps.
        warmup: Untimed steps before timing starts.

    Returns:
        A JSON-serialisable record: ``tasks_per_second`` and ``tasks_per_hour``, the
        ``sample_seconds``/``step_seconds`` split, ``peak_memory_bytes`` (CUDA's peak
        allocated by tensors during the timed steps, else ``None``), ``peak_rss_bytes`` (the
        process's lifetime peak resident set, where the platform reports one), and the
        settings that produced the numbers.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")
    rng = np.random.default_rng(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)
    model = FinancialTFM(model_cfg).to(train_cfg.device)
    model.feature_chunk = train_cfg.feature_chunk
    opt, sched = _build_optimizer(model, train_cfg)
    _autocast(model, train_cfg.amp)
    scaler = _grad_scaler(train_cfg.amp)
    device = next(model.parameters()).device
    objectives: set[str] = set()
    _set_optimizer_mode(opt, train_cfg.optimizer, "train")
    model.train()
    for _ in range(warmup):
        batch = sample_batch(rng, prior_cfg, train_cfg.batch_size).to(device)
        _training_step(model, batch, opt, sched, train_cfg.amp, objectives, scaler).item()
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sample_s = step_s = 0.0
    for _ in range(n_steps):
        t0 = time.perf_counter()
        batch = sample_batch(rng, prior_cfg, train_cfg.batch_size).to(device)
        _synchronize(device)
        t1 = time.perf_counter()
        _training_step(model, batch, opt, sched, train_cfg.amp, objectives, scaler).item()
        _synchronize(device)
        sample_s += t1 - t0
        step_s += time.perf_counter() - t1
    total = sample_s + step_s
    tasks = n_steps * train_cfg.batch_size
    peak_rss: int | None = None
    try:
        import resource  # Unix only

        # ru_maxrss is bytes on macOS and kilobytes on Linux
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss = int(rss if sys.platform == "darwin" else rss * 1024)
    except ImportError:
        pass
    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "amp": train_cfg.amp,
        "optimizer": train_cfg.optimizer,
        "batch_size": train_cfg.batch_size,
        "parameters": model.num_parameters(),
        "warmup_steps": warmup,
        "steps": n_steps,
        "tasks": tasks,
        "seconds": total,
        "sample_seconds": sample_s,
        "step_seconds": step_s,
        "tasks_per_second": tasks / total,
        "tasks_per_hour": 3600.0 * tasks / total,
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "peak_rss_bytes": peak_rss,
    }


def main(argv: list[str] | None = None) -> None:
    """Pretrain a checkpoint on synthetic tasks.

    Entry point for the ``train`` console script; see
    ``--help`` for the flags. Writes its record as JSON under ``--out``. With
    ``--benchmark-steps N`` it instead measures throughput and prints one JSON line, saving
    nothing.

    Args:
        argv: Arguments to parse; ``None`` reads ``sys.argv``, as the console script does.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="linear learning-rate warmup before the cosine decay (audit AU5.10); defaults "
        "to the config's train.warmup_steps, 500, the value every checkpoint used",
    )
    p.add_argument(
        "--decay-norm-bias",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="apply AdamW weight decay to LayerNorm gains, biases, embeddings and token "
        "parameters too (audit AU5.10). Defaults to the config's train.decay_norm_bias, "
        "true, which is every checkpoint's behaviour; --no-decay-norm-bias exempts them",
    )
    p.add_argument(
        "--amp",
        type=str,
        default=None,
        choices=AMP_MODES,
        help="mixed precision for the training forward pass (audit AU5.2): 'none' (float32, "
        "the config default and every checkpoint's), 'bf16' autocast on CUDA (compute "
        "capability 8.0+) or CPU, or 'fp16' autocast with loss scaling on CUDA only, for "
        "GPUs without native bf16 (T4, P100). Binned classification objective only. "
        "Weights, loss reduction and optimiser stay float32; refused on MPS",
    )
    p.add_argument(
        "--muon-lr",
        type=float,
        default=None,
        help="base learning rate of Muon's matrix groups under --optimizer muon (the AdamW "
        "side keeps --lr); defaults to the config's train.muon_lr, 0.02",
    )
    p.add_argument(
        "--benchmark-steps",
        type=int,
        default=None,
        help="measure throughput instead of training: run 3 untimed then N timed steps of "
        "this exact configuration and print one JSON line (tasks/second, tasks/hour, the "
        "prior-sampling vs device-step split, peak CUDA memory). Saves nothing; --out, "
        "--resume and evaluation are ignored",
    )
    p.add_argument("--n-rows", type=int, default=256)
    p.add_argument(
        "--n-rows-choices",
        type=str,
        default=None,
        help="comma-separated task sizes sampled per batch, e.g. 256,1024,2048. The "
        "base-rate floor scales with task size, so a fixed small value caps how "
        "imbalanced any task can be (docs/results/FINDINGS.md #26)",
    )
    p.add_argument("--max-features", type=int, default=24)
    p.add_argument(
        "--p-trivial",
        type=float,
        default=0.0,
        help="probability of a deliberately trivial task (docs/results/FINDINGS.md §53): the control "
        "that separates 'our prior is too hard' from 'the model cannot learn'",
    )
    p.add_argument(
        "--p-regression",
        type=float,
        default=0.0,
        help="probability of drawing a regression task: the SCM prior's continuous latent "
        "kept rather than thresholded, binned on context quantiles. Bins share the "
        "classification head, so this needs no architecture change (see "
        "fintfm/inference/binning.py)",
    )
    p.add_argument(
        "--scm-legacy",
        action="store_true",
        help="draw SCM tasks from the pre-48.17/48.19 prior (uniform edge sparsity, five "
        "activations); the control arm for the widened prior. Inert at --p-financial 1.0",
    )
    p.add_argument(
        "--p-tree",
        type=float,
        default=None,
        help="probability of drawing a tree-structured task (axis-aligned boundaries); "
        "selected on distinctiveness, see docs/results/FINDINGS.md 111",
    )
    p.add_argument(
        "--p-learnability-filter",
        type=float,
        default=None,
        help="probability of judging a drawn task with a held-out ExtraTrees and resampling "
        "it on rejection (task 48.5); 0.0 reproduces every prior checkpoint's behaviour, "
        "see docs/results/FINDINGS.md 125/129",
    )
    p.add_argument(
        "--min-ctx-per-class",
        type=int,
        default=None,
        help="context rows guaranteed to every class once the batch's split is drawn (audit "
        "AU1.16); defaults to the config's prior.min_ctx_per_class, 0, which moves nothing",
    )
    p.add_argument(
        "--scm-reuse-graph",
        type=int,
        default=None,
        help="tasks drawn per SCM graph (task 48.12); >1 gives ~2.5x tasks/second at 4",
    )
    p.add_argument(
        "--p-crossed",
        type=float,
        default=0.0,
        help="probability of a crossed-design task (docs/results/FINDINGS.md §66, prior/crossed.py): "
        "SCM features under the financial prior's label mechanism, isolating whether the "
        "financial prior's teaching failure tracks its features or its label function",
    )
    p.add_argument(
        "--p-task-family",
        type=float,
        default=0.0,
        help="probability of drawing one of prior/task_families.py's task 40.2-40.4 "
        "constructions instead (uniformly over the nine named families plus the "
        "AND-composition, Bayes-AUC target drawn per task); task 40.7, scoped only after "
        "40.2-40.6 validated on existing checkpoints, see docs/results/FINDINGS.md §137-§140",
    )
    p.add_argument(
        "--p-real-edgar",
        type=float,
        default=0.0,
        help="probability of drawing a task from a real SEC EDGAR firm-quarter panel instead "
        "of a synthetic one (FinTFM-R, task A2/A3); requires --real-edgar-panel-path and the "
        "'real' extra (uv sync --extra real). 0.0 (default) never opens the panel file",
    )
    p.add_argument(
        "--real-edgar-panel-path",
        type=str,
        default=None,
        help="parquet file built by scripts/edgar/build_panel.py; ignored unless "
        "--p-real-edgar > 0",
    )
    p.add_argument(
        "--p-real-mortgage",
        type=float,
        default=0.0,
        help="probability of drawing a task from a real Freddie Mac loan-level panel instead "
        "of a synthetic one (FinTFM-R, task B3/B4); requires --real-mortgage-panel-path and "
        "the 'real' extra (uv sync --extra real). 0.0 (default) never opens the panel file",
    )
    p.add_argument(
        "--real-mortgage-panel-path",
        type=str,
        default=None,
        help="parquet file built by scripts/freddie_mac/build_panel.py; ignored unless "
        "--p-real-mortgage > 0",
    )
    p.add_argument(
        "--p-real-lendingclub",
        type=float,
        default=0.0,
        help="probability of drawing a task from a real LendingClub panel instead of a "
        "synthetic one (D17's third real source); requires --real-lendingclub-panel-path and "
        "the 'real' extra (uv sync --extra real). 0.0 (default) never opens the panel file",
    )
    p.add_argument(
        "--real-lendingclub-panel-path",
        type=str,
        default=None,
        help="parquet file built by scripts/lendingclub/build_panel.py; ignored unless "
        "--p-real-lendingclub > 0",
    )
    p.add_argument(
        "--real-panels",
        type=str,
        default="",
        help="further real sources as 'path=prob,path=prob' (panels from "
        "scripts/candidates/build_panels.py); each an independent draw. Empty (default) "
        "opens no file",
    )
    p.add_argument(
        "--identity-shuffle",
        action="store_true",
        help="expose financial-task columns from independently-per-account-permuted accounts "
        "(docs/results/FINDINGS.md §67-§71, task 38.11): breaks cross-account identities while "
        "leaving the label and each column's marginal untouched",
    )
    p.add_argument(
        "--discretize-frac",
        type=float,
        default=0.0,
        help="fraction of a financial task's exposed columns to replace with their own "
        "5-quantile-bin means (task 48.6, Nori's 'discretized features')",
    )
    p.add_argument(
        "--n-noise-features",
        type=int,
        default=0,
        help="pure-Gaussian columns appended to every financial task, beyond the prior's "
        "existing unconditional 0-4 (task 48.6, Nori's 'noise features')",
    )
    p.add_argument(
        "--n-correlated-block-features",
        type=int,
        default=0,
        help="near-duplicate columns of one shared source appended to every financial task "
        "(task 48.6, Nori's 'correlated blocks')",
    )
    p.add_argument(
        "--label-noise-rate",
        type=float,
        default=0.0,
        help="fraction of labels flipped after being finalised, on every financial task "
        "(task 48.6, Nori's 'label noise')",
    )
    p.add_argument(
        "--p-financial",
        type=float,
        default=None,
        help="probability of drawing a financial rather than a generic SCM task; the "
        "survival objective forces 1.0 regardless (docs/results/FINDINGS.md §14)",
    )
    p.add_argument("--max-classes", type=int, default=10)
    p.add_argument(
        "--n-horizons",
        type=int,
        default=None,
        help="train a monotone PD term structure over K periods (requires p_financial=1.0)",
    )
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--d-cell", type=int, default=64)
    p.add_argument(
        "--pooling",
        type=str,
        default="meanmax",
        choices=("meanmax", "attention"),
        help="how feature tokens become a row vector; 'attention' can weight features "
        "where a mean cannot (docs/results/FINDINGS.md §50)",
    )
    p.add_argument(
        "--column-id-dim",
        type=int,
        default=None,
        help="width of the random per-task column identity; defaults to d_cell//4, and 0 "
        "reproduces the pre-fix architecture. Without it the row encoder is a symmetric "
        "function of the row's values and cannot represent 'column j matters' "
        "(docs/results/FINDINGS.md §54)",
    )
    p.add_argument(
        "--n-cell-blocks",
        type=int,
        default=0,
        help="alternating two-way cell-attention blocks before pooling; 0 (default) "
        "reproduces every checkpoint trained before this existed. Task 39.1, testing "
        "whether §74's capacity cap is architectural (docs/results/FINDINGS.md §74, §76)",
    )
    p.add_argument(
        "--run-steps",
        type=int,
        default=None,
        help="steps to run in THIS invocation; the LR schedule still spans --steps. Use when "
        "a run is longer than one job's wall-clock: each job saves <out>.state and "
        "prints the --resume line to continue with",
    )
    p.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="initialise weights from a model checkpoint with a fresh optimiser and schedule "
        "(continued pretraining for matched-pair ablations). The checkpoint's architecture "
        "must match this run's exactly; mutually exclusive with --resume",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="continue from a <out>.state file, restoring optimiser moments, schedule "
        "position, both RNG streams and the step counter. Resuming from a plain model "
        "checkpoint instead would restart AdamW cold and replay the same synthetic "
        "tasks, neither of which shows up in a loss curve",
    )
    p.add_argument(
        "--feature-chunk",
        type=int,
        default=None,
        help="features per row-within-feature attention call (--n-cell-blocks > 0 only). "
        "Identity-preserving, so it changes no number and only bounds memory "
        "(docs/results/FINDINGS.md S81): that stage's attention is (batch*F, heads, N, N), "
        "which is what made S78's run OOM on a T4 and forced its protocol deviation. "
        "Try 16 if a cell-attention run does not fit",
    )
    p.add_argument(
        "--cell-labels",
        action="store_true",
        help="inject labels per-cell before the cell-attention blocks, not only after "
        "pooling; ignored unless --n-cell-blocks > 0 (task 39.2)",
    )
    p.add_argument(
        "--mask-embedding",
        action="store_true",
        help="dedicated missingness embedding (nn.Embedding(2, d_cell), added to a "
        "value-only embedding) instead of the default joint [value, missing] cell MLP "
        "(task 48.10)",
    )
    p.add_argument(
        "--n-heads",
        type=int,
        default=None,
        help="attention heads in every stage; must divide --d-cell and --d-model. Defaults "
        "to the config's model.n_heads, 4, every checkpoint's value",
    )
    p.add_argument(
        "--n-cell-heads",
        type=int,
        default=None,
        help="attention heads in the cell-width stages (column, row-within-feature, "
        "pooling); defaults to the config's model.n_cell_heads, null, which shares "
        "--n-heads (audit AU5.3)",
    )
    p.add_argument(
        "--n-summary-tokens",
        type=int,
        default=None,
        help="K learned summary tokens per row replacing mean+max pooling (audit AU5.3); "
        "the config default 1 keeps today's pooling; incompatible with --pooling attention",
    )
    p.add_argument(
        "--value-embedding",
        type=str,
        default=None,
        choices=("mlp", "rbf", "fourier"),
        help="basis expansion of the normalised cell value before the cell MLP (audit "
        "AU5.16); the config default 'mlp' is today's embedding",
    )
    p.add_argument(
        "--value-basis-size",
        type=int,
        default=None,
        help="RBF centres or Fourier frequencies (config default 16); ignored for "
        "--value-embedding mlp",
    )
    p.add_argument(
        "--value-basis-learnable",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="train the RBF centres and widths or the Fourier frequencies (audit AU5.16); "
        "config default false",
    )
    p.add_argument(
        "--value-ecdf",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="add a context-only ECDF-rank channel to every cell embedding (audit AU5.16); "
        "config default false",
    )
    p.add_argument(
        "--attn-length-scale",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="multiply row-attention logits by log(n_ctx)/log(--attn-length-ref) (audit "
        "AU5.17); config default false",
    )
    p.add_argument(
        "--attn-length-ref",
        type=int,
        default=None,
        help="context length at which --attn-length-scale is the identity (config default "
        "512)",
    )
    p.add_argument(
        "--hazard-period-context",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="context rows of a hazard model embed their default period, or a censored "
        "token, beside the label (audit AU5.9); requires --n-horizons; config default "
        "false. Serving such a checkpoint needs FinancialTFMClassifier.fit(X, y, "
        "periods=...), which the V4FinBench harnesses pass",
    )
    p.add_argument(
        "--head-type",
        type=str,
        default="binned",
        choices=["binned", "quantile"],
        help="'binned' (default): classify a target binned into --max-classes quantile "
        "bins. 'quantile' (task 48.3): a genuine pinball-loss head over --n-quantiles "
        "levels, no binning -- requires --p-regression 1.0, since this head has no "
        "class-index path for a classification task",
    )
    p.add_argument(
        "--n-quantiles",
        type=int,
        default=99,
        help="output width of a --head-type quantile head; ignored for 'binned'",
    )
    p.add_argument(
        "--d-ff",
        type=int,
        default=None,
        help="feed-forward width; defaults to 4 x d_model, the transformer convention. "
        "Leaving it pinned while d_model grows makes the FFN a bottleneck and distorts "
        "any scaling comparison (docs/infra/HF_JOBS.md)",
    )
    p.add_argument("--n-layers", type=int, default=6, help="row-attention layers")
    p.add_argument("--n-col-layers", type=int, default=2, help="column-attention layers")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="cap torch CPU threads; set this on a shared machine (see CLAUDE.md)",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="save every N steps as well as at the end; 0 disables. Essential for any run "
        "long enough that losing it would hurt",
    )
    p.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=list(OPTIMIZERS),
        help="'adamw' (default, cosine-annealed), 'muon' (audit AU5.17: Muon on the "
        "attention and feed-forward matrices at --muon-lr, AdamW at --lr on the rest, same "
        "cosine schedule) or 'schedulefree' (task 48.9, needs the "
        "schedulefree extra: uv sync --extra schedulefree); the latter carries no external "
        "LR curve, so a run under it can be resumed past its original --steps. --lr's "
        "default (3e-4) is AdamW's tuned value (section 127), not schedulefree's own -- "
        "AdamWScheduleFree's package default is 0.0025, substantially higher; pass --lr "
        "explicitly for a schedulefree run rather than inheriting AdamW's tuning "
        "(section 146's addendum measured a negative result at the untuned default)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="runs/v0.pt")
    p.add_argument("--config", type=str, default=None, help="YAML overriding the defaults")
    add_prior_lever_arguments(p)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    model_cfg = ModelConfig(
        pooling=args.pooling,
        column_id_dim=args.column_id_dim,
        n_cell_blocks=args.n_cell_blocks,
        cell_labels=args.cell_labels,
        mask_embedding=args.mask_embedding,
        head_type=args.head_type,
        n_quantiles=args.n_quantiles,
        d_ff=args.d_ff if args.d_ff is not None else 4 * args.d_model,
        max_features=args.max_features,
        max_classes=args.max_classes,
        d_cell=args.d_cell,
        d_model=args.d_model,
        n_col_layers=args.n_col_layers,
        n_layers=args.n_layers,
        n_horizons=args.n_horizons,
        # the Phase 5 architecture levers and n_heads: an explicit flag wins over the
        # config's model section, whose defaults build today's architecture
        **{
            name: getattr(cfg.model, name) if getattr(args, name) is None else getattr(args, name)
            for name in _MODEL_CONFIG_KNOBS
        },
    )
    prior_cfg = PriorConfig(
        max_features=args.max_features,
        max_classes=args.max_classes,
        n_rows=args.n_rows,
        n_rows_choices=(
            tuple(int(v) for v in args.n_rows_choices.split(",")) if args.n_rows_choices else None
        ),
        # the survival objective needs every task to carry a period
        scm_legacy=args.scm_legacy,
        scm_reuse_graph=(
            PriorConfig.scm_reuse_graph if args.scm_reuse_graph is None else args.scm_reuse_graph
        ),
        p_tree=(PriorConfig.p_tree if args.p_tree is None else args.p_tree),
        p_learnability_filter=(
            PriorConfig.p_learnability_filter
            if args.p_learnability_filter is None
            else args.p_learnability_filter
        ),
        p_financial=(
            1.0
            if args.n_horizons
            else (PriorConfig.p_financial if args.p_financial is None else args.p_financial)
        ),
        n_horizons=args.n_horizons,
        p_trivial=args.p_trivial,
        p_crossed=args.p_crossed,
        p_task_family=args.p_task_family,
        p_regression=args.p_regression,
        identity_shuffle=args.identity_shuffle,
        discretize_frac=args.discretize_frac,
        n_noise_features=args.n_noise_features,
        n_correlated_block_features=args.n_correlated_block_features,
        label_noise_rate=args.label_noise_rate,
        head_type=args.head_type,
        p_real_edgar=args.p_real_edgar,
        p_real_mortgage=args.p_real_mortgage,
        p_real_lendingclub=args.p_real_lendingclub,
        real_panels=args.real_panels,
        real_edgar_panel_path=(
            PriorConfig.real_edgar_panel_path
            if args.real_edgar_panel_path is None
            else args.real_edgar_panel_path
        ),
        real_mortgage_panel_path=(
            PriorConfig.real_mortgage_panel_path
            if args.real_mortgage_panel_path is None
            else args.real_mortgage_panel_path
        ),
        real_lendingclub_panel_path=(
            PriorConfig.real_lendingclub_panel_path
            if args.real_lendingclub_panel_path is None
            else args.real_lendingclub_panel_path
        ),
        # the default-rate envelope comes from configuration, because a prior that cannot
        # generate the regime being evaluated is the defect behind docs/results/FINDINGS.md §26
        sharpness_min=cfg.prior.sharpness_min,
        sharpness_max=cfg.prior.sharpness_max,
        min_expected_positives=cfg.prior.min_expected_positives,
        absolute_rate_floor=cfg.prior.absolute_rate_floor,
        rate_ceiling=cfg.prior.rate_ceiling,
        n_sectors_max=cfg.prior.n_sectors,
        min_ctx_per_class=(
            cfg.prior.min_ctx_per_class
            if args.min_ctx_per_class is None
            else args.min_ctx_per_class
        ),
        **prior_lever_values(args, cfg.prior),
    )
    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        feature_chunk=args.feature_chunk,
        run_steps=args.run_steps,
        resume=args.resume,
        init_from=args.init_from,
        optimizer=args.optimizer,
        task_dump=args.task_dump,
        warmup_steps=cfg.train.warmup_steps if args.warmup_steps is None else args.warmup_steps,
        decay_norm_bias=(
            cfg.train.decay_norm_bias if args.decay_norm_bias is None else args.decay_norm_bias
        ),
        amp=cfg.train.amp if args.amp is None else args.amp,
        muon_lr=cfg.train.muon_lr if args.muon_lr is None else args.muon_lr,
    )
    print(f"config: {cfg.provenance()}")
    if args.benchmark_steps is not None:
        record = benchmark(model_cfg, prior_cfg, train_cfg, args.benchmark_steps)
        print(json.dumps(record), flush=True)
        return
    train(model_cfg, prior_cfg, train_cfg, args.out)


if __name__ == "__main__":
    main()
