"""Pretraining loop: sample a fresh synthetic batch every step, no epochs.

Usage:
    fintfm-train --steps 20000 --batch-size 64 --out runs/v0.pt
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from fintfm.config import load_config
from fintfm.modeling.model import FinancialTFM, ModelConfig
from fintfm.prior import PriorConfig
from fintfm.prior.mixture import sample_batch


@dataclass
class TrainConfig:
    steps: int = 20_000
    batch_size: int = 64
    lr: float = 3e-4
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
    #: Features per row-within-feature attention call, for ``n_cell_blocks > 0`` models.
    #: Identity-preserving (``docs/FINDINGS.md`` §81), so it changes no number and only
    #: bounds memory. ``None`` keeps the unchunked path, which is what every checkpoint
    #: before §81 trained with; it is also what made §78's run OOM on a T4 and forced that
    #: finding's documented protocol deviation (§79).
    feature_chunk: int | None = None


def _lr_schedule(step: int, cfg: TrainConfig) -> float:
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
    progress (``docs/FINDINGS.md`` §42). ``CLAUDE.md`` already said accuracy is not a proper
    scoring rule and that this is why training optimises cross-entropy; the evaluation inside
    the training loop did not follow the repository's own rule for three runs.

    Reports instead:

    - **AUC**, pooled over every query row of every task. Kept for continuity with earlier
      runs, and **it is the inflated one**: pooling across tasks whose base rates differ lets
      a model score well by predicting each task's base rate without discriminating *within*
      any task. Measured on the financial prior, whose per-task rates span 0.003 to 0.986,
      pooled AUC read 0.9080 where the per-task mean was 0.6426 -- an overstatement of 0.26,
      and the reason a checkpoint could log "held-out AUC 0.943" while scoring 0.63-0.69 on
      every probe (``docs/FINDINGS.md`` §57). The trivial prior hides this, because every one
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
    # (docs/FINDINGS.md §42).
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


def _save_training_state(
    path: str,
    model: FinancialTFM,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    rng: np.random.Generator,
    step: int,
    objectives: set[str],
    total_steps: int,
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
            different value would silently change the learning-rate curve, so it is recorded
            and checked.
    """
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model.cfg),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "numpy_rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "step": step,
            "objectives": sorted(objectives),
            "total_steps": total_steps,
        },
        path,
    )


def train(model_cfg: ModelConfig, prior_cfg: PriorConfig, train_cfg: TrainConfig, out_path: str) -> FinancialTFM:
    """Run pretraining and save the final checkpoint to ``out_path``."""
    rng = np.random.default_rng(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)
    model = FinancialTFM(model_cfg).to(train_cfg.device)
    model.feature_chunk = train_cfg.feature_chunk
    print(f"model parameters: {model.num_parameters():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr_schedule(s, train_cfg))
    t0 = time.time()
    running = 0.0
    # recorded into the checkpoint: the two objectives are exclusive per step, so a hazard
    # run leaves the classification head untrained and predict_proba would silently serve it
    objectives: set[str] = set()

    start_step = 0
    if train_cfg.resume is not None:
        state = torch.load(train_cfg.resume, map_location=train_cfg.device, weights_only=False)
        if state["total_steps"] != train_cfg.steps:
            # The cosine schedule is a function of total steps, so resuming against a
            # different total silently trains under a different curve than the one the
            # earlier steps used. Refuse rather than produce a run nobody can interpret.
            raise ValueError(
                f"{train_cfg.resume} was written for a {state['total_steps']}-step schedule, "
                f"but --steps is {train_cfg.steps}. Pass --steps {state['total_steps']} to "
                f"continue that run, or start a fresh one."
            )
        model.load_state_dict(state["model_state"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"].cpu() if hasattr(state["torch_rng"], "cpu") else state["torch_rng"])
        start_step = int(state["step"])
        objectives = set(state["objectives"])
        print(f"resumed from {train_cfg.resume} at step {start_step}/{train_cfg.steps}")

    stop_step = train_cfg.steps
    if train_cfg.run_steps is not None:
        stop_step = min(train_cfg.steps, start_step + train_cfg.run_steps)
    for step in range(start_step, stop_step):
        batch = sample_batch(rng, prior_cfg, train_cfg.batch_size).to(train_cfg.device)
        # survival objective when the model has a hazard head and the prior emits periods
        if model.hazard is not None and batch.period is not None:
            loss = model.survival_loss(batch.X, batch.y, batch.period, batch.n_ctx)
            objectives.add("survival")
        else:
            loss = model.loss(batch.X, batch.y, batch.n_ctx, batch.n_classes)
            objectives.add("classification")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running += loss.item()
        if (step + 1) % train_cfg.log_every == 0:
            elapsed = time.time() - t0
            print(f"step {step + 1}/{train_cfg.steps}  loss {running / train_cfg.log_every:.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {elapsed:.0f}s")
            running = 0.0
        if train_cfg.checkpoint_every and (step + 1) % train_cfg.checkpoint_every == 0:
            # written to a sibling path, not out_path: a run killed *during* a save would
            # otherwise leave a truncated file where the final checkpoint belongs
            partial = f"{out_path}.step{step + 1}"
            model.save(partial, trained_objectives=tuple(sorted(objectives)))
            _save_training_state(
                f"{out_path}.state", model, opt, sched, rng, step + 1, objectives,
                train_cfg.steps,
            )
            print(f"  checkpoint at step {step + 1}: {partial}", flush=True)
        if (step + 1) % train_cfg.eval_every == 0:
            q = _eval_quality(model, prior_cfg, rng, batch_size=train_cfg.batch_size)
            print(
                f"  held-out: AUC/task {q['auc_per_task']:.3f}  AUC pooled {q['auc']:.3f}"
                f"  Brier skill vs base rate {q['brier_skill']:+.3f}"
                f"  (base rate {q['base_rate']:.3f})"
            )
    model.save(out_path, trained_objectives=tuple(sorted(objectives)))
    _save_training_state(
        f"{out_path}.state", model, opt, sched, rng, stop_step, objectives, train_cfg.steps
    )
    print(f"saved checkpoint to {out_path}")
    if stop_step < train_cfg.steps:
        print(
            f"ran {start_step}->{stop_step} of {train_cfg.steps}; continue with "
            f"--resume {out_path}.state --steps {train_cfg.steps}"
        )
    return model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-rows", type=int, default=256)
    p.add_argument(
        "--n-rows-choices", type=str, default=None,
        help="comma-separated task sizes sampled per batch, e.g. 256,1024,2048. The "
             "base-rate floor scales with task size, so a fixed small value caps how "
             "imbalanced any task can be (docs/FINDINGS.md #26)",
    )
    p.add_argument("--max-features", type=int, default=24)
    p.add_argument(
        "--p-trivial", type=float, default=0.0,
        help="probability of a deliberately trivial task (docs/FINDINGS.md §53): the control "
             "that separates 'our prior is too hard' from 'the model cannot learn'",
    )
    p.add_argument(
        "--p-regression", type=float, default=0.0,
        help="probability of drawing a regression task: the SCM prior's continuous latent "
             "kept rather than thresholded, binned on context quantiles. Bins share the "
             "classification head, so this needs no architecture change (see "
             "fintfm/inference/binning.py)",
    )
    p.add_argument(
        "--scm-legacy", action="store_true",
        help="draw SCM tasks from the pre-48.17/48.19 prior (uniform edge sparsity, five "
             "activations); the control arm for the widened prior. Inert at --p-financial 1.0",
    )
    p.add_argument(
        "--p-tree", type=float, default=None,
        help="probability of drawing a tree-structured task (axis-aligned boundaries); "
             "selected on distinctiveness, see docs/FINDINGS.md 111",
    )
    p.add_argument(
        "--p-crossed", type=float, default=0.0,
        help="probability of a crossed-design task (docs/FINDINGS.md §66, prior/crossed.py): "
             "SCM features under the financial prior's label mechanism, isolating whether the "
             "financial prior's teaching failure tracks its features or its label function",
    )
    p.add_argument(
        "--identity-shuffle", action="store_true",
        help="expose financial-task columns from independently-per-account-permuted accounts "
             "(docs/FINDINGS.md §67-§71, task 38.11): breaks cross-account identities while "
             "leaving the label and each column's marginal untouched",
    )
    p.add_argument(
        "--p-financial", type=float, default=None,
        help="probability of drawing a financial rather than a generic SCM task; the "
             "survival objective forces 1.0 regardless (docs/FINDINGS.md §14)",
    )
    p.add_argument("--max-classes", type=int, default=10)
    p.add_argument(
        "--n-horizons", type=int, default=None,
        help="train a monotone PD term structure over K periods (requires p_financial=1.0)",
    )
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--d-cell", type=int, default=64)
    p.add_argument(
        "--pooling", type=str, default="meanmax", choices=("meanmax", "attention"),
        help="how feature tokens become a row vector; 'attention' can weight features "
             "where a mean cannot (docs/FINDINGS.md §50)",
    )
    p.add_argument(
        "--column-id-dim", type=int, default=None,
        help="width of the random per-task column identity; defaults to d_cell//4, and 0 "
             "reproduces the pre-fix architecture. Without it the row encoder is a symmetric "
             "function of the row's values and cannot represent 'column j matters' "
             "(docs/FINDINGS.md §54)",
    )
    p.add_argument(
        "--n-cell-blocks", type=int, default=0,
        help="alternating two-way cell-attention blocks before pooling; 0 (default) "
             "reproduces every checkpoint trained before this existed. Task 39.1, testing "
             "whether §74's capacity cap is architectural (docs/FINDINGS.md §74, §76)",
    )
    p.add_argument(
        "--run-steps", type=int, default=None,
        help="steps to run in THIS invocation; the LR schedule still spans --steps. Use when "
             "a run is longer than one job's wall-clock: each job saves <out>.state and "
             "prints the --resume line to continue with",
    )
    p.add_argument(
        "--resume", type=str, default=None,
        help="continue from a <out>.state file, restoring optimiser moments, schedule "
             "position, both RNG streams and the step counter. Resuming from a plain model "
             "checkpoint instead would restart AdamW cold and replay the same synthetic "
             "tasks, neither of which shows up in a loss curve",
    )
    p.add_argument(
        "--feature-chunk", type=int, default=None,
        help="features per row-within-feature attention call (--n-cell-blocks > 0 only). "
             "Identity-preserving, so it changes no number and only bounds memory "
             "(docs/FINDINGS.md S81): that stage's attention is (batch*F, heads, N, N), "
             "which is what made S78's run OOM on a T4 and forced its protocol deviation. "
             "Try 16 if a cell-attention run does not fit",
    )
    p.add_argument(
        "--cell-labels", action="store_true",
        help="inject labels per-cell before the cell-attention blocks, not only after "
             "pooling; ignored unless --n-cell-blocks > 0 (task 39.2)",
    )
    p.add_argument(
        "--d-ff", type=int, default=None,
        help="feed-forward width; defaults to 4 x d_model, the transformer convention. "
             "Leaving it pinned while d_model grows makes the FFN a bottleneck and distorts "
             "any scaling comparison (docs/HF_JOBS.md)",
    )
    p.add_argument("--n-layers", type=int, default=6, help="row-attention layers")
    p.add_argument("--n-col-layers", type=int, default=2, help="column-attention layers")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--threads", type=int, default=None,
        help="cap torch CPU threads; set this on a shared machine (see CLAUDE.md)",
    )
    p.add_argument(
        "--checkpoint-every", type=int, default=0,
        help="save every N steps as well as at the end; 0 disables. Essential for any run "
             "long enough that losing it would hurt",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="runs/v0.pt")
    p.add_argument("--config", type=str, default=None, help="YAML overriding the defaults")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    model_cfg = ModelConfig(
        pooling=args.pooling,
        column_id_dim=args.column_id_dim,
        n_cell_blocks=args.n_cell_blocks,
        cell_labels=args.cell_labels,
        d_ff=args.d_ff if args.d_ff is not None else 4 * args.d_model,
        max_features=args.max_features,
        max_classes=args.max_classes,
        d_cell=args.d_cell,
        d_model=args.d_model,
        n_col_layers=args.n_col_layers,
        n_layers=args.n_layers,
        n_horizons=args.n_horizons,
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
        p_tree=(PriorConfig.p_tree if args.p_tree is None else args.p_tree),
        p_financial=(
            1.0
            if args.n_horizons
            else (PriorConfig.p_financial if args.p_financial is None else args.p_financial)
        ),
        n_horizons=args.n_horizons,
        p_trivial=args.p_trivial,
        p_crossed=args.p_crossed,
        p_regression=args.p_regression,
        identity_shuffle=args.identity_shuffle,
        # the default-rate envelope comes from configuration, because a prior that cannot
        # generate the regime being evaluated is the defect behind docs/FINDINGS.md §26
        sharpness_min=cfg.prior.sharpness_min,
        sharpness_max=cfg.prior.sharpness_max,
        min_expected_positives=cfg.prior.min_expected_positives,
        absolute_rate_floor=cfg.prior.absolute_rate_floor,
        rate_ceiling=cfg.prior.rate_ceiling,
        n_sectors_max=cfg.prior.n_sectors,
    )
    train_cfg = TrainConfig(
        steps=args.steps, batch_size=args.batch_size, lr=args.lr, device=args.device,
        seed=args.seed, checkpoint_every=args.checkpoint_every,
        feature_chunk=args.feature_chunk,
        run_steps=args.run_steps, resume=args.resume,
    )
    print(f"config: {cfg.provenance()}")
    train(model_cfg, prior_cfg, train_cfg, args.out)


if __name__ == "__main__":
    main()
