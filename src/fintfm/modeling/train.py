"""Pretraining loop: sample a fresh synthetic batch every step, no epochs.

Usage:
    fintfm-train --steps 20000 --batch-size 64 --out runs/v0.pt
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import torch

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
    seed: int = 0
    device: str = "cpu"


def _lr_schedule(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return 0.5 * (1 + np.cos(np.pi * progress))


@torch.no_grad()
def _eval_accuracy(model: FinancialTFM, prior_cfg: PriorConfig, rng: np.random.Generator, n_batches: int = 5) -> float:
    model.eval()
    correct, total = 0, 0
    for _ in range(n_batches):
        batch = sample_batch(rng, prior_cfg, batch_size=16)
        logits = model(batch.X, batch.y, batch.n_ctx, batch.n_classes)[:, batch.n_ctx :]
        pred = logits.argmax(dim=-1)
        target = batch.y[:, batch.n_ctx :]
        correct += (pred == target).sum().item()
        total += target.numel()
    model.train()
    return correct / total


def train(model_cfg: ModelConfig, prior_cfg: PriorConfig, train_cfg: TrainConfig, out_path: str) -> FinancialTFM:
    """Run pretraining and save the final checkpoint to ``out_path``."""
    rng = np.random.default_rng(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)
    model = FinancialTFM(model_cfg).to(train_cfg.device)
    print(f"model parameters: {model.num_parameters():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr_schedule(s, train_cfg))
    t0 = time.time()
    running = 0.0
    for step in range(train_cfg.steps):
        batch = sample_batch(rng, prior_cfg, train_cfg.batch_size).to(train_cfg.device)
        loss = model.loss(batch.X, batch.y, batch.n_ctx, batch.n_classes)
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
        if (step + 1) % train_cfg.eval_every == 0:
            acc = _eval_accuracy(model, prior_cfg, rng)
            print(f"  held-out synthetic query accuracy: {acc:.3f}")
    model.save(out_path)
    print(f"saved checkpoint to {out_path}")
    return model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-rows", type=int, default=256)
    p.add_argument("--max-features", type=int, default=24)
    p.add_argument("--max-classes", type=int, default=10)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--d-cell", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=6, help="row-attention layers")
    p.add_argument("--n-col-layers", type=int, default=2, help="column-attention layers")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--threads", type=int, default=None,
        help="cap torch CPU threads; set this on a shared machine (see CLAUDE.md)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="runs/v0.pt")
    args = p.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    model_cfg = ModelConfig(
        max_features=args.max_features,
        max_classes=args.max_classes,
        d_cell=args.d_cell,
        d_model=args.d_model,
        n_col_layers=args.n_col_layers,
        n_layers=args.n_layers,
    )
    prior_cfg = PriorConfig(max_features=args.max_features, max_classes=args.max_classes, n_rows=args.n_rows)
    train_cfg = TrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr, device=args.device, seed=args.seed)
    train(model_cfg, prior_cfg, train_cfg, args.out)


if __name__ == "__main__":
    main()
