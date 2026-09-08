"""Does fitting the whole curve also help discrimination, or only coherence?

`docs/FINDINGS.md` §20 established that a hazard parameterisation makes the PD term structure
monotone **by construction**, fixing the 39% incoherence measured in §11. That is a coherence
result and it says nothing about accuracy.

This experiment asks the remaining question. Two ways to produce a term structure over ``K``
horizons:

**joint** — one model with a hazard head, trained on the discrete-time survival likelihood.
All horizons share parameters and the loss fits the whole curve at once.

**per-horizon** — ``K`` independent models, each trained on the binary label "defaulted by
horizon k". This is what the field does, and what §11 measured as incoherent.

Compute is matched two ways, because the fair comparison is genuinely ambiguous and reporting
only one would flatter a conclusion:

- **matched total** — the per-horizon arm gets ``steps/K`` per model, so both arms consume the
  same gradient steps overall. This is the practically relevant budget and it favours joint.
- **matched per-model** — the per-horizon arm gets the full ``steps`` per model, so it
  consumes ``K×`` the compute. This is the strongest form of the baseline.

Evaluation is on held-out synthetic tasks, since the term structure cannot be scored on the
real panels at all: they carry no firm identifiers (§7), so there is no per-firm hazard path
to compare against. That needs V4FinBench.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from fintfm.modeling.model import FinancialTFM, ModelConfig
from fintfm.prior import PriorConfig
from fintfm.prior.mixture import sample_batch


@dataclass
class ArmResult:
    """Outcome for one way of producing a term structure.

    Attributes:
        name: Arm label.
        total_steps: Gradient steps consumed across all models in the arm.
        n_models: Number of models trained (1 for joint, K for per-horizon).
        auc_by_horizon: Held-out AUC of cumulative PD at each horizon.
        mean_auc: Mean of the above.
        violation_rate: Fraction of (row, step) pairs where cumulative PD decreases.
        fully_monotone: Fraction of rows whose whole curve is non-decreasing.
    """

    name: str
    total_steps: int
    n_models: int
    auc_by_horizon: list[float]
    mean_auc: float
    violation_rate: float
    fully_monotone: float


def _cfg(max_features: int, n_horizons: int | None) -> ModelConfig:
    return ModelConfig(
        max_features=max_features, max_classes=2, d_cell=24, d_model=48, n_heads=4,
        n_col_layers=1, n_layers=3, d_ff=96, n_horizons=n_horizons,
    )


def _eval_batches(
    rng: np.random.Generator, prior: PriorConfig, n_batches: int, batch_size: int
) -> list:
    """Held-out evaluation batches, drawn once so every arm sees identical data."""
    return [sample_batch(rng, prior, batch_size) for _ in range(n_batches)]


def _auc_safe(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def train_joint(
    prior: PriorConfig, steps: int, batch_size: int, device: str, seed: int
) -> FinancialTFM:
    """Train one hazard-head model on the survival likelihood."""
    torch.manual_seed(seed)
    model = FinancialTFM(_cfg(prior.max_features, prior.n_horizons)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        b = sample_batch(rng, prior, batch_size).to(device)
        loss = model.survival_loss(b.X, b.y, b.period, b.n_ctx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval()


def train_per_horizon(
    prior: PriorConfig, steps_each: int, batch_size: int, device: str, seed: int
) -> list[FinancialTFM]:
    """Train K independent binary models, one per cumulative horizon.

    Each model sees the label "defaulted by horizon k", derived from the same sampled
    periods, so the arms differ only in how the objective is factorised.
    """
    models = []
    for k in range(prior.n_horizons or 1):
        torch.manual_seed(seed + k)
        model = FinancialTFM(_cfg(prior.max_features, None)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        rng = np.random.default_rng(seed + 1000 * k)
        for _ in range(steps_each):
            b = sample_batch(rng, prior, batch_size).to(device)
            by_k = ((b.period >= 0) & (b.period <= k)).long()
            loss = model.loss(b.X, by_k, b.n_ctx, b.n_classes)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        models.append(model.eval())
    return models


@torch.no_grad()
def evaluate(
    arm: str, models: FinancialTFM | list[FinancialTFM], batches: list, device: str,
    n_horizons: int, total_steps: int,
) -> ArmResult:
    """Score an arm's cumulative-PD curves on the shared held-out batches."""
    curves, truths = [], []
    for b in batches:
        bb = b.to(device)
        if isinstance(models, list):
            cols = [
                torch.softmax(m(bb.X, ((bb.period >= 0) & (bb.period <= k)).long(),
                                bb.n_ctx, bb.n_classes)[:, bb.n_ctx:, :2], dim=-1)[..., 1]
                for k, m in enumerate(models)
            ]
            pd_curve = torch.stack(cols, dim=-1)
        else:
            pd_curve = models.term_structure(bb.X, bb.y, bb.n_ctx)
        curves.append(pd_curve.reshape(-1, n_horizons).cpu().numpy())
        p = bb.period[:, bb.n_ctx:].reshape(-1).cpu().numpy()
        truths.append(np.stack([((p >= 0) & (p <= k)).astype(int) for k in range(n_horizons)], 1))

    C = np.concatenate(curves)
    T = np.concatenate(truths)
    aucs = [_auc_safe(T[:, k], C[:, k]) for k in range(n_horizons)]
    d = np.diff(C, axis=1)
    return ArmResult(
        name=arm,
        total_steps=total_steps,
        n_models=len(models) if isinstance(models, list) else 1,
        auc_by_horizon=[float(a) for a in aucs],
        mean_auc=float(np.nanmean(aucs)),
        violation_rate=float((d < 0).mean()),
        fully_monotone=float((d >= 0).all(axis=1).mean()),
    )


def run(
    out_dir: Path, steps: int = 1500, batch_size: int = 16, n_rows: int = 128,
    max_features: int = 32, n_horizons: int = 5, device: str = "cpu", seed: int = 0,
) -> dict:
    """Run both arms under both compute conventions and write the comparison."""
    prior = PriorConfig(
        max_features=max_features, max_classes=2, p_financial=1.0,
        n_rows=n_rows, n_horizons=n_horizons,
    )
    # one shared held-out set, drawn from a seed no arm trains on
    batches = _eval_batches(np.random.default_rng(9_999), prior, n_batches=6, batch_size=batch_size)

    results: list[ArmResult] = []
    print(f"joint: 1 model x {steps} steps")
    joint = train_joint(prior, steps, batch_size, device, seed)
    results.append(evaluate("joint", joint, batches, device, n_horizons, steps))

    each = max(1, steps // n_horizons)
    print(f"per-horizon (matched total): {n_horizons} models x {each} steps")
    ph_total = train_per_horizon(prior, each, batch_size, device, seed)
    results.append(
        evaluate("per_horizon_matched_total", ph_total, batches, device, n_horizons, each * n_horizons)
    )

    print(f"per-horizon (matched per-model): {n_horizons} models x {steps} steps")
    ph_each = train_per_horizon(prior, steps, batch_size, device, seed)
    results.append(
        evaluate("per_horizon_matched_per_model", ph_each, batches, device, n_horizons, steps * n_horizons)
    )

    record = {
        "config": {
            "steps": steps, "batch_size": batch_size, "n_rows": n_rows,
            "max_features": max_features, "n_horizons": n_horizons, "seed": seed,
        },
        "arms": [asdict(r) for r in results],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "term_structure.json").write_text(json.dumps(record, indent=2))
    return record


def summarise(record: dict) -> str:
    """Render the comparison, coherence first because that is the settled result."""
    lines = [
        (
            f"{'arm':>30} {'models':>7} {'steps':>7} {'mean AUC':>9} "
            f"{'violations':>11} {'monotone':>9}"
        )
    ]
    for a in record["arms"]:
        lines.append(
            f"{a['name']:>30} {a['n_models']:>7} {a['total_steps']:>7} "
            f"{a['mean_auc']:>9.4f} {a['violation_rate']:>11.2%} {a['fully_monotone']:>9.1%}"
        )
    by = {a["name"]: a for a in record["arms"]}
    j = by["joint"]
    lines.append("")
    lines.append("AUC by horizon:")
    for a in record["arms"]:
        lines.append(f"  {a['name']:>30} " + " ".join(f"{x:.4f}" for x in a["auc_by_horizon"]))
    for other in ("per_horizon_matched_total", "per_horizon_matched_per_model"):
        if other in by:
            d = j["mean_auc"] - by[other]["mean_auc"]
            lines.append(f"\njoint - {other}: {d:+.4f} mean AUC")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, default="runs/term-structure")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-rows", type=int, default=128)
    p.add_argument("--max-features", type=int, default=32)
    p.add_argument("--n-horizons", type=int, default=5)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    record = run(
        Path(args.out), steps=args.steps, batch_size=args.batch_size, n_rows=args.n_rows,
        max_features=args.max_features, n_horizons=args.n_horizons, device=args.device,
        seed=args.seed,
    )
    print("\n" + summarise(record))


if __name__ == "__main__":
    main()
