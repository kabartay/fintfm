"""Does fitting the whole curve also help discrimination, or only coherence?

`docs/results/FINDINGS.md` §20 established that a hazard parameterisation makes the PD term structure
monotone **by construction**, fixing the incoherence of per-horizon models
(98.7% of firms out of time on V4FinBench, §229). That is a coherence
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

from fintfm.evaluation.metrics import holm_adjusted_p, paired_auc_difference
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
    """A small matched configuration, so the two arms differ only in the hazard head.

    Args:
        max_features: Feature cap for both arms.
        n_horizons: Horizon grid length for the survival arm; ``None`` for the binary one.

    Returns:
        The configuration.
    """
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
    """ROC-AUC, or NaN when the split has one class and the metric is undefined.

    Args:
        y: ``(n,)`` labels.
        p: ``(n,)`` scores.

    Returns:
        ROC-AUC, or NaN. NaN rather than 0.0 so a degenerate split is visibly unscorable
        instead of silently contributing a worst-case value to a mean.
    """
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
def collect_curves(
    models: FinancialTFM | list[FinancialTFM], batches: list, device: str, n_horizons: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative-PD curves, truths and task labels on the shared held-out batches.

    Returns:
        ``(C, T, tasks)``: ``(n, K)`` predicted curves, ``(n, K)`` cumulative-default labels,
        and ``(n,)`` integer task ids. Rows of one synthetic task share a context and a
        generating process, so the paired bootstrap resamples tasks, not rows.
    """
    curves, truths, tasks = [], [], []
    offset = 0
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
            pd_curve = models.term_structure(bb.X, bb.y, bb.n_ctx, period=bb.period)
        curves.append(pd_curve.reshape(-1, n_horizons).cpu().numpy())
        per = bb.period[:, bb.n_ctx:]
        p = per.reshape(-1).cpu().numpy()
        truths.append(np.stack([((p >= 0) & (p <= k)).astype(int) for k in range(n_horizons)], 1))
        tasks.append(np.repeat(np.arange(per.shape[0]) + offset, per.shape[1]))
        offset += per.shape[0]
    return np.concatenate(curves), np.concatenate(truths), np.concatenate(tasks)


def summarise_curves(
    arm: str, C: np.ndarray, T: np.ndarray, n_models: int, total_steps: int
) -> ArmResult:
    """Score one arm's curves: AUC per horizon and the two coherence rates."""
    n_horizons = C.shape[1]
    aucs = [_auc_safe(T[:, k], C[:, k]) for k in range(n_horizons)]
    d = np.diff(C, axis=1)
    return ArmResult(
        name=arm,
        total_steps=total_steps,
        n_models=n_models,
        auc_by_horizon=[float(a) for a in aucs],
        mean_auc=float(np.nanmean(aucs)),
        violation_rate=float((d < 0).mean()),
        fully_monotone=float((d >= 0).all(axis=1).mean()),
    )


def evaluate(
    arm: str, models: FinancialTFM | list[FinancialTFM], batches: list, device: str,
    n_horizons: int, total_steps: int,
) -> ArmResult:
    """Score an arm's cumulative-PD curves on the shared held-out batches."""
    C, T, _ = collect_curves(models, batches, device, n_horizons)
    return summarise_curves(arm, C, T, len(models) if isinstance(models, list) else 1, total_steps)


def paired_arm_comparisons(
    curves: dict[str, np.ndarray],
    truth: np.ndarray,
    tasks: np.ndarray | None,
    reference: str = "joint",
    n_boot: int = 1000,
    seed: int = 0,
) -> list[dict]:
    """Holm-corrected paired bootstrap of the reference arm against every other arm (11.8).

    Uses the shared :func:`fintfm.evaluation.metrics.paired_auc_difference` per horizon (whole
    tasks resampled when ``tasks`` is given) and adjusts the whole family, every other arm at
    every horizon, together with :func:`fintfm.evaluation.metrics.holm_adjusted_p`.

    Args:
        curves: Arm name -> ``(n, K)`` predicted curve on the same rows.
        truth: ``(n, K)`` cumulative-default labels.
        tasks: Task id per row, or ``None`` for a row bootstrap.
        reference: Arm compared against all the others.
        n_boot: Bootstrap resamples per comparison.
        seed: Bootstrap seed.

    Returns:
        One dict per (other arm, horizon): ``delta`` (reference minus other), ``ci_lo``,
        ``ci_hi``, ``p_raw``, ``p_holm`` and ``significant`` (``p_holm < 0.05``).
    """
    rows: list[dict] = []
    for other in sorted(a for a in curves if a != reference):
        for k in range(truth.shape[1]):
            d, (lo, hi), p = paired_auc_difference(
                truth[:, k], curves[reference][:, k], curves[other][:, k],
                n_boot=n_boot, seed=seed, groups=tasks,
            )
            rows.append({"reference": reference, "other": other, "horizon": k, "delta": d,
                         "ci_lo": lo, "ci_hi": hi, "p_raw": p})
    for r, adj in zip(rows, holm_adjusted_p([r["p_raw"] for r in rows]), strict=True):
        r["p_holm"] = adj
        r["significant"] = bool(adj < 0.05)
    return rows


def run(
    out_dir: Path, steps: int = 1500, batch_size: int = 16, n_rows: int = 128,
    max_features: int = 32, n_horizons: int = 5, device: str = "cpu", seed: int = 0,
    n_boot: int = 1000,
) -> dict:
    """Run both arms under both compute conventions and write the comparison."""
    prior = PriorConfig(
        max_features=max_features, max_classes=2, p_financial=1.0,
        n_rows=n_rows, n_horizons=n_horizons,
    )
    # one shared held-out set, drawn from a seed no arm trains on
    batches = _eval_batches(np.random.default_rng(9_999), prior, n_batches=6, batch_size=batch_size)

    results: list[ArmResult] = []
    curves: dict[str, np.ndarray] = {}
    truth = tasks = None

    def score(name: str, models, total: int) -> None:
        nonlocal truth, tasks
        C, truth, tasks = collect_curves(models, batches, device, n_horizons)
        curves[name] = C
        n_models = len(models) if isinstance(models, list) else 1
        results.append(summarise_curves(name, C, truth, n_models, total))

    print(f"joint: 1 model x {steps} steps")
    joint = train_joint(prior, steps, batch_size, device, seed)
    score("joint", joint, steps)

    each = max(1, steps // n_horizons)
    print(f"per-horizon (matched total): {n_horizons} models x {each} steps")
    ph_total = train_per_horizon(prior, each, batch_size, device, seed)
    score("per_horizon_matched_total", ph_total, each * n_horizons)

    print(f"per-horizon (matched per-model): {n_horizons} models x {steps} steps")
    ph_each = train_per_horizon(prior, steps, batch_size, device, seed)
    score("per_horizon_matched_per_model", ph_each, steps * n_horizons)

    record = {
        "config": {
            "steps": steps, "batch_size": batch_size, "n_rows": n_rows,
            "max_features": max_features, "n_horizons": n_horizons, "seed": seed,
        },
        "arms": [asdict(r) for r in results],
        "comparisons": (
            paired_arm_comparisons(curves, truth, tasks, n_boot=n_boot) if n_boot > 0 else []
        ),
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
    cmps = record.get("comparisons") or []
    if cmps:
        lines.append("\npaired bootstrap, tasks resampled, Holm-adjusted across all cells:")
        lines.append(f"{'vs':>30} {'h':>3} {'dAUC':>9} {'95% CI':>20} {'p raw':>8} {'p Holm':>8}")
        for c in cmps:
            lines.append(
                f"{c['other']:>30} {c['horizon']:>3} {c['delta']:>+9.4f} "
                f"[{c['ci_lo']:>+8.4f},{c['ci_hi']:>+8.4f}] {c['p_raw']:>8.4f} {c['p_holm']:>8.4f}"
            )
    return "\n".join(lines)


def main() -> None:
    """Measure PD term-structure coherence against a per-horizon baseline.

    Entry point for the ``term-structure`` console script; see
    ``--help`` for the flags. Writes its record as JSON under ``--out``.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, default="runs/term-structure")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-rows", type=int, default=128)
    p.add_argument("--max-features", type=int, default=32)
    p.add_argument("--n-horizons", type=int, default=5)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=1000,
                   help="bootstrap resamples per comparison; 0 skips the Holm table")
    args = p.parse_args()
    record = run(
        Path(args.out), steps=args.steps, batch_size=args.batch_size, n_rows=args.n_rows,
        max_features=args.max_features, n_horizons=args.n_horizons, device=args.device,
        seed=args.seed, n_boot=args.n_boot,
    )
    print("\n" + summarise(record))


if __name__ == "__main__":
    main()
