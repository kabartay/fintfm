"""Phase 1: does a financial prior beat a generic one at matched compute?

This is the experiment that decides whether the project's core bet is real
(``docs/STRATEGY.md``). Everything else — the architecture, the calibration work, the
positioning — is downstream of its answer.

**The design.** Train several models that differ in *one* variable, the pretraining task
distribution, holding architecture, parameter count, optimiser, step count, batch size, seed
and evaluation identical. Then score them all on the same held-out real credit panels.

    financial   p_financial = 1.0   only synthetic company financials
    mixed       p_financial = 0.7   the library default
    generic     p_financial = 0.0   only structural causal models, no finance at all

**The exit condition, stated before the run so it cannot be moved afterwards:** the
financial variant must beat the generic variant on real credit data, on calibration as well
as ranking. If it does not, domain-specific pretraining buys nothing here, and the honest
response is to publish that and drop to the validation layer, which needs no model of our
own. A null result is a result; it is cheaper to learn it now than after raising money on it.

**Results are written to JSON, not just printed**, with the config and git commit that
produced them, so any number in a document can be re-derived rather than recalled.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from fintfm.evaluation.datasets import load_polish_bankruptcy
from fintfm.evaluation.metrics import evaluate_binary, holm_bonferroni, paired_auc_difference
from fintfm.inference.classifier import ContextStrategy, FinancialTFMClassifier
from fintfm.modeling.model import FinancialTFM, ModelConfig
from fintfm.modeling.train import TrainConfig, train
from fintfm.prior import PriorConfig

#: The one variable under test. Everything else is held fixed across variants.
VARIANTS: dict[str, float] = {
    "financial": 1.0,
    "mixed": 0.7,
    "generic": 0.0,
}


@dataclass
class VariantResult:
    """Outcome for one pretraining variant.

    Attributes:
        name: Variant key from :data:`VARIANTS`.
        p_financial: Probability of drawing a financial task during pretraining.
        n_parameters: Model size, recorded to prove compute was matched.
        train_seconds: Wall-clock training time.
        final_loss: Mean training loss over the last logging window.
        metrics: Real-data scores keyed by ``"<dataset>/<context_strategy>"``.
    """

    name: str
    p_financial: float
    n_parameters: int
    train_seconds: float
    final_loss: float
    metrics: dict[str, dict] = field(default_factory=dict)


def _git_commit() -> str:
    """Return the current commit, or ``"unknown"`` outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def evaluate_on_credit(
    model_path: str,
    horizons: tuple[int, ...] = (1, 3, 5),
    strategies: tuple[ContextStrategy, ...] = ("uniform", "balanced"),
    seed: int = 0,
) -> dict[str, dict]:
    """Score one checkpoint on the real corporate-default panels.

    Args:
        model_path: Checkpoint to evaluate.
        horizons: Bankruptcy forecast horizons, in years.
        strategies: Context-construction strategies to try.
        seed: Split seed, identical across variants so the comparison is paired.

    Returns:
        Mapping of ``"<dataset>/<strategy>"`` to a :class:`CreditMetrics` as a dict. A panel
        wider than the model is skipped with a printed note rather than raising: a narrow
        model is a legitimate configuration, and one unusable panel must not discard the
        results for every other.
    """
    out: dict[str, dict] = {}
    cfg = FinancialTFM.load(model_path).cfg
    for horizon in horizons:
        ds = load_polish_bankruptcy(horizon)
        if ds.X.shape[1] > cfg.max_features:
            print(f"    skipping {ds.name}: needs max_features >= {ds.X.shape[1]}, "
                  f"model has {cfg.max_features}")
            continue
        X_train, X_test, y_train, y_test = train_test_split(
            ds.X, ds.y, test_size=0.3, random_state=seed, stratify=ds.y
        )
        for strategy in strategies:
            clf = FinancialTFMClassifier(model_path, context_strategy=strategy).fit(
                X_train, y_train
            )
            proba = clf.predict_proba(X_test)[:, 1]
            metrics = evaluate_binary(y_test, proba)
            cell = asdict(metrics)
            # keep predictions and labels so variants can be compared as PAIRED samples
            # afterwards; a win count across cells is not a result (docs/FINDINGS.md §9).
            cell["_y_true"] = y_test.tolist()
            cell["_proba"] = proba.tolist()
            out[f"{ds.name}/{strategy}"] = cell
    return out


def run_ablation(
    out_dir: Path,
    steps: int,
    batch_size: int,
    n_rows: int,
    model_cfg: ModelConfig,
    seed: int = 0,
    variants: dict[str, float] | None = None,
    threads: int | None = None,
    horizons: tuple[int, ...] = (1, 3, 5),
    device: str = "cpu",
) -> dict:
    """Train one model per variant at matched compute and score them identically.

    Args:
        out_dir: Directory for checkpoints and ``results.json``.
        steps: Gradient steps per variant. Identical across variants by construction.
        batch_size: Tasks per step.
        n_rows: Rows per synthetic task.
        model_cfg: Architecture, shared by every variant.
        seed: Seed for training and evaluation splits.
        variants: Override :data:`VARIANTS`.
        threads: Cap on torch CPU threads. Set this on a shared machine.
        horizons: Bankruptcy horizons to evaluate on. Pass ``()`` to train only.
        device: Training device. ``"mps"`` uses the Apple GPU, which is ~3.5x faster than
            CPU here *and* leaves the CPU cores to whatever else shares the machine — see
            ``docs/COMPUTE.md`` for measured step times.

    Returns:
        The results record that was written to ``out_dir / "results.json"``.
    """
    if threads is not None:
        torch.set_num_threads(threads)
    variants = variants or VARIANTS
    out_dir.mkdir(parents=True, exist_ok=True)

    record: dict = {
        "created": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(),
        "device": device,
        "config": {
            "steps": steps,
            "batch_size": batch_size,
            "n_rows": n_rows,
            "seed": seed,
            "model": asdict(model_cfg),
        },
        "variants": {},
    }

    for name, p_financial in variants.items():
        print(f"\n=== variant {name!r} (p_financial={p_financial}) ===")
        prior_cfg = PriorConfig(
            max_features=model_cfg.max_features,
            max_classes=model_cfg.max_classes,
            p_financial=p_financial,
            n_rows=n_rows,
        )
        train_cfg = TrainConfig(steps=steps, batch_size=batch_size, seed=seed, device=device)
        ckpt = out_dir / f"{name}.pt"
        started = time.time()
        model = train(model_cfg, prior_cfg, train_cfg, str(ckpt))
        elapsed = time.time() - started

        result = VariantResult(
            name=name,
            p_financial=p_financial,
            n_parameters=model.num_parameters(),
            train_seconds=elapsed,
            final_loss=float("nan"),
            metrics=evaluate_on_credit(str(ckpt), horizons=horizons, seed=seed),
        )
        record["variants"][name] = asdict(result)

    sizes = {v["n_parameters"] for v in record["variants"].values()}
    if len(sizes) > 1:
        raise RuntimeError(f"compute was not matched across variants: parameter counts {sizes}")

    (out_dir / "results.json").write_text(json.dumps(record, indent=2))
    print(f"\nwrote {out_dir / 'results.json'}")
    return record


def summarise(record: dict) -> str:
    """Render an ablation record as a comparison table.

    Args:
        record: Output of :func:`run_ablation`.

    Returns:
        A printable summary, including the verdict against the pre-stated exit condition.
    """
    variants = record["variants"]
    keys = sorted({k for v in variants.values() for k in v["metrics"]})
    lines = [
        (
            f"parameters: {next(iter(variants.values()))['n_parameters']:,} "
            f"(identical across variants) | steps: {record['config']['steps']}"
        ),
        "",
        (f"{'dataset/strategy':38s} " + " ".join(f"{n:>22s}" for n in variants)),
    ]
    for key in keys:
        cells = []
        for name in variants:
            m = variants[name]["metrics"].get(key)
            cells.append("".ljust(22) if m is None else f"AUC {m['roc_auc']:.4f} ECE {m['ece']:.4f}")
        lines.append(f"{key:38s} " + " ".join(f"{c:>22s}" for c in cells))

    if "financial" in variants and "generic" in variants:
        fin, gen = variants["financial"]["metrics"], variants["generic"]["metrics"]
        shared = [k for k in keys if k in fin and k in gen]
        if not shared:
            lines += [
                "",
                "No panel was scored, so the exit condition cannot be evaluated. Either every",
                "panel was skipped (model narrower than the data) or horizons was empty.",
            ]
            return "\n".join(lines)
        auc_wins = sum(fin[k]["roc_auc"] > gen[k]["roc_auc"] for k in shared)
        ece_wins = sum(fin[k]["ece"] < gen[k]["ece"] for k in shared)
        mean_fin = float(np.mean([fin[k]["roc_auc"] for k in shared]))
        mean_gen = float(np.mean([gen[k]["roc_auc"] for k in shared]))
        lines += [
            "",
            (
                f"financial vs generic: AUC better on {auc_wins}/{len(shared)}, "
                f"ECE better on {ece_wins}/{len(shared)} (win counts only, NOT a result)"
            ),
            (
                f"mean AUC: financial {mean_fin:.4f} vs generic {mean_gen:.4f} "
                f"(delta {mean_fin - mean_gen:+.4f})"
            ),
        ]

        # Paired bootstrap per cell, then family-wise correction. Without this a sweep over
        # panels and strategies produces a "winner" by chance roughly one time in twenty.
        rows, pvals = [], []
        for k in shared:
            y = fin[k].get("_y_true")
            if y is None or gen[k].get("_proba") is None:
                continue
            delta, (lo, hi), p = paired_auc_difference(
                np.asarray(y), np.asarray(fin[k]["_proba"]), np.asarray(gen[k]["_proba"])
            )
            rows.append((k, delta, lo, hi, p))
            pvals.append(p)
        if rows:
            survives = holm_bonferroni(pvals)
            lines += [
                "",
                (
                    "paired AUC difference (financial - generic), 95% bootstrap CI, "
                    "Holm-corrected across cells:"
                ),
            ]
            for (k, delta, lo, hi, p), ok in zip(rows, survives, strict=True):
                mark = "significant" if ok else "not significant"
                lines.append(
                    f"  {k:38s} {delta:+.4f}  [{lo:+.4f}, {hi:+.4f}]  p={p:.4f}  {mark}"
                )
            n_sig = sum(survives)
            n_sig_fav = sum(
                ok and delta > 0 for (_k, delta, _lo, _hi, _p), ok in zip(rows, survives, strict=True)
            )
            lines += [
                "",
                (
                    f"{n_sig}/{len(rows)} cells significant after correction; "
                    f"{n_sig_fav} favour the financial prior."
                ),
            ]

        lines += [
            "",
            "EXIT CONDITION (docs/STRATEGY.md Phase 1): the financial prior must beat the",
            "generic one on real credit data, and the difference must survive the paired test",
            "above with family-wise correction. Baesens et al. (arXiv:2605.18147) found only",
            "22 of 406 pairwise comparisons significant in this domain, so expect small",
            "effects and do not read a win count as a verdict. A null result says",
            "domain-specific pretraining buys nothing here and the thesis needs revising.",
        ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, default="runs/phase1")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-rows", type=int, default=256)
    p.add_argument("--max-features", type=int, default=64)
    p.add_argument("--d-cell", type=int, default=64)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-col-layers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="cap torch CPU threads; set this on a shared machine (see CLAUDE.md)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu, mps (Apple GPU, ~3.5x faster and spares the CPU cores), or cuda",
    )
    args = p.parse_args()

    model_cfg = ModelConfig(
        max_features=args.max_features,
        max_classes=2,
        d_cell=args.d_cell,
        d_model=args.d_model,
        n_col_layers=args.n_col_layers,
        n_layers=args.n_layers,
    )
    record = run_ablation(
        out_dir=Path(args.out),
        steps=args.steps,
        batch_size=args.batch_size,
        n_rows=args.n_rows,
        model_cfg=model_cfg,
        seed=args.seed,
        threads=args.threads,
        device=args.device,
    )
    print("\n" + summarise(record))


if __name__ == "__main__":
    main()
