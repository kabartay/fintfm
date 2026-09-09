"""How much does sharing a retrieved context with other queries cost?

``docs/FINDINGS.md`` §32 and §35 report retrieval as the largest accuracy gain in this
project, and both rest on an approximation that has never been measured. Per-query retrieval
is the correct operation — every firm gets the context of *its own* nearest neighbours — but it
costs one forward pass per firm, roughly 500× the work of chunked scoring at a 2,000-row
context, which is unusable on 47,378 firms. So queries are clustered and each **group** shares
one retrieved context.

That trades away a property the blind strategies keep: a prediction now depends on which other
queries were scored alongside it. The size of the error this introduces was stated as unknown
in ``fintfm.inference.retrieval``, and an unmeasured approximation under a headline result is
a liability rather than a caveat.

**What this measures.** Exact per-query retrieval is the reference. Grouped retrieval is then
run at a range of *group sizes* — not group counts, which are not comparable across
subsample sizes — and compared against that reference row by row:

- **mean and maximum absolute deviation** in cumulative PD, the direct error;
- **Spearman correlation** with the exact prediction, since ranking is what AUC consumes;
- **AUC**, to show whether the deviation actually costs anything measurable.

The subsample is **stratified** rather than random: at a 0.19% default rate a few thousand
random firms carry too few defaults to measure AUC at all, so every positive is kept and
negatives are sampled. That makes the AUC column comparable *between arms here* and not
comparable with the full-panel numbers in §35, which is stated in the output rather than left
for a reader to infer.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fintfm.config import Config, load_config
from fintfm.evaluation.datasets import load_v4finbench
from fintfm.evaluation.metrics import evaluate_binary
from fintfm.experiments.v4_out_of_time import _curve_truth, time_split
from fintfm.inference.classifier import FinancialTFMClassifier


@dataclass
class GroupingScore:
    """Agreement between grouped retrieval and the exact per-query reference."""

    group_size: int
    n_groups: int
    seconds: float
    mean_abs_deviation: float
    max_abs_deviation: float
    spearman: float
    mean_auc: float
    auc_by_horizon: list[float]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation between two flattened score arrays."""
    from scipy.stats import spearmanr

    r = spearmanr(a.ravel(), b.ravel()).statistic
    return float(r) if np.isfinite(r) else float("nan")


def _score(
    curve: np.ndarray, truth: np.ndarray, seen: np.ndarray, min_rows: int
) -> tuple[float, list[float]]:
    """Mean AUC across scoreable horizons, and the per-horizon values."""
    aucs: list[float] = []
    for k in range(curve.shape[1]):
        m = seen[:, k]
        if m.sum() < min_rows or len(np.unique(truth[m, k])) < 2:
            continue
        aucs.append(evaluate_binary(truth[m, k], curve[m, k]).roc_auc)
    return (float(np.mean(aucs)) if aucs else float("nan")), aucs


def run(
    model_path: str,
    out_dir: Path,
    n_negatives: int = 2000,
    n_positives: int = 0,
    group_sizes: tuple[int, ...] = (2, 8, 32, 128, 512),
    seed: int = 0,
    cfg: Config | None = None,
) -> dict:
    """Measure grouped retrieval against exact per-query retrieval.

    Args:
        model_path: Checkpoint carrying a hazard head.
        out_dir: Directory for ``retrieval_grouping.json``.
        n_negatives: Non-defaulting firms sampled into the subsample.
        n_positives: Defaulting firms sampled in; ``0`` keeps every one of them. Capping
            them is what makes a complete run affordable: the exact reference costs
            **1.05 s per query** (measured), so the subsample size sets the wall clock almost
            entirely. The primary metric here is per-row deviation from the reference, which
            does not need positives at all; they are kept only so the secondary AUC column
            means something between arms.
        group_sizes: Queries per shared context to test. Group *size* rather than group
            count, so the numbers mean the same thing at any subsample size.
        seed: Seed for the negative sample and for context selection.
        cfg: Configuration; loaded from the packaged default when omitted.

    Returns:
        The recorded result dictionary.
    """
    cfg = cfg or load_config()
    ds = load_v4finbench(max_rows=cfg.v4finbench.max_rows)
    tr, te = time_split(ds, cfg.v4finbench.train_until, cfg.v4finbench.test_from)
    rng = np.random.default_rng(seed)

    # keep every positive, sample the negatives: a random draw at this base rate carries too
    # few defaults to score, and an unscoreable reference measures nothing
    pos = te[ds.y[te] == 1]
    neg = te[ds.y[te] == 0]
    if n_positives and n_positives < pos.size:
        pos = rng.choice(pos, size=n_positives, replace=False)
    keep = np.sort(
        np.concatenate([pos, rng.choice(neg, size=min(n_negatives, neg.size), replace=False)])
    )
    truth, seen = _curve_truth(ds, keep)
    print(
        f"{ds.name}: subsample {len(keep):,} test firms "
        f"({len(pos):,} defaulting, {len(keep) - len(pos):,} not) against a "
        f"{len(tr):,}-row retrieval pool"
    )

    def fitted(groups: int) -> FinancialTFMClassifier:
        return FinancialTFMClassifier(
            model_path,
            max_context=cfg.v4finbench.max_context,
            context_strategy="retrieval",
            correct_prior=True,
            random_state=seed,
            retrieval_groups=groups,
            retrieval_min_positive=cfg.inference.retrieval_min_positive,
            feature_transform=cfg.inference.feature_transform,
        ).fit(ds.X[tr], ds.y[tr])

    started = time.perf_counter()
    reference = fitted(0).predict_term_structure(ds.X[keep])  # groups=0 -> one query each
    ref_seconds = time.perf_counter() - started
    ref_auc, ref_by_h = _score(
        reference, truth, seen, cfg.evaluation.min_rows_per_horizon
    )
    print(
        f"  exact per-query reference: {ref_seconds:.0f}s for {len(keep):,} queries "
        f"({ref_seconds / max(len(keep), 1):.3f} s/query), mean AUC {ref_auc:.4f}"
    )

    rows: list[GroupingScore] = []
    for size in group_sizes:
        n_groups = max(1, int(np.ceil(len(keep) / size)))
        started = time.perf_counter()
        curve = fitted(n_groups).predict_term_structure(ds.X[keep])
        elapsed = time.perf_counter() - started
        dev = np.abs(curve - reference)
        auc, by_h = _score(curve, truth, seen, cfg.evaluation.min_rows_per_horizon)
        row = GroupingScore(
            group_size=size,
            n_groups=n_groups,
            seconds=float(elapsed),
            mean_abs_deviation=float(dev.mean()),
            max_abs_deviation=float(dev.max()),
            spearman=_spearman(curve, reference),
            mean_auc=auc,
            auc_by_horizon=[float(a) for a in by_h],
        )
        rows.append(row)
        print(
            f"  ~{size:>4} queries/context ({n_groups:>4} groups): {elapsed:>5.0f}s  "
            f"mean|dev| {row.mean_abs_deviation:.5f}  max {row.max_abs_deviation:.4f}  "
            f"rho {row.spearman:.5f}  AUC {auc:.4f} ({auc - ref_auc:+.4f})",
            flush=True,
        )

    record = {
        "dataset": ds.name,
        "attribution": ds.attribution,
        "config": {
            "model": model_path,
            "n_test_subsample": len(keep),
            "n_positive": len(pos),
            "pool_rows": len(tr),
            "max_context": cfg.v4finbench.max_context,
            "feature_transform": cfg.inference.feature_transform,
            "seed": seed,
            "config_sources": list(cfg.sources),
        },
        "reference": {
            "mode": "exact per-query retrieval",
            "seconds": float(ref_seconds),
            "mean_auc": ref_auc,
            "auc_by_horizon": [float(a) for a in ref_by_h],
        },
        "grouped": [asdict(r) for r in rows],
        "note": (
            "AUC here is on a stratified subsample (all positives, sampled negatives) so it "
            "is comparable between these arms and NOT with the full-panel numbers in "
            "docs/FINDINGS.md §35."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrieval_grouping.json").write_text(json.dumps(record, indent=2))
    return record


def summarise(record: dict) -> str:
    """Render the approximation error against the exact reference."""
    r = record["reference"]
    header = (
        f"{'queries/ctx':>12} {'groups':>7} {'secs':>6} {'mean|dev|':>10} {'max|dev|':>9} "
        f"{'rho':>8} {'meanAUC':>8} {'vs exact':>9}"
    )
    lines = [
        f"exact per-query reference: {r['seconds']:.0f}s, mean AUC {r['mean_auc']:.4f}",
        "",
        header,
    ]
    for g in record["grouped"]:
        lines.append(
            f"{g['group_size']:>12} {g['n_groups']:>7} {g['seconds']:>6.0f} "
            f"{g['mean_abs_deviation']:>10.5f} {g['max_abs_deviation']:>9.4f} "
            f"{g['spearman']:>8.5f} {g['mean_auc']:>8.4f} "
            f"{g['mean_auc'] - r['mean_auc']:>+9.4f}"
        )
    lines += ["", record["note"]]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=str, default="runs/retrieval-grouping")
    p.add_argument("--config", type=str, default=None, help="YAML overriding the defaults")
    p.add_argument("--n-negatives", type=int, default=2000)
    p.add_argument(
        "--n-positives", type=int, default=0, help="0 keeps every defaulting firm"
    )
    p.add_argument(
        "--group-sizes", type=str, default="2,8,32,128,512", help="comma-separated"
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    cfg = load_config(args.config)
    record = run(
        args.model,
        Path(args.out),
        n_negatives=args.n_negatives,
        n_positives=args.n_positives,
        group_sizes=tuple(int(v) for v in args.group_sizes.split(",")),
        seed=args.seed,
        cfg=cfg,
    )
    print("\n" + summarise(record))
    print(f"\nconfig: {cfg.provenance()}")
    print(f"\n{record['attribution']}")


if __name__ == "__main__":
    main()
