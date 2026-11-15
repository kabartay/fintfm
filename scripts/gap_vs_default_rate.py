"""Gap between context strategies against the default rate of the slice, from stored outputs.

Task 32.4 of ``openspec/changes/revisit-context-strategy`` asks whether the gap between context
strategies shrinks as the portfolio default rate rises. This script reads outputs that already
exist (no inference) and reports, for each slice, the default rate and the strategy gap:

* ``folds``: the five V4FinBench published-protocol folds, comparing the stored arms
  ``runs/lrsweep/eval-published`` (uniform, 2,000 rows), ``runs/4824/maj4000`` (every positive
  plus a uniform draw of negatives, 4,000 rows) and ``runs/au42/prototype``. The slice rate is
  the fold's ``y_true`` mean; the metrics are average precision and ROC AUC of ``pred_fintfm``.
* ``horizons``: the context sweep ``runs/context-sweep-3seed-emb/context_sweep.json``, uniform
  minus balanced mean AUC per scored horizon, against that horizon's observed default rate.

Both are slices of one portfolio. Neither can fit the relationship across portfolios, which is
what the task's mechanism needs; the output states the slice count and the span of rates so the
caller can see that.

* ``--binary-sweep``: the output of ``fintfm-binctx``
  (``runs/binary-context-sweep/binary_context_sweep.json``), one row per (panel, context size)
  with the seed-mean strategy gap in AP and AUC against the slice default rate. This spans
  several panels, so it is the input the task's cross-portfolio relationship needs; the
  Spearman correlation is printed with the number of (panel, size) rows behind it.

Usage:
    uv run python scripts/gap_vs_default_rate.py --runs /path/to/runs
    uv run python scripts/gap_vs_default_rate.py --binary-sweep path/to/binary_context_sweep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ARMS = {
    "default": "lrsweep/eval-published",
    "maj4000": "4824/maj4000",
    "prototype": "au42/prototype",
}


def fold_table(runs: Path) -> list[dict[str, float]]:
    """Per-fold default rate and arm metrics for the stored V4FinBench arms.

    Args:
        runs: The ``runs`` directory.

    Returns:
        One row per fold with ``rate``, and ``ap_<arm>`` / ``auc_<arm>`` for each arm in ``ARMS``.

    Raises:
        ValueError: If an arm's ``y_true`` differs from the default arm's on a fold.
    """
    rows = []
    for k in range(5):
        data = {a: np.load(runs / d / f"predictions_fold{k}.npz") for a, d in ARMS.items()}
        y = data["default"]["y_true"]
        row: dict[str, float] = {"fold": k, "n": len(y), "positives": int(y.sum())}
        row["rate"] = float(y.mean())
        for arm, d in data.items():
            if not np.array_equal(d["y_true"], y):
                raise ValueError(f"fold {k}: {arm} scores different rows")
            p = d["pred_fintfm"]
            row[f"ap_{arm}"] = float(average_precision_score(y, p))
            row[f"auc_{arm}"] = float(roc_auc_score(y, p))
        rows.append(row)
    return rows


def horizon_table(path: Path) -> list[dict[str, float]]:
    """Uniform minus balanced mean AUC per scored horizon against the observed rate.

    Args:
        path: A ``context_sweep.json`` from ``fintfm-ctxsweep``.

    Returns:
        One row per (max_context, horizon) with the seed-mean gap and the observed rate.
    """
    cells = json.loads(path.read_text())["cells"]
    out = []
    for size in sorted({c["max_context"] for c in cells}):
        for h in range(max(c["n_horizons_scored"] for c in cells)):
            gaps, rates = [], []
            for seed in sorted({c["seed"] for c in cells}):
                pick = {
                    c["strategy"]: c
                    for c in cells
                    if c["max_context"] == size and c["seed"] == seed
                }
                if "uniform" in pick and "balanced" in pick:
                    u, b = pick["uniform"], pick["balanced"]
                    gaps.append(u["auc_by_horizon"][h] - b["auc_by_horizon"][h])
                    rates.append(u["observed_by_horizon"][h])
            if gaps:
                out.append(
                    {
                        "size": size,
                        "horizon": h,
                        "rate": float(np.mean(rates)),
                        "gap": float(np.mean(gaps)),
                        "seeds": len(gaps),
                    }
                )
    return out


def binary_table(path: Path, a: str = "uniform", b: str = "balanced") -> list[dict[str, float]]:
    """Strategy gap against slice default rate from a ``fintfm-binctx`` record.

    Args:
        path: A ``binary_context_sweep.json``.
        a: Strategy whose score is the minuend.
        b: Strategy subtracted.

    Returns:
        One row per (panel, context size) holding both strategies in at least one seed:
        ``panel``, ``size``, ``rate`` (seed-mean slice default rate), ``gap_ap`` and
        ``gap_auc`` (seed-mean ``a - b``, paired by seed) and ``seeds``.
    """
    cells = json.loads(path.read_text())["cells"]
    by = {(c["panel"], c["max_context"], c["seed"], c["strategy"]): c for c in cells}
    out: list[dict[str, float]] = []
    for panel, size in sorted({(c["panel"], c["max_context"]) for c in cells}):
        pairs = [
            (by[(panel, size, sd, a)], by[(panel, size, sd, b)])
            for sd in sorted({c["seed"] for c in cells})
            if (panel, size, sd, a) in by and (panel, size, sd, b) in by
        ]
        if pairs:
            out.append(
                {
                    "panel": panel,
                    "size": size,
                    "rate": float(np.mean([x["slice_default_rate"] for x, _ in pairs])),
                    "gap_ap": float(np.mean([x["ap"] - y["ap"] for x, y in pairs])),
                    "gap_auc": float(np.mean([x["auc"] - y["auc"] for x, y in pairs])),
                    "seeds": len(pairs),
                }
            )
    return out


def print_binary(path: Path, a: str, b: str) -> None:
    """Print the binary-sweep table and Spearman correlations of gap with rate."""
    rows = binary_table(path, a, b)
    print(f"BINARY SWEEP ({path.name}): {a} - {b}")
    print("panel max_context rate% seeds gapAP gapAUC")
    for r in rows:
        print(
            f"{r['panel']} {r['size']} {100 * r['rate']:.3f} {r['seeds']} "
            f"{r['gap_ap']:+.4f} {r['gap_auc']:+.4f}"
        )
    if not rows:
        print("no (panel, size) holds both strategies")
        return
    rates = [r["rate"] for r in rows]
    print(f"rate span {100 * min(rates):.3f}% to {100 * max(rates):.3f}%")
    for size in [None, *sorted({r["size"] for r in rows})]:
        sub = [r for r in rows if size is None or r["size"] == size]
        label = "all sizes" if size is None else f"max_context {size}"
        if len(sub) < 3:
            print(f"{label}: n={len(sub)}, too few rows for a rank correlation")
            continue
        for m in ("ap", "auc"):
            rho = spearmanr([r["rate"] for r in sub], [r[f"gap_{m}"] for r in sub])
            print(f"{label}: Spearman {m} gap vs rate {rho.statistic:.2f} (n={len(sub)})")


def main() -> None:
    """Print both tables and the rank correlation of gap with rate."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=None)
    ap.add_argument("--binary-sweep", type=Path, default=None)
    ap.add_argument("--gap-a", default="uniform")
    ap.add_argument("--gap-b", default="balanced")
    args = ap.parse_args()
    if args.binary_sweep is not None:
        print_binary(args.binary_sweep, args.gap_a, args.gap_b)
        if args.runs is None:
            return
        print()
    if args.runs is None:
        ap.error("give --runs, --binary-sweep, or both")

    rows = fold_table(args.runs)
    print("FOLDS (V4FinBench published protocol)")
    print("fold n positives rate% | AP default/maj4000/prototype | AUC default/maj4000/prototype")
    for r in rows:
        print(
            f"{r['fold']} {r['n']} {r['positives']} {100 * r['rate']:.4f} | "
            f"{r['ap_default']:.4f}/{r['ap_maj4000']:.4f}/{r['ap_prototype']:.4f} | "
            f"{r['auc_default']:.4f}/{r['auc_maj4000']:.4f}/{r['auc_prototype']:.4f}"
        )
    rates = [r["rate"] for r in rows]
    print(f"rate span {100 * min(rates):.4f}% to {100 * max(rates):.4f}%")
    for metric in ("ap", "auc"):
        gaps = [r[f"{metric}_maj4000"] - r[f"{metric}_default"] for r in rows]
        rho = spearmanr(rates, gaps)
        print(
            f"maj4000 - default {metric}: {[round(g, 4) for g in gaps]}, "
            f"Spearman vs rate {rho.statistic:.2f} (p {rho.pvalue:.2f}, n=5)"
        )

    hs = horizon_table(args.runs / "context-sweep-3seed-emb" / "context_sweep.json")
    print("\nHORIZONS (context sweep, embargoed split, 3 seeds): uniform - balanced AUC")
    print("max_context horizon rate% gap")
    for r in hs:
        print(f"{r['size']} {r['horizon']} {100 * r['rate']:.3f} {r['gap']:+.4f}")
    for size in sorted({r["size"] for r in hs}):
        sub = [r for r in hs if r["size"] == size]
        rho = spearmanr([r["rate"] for r in sub], [r["gap"] for r in sub])
        print(f"max_context {size}: Spearman gap vs rate {rho.statistic:.2f} (n={len(sub)})")
    rho = spearmanr([r["rate"] for r in hs], [r["gap"] for r in hs])
    print(f"all cells: Spearman {rho.statistic:.2f} (p {rho.pvalue:.2f}, n={len(hs)})")


if __name__ == "__main__":
    main()
