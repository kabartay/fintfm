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

Usage:
    uv run python scripts/gap_vs_default_rate.py --runs /path/to/runs
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


def main() -> None:
    """Print both tables and the rank correlation of gap with rate."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, required=True)
    args = ap.parse_args()

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
