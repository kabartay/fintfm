"""Compare every pilot arm against the control on identical V4FinBench rows.

Reads `runs/pilot/<arm>.score/predictions_fold0.npz` for each arm, checks the rows are the
same rows (an arm scored on a different subsample would make every difference meaningless),
then reports each arm's AP and ROC-AUC, and its paired-bootstrap AP difference against the
control with Holm-corrected p-values across all arms -- AP first, per D13, and never a win
count (`fintfm.evaluation.metrics.paired_auc_difference`'s own docstring explains why).

Usage:
    uv run python scripts/candidates/pilot_compare.py [--n-boot 2000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from fintfm.evaluation.metrics import holm_adjusted_p, paired_auc_difference

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "runs" / "pilot"


def main() -> None:
    """Entry point: print the comparison table and write it as JSON beside the runs."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    preds = {
        d.name.removesuffix(".score"): np.load(d / "predictions_fold0.npz")
        for d in sorted(PILOT.glob("*.score")) if (d / "predictions_fold0.npz").exists()
    }
    if "control" not in preds:
        raise SystemExit("control arm has not been scored yet")
    y = preds["control"]["y_true"]
    for arm, d in preds.items():
        if not np.array_equal(d["y_true"], y):
            raise SystemExit(f"{arm} was scored on different rows than control -- not comparable")

    ctrl = preds["control"]["pred_fintfm"]
    lr = preds["control"]["pred_logistic_regression"]
    rows = []
    for arm, d in preds.items():
        s = d["pred_fintfm"]
        row = {"arm": arm, "ap": average_precision_score(y, s), "auc": roc_auc_score(y, s)}
        if arm != "control":
            delta, (lo, hi), pval = paired_auc_difference(
                y, s, ctrl, n_boot=args.n_boot, metric=average_precision_score
            )
            row.update(d_ap=delta, lo=lo, hi=hi, p=pval)
        rows.append(row)
    tested = [r for r in rows if "p" in r]
    for r, adj in zip(tested, holm_adjusted_p([r["p"] for r in tested]), strict=True):
        r["p_holm"] = adj

    print(f"V4FinBench horizon 0, fold 0, {len(y):,} test rows, {int(y.sum())} positives "
          f"(untuned logistic regression on the same rows: AP {average_precision_score(y, lr):.4f})")
    print(f"{'arm':12s} {'AP':>7s} {'ROC-AUC':>8s} {'dAP vs ctrl':>12s} {'95% CI':>20s} {'p_holm':>7s}")
    for r in sorted(rows, key=lambda r: -r["ap"]):
        tail = (f"{r['d_ap']:+12.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] {r['p_holm']:7.3f}"
                if "d_ap" in r else f"{'(control)':>12s}")
        print(f"{r['arm']:12s} {r['ap']:7.4f} {r['auc']:8.4f} {tail}")
    (PILOT / "comparison.json").write_text(json.dumps(rows, indent=2, default=float))


if __name__ == "__main__":
    main()
