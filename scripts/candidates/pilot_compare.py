"""Compare every arm against a control arm on identical V4FinBench fold-0 rows.

Reads `<root>/<arm><suffix>/predictions_fold0.npz` for each arm (by default the §157 pilot's
`runs/pilot/<arm>.score/`; §162's context diagnostic uses `--root runs/ctxdiag --suffix ""
--control ctx2000`), checks the rows are the
same rows (an arm scored on a different subsample would make every difference meaningless),
then reports each arm's AP and ROC-AUC, and its paired-bootstrap AP difference against the
control with Holm-corrected p-values across all arms -- AP first, per D13, and never a win
count (`fintfm.evaluation.metrics.paired_auc_difference`'s own docstring explains why).

Usage:
    uv run python scripts/candidates/pilot_compare.py [--n-boot 2000]
    uv run python scripts/candidates/pilot_compare.py --root runs/ctxdiag --suffix '' --control ctx2000

Arms are compared on fold 0's **validation** rows by default. Choosing an arm on fold 0's test
rows and then reporting a five-fold mean that includes fold 0 selects the configuration on its
own test fold (audit AU2.4); `--split test` reproduces the older behaviour, for runs written
before validation predictions were saved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from fintfm.evaluation.metrics import holm_adjusted_p, paired_auc_difference

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Entry point: print the comparison table and write it as JSON beside the runs."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--root", type=Path, default=ROOT / "runs" / "pilot")
    p.add_argument("--suffix", default=".score", help="arm directory suffix")
    p.add_argument("--control", default="control", help="arm every other is compared against")
    p.add_argument("--split", choices=("val", "test"), default="val",
                   help="fold 0's validation rows (default) or its test rows")
    args = p.parse_args()
    ctl = args.control
    y_key, pre = ("y_val", "val_") if args.split == "val" else ("y_true", "pred_")

    preds = {
        d.name.removesuffix(args.suffix) if args.suffix else d.name: np.load(d / "predictions_fold0.npz")
        for d in sorted(args.root.glob(f"*{args.suffix}"))
        if d.is_dir() and (d / "predictions_fold0.npz").exists()
    }
    if ctl not in preds:
        raise SystemExit(f"control arm {ctl!r} has not been scored yet")
    missing = [arm for arm, d in preds.items() if y_key not in d.files]
    if missing:
        raise SystemExit(f"{missing} saved no {args.split} predictions; rerun them, or pass "
                         "--split test and say the arm was chosen on fold 0's test rows")
    y = preds[ctl][y_key]
    for arm, d in preds.items():
        if not np.array_equal(d[y_key], y):
            raise SystemExit(f"{arm} was scored on different rows than {ctl} -- not comparable")

    ctrl = preds[ctl][pre + "fintfm"]
    lr = preds[ctl][pre + "logistic_regression"]
    rows = []
    for arm, d in preds.items():
        s = d[pre + "fintfm"]
        row = {"arm": arm, "ap": average_precision_score(y, s), "auc": roc_auc_score(y, s)}
        if arm != ctl:
            delta, (lo, hi), pval = paired_auc_difference(
                y, s, ctrl, n_boot=args.n_boot, metric=average_precision_score
            )
            row.update(d_ap=delta, lo=lo, hi=hi, p=pval)
        rows.append(row)
    tested = [r for r in rows if "p" in r]
    for r, adj in zip(tested, holm_adjusted_p([r["p"] for r in tested]), strict=True):
        r["p_holm"] = adj

    print(f"V4FinBench horizon 0, fold 0, {len(y):,} {args.split} rows, {int(y.sum())} positives "
          f"(untuned logistic regression on the same rows: AP {average_precision_score(y, lr):.4f})")
    print(f"{'arm':12s} {'AP':>7s} {'ROC-AUC':>8s} {'dAP vs ctrl':>12s} {'95% CI':>20s} {'p_holm':>7s}")
    for r in sorted(rows, key=lambda r: -r["ap"]):
        tail = (f"{r['d_ap']:+12.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] {r['p_holm']:7.3f}"
                if "d_ap" in r else f"{'(control)':>12s}")
        print(f"{r['arm']:12s} {r['ap']:7.4f} {r['auc']:8.4f} {tail}")
    (args.root / f"comparison_{args.split}.json").write_text(json.dumps(rows, indent=2, default=float))


if __name__ == "__main__":
    main()
