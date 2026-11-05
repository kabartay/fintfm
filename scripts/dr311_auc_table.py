"""Per-fold ROC-AUC and AP of two `fintfm-v4protocol` runs side by side (DR3.11).

`compare_v4_scores.py` gives the paired AP bootstrap; this adds the ROC-AUC point estimates the
task asks to see beside it, and the §163 rule's counts (mean difference, folds positive).

Usage:
    uv run python scripts/dr311_auc_table.py --control runs/dr31/partition RUN_DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _scores(run: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load ``(y_true, pred_fintfm)`` for every `predictions_fold<k>.npz` in a run directory.

    Args:
        run: A `fintfm-v4protocol` output directory.

    Returns:
        Fold index -> ``(y_true, pred_fintfm)``.
    """
    out = {}
    for f in sorted(run.glob("predictions_fold*.npz")):
        d = np.load(f)
        out[int(f.stem.removeprefix("predictions_fold"))] = (d["y_true"], d["pred_fintfm"])
    return out


def main() -> None:
    """Print a per-fold AP / ROC-AUC table for the control and the arm, with the §163 counts."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", required=True, type=Path)
    p.add_argument("arm", type=Path)
    args = p.parse_args()
    ctrl, arm = _scores(args.control), _scores(args.arm)
    print("| fold | control AP | arm AP | dAP | control AUC | arm AUC | dAUC |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    dap, dauc = [], []
    for k in sorted(set(ctrl) & set(arm)):
        (y, c), (y2, a) = ctrl[k], arm[k]
        if not np.array_equal(y, y2):
            raise SystemExit(f"fold {k}: rows differ")
        ap_c, ap_a = average_precision_score(y, c), average_precision_score(y, a)
        au_c, au_a = roc_auc_score(y, c), roc_auc_score(y, a)
        dap.append(ap_a - ap_c)
        dauc.append(au_a - au_c)
        print(f"| {k} | {ap_c:.4f} | {ap_a:.4f} | {ap_a - ap_c:+.4f} | {au_c:.4f} | {au_a:.4f} "
              f"| {au_a - au_c:+.4f} |")
    print(f"mean dAP {np.mean(dap):+.4f}, {sum(d > 0 for d in dap)}/{len(dap)} positive; "
          f"mean dAUC {np.mean(dauc):+.4f}, {sum(d > 0 for d in dauc)}/{len(dauc)} positive")


if __name__ == "__main__":
    main()
