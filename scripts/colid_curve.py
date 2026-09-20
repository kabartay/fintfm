"""Assemble the `column_id_dim` curve from saved per-fold predictions (task 39.30).

Why a script rather than a notebook cell
----------------------------------------

§97 measured `column_id_dim` 12 → 16 as worth +0.0126 average precision on 5 of 5 V4FinBench
folds, against nulls everywhere else, and noted that 16 was chosen **by accident** while
setting up an unrelated experiment. Two values had ever been tried. This assembles the full
sweep — 12, 16, 20, 24, 32 — from prediction files each protocol run already writes, so the
curve costs no additional scoring.

**The comparison is paired at the row level.** Every checkpoint is scored on identical rows in
identical order, so `paired_auc_difference` keeps that pairing rather than treating the two
average precisions as independent samples, which would overstate the uncertainty. Holm
correction is applied across the adjacent-pair comparisons, because a sweep of five values
manufactures a winner by chance otherwise (`evaluation/metrics.py` cites the credit-risk
literature on exactly this).

**A peak is as publishable as a climb.** Task 39.30 says so explicitly, and this script prints
the shape it finds rather than testing a preferred direction.
"""

from __future__ import annotations

import glob
import sys

import numpy as np
from sklearn.metrics import average_precision_score

from fintfm.evaluation.metrics import holm_adjusted_p, paired_auc_difference

#: `column_id_dim` → directory of per-fold prediction files. The 12 and 16 entries are §97's
#: own runs, reused rather than re-scored so the curve's first two points are literally the
#: numbers that finding reported.
ARMS: dict[int, str] = {
    12: "runs/v4-colid12-full",
    16: "runs/v4-scale-48k",
    20: "runs/colid-sweep-20",
    24: "runs/colid-sweep-24",
    32: "runs/colid-sweep-32",
}


def fold_scores(directory: str) -> list[float]:
    """Average precision per fold, in fold order."""
    files = sorted(glob.glob(f"{directory}/predictions_fold*.npz"))
    out = []
    for f in files:
        z = np.load(f)
        out.append(float(average_precision_score(z["y_true"], z["pred_fintfm"])))
    return out


def paired_delta(dir_a: str, dir_b: str) -> tuple[float, int, int, list[float]]:
    """Per-fold paired AP difference between two arms, plus the per-fold p-values."""
    fa = sorted(glob.glob(f"{dir_a}/predictions_fold*.npz"))
    fb = sorted(glob.glob(f"{dir_b}/predictions_fold*.npz"))
    deltas, pvals = [], []
    for a, b in zip(fa, fb, strict=True):
        za, zb = np.load(a), np.load(b)
        if not np.array_equal(za["y_true"], zb["y_true"]):
            raise ValueError(f"{a} and {b} are not the same rows; the pairing is invalid")
        d, _ci, p = paired_auc_difference(
            za["y_true"], za["pred_fintfm"], zb["pred_fintfm"],
            n_boot=2000, seed=0, metric=average_precision_score,
        )
        deltas.append(d)
        pvals.append(p)
    return float(np.mean(deltas)), sum(d > 0 for d in deltas), len(deltas), pvals


def main() -> None:
    have = {k: v for k, v in ARMS.items() if len(glob.glob(f"{v}/predictions_fold*.npz")) == 5}
    missing = sorted(set(ARMS) - set(have))
    if missing:
        print(f"incomplete (5 folds each required), missing: {missing}", file=sys.stderr)
    if len(have) < 2:
        sys.exit("need at least two complete arms")

    print(f"{'column_id_dim':>14} " + " ".join(f"{'f'+str(i):>8}" for i in range(5)) + f"{'mean':>9}")
    means = {}
    for dim in sorted(have):
        s = fold_scores(have[dim])
        means[dim] = float(np.mean(s))
        print(f"{dim:>14} " + " ".join(f"{v:>8.4f}" for v in s) + f"{means[dim]:>9.4f}")

    dims = sorted(have)
    print(f"\n{'comparison':>22}{'dAP':>10}{'folds':>8}{'Holm p (per fold)':>34}")
    raw: list[float] = []
    rows = []
    for a, b in zip(dims[1:], dims[:-1], strict=True):
        d, w, n, pv = paired_delta(have[a], have[b])
        rows.append((f"{b} -> {a}", d, w, n))
        raw.extend(pv)
    adj = holm_adjusted_p(raw)
    for i, (label, d, w, n) in enumerate(rows):
        block = adj[i * 5:(i + 1) * 5]
        print(f"{label:>22}{d:>+10.4f}{f'{w}/{n}':>8}   " + " ".join(f"{p:.3f}" for p in block))

    best = max(means, key=means.get)
    print()
    if best == dims[0]:
        shape = "falls monotonically from the smallest value tested"
    elif best == dims[-1]:
        shape = "still climbing at the largest value tested -- the sweep did not bracket it"
    else:
        shape = f"peaks at {best}, bracketed on both sides"
    print(f"shape: {shape}  (best mean AP {means[best]:.4f} at column_id_dim={best})")
    print("A peak is a result, not a failure: it closes the lever rather than leaving it open.")


if __name__ == "__main__":
    main()
