"""Paired per-fold comparison of `fintfm-v4protocol` outputs against a control, §144-style.

Reproduces the table format §144-§147 used -- control AP, arm AP, paired-bootstrap difference
and Holm-adjusted p per fold, Holm across that arm's folds -- so a re-score (§158's re-adjudication)
is judged exactly the way the original results were. Refuses to compare a fold whose rows differ
between control and arm, since a difference on different rows means nothing.

Usage:
    uv run python scripts/compare_v4_scores.py \\
        --control runs/task48-6-kaggle/control_rescore.json \\
        s1=runs/cellattn-s1-kaggle/s1_rescore.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from fintfm.evaluation.metrics import holm_adjusted_p, paired_auc_difference


def _folds(run: Path) -> dict[int, dict[str, np.ndarray]]:
    """Load every `predictions_fold<k>.npz` in a v4protocol output directory.

    Args:
        run: The directory `fintfm-v4protocol --out` wrote.

    Returns:
        Fold index -> arrays (`y_true`, `pred_fintfm`, ...).
    """
    out = {}
    for f in sorted(run.glob("predictions_fold*.npz")):
        d = np.load(f)
        out[int(f.stem.removeprefix("predictions_fold"))] = {k: d[k] for k in d.files}
    return out


def main() -> None:
    """Entry point: print one §144-style table per arm."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", required=True, type=Path)
    p.add_argument("arms", nargs="+", help="name=directory")
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()
    ctrl = _folds(args.control)
    for spec in args.arms:
        name, _, path = spec.partition("=")
        arm = _folds(Path(path))
        folds = sorted(set(ctrl) & set(arm))
        if not folds:
            raise SystemExit(f"{name}: no folds in common with the control")
        rows = []
        for k in folds:
            y = ctrl[k]["y_true"]
            if not np.array_equal(y, arm[k]["y_true"]):
                raise SystemExit(f"{name}: fold {k} rows differ from the control's -- not comparable")
            a, c = arm[k]["pred_fintfm"], ctrl[k]["pred_fintfm"]
            d, (lo, hi), pv = paired_auc_difference(
                y, a, c, n_boot=args.n_boot, metric=average_precision_score
            )
            rows.append((k, average_precision_score(y, c), average_precision_score(y, a), d, lo, hi, pv))
        holm = holm_adjusted_p([r[-1] for r in rows])
        print(f"\n{name} vs control ({args.control})")
        print("| fold | control AP | arm AP | diff | 95% CI | Holm-adjusted p |")
        print("| --- | --- | --- | --- | --- | --- |")
        for (k, ca, aa, d, lo, hi, _), h in zip(rows, holm, strict=True):
            print(f"| {k} | {ca:.4f} | {aa:.4f} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {h:.3f} |")
        mean_d = float(np.mean([r[3] for r in rows]))
        print(f"mean diff {mean_d:+.4f} over {len(rows)} fold(s); "
              f"{sum(r[3] < 0 for r in rows)} negative, {sum(r[3] > 0 for r in rows)} positive")


if __name__ == "__main__":
    main()
