"""Apply §159's pre-registered decision rules to V4FinBench scores, mechanically.

§159 fixed, before any of the queued GPU runs trained, what each result would have to show. This
script is those rules as code, so the verdict is read off rather than argued after the numbers
are known. The thresholds are §158's measured no-change gap and are not parameters on purpose:
changing them after seeing a result is exactly what pre-registration exists to prevent.

Two modes:

* ``pair`` (Rules 1 and 2) -- a real-source arm at two seeds, each against the no-change control
  at the same seed.
* ``single`` (Rule 3) -- one ablation against one matched control.

Usage:
    uv run python scripts/c5_verdict.py pair sba \\
        runs/kaggle-fintfm-task-158-matched-control/score.json runs/kaggle-fintfm-r-sba-s0/score.json \\
        runs/cellattn-s1-kaggle/s1_rescore.json runs/kaggle-fintfm-r-sba-s1/score.json
    uv run python scripts/c5_verdict.py single schedfree146 \\
        runs/kaggle-fintfm-task-158-matched-control/score.json runs/task48-9-kaggle/schedulefree48_9_score.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from compare_v4_scores import _folds
from sklearn.metrics import average_precision_score

#: §158: a no-change ``max_classes 10`` replicate's five-fold mean AP gap to the control.
YARDSTICK = 0.0073
#: §159 Rule 3: folds that must be negative, out of five, for an ablation to count as a loss.
MIN_NEGATIVE_FOLDS = 4


def fold_diffs(control: Path, arm: Path) -> list[float]:
    """Per-fold AP difference (arm minus control) on identical rows.

    Args:
        control: A `fintfm-v4protocol` output directory.
        arm: Another, scored on the same folds.

    Returns:
        One difference per fold, in fold order.

    Raises:
        SystemExit: If the two do not cover the same five folds on the same rows.
    """
    c, a = _folds(control), _folds(arm)
    if sorted(c) != sorted(a) or len(c) != 5:
        raise SystemExit(f"{arm} vs {control}: folds {sorted(a)} vs {sorted(c)}, need the same five")
    out = []
    for k in sorted(c):
        y = c[k]["y_true"]
        if not np.array_equal(y, a[k]["y_true"]):
            raise SystemExit(f"fold {k}: rows differ between {control} and {arm}")
        out.append(average_precision_score(y, a[k]["pred_fintfm"])
                   - average_precision_score(y, c[k]["pred_fintfm"]))
    return out


def rule1(mean_s0: float, mean_s1: float) -> str:
    """§159 Rule 1: does a real source help, hurt, or neither?

    Args:
        mean_s0: Five-fold mean AP difference at seed 0.
        mean_s1: The same at seed 1.

    Returns:
        ``"helps"``, ``"hurts"`` or ``"null"``.
    """
    avg = (mean_s0 + mean_s1) / 2
    if mean_s0 > 0 and mean_s1 > 0 and avg > YARDSTICK:
        return "helps"
    if mean_s0 < 0 and mean_s1 < 0 and avg < -YARDSTICK:
        return "hurts"
    return "null"


def rule2(mean_s0: float, mean_s1: float) -> bool:
    """§159 Rule 2: is the next source (PPDai) worth launching?

    Args:
        mean_s0: Five-fold mean AP difference at seed 0.
        mean_s1: The same at seed 1.

    Returns:
        True only if the two seeds agree in sign.
    """
    return (mean_s0 > 0) == (mean_s1 > 0) and mean_s0 != 0 and mean_s1 != 0


def rule3(diffs: list[float]) -> str:
    """§159 Rule 3: is an ablation a loss against its matched control?

    Args:
        diffs: Five per-fold AP differences, ablation minus control.

    Returns:
        ``"loss"`` or ``"null"``.
    """
    negative = sum(d < 0 for d in diffs)
    return "loss" if np.mean(diffs) < -YARDSTICK and negative >= MIN_NEGATIVE_FOLDS else "null"


def _fmt(diffs: list[float]) -> str:
    """Render per-fold differences and their mean on one line."""
    return " ".join(f"{d:+.4f}" for d in diffs) + f"  | mean {np.mean(diffs):+.4f}"


def main() -> None:
    """Entry point: print the verdict and the numbers it was read from."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    pr = sub.add_parser("pair", help="Rules 1-2: an arm at seeds 0 and 1")
    pr.add_argument("name")
    pr.add_argument("control_s0", type=Path)
    pr.add_argument("arm_s0", type=Path)
    pr.add_argument("control_s1", type=Path)
    pr.add_argument("arm_s1", type=Path)
    sg = sub.add_parser("single", help="Rule 3: one ablation vs one matched control")
    sg.add_argument("name")
    sg.add_argument("control", type=Path)
    sg.add_argument("arm", type=Path)
    args = p.parse_args()

    if args.mode == "pair":
        d0 = fold_diffs(args.control_s0, args.arm_s0)
        d1 = fold_diffs(args.control_s1, args.arm_s1)
        m0, m1 = float(np.mean(d0)), float(np.mean(d1))
        print(f"{args.name} seed 0: {_fmt(d0)}")
        print(f"{args.name} seed 1: {_fmt(d1)}")
        print(f"average of seed means {(m0 + m1) / 2:+.4f} (yardstick ±{YARDSTICK})")
        print(f"Rule 1: {rule1(m0, m1)}")
        print(f"Rule 2: next source {'eligible' if rule2(m0, m1) else 'NOT eligible'} "
              f"(seed signs {'agree' if rule2(m0, m1) else 'disagree'})")
    else:
        d = fold_diffs(args.control, args.arm)
        print(f"{args.name}: {_fmt(d)}  ({sum(x < 0 for x in d)}/5 negative)")
        print(f"Rule 3: {rule3(d)}")


if __name__ == "__main__":
    main()
