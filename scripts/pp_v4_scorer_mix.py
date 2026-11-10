"""Test FINDINGS §218's hypothesis: the probe per leaf loses AP by pooling scorers on different scales.

§218 found the probe per partition leaf at -0.0570 mean AP on V4FinBench while AUC stayed at
0.98-0.99, and proposed that each fold ranks rows from two or three scorers whose outputs are
not on one scale. This script tests that from the saved predictions alone, with no refit.

Which scorer handled each test row is recoverable exactly. ``representation_transfer._leaf_probe``
copies the global probe's probabilities into rows routed to a fallback leaf, and scores a leaf
whose probe sample held one class with that leaf's own head, which is the partition-head arm's
output for those rows. So, per row:

* ``fallback``: the leaf-probe prediction equals the global probe's, bit for bit;
* ``leaf head``: otherwise it equals the partition head's;
* ``leaf probe``: neither.

For each fold it reports every group's size, defaults and mean score, then the pooled AP of
four label-free repairs:

* ``as run``: the leaf-probe arm unchanged (the §218 number);
* ``head -> global``: single-class-leaf rows scored by the global probe instead;
* ``rescaled``: each group's scores moved by one logit shift, so the group's mean score equals
  the global probe's mean on the same rows;
* ``both``: the two together.

The shift reads no label; the global probe supplies it. Within-group AP of each scorer is
printed beside it. If the hypothesis holds, the leaf probe ranks well inside its own group and
the repairs recover most of the pooled AP.

Usage, from the repository root::

    uv run python scripts/pp_v4_scorer_mix.py [runs/pp-v4]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-7


def _logit(p: np.ndarray) -> np.ndarray:
    """Log-odds of ``p``, clipped away from 0 and 1."""
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_logit`."""
    return 1.0 / (1.0 + np.exp(-z))


def _match_mean(scores: np.ndarray, target_mean: float) -> np.ndarray:
    """Shift ``scores`` by one logit so their mean equals ``target_mean``; ranking unchanged.

    Args:
        scores: Probabilities of one group of rows.
        target_mean: The mean probability the group should have after the shift.

    Returns:
        The shifted probabilities, found by bisection on the shift.
    """
    z = _logit(scores)
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() < target_mean:
            lo = mid
        else:
            hi = mid
    return _sigmoid(z + (lo + hi) / 2)


def _ap(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision, NaN when ``y`` holds one class."""
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def groups(d: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    """Boolean masks of the rows each scorer handled, recovered by exact equality.

    Args:
        d: One fold's ``predictions_fold<k>.npz`` from the ``--partition`` run.

    Returns:
        ``{"fallback", "leaf head", "leaf probe"}`` masks that partition the test rows.
    """
    leaf, glob, head = d["pred_representation"], d["pred_representation_global"], d["pred_own_head"]
    fallback = leaf == glob
    leaf_head = ~fallback & (leaf == head)
    return {"fallback": fallback, "leaf head": leaf_head, "leaf probe": ~fallback & ~leaf_head}


def fold_report(path: Path) -> dict[str, float]:
    """Print one fold's group table and repairs, and return the pooled AP of each repair.

    Args:
        path: The fold's ``predictions_fold<k>.npz``.

    Returns:
        Pooled AP keyed by repair name, plus the global head's and global probe's AP.
    """
    d = np.load(path)
    y = d["y_true"].astype(int)
    leaf, glob, head = d["pred_representation"], d["pred_representation_global"], d["pred_own_head"]
    masks = groups(d)
    print(f"\n{path.parent.name}: {len(y):,} rows, {y.sum():,} defaults, AUC as run {roc_auc_score(y, leaf):.4f}")
    print("| group | rows | defaults | mean leaf-probe score | mean global-probe score | AP leaf-probe arm | AP global probe | AP partition head |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, m in masks.items():
        if not m.any():
            continue
        print(f"| {name} | {m.sum():,} | {y[m].sum():,} | {leaf[m].mean():.4f} | {glob[m].mean():.4f} "
              f"| {_ap(y[m], leaf[m]):.4f} | {_ap(y[m], glob[m]):.4f} | {_ap(y[m], head[m]):.4f} |")
    head_to_global = np.where(masks["leaf head"], glob, leaf)
    rescaled = leaf.copy()
    both = head_to_global.copy()
    for m in masks.values():
        if m.any():
            rescaled[m] = _match_mean(leaf[m], glob[m].mean())
            both[m] = _match_mean(head_to_global[m], glob[m].mean())
    out = {
        "as run": _ap(y, leaf),
        "head -> global": _ap(y, head_to_global),
        "rescaled": _ap(y, rescaled),
        "both": _ap(y, both),
        "global probe": _ap(y, glob),
        "global head": _ap(y, d["pred_own_head_global"]),
    }
    print("pooled AP: " + ", ".join(f"{k} {v:.4f}" for k, v in out.items()))
    return out


def main(root: Path) -> None:
    """Report every fold and the five-fold means."""
    rows = [fold_report(root / f"f{k}" / f"predictions_fold{k}.npz") for k in range(5)]
    print("\n| repair | mean AP | change against the global head | folds above the global head |")
    print("| --- | --- | --- | --- |")
    base = np.array([r["global head"] for r in rows])
    for key in ("as run", "head -> global", "rescaled", "both", "global probe", "global head"):
        v = np.array([r[key] for r in rows])
        print(f"| {key} | {v.mean():.4f} | {v.mean() - base.mean():+.4f} | {(v > base).sum()} of 5 |")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/pp-v4"))
