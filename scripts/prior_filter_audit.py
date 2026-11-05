"""Audit 48.23: does the published prior drop weak-signal or low-base-rate tasks?

TabAbax (arXiv:2611.00562) filters generated tasks and names the loss of weak-signal tasks as
its own limitation. This script draws binary tasks from the published recipe's prior section
(``configs/published/v4-cellattn-labels.yaml``) with a fixed seed, and reports, without
changing generator behaviour:

* the distribution of the positive rate;
* the distribution of a held-out learnability AUC (the ``fintfm.prior.learnability`` judge,
  recomputed here as a number, since :func:`is_learnable` returns only a bool);
* how often the in-generator both-classes guards fire (counted by wrapping ``Generator.choice``);
* how many tasks the opt-in learnability filter *would* reject (it is off in the recipe).

Usage::

    nice -n 19 uv run --frozen python scripts/prior_filter_audit.py --n-tasks 2000 --seed 0
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

from fintfm.prior.base import Task
from fintfm.prior.learnability import _JUDGE_SEED, _N_ESTIMATORS, _TRAIN_FRACTION
from fintfm.prior.mixture import PriorConfig, sample_task

RECIPE = Path(__file__).resolve().parents[1] / "configs/published/v4-cellattn-labels.yaml"


class _CountingRng(np.random.Generator):
    """Generator that counts ``choice(n, size=k, replace=False)`` calls the guard makes."""

    guard_calls: int = 0
    last_n_rows: int = 0

    def choice(self, a: Any, size: Any = None, replace: bool = True, *args: Any, **kw: Any) -> Any:
        """Delegate to ``Generator.choice``, counting the financial both-classes guard.

        Args:
            a: As :meth:`numpy.random.Generator.choice`.
            size: As above.
            replace: As above.

        Returns:
            The drawn sample.
        """
        if (
            not replace
            and isinstance(a, (int, np.integer))
            and size == max(1, int(a) // 50)
            and int(a) in (256, 512, 1024)
        ):
            self.guard_calls += 1
        return super().choice(a, size, replace, *args, **kw)


def judge_auc(task: Task) -> float:
    """Held-out AUC of the repo's ExtraTrees judge (70/30 row split); NaN if unscorable.

    Args:
        task: A binary task.

    Returns:
        The AUC, or ``nan`` when a side of the split is single-class.
    """
    X = np.nan_to_num(np.asarray(task.X, dtype=float))
    y = np.asarray(task.y).ravel()
    cut = int(len(y) * _TRAIN_FRACTION)
    if len(np.unique(y[:cut])) < 2 or len(np.unique(y[cut:])) < 2:
        return float("nan")
    m = ExtraTreesClassifier(n_estimators=_N_ESTIMATORS, random_state=_JUDGE_SEED, n_jobs=1)
    m.fit(X[:cut], y[:cut])
    return float(roc_auc_score(y[cut:], m.predict_proba(X[cut:])[:, 1]))


def main() -> None:
    """Draw tasks, print the MEASURED distributions."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-tasks", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-seconds", type=float, default=600.0)
    a = ap.parse_args()
    raw = yaml.safe_load(RECIPE.read_text())
    prior = dict(raw["prior"])
    prior["n_sectors_max"] = prior.pop("n_sectors")  # the config file's name for the field
    cfg = PriorConfig(
        max_features=raw["model"]["max_features"], max_classes=raw["model"]["max_classes"],
        **prior,
    )
    rng = _CountingRng(np.random.PCG64(a.seed))
    rows: list[tuple[int, float, float]] = []
    t0 = time.time()
    for _ in range(a.n_tasks):
        n = int(rng.choice(cfg.n_rows_choices))
        task = sample_task(rng, cfg, n_rows=n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows.append((n, float(task.y.mean()), judge_auc(task)))
        if time.time() - t0 > a.budget_seconds:
            break
    n_done = len(rows)
    rate = np.array([r[1] for r in rows])
    auc = np.array([r[2] for r in rows])
    ok = ~np.isnan(auc)
    print(f"MEASURED seed={a.seed} tasks={n_done} secs={time.time() - t0:.0f}")
    print("positive rate pct 1/5/25/50/75/95/99:",
          np.round(np.percentile(rate, [1, 5, 25, 50, 75, 95, 99]), 4).tolist())
    for lo, hi in [(0, .005), (.005, .02), (.02, .1), (.1, 1.01)]:
        print(f"  rate in [{lo},{hi}): {np.mean((rate >= lo) & (rate < hi)):.3f}")
    print(f"unscorable (single-class on a split side): {np.mean(~ok):.3f}")
    print("judge AUC pct 5/25/50/75/95 (scorable):",
          np.round(np.percentile(auc[ok], [5, 25, 50, 75, 95]), 3).tolist())
    rej = ~ok | (auc <= cfg.learnability_auc_floor)
    print(f"would-reject at floor {cfg.learnability_auc_floor}: {rej.mean():.3f}")
    print(f"  AUC<=floor (scorable): {np.mean(ok & (auc <= cfg.learnability_auc_floor)):.3f}")
    print(f"  among rate<0.02: reject share {rej[rate < 0.02].mean():.3f} "
          f"(n={int((rate < 0.02).sum())}); rate>=0.02: {rej[rate >= 0.02].mean():.3f}")
    print(f"both-classes guard firings (choice w/o replace, size n//50): {rng.guard_calls}")
    print(f"actual rejections/resamples with recipe (p_learnability_filter="
          f"{cfg.p_learnability_filter}): 0 by construction")


if __name__ == "__main__":
    main()
