"""Describe what a prior mixture actually emits: balance, categoricals, missingness, tails.

Audit AU5.11 lists diversity gaps in the pretraining prior found by reading the code: binary
tasks with few categoricals at the shipped ``p_financial 1.0``, categoricals that are always
ordered low-cardinality codes, missingness that is only ever MCAR, near-Gaussian marginals.
Its Verify clause asks for the sampler's statistics before and after any change, so this
measures them on drawn tasks rather than reasoning about the code. It reports; it changes no
default.

Statistics, per mixture, over ``--n-tasks`` tasks drawn with :func:`sample_task` at sizes from
``--n-rows-choices``:

* **class balance**: the smallest class's share of a task's rows, and the class count;
* **categoricals**: the share of a task's columns flagged categorical, and each categorical
  column's number of distinct values;
* **missingness**: the share of missing cells, and whether it is label-informative -- a
  binary task counts as informative when some column's missingness and the label fail
  Fisher's exact test of independence at 0.01, Bonferroni-corrected over the task's columns,
  so an MCAR task is flagged at most 1% of the time. (A two-proportion z-test was tried first
  and is wrong here: with two positives, both landing in a 7%-missing column happens by
  chance in about 0.5% of columns, which the normal approximation scores as |z| > 5.)
* **tails**: each numeric column's excess kurtosis (0 for a Gaussian), over columns with at
  least 20 finite, non-constant values.

Usage::

    uv run python scripts/prior_stats.py --n-tasks 300
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import fisher_exact, kurtosis

from fintfm.prior.base import Task
from fintfm.prior.mixture import PriorConfig, sample_task

#: The mixtures compared by default. "default" is :class:`PriorConfig`'s own weights; "shipped"
#: is the published checkpoint's ``--p-financial 1.0`` (FINDINGS §127's recipe); "broader" is
#: AU5.11's proposed arm, SCM + tree + task families with ``p_financial`` at most 0.3.
MIXTURES: dict[str, dict[str, float]] = {
    "default": {},
    "shipped": {"p_financial": 1.0},
    "broader": {"p_financial": 0.3, "p_tree": 0.3, "p_task_family": 0.3},
}


@dataclass
class _Acc:
    """Running per-mixture accumulators."""

    minority: list[float] = field(default_factory=list)
    n_classes: list[int] = field(default_factory=list)
    n_features: list[int] = field(default_factory=list)
    cat_share: list[float] = field(default_factory=list)
    cardinality: list[int] = field(default_factory=list)
    miss_share: list[float] = field(default_factory=list)
    informative: list[bool] = field(default_factory=list)
    kurt: list[float] = field(default_factory=list)
    sources: dict[str, int] = field(default_factory=dict)


def _accumulate(acc: _Acc, t: Task) -> None:
    """Add one task's statistics to ``acc``.

    Args:
        acc: Accumulators to extend.
        t: A drawn task.
    """
    src = "family" if t.source.startswith("family:") else t.source
    acc.sources[src] = acc.sources.get(src, 0) + 1
    counts = np.bincount(t.y, minlength=t.n_classes)
    present = counts[counts > 0]
    acc.minority.append(float(present.min() / counts.sum()))
    acc.n_classes.append(len(present))
    acc.n_features.append(t.n_features)
    cat = np.asarray(t.is_categorical, dtype=bool)
    acc.cat_share.append(float(cat.mean()) if len(cat) else 0.0)
    missing = np.isnan(t.X)
    acc.miss_share.append(float(missing.mean()))
    for j in np.flatnonzero(cat):
        acc.cardinality.append(len(np.unique(t.X[~missing[:, j], j])))
    if len(present) == 2:
        ys = t.y == int(np.flatnonzero(counts)[0])
        cols = np.flatnonzero(missing.any(axis=0) & ~missing.all(axis=0))
        alpha = 0.01 / max(t.n_features, 1)
        hit = False
        for j in cols:
            m = missing[:, j]
            table = [[int((m & ys).sum()), int((~m & ys).sum())],
                     [int((m & ~ys).sum()), int((~m & ~ys).sum())]]
            if fisher_exact(table)[1] < alpha:
                hit = True
                break
        acc.informative.append(hit)
    for j in np.flatnonzero(~cat):
        col = t.X[:, j]
        col = col[np.isfinite(col)]
        if len(col) >= 20 and np.ptp(col) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                k = float(kurtosis(col, fisher=True))
            if np.isfinite(k):
                acc.kurt.append(k)


def _q(values: list[float] | list[int], qs: tuple[float, ...] = (5, 50, 95)) -> str:
    """Format percentiles of ``values``, or ``-`` when empty."""
    if not values:
        return "-"
    return "/".join(f"{v:.3g}" for v in np.percentile(np.asarray(values, float), qs))


def describe(name: str, cfg: PriorConfig, n_tasks: int, sizes: tuple[int, ...], seed: int) -> None:
    """Draw ``n_tasks`` tasks from ``cfg`` and print their statistics.

    Args:
        name: Label for the output.
        cfg: Prior configuration.
        n_tasks: Tasks to draw.
        sizes: Task sizes drawn uniformly per task.
        seed: Generator seed.
    """
    rng = np.random.default_rng(seed)
    acc = _Acc()
    t0 = time.perf_counter()
    for _ in range(n_tasks):
        n = int(rng.choice(sizes))
        _accumulate(acc, sample_task(rng, cfg, n_rows=n))
    secs = time.perf_counter() - t0
    k = np.asarray(acc.kurt)
    print(f"\n## {name}  ({n_tasks} tasks, {secs:.0f}s)  weights {MIXTURES.get(name, {})}")
    print(f"sources: {dict(sorted(acc.sources.items()))}")
    print(f"classes: binary share {np.mean(np.asarray(acc.n_classes) == 2):.3f}, "
          f"n_classes p5/p50/p95 {_q(acc.n_classes)}")
    print(f"minority class share p5/p50/p95 {_q(acc.minority)}, "
          f"range {min(acc.minority):.4f}-{max(acc.minority):.4f}")
    print(f"features p5/p50/p95 {_q(acc.n_features)}")
    print(f"categorical column share: mean {np.mean(acc.cat_share):.4f}, "
          f"tasks with any {np.mean(np.asarray(acc.cat_share) > 0):.3f}; "
          f"cardinality p5/p50/p95 {_q(acc.cardinality)}, max "
          f"{max(acc.cardinality) if acc.cardinality else '-'}")
    print(f"missing cells: mean {np.mean(acc.miss_share):.4f}, tasks with any "
          f"{np.mean(np.asarray(acc.miss_share) > 0):.3f}; binary tasks with label-informative "
          f"missingness (Fisher, Bonferroni 0.01) {np.mean(acc.informative) if acc.informative else 0:.3f}")
    print(f"excess kurtosis over {len(k)} numeric columns p5/p50/p95 {_q(acc.kurt)}, "
          f"share > 3 {np.mean(k > 3) if len(k) else 0:.3f}, share > 10 "
          f"{np.mean(k > 10) if len(k) else 0:.3f}")


def main() -> None:
    """Parse arguments and describe each mixture."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-tasks", type=int, default=300)
    p.add_argument("--n-rows-choices", type=str, default="256,512,1024")
    p.add_argument("--max-features", type=int, default=136)
    p.add_argument("--max-classes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mixtures", type=str, default=",".join(MIXTURES))
    args = p.parse_args()
    sizes = tuple(int(v) for v in args.n_rows_choices.split(","))
    for name in args.mixtures.split(","):
        cfg = PriorConfig(max_features=args.max_features, max_classes=args.max_classes,
                          **MIXTURES[name])
        describe(name, cfg, args.n_tasks, sizes, args.seed)


if __name__ == "__main__":
    main()
