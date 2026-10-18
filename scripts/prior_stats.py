"""Describe what a prior mixture actually emits: balance, categoricals, missingness, tails.

Audit AU5.11 lists diversity gaps in the pretraining prior found by reading the code: binary
tasks with few categoricals at the shipped ``p_financial 1.0``, categoricals that are always
ordered low-cardinality codes, missingness that is only ever MCAR, near-Gaussian marginals.
Its Verify clause asks for the sampler's statistics before and after any change, so this
measures them on drawn tasks rather than reasoning about the code. It reports; it changes no
default.

Statistics, per mixture, over ``--n-tasks`` tasks drawn with :func:`sample_task` at sizes from
``--n-rows-choices``:

* **sources**: the realised source counts beside the shares :func:`source_shares` says the
  configuration produces;
* **class balance**: the smallest class's share of a task's rows, its row count, and the
  class count;
* **width**: features per task and features per row;
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

**Design review DR2.2/DR2.3 checks** (``--checks``): ``--n-tasks`` tasks *per source* (SCM,
tree, task families, regression, the gated prior, the financial prior gated, with each
ratio's sign a per-task coin, and ungated),
drawn under :data:`V2_KNOBS`, each knob distribution compared with its target and printed as
PASS or FAIL; the gated financial source's tree-versus-rank-logistic average precision; the
"no column points the same way in more than 75% of tasks" check (``CLAUDE.md``, §47); and
the generation time per batch of 8 at the reference recipe. The targets are stated beside
each line, so a FAIL says which distribution missed and by how much.

Usage::

    uv run python scripts/prior_stats.py --n-tasks 300
    uv run python scripts/prior_stats.py --n-tasks 300 --mixtures default,v2 --checks
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy.stats import fisher_exact, kstest, kurtosis, spearmanr

from fintfm.prior.base import Task
from fintfm.prior.knobs import (
    CATEGORICAL_LEVELS,
    MIN_ROWS_PER_CLASS,
    MISSING_MECHANISMS,
    MISSING_RATE_RANGE,
    TIE_KINDS,
)
from fintfm.prior.mixture import PriorConfig, sample_batch, sample_task, source_shares

#: The DR2.3 knob settings the checks draw under, and the "v2" mixture's: the design review's
#: proposal (p_binary 0.7, narrow-skewed width bands) with every other knob on at a moderate
#: value. These are the values the checks measure, not new defaults; every default stays off.
V2_KNOBS: dict[str, Any] = {
    "feature_ratio_bands": ((0.5, 0.0, 0.05), (0.35, 0.05, 0.2), (0.15, 0.2, 0.5)),
    "p_binary": 0.7,
    "imbalance_draws": True,
    "categorical_mode": "unordered",
    "missing_mode": "mixed",
    "tie_column_prob": 0.15,
    "irrelevant_frac_max": 0.2,
    "duplicate_frac_max": 0.1,
    "label_flip_max": 0.05,
}

#: The mixtures compared by default. "default" is :class:`PriorConfig`'s own weights; "shipped"
#: is the published checkpoint's ``--p-financial 1.0`` (FINDINGS §127's recipe); "broader" is
#: AU5.11's proposed arm, SCM + tree + task families with ``p_financial`` at most 0.3. "v2" is
#: the design review's prior v2 shape (DR2): normalised weights over a general trunk plus the
#: gated and financial sources, half the financial labels gated and each financial ratio's
#: sign a per-task coin, under :data:`V2_KNOBS`.
MIXTURES: dict[str, dict[str, Any]] = {
    "default": {},
    "shipped": {"p_financial": 1.0},
    "broader": {"p_financial": 0.3, "p_tree": 0.3, "p_task_family": 0.3},
    "v2": {
        "source_weights": {"scm": 0.3, "tree": 0.2, "task_family": 0.1, "gated": 0.1,
                           "financial": 0.3},
        "financial_gate_prob": 0.5,
        "financial_sign_flip_prob": 0.5,
        **V2_KNOBS,
    },
}

#: The sources the checks draw one at a time: name -> PriorConfig overrides.
CHECK_SOURCES: dict[str, dict[str, Any]] = {
    "scm": {"source_weights": {"scm": 1.0}},
    "tree": {"source_weights": {"tree": 1.0}},
    "task_family": {"source_weights": {"task_family": 1.0}},
    "regression": {"source_weights": {"regression": 1.0}},
    "gated": {"source_weights": {"gated": 1.0}},
    # the ratio sign flip is on with the gate: without it long_term_debt/equity points the
    # same way in 87.3% of the tasks where it is significant (DR2.2)
    "financial_gated": {"source_weights": {"financial": 1.0}, "financial_gate_prob": 1.0,
                        "financial_sign_flip_prob": 0.5},
    "financial": {"source_weights": {"financial": 1.0}},
}

#: Most tasks any one column may point the same way in (``CLAUDE.md``, "check the prior
#: admits no global rule"; DR2.2's Verify).
MAX_SAME_DIRECTION = 0.75

#: Fewest tasks a column must be significant in before its same-direction share is judged.
#: At 20, the largest share over some 40 column positions passes 0.75 by chance alone: the
#: SCM and tree references, which have no gate, reach 0.70 there. At 50 a fair coin reaches
#: 0.76 in a given column with probability about 2e-4, so a share above 0.75 is a rule.
MIN_DIRECTION_TASKS = 50

#: Generation budget per batch of 8 at the reference recipe (DR2.3's Verify).
MAX_SECONDS_PER_BATCH = 0.25


@dataclass
class _Acc:
    """Running per-mixture accumulators."""

    minority: list[float] = field(default_factory=list)
    min_count: list[int] = field(default_factory=list)
    n_classes: list[int] = field(default_factory=list)
    n_features: list[int] = field(default_factory=list)
    ratio: list[float] = field(default_factory=list)
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
    acc.min_count.append(int(present.min()))
    acc.n_classes.append(len(present))
    acc.n_features.append(t.n_features)
    acc.ratio.append(t.n_features / t.n_rows)
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


def _config(overrides: dict[str, Any], max_features: int, max_classes: int) -> PriorConfig:
    """Build a prior configuration at the given shape.

    Args:
        overrides: :class:`PriorConfig` fields.
        max_features: Feature width.
        max_classes: Class-head width.

    Returns:
        The configuration.
    """
    return PriorConfig(max_features=max_features, max_classes=max_classes, **overrides)


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
    shares = source_shares(cfg)
    configured = {s: round(v, 3) for s, v in shares["realised"].items() if v > 0}
    print(f"\n## {name}  ({n_tasks} tasks, {secs:.0f}s)  weights {MIXTURES.get(name, {})}")
    print(f"sources: {dict(sorted(acc.sources.items()))}; configured shares ({shares['mode']}) "
          f"{configured}")
    print(f"classes: binary share {np.mean(np.asarray(acc.n_classes) == 2):.3f}, "
          f"n_classes p5/p50/p95 {_q(acc.n_classes)}")
    print(f"minority class share p5/p50/p95 {_q(acc.minority)}, "
          f"range {min(acc.minority):.4f}-{max(acc.minority):.4f}")
    mc = np.asarray(acc.min_count)
    print(f"smallest class rows p5/p50/p95 {_q(acc.min_count)}, tasks with a class of <= 2 "
          f"rows {np.mean(mc <= 2):.3f}")
    print(f"features p5/p50/p95 {_q(acc.n_features)}; features per row p5/p50/p95 "
          f"{_q(acc.ratio)}, max {max(acc.ratio):.3f}")
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


# --- DR2.2 / DR2.3 checks ------------------------------------------------------------------


class _Report:
    """Collects PASS/FAIL lines so the run ends with a count of failures."""

    def __init__(self) -> None:
        """Start with no checks."""
        self.failures: list[str] = []
        self.n = 0

    def check(self, label: str, ok: bool, detail: str) -> None:
        """Print one check.

        Args:
            label: What is checked.
            ok: Whether it met its target.
            detail: The measured value beside the target.
        """
        self.n += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
        if not ok:
            self.failures.append(label)


def _within(share: float, p: float, n: int, sigmas: float = 4.0) -> bool:
    """Whether an observed share is within ``sigmas`` binomial errors of ``p`` at ``n``."""
    if n == 0:
        return False
    return abs(share - p) <= sigmas * np.sqrt(max(p * (1 - p), 1e-12) / n)


def _draw(cfg: PriorConfig, n_tasks: int, sizes: tuple[int, ...], seed: int) -> list[Task]:
    """Draw ``n_tasks`` tasks at sizes from ``sizes``.

    Args:
        cfg: Prior configuration.
        n_tasks: Tasks to draw.
        sizes: Task sizes, drawn uniformly per task.
        seed: Generator seed.

    Returns:
        The tasks.
    """
    rng = np.random.default_rng(seed)
    return [sample_task(rng, cfg, n_rows=int(rng.choice(sizes))) for _ in range(n_tasks)]


def _check_width(rep: _Report, tasks: list[Task], slack: int) -> None:
    """Width bands: each task's features per row lies in the band it drew, or at a width limit.

    Args:
        rep: Report.
        tasks: Tasks drawn with ``feature_ratio_bands``.
        slack: Columns a task may exceed its band's upper end by (the financial prior adds up
            to four redundant/noise columns and the gate column after the band's width).
    """
    bands = V2_KNOBS["feature_ratio_bands"]
    hits, drawn = 0, Counter()
    for t in tasks:
        b = (t.meta or {}).get("width_band")
        if b is None:
            continue
        drawn[b] += 1
        _, lo, hi = bands[b]
        f, n = t.n_features, t.n_rows
        if lo * n - 1 <= f <= hi * n + 1 + slack or f <= 4 or f >= 136 - slack:
            hits += 1
    total = sum(drawn.values())
    rep.check("width in its drawn band (or at a width limit)", total > 0 and hits / total >= 0.98,
              f"{hits}/{total}; target >= 98%")
    weights = np.array([b[0] for b in bands]) / sum(b[0] for b in bands)
    shares = [drawn[i] / max(total, 1) for i in range(len(bands))]
    rep.check("band shares match their weights",
              all(_within(s, w, total) for s, w in zip(shares, weights)),
              f"{[round(s, 3) for s in shares]} against {[round(w, 3) for w in weights]} (4 sigma)")


def _check_labels(rep: _Report, tasks: list[Task]) -> None:
    """``p_binary``, the binary minority share LogU(4/N, 0.5), and per-class minimum counts.

    Args:
        rep: Report.
        tasks: SCM or tree tasks drawn under :data:`V2_KNOBS` with label flips off.
    """
    n_cls = np.array([t.n_classes for t in tasks])
    binary = float(np.mean(n_cls == 2))
    p = V2_KNOBS["p_binary"]
    rep.check("binary share", _within(binary, p, len(tasks)),
              f"{binary:.3f}; target {p} within 4 sigma ({len(tasks)} tasks)")
    u: list[float] = []
    for t in tasks:
        if t.n_classes != 2:
            continue
        counts = np.bincount(t.y, minlength=2)
        floor = MIN_ROWS_PER_CLASS / t.n_rows
        m = counts.min() / counts.sum()
        u.append((np.log(m) - np.log(floor)) / (np.log(0.5) - np.log(floor)))
    ua = np.asarray(u)
    ks = kstest(np.clip(ua, 0, 1), "uniform").pvalue if len(ua) else 0.0
    rep.check("binary minority share ~ LogU(4/N, 0.5)",
              len(ua) > 0 and ua.min() >= -0.05 and ua.max() <= 1.0 + 1e-9 and ks > 0.001,
              f"log-position p5/p50/p95 {_q(list(ua))}, min {ua.min():.3f}, KS p {ks:.3g}; "
              "target in [0, 1], uniform (KS p > 0.001)")
    mins = [int(np.bincount(t.y, minlength=t.n_classes).min()) for t in tasks]
    # flips (label_flip_max) can move a few rows out of a 4-row class
    rep.check("per-class minimum count", min(mins) >= MIN_ROWS_PER_CLASS - 1,
              f"min {min(mins)}, p5/p50 {_q(mins, (5, 50))}; target >= {MIN_ROWS_PER_CLASS} "
              "(one row of slack for a tie at a quantile cut)")
    multi = [t for t in tasks if t.n_classes > 2]
    rep.check("multiclass tasks present (Dirichlet shares)", len(multi) > 0,
              f"{len(multi)} tasks, smallest class share p5/p50 "
              f"{_q([np.bincount(t.y).min() / t.n_rows for t in multi], (5, 50))}")


def _expected_round_share(frac_max: float, width: int) -> float:
    """``E[round(U(0, frac_max) * width)] / width``: the expected added share at one width.

    Args:
        frac_max: Upper end of the drawn share.
        width: The width the share is taken of.

    Returns:
        The expected share of ``width`` the rounded count makes up.
    """
    grid = (np.arange(4000) + 0.5) / 4000 * frac_max * width
    return float(np.round(grid).mean() / width)


def _check_columns(rep: _Report, tasks: list[Task], labels: bool) -> None:
    """Categoricals, missingness, ties, irrelevant/duplicate columns and label flips.

    Args:
        rep: Report.
        tasks: Tasks from a general source drawn under :data:`V2_KNOBS`.
        labels: Whether label flips apply (not to regression).
    """
    recs = [t.meta["knobs"] for t in tasks if t.meta and "knobs" in t.meta]
    n = len(recs)
    any_cat = float(np.mean([bool(r.get("categorical_levels")) for r in recs]))
    rep.check("tasks with unordered categoricals", _within(any_cat, 0.5, n, 5.0),
              f"{any_cat:.3f}; target 0.5 less the tasks whose share draw hit no column")
    levels = [k for r in recs for k in r.get("categorical_levels", [])]
    realised: list[int] = []
    top_ratio: list[float] = []
    for t in tasks:
        for j in np.flatnonzero(t.is_categorical):
            col = t.X[:, j]
            vals, counts = np.unique(col[np.isfinite(col)], return_counts=True)
            if len(vals):
                realised.append(len(vals))
                top_ratio.append(counts.max() / counts.sum() * len(vals))
    lo, hi = CATEGORICAL_LEVELS
    med = float(np.median(levels)) if levels else 0.0
    rep.check("categorical levels LogInt(2, 64)",
              bool(levels) and min(levels) >= lo and max(levels) <= hi and 6 <= med <= 16,
              f"drawn p5/p50/p95 {_q(levels)}, realised max {max(realised) if realised else '-'}; "
              f"target in [{lo}, {hi}], median near 11")
    zipf = float(np.mean(top_ratio)) if top_ratio else 0.0
    rep.check("Zipf frequencies (top level over uniform share)", zipf > 1.5,
              f"mean {zipf:.2f}; target > 1.5 (1.0 is uniform)")
    miss = [r.get("missing", {}) for r in recs]
    with_miss = [m for m in miss if m.get("mechanism", "none") != "none"]
    share = len(with_miss) / max(n, 1)
    rep.check("tasks with missingness", _within(share, 0.5, n), f"{share:.3f}; target 0.5")
    mech = Counter(m["mechanism"] for m in with_miss)
    rep.check("MCAR/MAR/MNAR shares",
              all(_within(mech[k] / max(len(with_miss), 1), 1 / 3, len(with_miss))
                  for k in MISSING_MECHANISMS),
              f"{dict(mech)}; target a third each")
    rates = [m["rate"] for m in with_miss]
    rlo, rhi = MISSING_RATE_RANGE
    ru = [(np.log(r) - np.log(rlo)) / (np.log(rhi) - np.log(rlo)) for r in rates]
    ks = kstest(ru, "uniform").pvalue if ru else 0.0
    rep.check("missing rate LogU(0.01, 0.3)", bool(rates) and ks > 0.001,
              f"p5/p50/p95 {_q(rates)}, KS p {ks:.3g}")
    numeric = sum(r.get("numeric_columns", 0) for r in recs)
    kinds = Counter(k for r in recs for k in r.get("tie_kinds", []))
    tied = sum(kinds.values())
    p = V2_KNOBS["tie_column_prob"]
    rep.check("tie / point-mass / zero-inflation column share", _within(tied / max(numeric, 1), p,
              numeric), f"{tied}/{numeric} = {tied / max(numeric, 1):.3f}; target {p}")
    rep.check("tie kinds a third each",
              all(_within(kinds[k] / max(tied, 1), 1 / 3, tied) for k in TIE_KINDS),
              f"{dict(kinds)}")
    widths = [max(len(r["origin"]), 1) for r in recs]
    for label, key, knob in (("irrelevant", "n_irrelevant", "irrelevant_frac_max"),
                             ("near-duplicate", "n_duplicate", "duplicate_frac_max")):
        got = float(np.mean([r[key] / w for r, w in zip(recs, widths)]))
        # the count is round(U(0, max) * width), so a narrow task rounds to fewer columns
        # than max/2 of its width; the target is that expectation, averaged over the tasks
        want = float(np.mean([_expected_round_share(V2_KNOBS[knob], w) for w in widths]))
        rep.check(f"{label} column share", abs(got - want) <= 0.2 * want + 0.005,
                  f"mean {got:.4f}; target {want:.4f} (E[round(U(0, {V2_KNOBS[knob]}) x width)]"
                  " / width)")
    if labels:
        flips = [r["label_flip_rate"] for r in recs if "label_flip_rate" in r]
        want = V2_KNOBS["label_flip_max"]
        rep.check("label flip rate U(0, max)",
                  bool(flips) and abs(np.mean(flips) - want / 2) <= 0.25 * want
                  and max(flips) <= 2 * want,
                  f"mean {np.mean(flips):.4f}, max {max(flips):.4f}; target mean {want / 2}")


def _directions(tasks: list[Task], key: str) -> tuple[str, float, int, list[tuple]]:
    """How consistently each column points the same way across tasks.

    A column "points" up in a task when its Spearman correlation with the binary label is
    positive at p < 0.01, down when negative; insignificant columns do not count.

    Args:
        tasks: Binary tasks.
        key: ``"position"`` to key columns by index, ``"name"`` by the financial prior's
            ``meta["columns"]`` (with ``gate:``, ``redundant:`` and ``block:`` stripped, so
            a quantity is one column however it was exposed).

    Returns:
        ``(column, share, n, top5)``: the column with the largest same-direction share among
        those significant in at least :data:`MIN_DIRECTION_TASKS` tasks, that share, its task
        count, and the top five.
    """
    signs: dict[Any, list[int]] = defaultdict(list)
    for t in tasks:
        names = (t.meta or {}).get("columns") if key == "name" else None
        for j in range(t.n_features):
            col = t.X[:, j]
            ok = np.isfinite(col)
            if ok.sum() < 10 or np.ptp(col[ok]) == 0 or len(np.unique(t.y[ok])) < 2:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, pval = spearmanr(col[ok], t.y[ok])
            if not np.isfinite(pval) or pval >= 0.01:
                continue
            if names is not None:
                name = names[j]
                for prefix in ("gate:", "redundant:", "block:"):
                    while name.startswith(prefix):
                        name = name[len(prefix):]
                k: Any = name
            else:
                k = j
            signs[k].append(1 if rho > 0 else -1)
    rows = []
    for k, s in signs.items():
        if len(s) >= MIN_DIRECTION_TASKS:
            up = np.mean(np.asarray(s) > 0)
            rows.append((k, max(up, 1 - up), len(s)))
    rows.sort(key=lambda r: -r[1])
    if not rows:
        return "-", 0.0, 0, []
    return str(rows[0][0]), float(rows[0][1]), int(rows[0][2]), rows[:5]


def _rank_features(train: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Rank-transform each column by the training rows' empirical CDF, NaN at 0.5.

    Args:
        train: ``(n_train, F)`` fit rows.
        X: ``(n, F)`` rows to transform.

    Returns:
        ``(n, F)`` values in ``[0, 1]``.
    """
    out = np.full(X.shape, 0.5)
    for j in range(X.shape[1]):
        ref = np.sort(train[np.isfinite(train[:, j]), j])
        if len(ref) == 0:
            continue
        ok = np.isfinite(X[:, j])
        out[ok, j] = np.searchsorted(ref, X[ok, j], side="right") / len(ref)
    return out


def tree_vs_rank_logistic(tasks: list[Task], train_frac: float = 0.7) -> dict[str, float]:
    """Average precision of a tree ensemble against rank-logistic regression, per task.

    Rows are exchangeable in every prior, so the first ``train_frac`` of each task's rows fit
    and the rest score. A task is skipped when its fit rows hold fewer than two positives or
    its scored rows none.

    Args:
        tasks: Binary tasks.
        train_frac: Share of rows used to fit.

    Returns:
        ``{"n", "tree", "rank_logistic", "diff", "tree_wins"}``: tasks scored, mean AP of
        each, the mean paired difference and the share of tasks the tree wins.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score

    tree_ap, lr_ap = [], []
    for t in tasks:
        n_fit = int(train_frac * t.n_rows)
        Xf, yf, Xs, ys = t.X[:n_fit], t.y[:n_fit], t.X[n_fit:], t.y[n_fit:]
        if yf.sum() < 2 or ys.sum() < 1 or yf.sum() == len(yf):
            continue
        rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1)
        rf.fit(Xf, yf)
        tree_ap.append(average_precision_score(ys, rf.predict_proba(Xs)[:, 1]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr = LogisticRegression(max_iter=2000).fit(_rank_features(Xf, Xf), yf)
        lr_ap.append(average_precision_score(ys, lr.predict_proba(_rank_features(Xf, Xs))[:, 1]))
    a, b = np.asarray(tree_ap), np.asarray(lr_ap)
    return {"n": len(a), "tree": float(a.mean()), "rank_logistic": float(b.mean()),
            "diff": float((a - b).mean()), "tree_wins": float(np.mean(a > b))}


def generation_seconds(cfg: PriorConfig, n_batches: int, seed: int) -> list[float]:
    """Wall time of ``n_batches`` batches of 8 from ``cfg``.

    Args:
        cfg: Prior configuration, with ``n_rows_choices`` set.
        n_batches: Batches to time.
        seed: Generator seed.

    Returns:
        Seconds per batch.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_batches):
        t0 = time.perf_counter()
        sample_batch(rng, cfg, 8)
        out.append(time.perf_counter() - t0)
    return out


def run_checks(n_tasks: int, sizes: tuple[int, ...], max_features: int, max_classes: int,
               seed: int, n_batches: int) -> int:
    """Run every DR2.2/DR2.3 check and print a PASS/FAIL line per check.

    Args:
        n_tasks: Tasks per source.
        sizes: Task sizes.
        max_features: Feature width.
        max_classes: Class-head width.
        seed: Generator seed.
        n_batches: Batches to time per recipe.

    Returns:
        The number of failed checks.
    """
    rep = _Report()
    drawn: dict[str, list[Task]] = {}
    for name, overrides in CHECK_SOURCES.items():
        knobs = {} if name == "financial" else V2_KNOBS
        cfg = _config({**overrides, **knobs}, max_features, max_classes)
        t0 = time.perf_counter()
        drawn[name] = _draw(cfg, n_tasks, sizes, seed)
        print(f"\n### {name}: {n_tasks} tasks in {time.perf_counter() - t0:.0f}s"
              + ("" if knobs else " (knobs off, the reference for the gated variant)"))
        tasks = drawn[name]
        if name != "financial":
            _check_width(rep, tasks, slack=6 if name.startswith("financial") else 1)
        if name in ("scm", "tree"):
            # label flips move rows between classes after the cut, so the cut's own
            # distribution is checked on a draw with flips off (flips are checked below)
            no_flip = _config({**overrides, **V2_KNOBS, "label_flip_max": 0.0}, max_features,
                              max_classes)
            _check_labels(rep, _draw(no_flip, n_tasks, sizes, seed))
        if name in ("scm", "tree", "task_family", "regression", "gated"):
            _check_columns(rep, tasks, labels=name != "regression")
        if name == "gated":
            mins = [int(t.y.sum()) for t in tasks]
            shares = [t.meta["gate"]["share"] for t in tasks]
            sides = np.mean([d > 0 for t in tasks for d in t.meta["gate"]["directions"]])
            rep.check("gated: positives per task", min(mins) >= 2,
                      f"min {min(mins)}, p5/p50/p95 {_q(mins)}; target >= 2")
            rep.check("gated: gate share and side", 0.4 <= sides <= 0.6,
                      f"share p5/p50/p95 {_q(shares)}; high-side share {sides:.3f} "
                      "(target 0.5)")
        if name.startswith("financial"):
            mins = [int(t.y.sum()) for t in tasks]
            print(f"  positives per task p5/p50/p95 {_q(mins)}, min {min(mins)}")
        if name == "financial_gated":
            gates = [t.meta["gate"] for t in tasks]
            kinds = Counter(g["kind"] for g in gates)
            thr = Counter(g["threshold"] for g in gates)
            econ = [g["economic"] for g in gates if g["economic"] is not None]
            print(f"  gate kinds {dict(kinds)}; thresholds {dict(thr)}; economic side "
                  f"{np.mean(econ):.3f}; gate share p5/p50/p95 {_q([g['share'] for g in gates])};"
                  f" leak p5/p50/p95 {_q([g['leak'] for g in gates])}")
    print("\n### column directions (no global rule, CLAUDE.md / §47)")
    for name, key in (("gated", "position"), ("financial_gated", "name"),
                      ("financial", "name"), ("scm", "position"), ("tree", "position")):
        tasks = [t for t in drawn[name] if t.n_classes == 2]
        if key == "name" and not all((t.meta or {}).get("columns") for t in tasks):
            # the ungated reference carries no column names; a gate probability that never
            # fires attaches them while leaving every label ungated
            cfg = _config({"source_weights": {"financial": 1.0},
                           "financial_gate_prob": 1e-12}, max_features, max_classes)
            tasks = _draw(cfg, n_tasks, sizes, seed)
        col, share, n, top = _directions(tasks, key)
        detail = f"worst {col!r} {share:.3f} over {n} tasks; top {[(str(k), round(s, 3), c) for k, s, c in top[:3]]}"
        if name in ("gated", "financial_gated") and n == 0:
            # positions are columns only by index, and the gate's column and side are drawn
            # per task, so no position gathers enough significant tasks to judge; the gate's
            # high-side share is the check for this source
            print(f"  (not measurable) {name}: no column significant in "
                  f"{MIN_DIRECTION_TASKS} tasks; see the gate-side share check")
        elif name in ("gated", "financial_gated"):
            rep.check(f"{name}: no column points the same way in > 75% of tasks",
                      share <= MAX_SAME_DIRECTION, detail)
        else:
            print(f"  (reference) {name}: {detail}")
    print("\n### tree vs rank-logistic regression, average precision (70/30 split per task)")
    for name in ("financial_gated", "financial", "gated"):
        r = tree_vs_rank_logistic(drawn[name])
        detail = (f"tree {r['tree']:.4f} vs rank-logistic {r['rank_logistic']:.4f}, paired "
                  f"diff {r['diff']:+.4f}, tree wins {r['tree_wins']:.3f} of {r['n']} tasks")
        if name == "financial_gated":
            rep.check("financial_gated: tree AP > rank-logistic AP", r["diff"] > 0, detail)
        else:
            print(f"  (reference) {name}: {detail}")
    print("\n### generation time per batch of 8 "
          f"(n_rows_choices {sizes}, max_features {max_features}; load average "
          f"{os.getloadavg()[0]:.0f})")
    times: dict[str, float] = {}
    for name, overrides in (("default", {}), ("shipped", MIXTURES["shipped"]),
                            ("v2", MIXTURES["v2"])):
        cfg = replace(_config(overrides, max_features, max_classes), n_rows_choices=sizes)
        secs = generation_seconds(cfg, n_batches, seed)
        times[name] = float(np.mean(secs))
        print(f"  {name}: mean {np.mean(secs):.3f}s, p50 {np.median(secs):.3f}s, max "
              f"{max(secs):.3f}s over {n_batches} batches")
    rep.check("v2 generation time per batch", times["v2"] <= MAX_SECONDS_PER_BATCH,
              f"{times['v2']:.3f}s (default {times['default']:.3f}s, ratio "
              f"{times['v2'] / times['default']:.2f}); target <= {MAX_SECONDS_PER_BATCH}s")
    print(f"\n{rep.n - len(rep.failures)}/{rep.n} checks pass"
          + (f"; FAILED: {rep.failures}" if rep.failures else ""))
    return len(rep.failures)


def main() -> None:
    """Parse arguments, describe each mixture, and run the DR2 checks when asked."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-tasks", type=int, default=300)
    p.add_argument("--n-rows-choices", type=str, default="256,512,1024")
    p.add_argument("--max-features", type=int, default=136)
    p.add_argument("--max-classes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mixtures", type=str, default="default,shipped,broader")
    p.add_argument("--checks", action="store_true",
                   help="run the design review DR2.2/DR2.3 checks, --n-tasks per source")
    p.add_argument("--n-batches", type=int, default=20,
                   help="batches of 8 timed per recipe by --checks")
    args = p.parse_args()
    sizes = tuple(int(v) for v in args.n_rows_choices.split(","))
    for name in [m for m in args.mixtures.split(",") if m]:
        describe(name, _config(MIXTURES[name], args.max_features, args.max_classes),
                 args.n_tasks, sizes, args.seed)
    if args.checks:
        failed = run_checks(args.n_tasks, sizes, args.max_features, args.max_classes,
                            args.seed, args.n_batches)
        raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
