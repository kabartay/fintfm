"""Opt-in knob distributions for the general priors (design review DR2.3).

The design review (``docs/design/DESIGN_REVIEW_2026-10-16.md`` §1.5) found structures the
benchmarks carry and the general priors (SCM, tree, task families) never generate: feature
counts uniform on ``[2, max_features]`` at every row count, so a 256-row task can be 0.53
features per row; SCM tasks 89% multiclass with imbalance only from random cut points;
categoricals that are always ordered quantile codes with 2-7 levels; no ties, point masses or
zero inflation; MCAR missingness only. This module draws each of those as an explicit,
configurable distribution.

**Every knob defaults to today's behaviour and consumes no random draw when off**, so a
:class:`~fintfm.prior.mixture.PriorConfig` at its defaults reproduces the stored stream
fingerprints byte for byte (``tests/test_mixture.py``, ``tests/test_prior_levers.py``).
:func:`knobs_from` returns ``None`` when every knob is off, and every caller then takes the
original code path.

The knobs, and the distribution each draws when on:

* ``feature_ratio_bands``: ``(weight, lo, hi)`` bands of features per row; a task draws a band
  by weight, then a ratio uniformly inside it, and its width is that ratio times its rows,
  clipped to the source's width limits.
* ``p_binary``: probability a multiclass-capable task (SCM, tree) is binary; otherwise its
  class count is uniform on ``[3, max_classes]``.
* ``imbalance_draws``: a binary task's minority share is LogU(4/N, 0.5) on a random side; a
  multiclass task's class shares are a Dirichlet draw over a floor of 4 rows per class, in
  random class order. Cuts are made at the latent's quantiles, so the shares are realised.
* ``categorical_mode="unordered"``: half the tasks carry categoricals; in those, each numeric
  column becomes categorical with a share drawn U(0.05, 0.5), with LogInt(2, 64) levels whose
  frequencies follow a Zipf law (exponent U(0.5, 1.5)), and the codes are permuted so their
  order says nothing about the underlying value.
* ``missing_mode="mixed"``: half the tasks carry missingness; in those, one mechanism of MCAR,
  MAR (driven by another column) or MNAR (driven by the column's own value), at a LogU(0.01,
  0.3) rate, on a share U(0.2, 1.0) of the columns.
* ``tie_column_prob``: each numeric column, with this probability, is coarsened onto a grid of
  LogInt(3, 30) values (ties), given a point mass (a share U(0.05, 0.5) of rows set to one of
  its values) or zero-inflated (its lowest share U(0.05, 0.6) set to exactly zero, the rest
  shifted, on a random side).
* ``irrelevant_frac_max`` / ``duplicate_frac_max``: a share U(0, max) of the width added as
  row-permuted copies of existing columns (each keeps its marginal and loses any relation to
  the label) and as near-duplicates (a column plus noise at LogU(0.01, 0.2) of its spread).
  The columns are then shuffled, so the added ones sit nowhere in particular.
* ``label_flip_max``: a classification task's labels flip to a random other class at a rate
  U(0, max).

Reimplemented from the design review's description; no code from any tabular-foundation-
model project enters this repository (``CLAUDE.md``, licensing boundary).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from fintfm.prior.base import Task

#: Values :attr:`TaskKnobs.categorical_mode` accepts. ``"ordered"`` is each source's own
#: quantile coding with 2-7 levels, as every checkpoint trained.
CATEGORICAL_MODES: tuple[str, ...] = ("ordered", "unordered")

#: Values :attr:`TaskKnobs.missing_mode` accepts. ``"mcar"`` is each source's own MCAR step,
#: as every checkpoint trained.
MISSING_MODES: tuple[str, ...] = ("mcar", "mixed")

#: Rows a class is guaranteed by :attr:`TaskKnobs.imbalance_draws`: the binary minority share
#: is drawn from LogU(4/N, 0.5), and a multiclass Dirichlet sits over this floor per class.
MIN_ROWS_PER_CLASS = 4

#: Cardinality range of an unordered categorical, drawn log-uniformly on the integers.
CATEGORICAL_LEVELS: tuple[int, int] = (2, 64)

#: Rate range of :attr:`TaskKnobs.missing_mode` ``"mixed"``, drawn log-uniformly.
MISSING_RATE_RANGE: tuple[float, float] = (0.01, 0.3)

#: The mechanisms ``"mixed"`` missingness draws from, uniformly.
MISSING_MECHANISMS: tuple[str, ...] = ("mcar", "mar", "mnar")

#: The column treatments :attr:`TaskKnobs.tie_column_prob` draws from, uniformly.
TIE_KINDS: tuple[str, ...] = ("ties", "point_mass", "zero_inflation")


@dataclass(frozen=True)
class TaskKnobs:
    """The DR2.3 knobs a general prior's task is drawn under; see the module docstring.

    Attributes:
        feature_ratio_bands: ``(weight, lo, hi)`` features-per-row bands; ``()`` keeps each
            source's own width draw.
        p_binary: Probability a multiclass-capable task is binary; ``None`` keeps the uniform
            class-count draw on ``[2, max_classes]``.
        imbalance_draws: Draw class shares explicitly instead of the 50/50 choice between
            quantile and random cut points.
        categorical_mode: ``"ordered"`` (today) or ``"unordered"``.
        missing_mode: ``"mcar"`` (today) or ``"mixed"``.
        tie_column_prob: Per numeric column probability of ties, a point mass or zero
            inflation.
        irrelevant_frac_max: Upper end of the share of the width added as permuted copies.
        duplicate_frac_max: Upper end of the share of the width added as near-duplicates.
        label_flip_max: Upper end of the per-task label flip rate.
    """

    feature_ratio_bands: tuple[tuple[float, float, float], ...] = ()
    p_binary: float | None = None
    imbalance_draws: bool = False
    categorical_mode: str = "ordered"
    missing_mode: str = "mcar"
    tie_column_prob: float = 0.0
    irrelevant_frac_max: float = 0.0
    duplicate_frac_max: float = 0.0
    label_flip_max: float = 0.0

    @property
    def unordered(self) -> bool:
        """Whether sources skip their ordered categorical coding for the knob layer's."""
        return self.categorical_mode == "unordered"

    @property
    def mixed_missing(self) -> bool:
        """Whether sources skip their own MCAR step for the knob layer's mechanisms."""
        return self.missing_mode == "mixed"


def knobs_from(cfg: Any) -> TaskKnobs | None:
    """Collect the knobs from a prior configuration, or ``None`` when every one is off.

    ``None`` is what keeps the default path byte-identical: every caller tests it and, when it
    is ``None``, runs exactly the code that ran before the knobs existed.

    Args:
        cfg: A :class:`~fintfm.prior.mixture.PriorConfig` (typed loosely to avoid an import
            cycle).

    Returns:
        The knobs, or ``None``.
    """
    knobs = TaskKnobs(
        feature_ratio_bands=tuple(tuple(float(v) for v in b) for b in cfg.feature_ratio_bands),
        p_binary=cfg.p_binary,
        imbalance_draws=cfg.imbalance_draws,
        categorical_mode=cfg.categorical_mode,
        missing_mode=cfg.missing_mode,
        tie_column_prob=cfg.tie_column_prob,
        irrelevant_frac_max=cfg.irrelevant_frac_max,
        duplicate_frac_max=cfg.duplicate_frac_max,
        label_flip_max=cfg.label_flip_max,
    )
    return None if knobs == TaskKnobs() else knobs


def check_knobs(knobs: TaskKnobs) -> None:
    """Refuse a malformed knob setting before any draw.

    Args:
        knobs: The knobs.

    Raises:
        ValueError: On an unknown mode, a probability or share outside ``[0, 1]``, or a band
            that is not ``(weight >= 0, 0 <= lo <= hi)`` with weights summing above zero.
    """
    if knobs.categorical_mode not in CATEGORICAL_MODES:
        raise ValueError(
            f"categorical_mode must be one of {list(CATEGORICAL_MODES)}, "
            f"got {knobs.categorical_mode!r}"
        )
    if knobs.missing_mode not in MISSING_MODES:
        raise ValueError(
            f"missing_mode must be one of {list(MISSING_MODES)}, got {knobs.missing_mode!r}"
        )
    unit = {
        "p_binary": knobs.p_binary if knobs.p_binary is not None else 0.0,
        "tie_column_prob": knobs.tie_column_prob,
        "irrelevant_frac_max": knobs.irrelevant_frac_max,
        "duplicate_frac_max": knobs.duplicate_frac_max,
        "label_flip_max": knobs.label_flip_max,
    }
    bad = {k: v for k, v in unit.items() if not (math.isfinite(v) and 0.0 <= v <= 1.0)}
    if bad:
        raise ValueError(f"knob values must lie in [0, 1], got {bad}")
    if knobs.feature_ratio_bands:
        for band in knobs.feature_ratio_bands:
            if len(band) != 3:
                raise ValueError(f"feature_ratio_bands entry {band} is not (weight, lo, hi)")
            w, lo, hi = band
            if not (w >= 0 and 0 <= lo <= hi and math.isfinite(hi)):
                raise ValueError(
                    f"feature_ratio_bands entry {band} needs weight >= 0 and 0 <= lo <= hi"
                )
        if sum(b[0] for b in knobs.feature_ratio_bands) <= 0:
            raise ValueError("feature_ratio_bands weights sum to zero")


def parse_feature_ratio_bands(spec: str) -> tuple[tuple[float, float, float], ...]:
    """Parse a ``--feature-ratio-bands`` value, ``"0.5:0:0.05,0.35:0.05:0.2"``.

    Args:
        spec: Comma-separated ``weight:lo:hi`` entries; empty for none.

    Returns:
        The bands, in the order written.

    Raises:
        ValueError: On an entry that is not three numbers.
    """
    out: list[tuple[float, float, float]] = []
    for raw in spec.split(","):
        entry = raw.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) != 3:
            raise ValueError(f"feature_ratio_bands entry {entry!r} is not weight:lo:hi")
        w, lo, hi = (float(v) for v in parts)
        out.append((w, lo, hi))
    return tuple(out)


def log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    """One draw from LogU(lo, hi).

    Args:
        rng: NumPy random generator.
        lo: Lower end, positive.
        hi: Upper end, at least ``lo``.

    Returns:
        The draw; ``lo`` itself when ``hi <= lo``, still consuming one draw.
    """
    u = rng.uniform(np.log(lo), np.log(max(hi, lo)))
    return float(np.exp(u))


def log_int(rng: np.random.Generator, lo: int, hi: int) -> int:
    """One draw from LogInt(lo, hi): log-uniform on ``[lo, hi + 1)``, floored.

    Args:
        rng: NumPy random generator.
        lo: Smallest value, at least 1.
        hi: Largest value.

    Returns:
        An integer in ``[lo, hi]``.
    """
    return int(min(hi, np.floor(log_uniform(rng, lo, hi + 1))))


def draw_width(
    rng: np.random.Generator, knobs: TaskKnobs, n_rows: int, min_features: int,
    max_features: int,
) -> tuple[int, int]:
    """Draw a task's width from :attr:`TaskKnobs.feature_ratio_bands`.

    Args:
        rng: NumPy random generator.
        knobs: Knobs with bands set.
        n_rows: The task's rows.
        min_features: The source's narrowest width.
        max_features: The source's widest width.

    Returns:
        ``(width, band)``: the width, clipped to ``[min_features, max_features]``, and the
        index of the band drawn.
    """
    bands = knobs.feature_ratio_bands
    weights = np.array([b[0] for b in bands], dtype=np.float64)
    band = int(np.searchsorted(np.cumsum(weights / weights.sum()), rng.random(), side="right"))
    band = min(band, len(bands) - 1)
    _, lo, hi = bands[band]
    ratio = rng.uniform(lo, hi)
    width = int(np.clip(round(ratio * n_rows), min_features, max_features))
    return width, band


def draw_extra_counts(
    rng: np.random.Generator, knobs: TaskKnobs, width: int
) -> tuple[int, int]:
    """Draw how many irrelevant and near-duplicate columns a task of ``width`` carries.

    Args:
        rng: NumPy random generator; one draw per knob that is on.
        knobs: The knobs.
        width: The width the shares are taken of.

    Returns:
        ``(n_irrelevant, n_duplicate)``.
    """
    n_irr = (
        round(rng.uniform(0.0, knobs.irrelevant_frac_max) * width)
        if knobs.irrelevant_frac_max > 0 else 0
    )
    n_dup = (
        round(rng.uniform(0.0, knobs.duplicate_frac_max) * width)
        if knobs.duplicate_frac_max > 0 else 0
    )
    return n_irr, n_dup


def _draw_class_count(rng: np.random.Generator, max_classes: int, knobs: TaskKnobs | None) -> int:
    """A task's class count: uniform on ``[2, max_classes]``, or binary with ``p_binary``.

    Args:
        rng: NumPy random generator.
        max_classes: Upper bound on classes.
        knobs: The knobs, or ``None`` for today's draw.

    Returns:
        The class count to cut into.
    """
    if knobs is None or knobs.p_binary is None:
        return int(rng.integers(2, max_classes + 1))
    if max_classes < 3 or rng.random() < knobs.p_binary:
        return 2
    return int(rng.integers(3, max_classes + 1))


def cut_classes(
    rng: np.random.Generator, t: np.ndarray, max_classes: int, knobs: TaskKnobs | None = None
) -> tuple[np.ndarray, int]:
    """Cut a continuous latent into dense class labels; the SCM and tree priors' shared step.

    With ``knobs`` ``None``, or with neither ``p_binary`` nor ``imbalance_draws`` set, this is
    exactly the code both priors carried inline before DR2.3: a uniform class count, then a
    fair coin between quantile cuts (balanced) and uniform random cut points (imbalanced), the
    degenerate-label repair, and a dense remap.

    Args:
        rng: NumPy random generator.
        t: ``(n_rows,)`` continuous latent.
        max_classes: Upper bound on classes.
        knobs: The knobs, or ``None``.

    Returns:
        ``(y, n_classes)``: labels in ``[0, n_classes)`` and the count actually present.
    """
    n_rows = len(t)
    n_classes = _draw_class_count(rng, max_classes, knobs)
    if knobs is not None and knobs.imbalance_draws:
        n_classes = max(2, min(n_classes, n_rows // MIN_ROWS_PER_CLASS))
        floor = MIN_ROWS_PER_CLASS / n_rows
        if n_classes == 2:
            minority = log_uniform(rng, floor, 0.5) if floor < 0.5 else 0.5
            shares = np.array([minority, 1.0 - minority])
            if rng.random() < 0.5:
                shares = shares[::-1]
        else:
            alpha = log_uniform(rng, 0.3, 3.0)
            shares = floor + (1.0 - n_classes * floor) * rng.dirichlet(np.full(n_classes, alpha))
        edges = np.quantile(t, np.cumsum(shares)[:-1])
    elif rng.random() < 0.5:
        # discretise target either by quantiles (balanced) or random cut points (imbalanced)
        edges = np.quantile(t, np.linspace(0, 1, n_classes + 1)[1:-1])
    else:
        edges = np.sort(rng.uniform(t.min(), t.max(), size=n_classes - 1))
    y = np.searchsorted(edges, t).astype(np.int64)
    present = np.unique(y)
    if len(present) < 2:
        y[rng.choice(n_rows, size=max(1, n_rows // 20), replace=False)] = (y[0] + 1) % n_classes
        present = np.unique(y)
    # remap to a dense label space so n_classes reflects classes actually present
    remap = {c: i for i, c in enumerate(present)}
    y = np.vectorize(remap.get)(y).astype(np.int64)
    return y, len(present)


def _standardise(v: np.ndarray) -> np.ndarray:
    """Z-score each column over its finite values, with NaN as 0 (the mean).

    Args:
        v: ``(n_rows, k)`` values.

    Returns:
        ``(n_rows, k)`` standardised values; a constant or all-missing column is all zero.
    """
    finite = np.isfinite(v)
    count = np.maximum(finite.sum(axis=0), 1)
    filled = np.where(finite, v, 0.0)
    mu = filled.sum(axis=0) / count
    sd = np.sqrt(np.where(finite, (v - mu) ** 2, 0.0).sum(axis=0) / count)
    return np.where(finite, (filled - mu) / np.where(sd > 1e-12, sd, 1.0), 0.0)


def _intercepts_for_rate(z: np.ndarray, rate: float) -> np.ndarray:
    """Bisect, per column, for ``b`` with ``mean(sigmoid(z[:, j] + b[j])) == rate``.

    Vectorised over columns, since a wide task otherwise pays a Python-level bisection per
    column, which was a third of the knob layer's generation time.

    Args:
        z: ``(n_rows, k)`` logits.
        rate: Target mean probability.

    Returns:
        ``(k,)`` intercepts.
    """
    lo, hi = np.full(z.shape[1], -40.0), np.full(z.shape[1], 40.0)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        below = (1.0 / (1.0 + np.exp(-(z + mid)))).mean(axis=0) < rate
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def _flip_labels(
    rng: np.random.Generator, y: np.ndarray, n_classes: int, rate: float
) -> tuple[np.ndarray, int, float]:
    """Flip labels to a uniformly drawn other class at ``rate``; keep the label space dense.

    Args:
        rng: NumPy random generator.
        y: Labels in ``[0, n_classes)``.
        n_classes: Classes present.
        rate: Flip probability per row.

    Returns:
        ``(y, n_classes, realised_rate)``. The original labels come back unchanged if the
        flips would leave a single class.
    """
    flip = rng.random(len(y)) < rate
    shift = rng.integers(1, n_classes, size=len(y)) if n_classes > 1 else np.zeros(len(y), int)
    out = np.where(flip, (y + shift) % n_classes, y).astype(np.int64)
    present = np.unique(out)
    if len(present) < 2:
        return y, n_classes, 0.0
    if len(present) < n_classes:
        remap = np.zeros(n_classes, dtype=np.int64)
        remap[present] = np.arange(len(present))
        out = remap[out]
    return out, len(present), float(flip.mean())


def _unordered_code(rng: np.random.Generator, col: np.ndarray) -> tuple[np.ndarray, int]:
    """Recode a numeric column as an unordered categorical with Zipf level frequencies.

    The levels are quantile bins of the column whose sizes follow a Zipf law, assigned to the
    bins in random order (so the most frequent level is not always the lowest values), and the
    codes are then permuted, so a code's size says nothing about the value behind it.

    Args:
        rng: NumPy random generator.
        col: ``(n_rows,)`` values; NaN stays NaN.

    Returns:
        ``(codes, levels)``: float codes in ``[0, levels)`` and the level count drawn.
    """
    k = log_int(rng, *CATEGORICAL_LEVELS)
    exponent = rng.uniform(0.5, 1.5)
    weights = 1.0 / np.arange(1, k + 1) ** exponent
    weights = weights[rng.permutation(k)]
    weights /= weights.sum()
    finite = np.isfinite(col)
    out = np.full_like(col, np.nan)
    if finite.any():
        edges = np.quantile(col[finite], np.cumsum(weights)[:-1])
        codes = np.searchsorted(edges, col[finite], side="right")
        out[finite] = rng.permutation(k)[np.minimum(codes, k - 1)]
    return out, k


def _tie_column(rng: np.random.Generator, col: np.ndarray) -> tuple[np.ndarray, str]:
    """Give a numeric column ties, a point mass or zero inflation.

    Args:
        rng: NumPy random generator.
        col: ``(n_rows,)`` values; NaN stays NaN.

    Returns:
        ``(column, kind)``.
    """
    kind = TIE_KINDS[int(rng.integers(len(TIE_KINDS)))]
    finite = np.isfinite(col)
    out = col.copy()
    if finite.sum() < 2:
        return out, kind
    v = col[finite]
    if kind == "ties":
        levels = log_int(rng, 3, 30)
        lo, span = float(v.min()), float(np.ptp(v))
        if span > 0:
            out[finite] = np.round((v - lo) / span * (levels - 1)) / (levels - 1) * span + lo
    elif kind == "point_mass":
        share = rng.uniform(0.05, 0.5)
        value = v[int(rng.integers(len(v)))]
        hit = finite & (rng.random(len(col)) < share)
        out[hit] = value
    else:
        share = rng.uniform(0.05, 0.6)
        side = 1.0 if rng.random() < 0.5 else -1.0
        cut = float(np.quantile(side * v, share))
        out[finite] = np.maximum(side * v - cut, 0.0)
    return out, kind


def _missing_mask(
    rng: np.random.Generator, X: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Draw ``"mixed"`` missingness for a feature matrix.

    Args:
        rng: NumPy random generator.
        X: ``(n_rows, n_features)`` values, NaN already missing.

    Returns:
        ``(mask, record)``: cells to blank, and what was drawn.
    """
    n_rows, n_features = X.shape
    mask = np.zeros(X.shape, dtype=bool)
    if rng.random() < 0.5 or n_features == 0:
        return mask, {"mechanism": "none"}
    mechanism = MISSING_MECHANISMS[int(rng.integers(len(MISSING_MECHANISMS)))]
    rate = log_uniform(rng, *MISSING_RATE_RANGE)
    share = rng.uniform(0.2, 1.0)
    cols = np.flatnonzero(rng.random(n_features) < share)
    if len(cols) == 0:
        cols = np.array([int(rng.integers(n_features))])
    k = len(cols)
    if mechanism == "mcar":
        mask[:, cols] = rng.random((n_rows, k)) < rate
        return mask, {"mechanism": mechanism, "rate": rate, "columns": k}
    if mechanism == "mar" and n_features > 1:
        drivers = rng.integers(n_features - 1, size=k)
        drivers += drivers >= cols  # any column but the one going missing
    else:
        drivers = cols  # MNAR, or MAR with no other column to lean on
    slope = rng.uniform(1.0, 3.0, size=k) * rng.choice([-1.0, 1.0], size=k)
    z = _standardise(X[:, drivers]) * slope
    p = 1.0 / (1.0 + np.exp(-(z + _intercepts_for_rate(z, rate))))
    mask[:, cols] = rng.random((n_rows, k)) < p
    return mask, {"mechanism": mechanism, "rate": rate, "columns": len(cols)}


def apply_task_knobs(
    rng: np.random.Generator,
    task: Task,
    knobs: TaskKnobs,
    max_features: int,
    extra_counts: tuple[int, int] | None = None,
    label_knobs: bool = True,
) -> Task:
    """Apply the column- and label-side knobs to a finished task from a general prior.

    In a fixed order, each consuming draws only when on: label flips, irrelevant and
    near-duplicate columns (then a column shuffle), unordered categoricals, ties / point masses
    / zero inflation, then mixed missingness. The source's own ordered categoricals and MCAR
    step were skipped when ``categorical_mode`` / ``missing_mode`` replace them, so nothing is
    applied twice.

    Args:
        rng: NumPy random generator.
        task: The task.
        knobs: The knobs.
        max_features: Width cap; added columns never exceed it.
        extra_counts: ``(n_irrelevant, n_duplicate)`` already drawn against a band width, or
            ``None`` to draw them here against the task's own width.
        label_knobs: Apply label flips. ``False`` for a regression task, whose label is binned
            later from the context.

    Returns:
        A new task, with :attr:`~fintfm.prior.base.Task.meta` recording each draw.
    """
    X = task.X.astype(np.float64, copy=True)
    y, n_classes = task.y, task.n_classes
    is_cat = np.asarray(task.is_categorical, dtype=bool).copy()
    record: dict[str, Any] = {}
    if label_knobs and knobs.label_flip_max > 0 and task.y_continuous is None:
        rate = rng.uniform(0.0, knobs.label_flip_max)
        y, n_classes, realised = _flip_labels(rng, y, n_classes, rate)
        record["label_flip_rate"] = realised
    n_irr, n_dup = (
        extra_counts if extra_counts is not None
        else draw_extra_counts(rng, knobs, X.shape[1])
    )
    room = max(0, max_features - X.shape[1])
    n_irr = min(n_irr, room)
    n_dup = min(n_dup, room - n_irr)
    added: list[np.ndarray] = []
    added_cat: list[bool] = []
    n_rows = X.shape[0]
    for _ in range(n_irr):
        j = int(rng.integers(X.shape[1]))
        added.append(X[rng.permutation(n_rows), j])
        added_cat.append(bool(is_cat[j]))
    for _ in range(n_dup):
        j = int(rng.integers(X.shape[1]))
        src = X[:, j]
        if is_cat[j]:
            added.append(src.copy())
        else:
            spread = float(np.nanstd(src)) if np.isfinite(src).any() else 0.0
            added.append(src + rng.normal(0.0, log_uniform(rng, 0.01, 0.2) * spread + 1e-12,
                                          n_rows))
        added_cat.append(bool(is_cat[j]))
    origin = np.array(["base"] * X.shape[1] + ["irrelevant"] * n_irr + ["duplicate"] * n_dup)
    if added:
        X = np.concatenate([X, np.stack(added, axis=1)], axis=1)
        is_cat = np.concatenate([is_cat, np.array(added_cat, dtype=bool)])
        perm = rng.permutation(X.shape[1])
        X, is_cat, origin = X[:, perm], is_cat[perm], origin[perm]
    record["n_irrelevant"], record["n_duplicate"] = n_irr, n_dup
    record["origin"] = origin.tolist()
    if knobs.unordered:
        levels: list[int] = []
        if rng.random() < 0.5:
            share = rng.uniform(0.05, 0.5)
            for j in np.flatnonzero(~is_cat & (rng.random(X.shape[1]) < share)):
                X[:, j], k = _unordered_code(rng, X[:, j])
                is_cat[j] = True
                levels.append(k)
        record["categorical_levels"] = levels
    if knobs.tie_column_prob > 0:
        kinds: list[str] = []
        numeric = np.flatnonzero(~is_cat)
        for j in numeric[rng.random(len(numeric)) < knobs.tie_column_prob]:
            X[:, j], kind = _tie_column(rng, X[:, j])
            kinds.append(kind)
        record["tie_kinds"] = kinds
        record["numeric_columns"] = len(numeric)
    if knobs.mixed_missing:
        mask, record["missing"] = _missing_mask(rng, X)
        X[mask] = np.nan
    meta = dict(task.meta or {})
    meta["knobs"] = record
    return Task(
        X=X.astype(np.float32),
        y=y,
        n_classes=n_classes,
        is_categorical=is_cat,
        source=task.source,
        period=task.period,
        n_horizons=task.n_horizons,
        y_continuous=task.y_continuous,
        meta=meta,
    )
