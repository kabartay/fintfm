"""Real tasks sampled from any panel that follows the shared panel convention (FinTFM-R).

Why one generic sampler rather than one module per source
----------------------------------------------------------

`real_edgar.py`, `real_mortgage.py` and `real_lendingclub.py` differ only in the names of their
bookkeeping columns and the `source` label they stamp on a `Task` -- three copies of the same
twenty lines. The candidate pipeline (`scripts/candidates/build_panels.py`) writes every further
source to one convention: float feature columns plus an integer `label`. Adding a source is now a
builder function and a weight, not a new module, a new `PriorConfig` field pair, a new CLI flag
pair, and a new provenance exemption. The three existing modules stay as they are -- every
checkpoint and test that names them keeps working -- and this sampler also reads their panels,
since it recognises their bookkeeping columns too.

How a panel is chosen
---------------------

`PriorConfig.real_panels` is a spec string, ``"path=prob,path=prob"``. In
:func:`fintfm.prior.mixture.sample_task` each entry is checked in order, the same independent
"draw from this source with probability p" rule the dedicated `p_real_*` fields use; a draw that
misses every entry falls through to the synthetic priors. The empty default never parses a path
and never opens a file, so FinTFM (synthetic-only) is untouched.

The real data files never enter this repository -- see `.gitignore` (`data/real/`) and
`docs/research/data_candidates.csv` for every source's licence and provenance check.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

from fintfm.prior.base import Task

#: Columns that identify a row rather than describe it: this convention's `label`, plus the
#: bookkeeping columns the three dedicated real-source panels already use. Any column whose
#: name starts with an underscore is bookkeeping too, so a builder can carry an identifier
#: without it becoming a feature.
_BOOKKEEPING = frozenset({"label", "loan_id", "cik", "period_end"})

#: The column that identifies a firm in a panel with several rows per firm (EDGAR's filer id).
#: When present, a task takes at most one row per firm; see :func:`draw_rows`.
GROUP_COLUMN = "cik"


@functools.lru_cache(maxsize=16)
def load_panel(path: str) -> pd.DataFrame:
    """Read a panel parquet file once per path and cache it for the process lifetime.

    Args:
        path: Filesystem path to a panel parquet file.

    Returns:
        The panel, unmodified.

    Raises:
        FileNotFoundError: If no file exists at `path`.
    """
    return pd.read_parquet(path)


@functools.lru_cache(maxsize=16)
def parse_real_panels(spec: str) -> tuple[tuple[str, float], ...]:
    """Parse a ``"path=prob,path=prob"`` spec into validated ``(path, prob)`` pairs.

    Args:
        spec: Comma-separated ``path=probability`` entries. Empty means no real panels.

    Returns:
        The entries, in spec order.

    Raises:
        ValueError: On a malformed entry, a probability outside ``(0, 1]``, or a path listed
            twice -- each a misconfiguration better caught at startup than mid-run.
    """
    entries: list[tuple[str, float]] = []
    for raw in filter(None, (s.strip() for s in spec.split(","))):
        path, sep, prob = raw.rpartition("=")
        if not sep or not path:
            raise ValueError(f"real_panels entry {raw!r} is not of the form path=probability")
        p = float(prob)
        if not 0.0 < p <= 1.0:
            raise ValueError(f"real_panels probability for {path!r} must be in (0, 1], got {p}")
        if any(path == seen for seen, _ in entries):
            raise ValueError(f"real_panels lists {path!r} twice")
        entries.append((path, p))
    return tuple(entries)


def feature_columns(panel: pd.DataFrame) -> list[str]:
    """Every column in `panel` that is a feature rather than bookkeeping, in panel order.

    Args:
        panel: A panel DataFrame.

    Returns:
        Feature column names -- stable across calls, so every task drawn from one panel exposes
        the same columns in the same positions.
    """
    return [c for c in panel.columns if c not in _BOOKKEEPING and not c.startswith("_")]


def source_name(path: str) -> str:
    """The `Task.source` label for a panel: ``"real:"`` plus its directory name.

    Args:
        path: The panel's path, conventionally ``data/real/<source>/panel.parquet``.

    Returns:
        E.g. ``"real:sba"``.
    """
    return f"real:{Path(path).parent.name}"


def draw_rows(
    rng: np.random.Generator, n_panel: int, n_rows: int, groups: np.ndarray | None = None
) -> np.ndarray:
    """Draw a task's rows without replacement, at most one per group when groups are given.

    Without ``groups`` this is exactly ``rng.choice(n_panel, n_rows, replace=False)``, the
    draw every real sampler made before audit AU1.15.

    With ``groups`` (EDGAR's ``cik``), no group contributes two rows. A task's context/query
    boundary is drawn per batch, after the task exists (:func:`fintfm.prior.mixture.sample_batch`),
    so the sampler cannot split firms across a boundary it does not know; one row per firm
    makes *every* boundary firm-disjoint, which is the property
    :func:`fintfm.inference.classifier._finetune_rows` enforces for fine-tuning. Without it,
    73% of EDGAR rows share a firm with another row, and a task holding the same firm in
    context and query rewards lookup over inference. Rows are taken in a random order and a
    row is skipped when its firm is already in, so the draw stays as close to row-uniform as
    the constraint allows.

    Args:
        rng: NumPy random generator.
        n_panel: Rows in the panel.
        n_rows: Rows wanted.
        groups: Optional ``(n_panel,)`` group label per row.

    Returns:
        ``(n_rows,)`` row indices into the panel.

    Raises:
        ValueError: If the panel has fewer than ``n_rows`` groups.
    """
    if groups is None:
        return rng.choice(n_panel, size=n_rows, replace=False)
    # A candidate draw twice the size almost always holds enough distinct firms; the full
    # permutation is the fallback, not the common path, since it costs O(panel) per task.
    for candidates in (
        rng.choice(n_panel, size=min(n_panel, 2 * n_rows), replace=False),
        rng.permutation(n_panel),
    ):
        _, first = np.unique(groups[candidates], return_index=True)
        if len(first) >= n_rows:
            return candidates[np.sort(first)[:n_rows]]
    raise ValueError(
        f"requested n_rows={n_rows} exceeds the panel's {len(first)} distinct {GROUP_COLUMN} "
        "values; a task takes one row per firm so that no firm sits in both context and query"
    )


def _swap_in_class(
    rng: np.random.Generator,
    labels: np.ndarray,
    idx: np.ndarray,
    cls: int,
    floor: int,
    groups: np.ndarray | None,
) -> np.ndarray:
    """Raise the count of class ``cls`` in ``idx`` to ``floor`` with real rows from the panel.

    Rows of the other class are evicted at random and replaced by rows of ``cls`` that are not
    already in the task (and, with ``groups``, whose group is not already in it either). Labels
    are never edited, so every label stays the one the panel recorded for that row's features.

    Args:
        rng: NumPy random generator.
        labels: ``(n_panel,)`` binary labels of the whole panel.
        idx: The task's current row indices.
        cls: The class to top up.
        floor: Rows of ``cls`` wanted. Reached only as far as the panel can supply.
        groups: Optional group label per panel row, kept one-per-task.

    Returns:
        The (possibly) updated row indices; ``idx`` itself is not modified.
    """
    short = floor - int((labels[idx] == cls).sum())
    if short <= 0:
        return idx
    pool = np.flatnonzero(labels == cls)
    if groups is None:
        pool = np.setdiff1d(pool, idx, assume_unique=True)
    else:
        pool = pool[~np.isin(groups[pool], groups[idx])]
        pool = pool[rng.permutation(len(pool))]
        _, first = np.unique(groups[pool], return_index=True)
        pool = pool[np.sort(first)]
    evictable = np.flatnonzero(labels[idx] != cls)
    short = min(short, len(pool), len(evictable))
    if short <= 0:
        return idx
    idx = idx.copy()
    idx[rng.choice(evictable, size=short, replace=False)] = rng.choice(
        pool, size=short, replace=False
    )
    return idx


def ensure_both_classes(
    rng: np.random.Generator,
    labels: np.ndarray,
    idx: np.ndarray,
    min_positives: int = 0,
    groups: np.ndarray | None = None,
    name: str = "panel",
) -> np.ndarray:
    """Give a real task its positive floor and both classes, by resampling rows, never labels.

    Every real sampler used to repair a single-class draw by flipping 5% of its labels at
    random. On a rare-label panel that was the common case, not the edge case -- most 256-row
    mortgage tasks at the 0.048% rate, about 3% of 256-row EDGAR tasks -- so the model was
    trained on labels unrelated to the features beside them (audit AU1.14). Here the task is
    topped up with real rows instead: first to ``max(min_positives, 1)`` positives (capped at
    ``len(idx) - 1`` so a negative can remain), then to at least one negative.

    Args:
        rng: NumPy random generator.
        labels: ``(n_panel,)`` binary 0/1 labels of the whole panel.
        idx: The task's drawn row indices.
        min_positives: Positive floor; values below one still guarantee one positive.
        groups: Optional group label per panel row; swapped-in rows keep one row per group.
        name: Panel name for the error message.

    Returns:
        Row indices holding real rows of both classes.

    Raises:
        ValueError: If the panel cannot supply a row of each class -- a misconfigured panel,
            which a random relabelling would have hidden.
    """
    floor = min(max(min_positives, 1), len(idx) - 1)
    idx = _swap_in_class(rng, labels, idx, 1, floor, groups)
    idx = _swap_in_class(rng, labels, idx, 0, 1, groups)
    if len(np.unique(labels[idx])) < 2:
        raise ValueError(
            f"{name}: cannot draw a task with both classes from real rows; the panel holds "
            f"{int((labels == 1).sum())} positives and {int((labels == 0).sum())} negatives"
            + ("" if groups is None else " (one row per firm)")
        )
    return idx


def sample_real_panel_task(
    rng: np.random.Generator,
    n_rows: int,
    panel_path: str,
    max_features: int = 136,
    min_features: int = 4,
    panel: pd.DataFrame | None = None,
    min_positives: int = 0,
) -> Task:
    """Sample one binary task from a real panel.

    Args:
        rng: NumPy random generator.
        n_rows: Rows to sample for this task, without replacement.
        panel_path: Where the panel lives; also determines the task's `source` label.
        max_features: Upper bound the checkpoint's architecture accepts -- a hard ceiling,
            raised rather than truncated, matching `collate()`'s own invariant.
        min_features: Lower bound, checked for the same reason.
        panel: An already-loaded panel, for tests; when `None`, `panel_path` is read (cached).
        min_positives: Guarantee at least this many *real* positive rows per task by swapping
            randomly chosen negatives for positives drawn from the panel's own positive pool.
            The real-data counterpart of the synthetic prior's `min_expected_positives` floor:
            at a 256-row context a 0.17%-rate source averages under one positive, so without it
            most tasks would be single-class. Enriching a task's base rate is safe here for the
            same reason it is in the synthetic prior, and the resulting rate (2/256, ~0.8%) is
            inside V4FinBench's own regime. Whatever its value, every task carries at least one
            real row of each class; see :func:`ensure_both_classes`.

    Returns:
        A binary `Task`, `source` from :func:`source_name`, no survival framing. When the
        panel carries a ``cik`` column, no firm appears twice in the task (see
        :func:`draw_rows`).

    Raises:
        ValueError: If the feature count is outside ``[min_features, max_features]``,
            `n_rows` exceeds the panel's rows (or firms), or the panel cannot supply a real
            row of each class.
    """
    df = panel if panel is not None else load_panel(panel_path)
    cols = feature_columns(df)
    if not (min_features <= len(cols) <= max_features):
        raise ValueError(
            f"panel {panel_path!r} has {len(cols)} feature columns, expected between "
            f"{min_features} and {max_features} -- rebuild the panel or raise max_features"
        )
    if n_rows > len(df):
        raise ValueError(
            f"requested n_rows={n_rows} exceeds the panel's {len(df)} rows; sampling without "
            "replacement cannot supply more distinct real rows than the panel has"
        )
    groups = df[GROUP_COLUMN].to_numpy() if GROUP_COLUMN in df.columns else None
    idx = draw_rows(rng, len(df), n_rows, groups=groups)
    labels = df["label"].to_numpy(dtype=np.int64)
    idx = ensure_both_classes(rng, labels, idx, min_positives, groups=groups, name=panel_path)
    rows = df.iloc[idx]
    X = rows[cols].to_numpy(dtype=np.float64)
    y = rows["label"].to_numpy(dtype=np.int64, copy=True)
    return Task(
        X=X.astype(np.float32),
        y=y,
        n_classes=2,
        is_categorical=np.zeros(len(cols), dtype=bool),
        source=source_name(panel_path),
    )
