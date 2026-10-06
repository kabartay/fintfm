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
            most tasks would fall to the label-flip repair below -- random labels presented as
            real ones. Enriching a task's base rate is safe here for the same reason it is in
            the synthetic prior, and the resulting rate (2/256, ~0.8%) is inside V4FinBench's
            own regime.

    Returns:
        A binary `Task`, `source` from :func:`source_name`, no survival framing.

    Raises:
        ValueError: If the feature count is outside ``[min_features, max_features]`` or
            `n_rows` exceeds the panel's rows.
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
    idx = rng.choice(len(df), size=n_rows, replace=False)
    if min_positives > 0:
        labels = df["label"].to_numpy(dtype=np.int64)
        short = min_positives - int(labels[idx].sum())
        if short > 0:
            pool = np.setdiff1d(np.flatnonzero(labels == 1), idx, assume_unique=True)
            short = min(short, len(pool))
            negatives = np.flatnonzero(labels[idx] == 0)
            if short > 0 and len(negatives) >= short:
                idx = idx.copy()
                idx[rng.choice(negatives, size=short, replace=False)] = rng.choice(
                    pool, size=short, replace=False
                )
    rows = df.iloc[idx]
    X = rows[cols].to_numpy(dtype=np.float64)
    y = rows["label"].to_numpy(dtype=np.int64, copy=True)
    if len(np.unique(y)) < 2:
        # Same guarantee every other prior makes: a returned task is always scorable. Fires
        # often on the rarest sources (card fraud at 0.17%, mortgages at 0.05%) at small n_rows,
        # which is a reason to weight those sources low or draw them with larger contexts.
        flip = rng.choice(n_rows, size=max(1, n_rows // 20), replace=False)
        y[flip] = 1 - y[0]
    return Task(
        X=X.astype(np.float32),
        y=y,
        n_classes=2,
        is_categorical=np.zeros(len(cols), dtype=bool),
        source=source_name(panel_path),
    )
