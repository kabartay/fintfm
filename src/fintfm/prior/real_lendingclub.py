"""Real loan-level tasks sampled from LendingClub's granting-model dataset (D17's third source).

Why this is a *sampler*, not a *generator*
-------------------------------------------

Same reasoning as `fintfm.prior.real_edgar` and `fintfm.prior.real_mortgage`, which this
module mirrors field for field: LendingClub is one fixed, finite real world, so there is
nothing to regenerate. This module draws a random subset of real loans as one task's
in-context evidence, the same way `FinancialTFMClassifier` draws a real panel at inference
time.

Provenance, checked before this file was written
--------------------------------------------------

Source: "Lending Club loan dataset for granting models" (Ariza-Garzon, Sanz-Guerrero, Arroyo
Gallardo, Universidad Complutense de Madrid), Zenodo record 11295916, CC-BY-4.0 -- clean,
redistribution-friendly, attribution only. Absent from TabArena-v0.1's 51 datasets, unlike the
UCI/Kaggle credit sets ruled out for training (German Credit, Taiwan Credit Card, Give Me Some
Credit, FICO HELOC, Polish/Taiwanese bankruptcy) -- **but present in BeyondArena as
`lending_club_1m`**, found only after this file was first written: a FinTFM-R checkpoint
trained on this source cannot be scored cleanly on that BeyondArena task, and that task must
be excluded from any FinTFM-R BeyondArena number (`docs/research/data_candidates.csv`, the
same bounded-forfeit pattern D17 records for EDGAR). Unlike the raw Kaggle
LendingClub dump, this release is already filtered to application-time-only variables by its
own authors -- no post-approval leakage to catch by hand.

What this module does not do (yet)
-----------------------------------

No survival framing (`Task.period`) -- the panel carries a single binary label (the source's
own `Default` column), not a horizon grid. `scripts/lendingclub/build_panel.py` builds that
panel; this module only samples from it.

The real data file itself never enters this repository or its git history, matching
`CLAUDE.md`'s convention for trained weights -- see `.gitignore` (`data/lendingclub/`).
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from fintfm.prior.base import Task

#: Columns that identify a row rather than describe it. Never sampled as features.
_BOOKKEEPING_COLUMNS = ("loan_id", "label")

#: Default location a built panel is expected at. Overridable per call so tests can inject a
#: small fixture panel without touching this path, and so a real run can point at wherever
#: `scripts/lendingclub/build_panel.py` wrote its output.
DEFAULT_PANEL_PATH = "data/lendingclub/panel.parquet"


@functools.lru_cache(maxsize=4)
def _load_panel_cached(path: str) -> pd.DataFrame:
    """Read a panel parquet file once per path and cache it for the process lifetime.

    Args:
        path: Filesystem path to a parquet file built by `scripts/lendingclub/build_panel.py`.

    Returns:
        The panel, unmodified.

    Raises:
        FileNotFoundError: If no file exists at `path` -- raised here rather than inside
            `sample_real_lendingclub_task` so a misconfigured `--p-real-lendingclub` run fails
            at startup, not 500 steps in.
    """
    return pd.read_parquet(path)


def feature_columns(panel: pd.DataFrame) -> list[str]:
    """Every column in `panel` that is a feature rather than bookkeeping.

    Args:
        panel: A panel DataFrame, real or a test fixture, carrying the same schema
            `scripts/lendingclub/build_panel.py` writes.

    Returns:
        Feature column names, in the panel's own column order -- stable across calls so two
        tasks drawn from the same panel expose the same columns in the same positions.
    """
    return [c for c in panel.columns if c not in _BOOKKEEPING_COLUMNS]


def sample_real_lendingclub_task(
    rng: np.random.Generator,
    n_rows: int,
    max_features: int = 136,
    min_features: int = 4,
    panel: pd.DataFrame | None = None,
    panel_path: str = DEFAULT_PANEL_PATH,
) -> Task:
    """Sample one task from real LendingClub loans.

    Args:
        rng: NumPy random generator.
        n_rows: Rows to sample for this task.
        max_features: Upper bound this checkpoint's architecture accepts. The panel's feature
            set is fixed by `scripts/lendingclub/build_panel.py`, not chosen per task -- a hard
            ceiling checked once, never a sampling knob. Raising rather than truncating matches
            `collate()`'s own `n_features <= max_features` invariant.
        min_features: Lower bound, checked for the same reason.
        panel: An already-loaded panel, for tests and for callers that manage their own
            caching. When `None`, `panel_path` is read (and cached) instead.
        panel_path: Where to read the panel from when `panel` is not given directly. See
            `DEFAULT_PANEL_PATH`.

    Returns:
        A binary `Task` with `source="real_lendingclub"`, no survival framing (`period=None`).

    Raises:
        ValueError: If the panel's feature count is outside `[min_features, max_features]`,
            or if `n_rows` exceeds the panel's own row count (sampling without replacement
            must not silently fall back to sampling the same real loan twice into one task).
    """
    df = panel if panel is not None else _load_panel_cached(panel_path)
    cols = feature_columns(df)
    if not (min_features <= len(cols) <= max_features):
        raise ValueError(
            f"panel has {len(cols)} feature columns, expected between {min_features} and "
            f"{max_features} -- rebuild the panel or adjust the checkpoint's max_features"
        )
    if n_rows > len(df):
        raise ValueError(
            f"requested n_rows={n_rows} exceeds the panel's {len(df)} rows; sampling without "
            "replacement cannot supply more distinct real rows than the panel has"
        )
    idx = rng.choice(len(df), size=n_rows, replace=False)
    rows = df.iloc[idx]
    X = rows[cols].to_numpy(dtype=np.float64)
    y = rows["label"].to_numpy(dtype=np.int64, copy=True)
    present = np.unique(y)
    if len(present) < 2:
        # A sampled task with only one class present is unscorable -- the same guarantee
        # real_edgar.py/real_mortgage.py make. Much less likely to fire here than for the
        # mortgage source: this panel's base rate is ~20%, not ~0.05%.
        flip = rng.choice(n_rows, size=max(1, n_rows // 20), replace=False)
        y[flip] = 1 - y[0]
    return Task(
        X=X.astype(np.float32),
        y=y,
        n_classes=2,
        is_categorical=np.zeros(len(cols), dtype=bool),
        source="real_lendingclub",
    )
