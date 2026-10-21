"""Real loan-level tasks sampled from LendingClub's granting-model dataset (D17's third source).

**Retired (AU0.1, 2026-10-20; `docs/design/DECISIONS.md` D19).** LendingClub is BeyondArena's
``lending_club_1m``, and this project never trains on a TabArena or BeyondArena dataset. The
bounded forfeit recorded below (exclude that one task from any FinTFM-R BeyondArena number) is
withdrawn: :func:`sample_real_lendingclub_task` raises :class:`RetiredSourceError`, and
``p_real_lendingclub`` above zero or a ``real_lendingclub`` source weight is refused before any
draw (:func:`fintfm.prior.mixture._check_levers`). The §157 pilot arm that trained on it is not
held out on that task. The text below is the module as it was written.

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

import numpy as np
import pandas as pd

from fintfm.prior.base import Task

#: Why every draw from this source is refused (AU0.1).
RETIRED = (
    "LendingClub is retired as a training source (AU0.1, 2026-10-20; DECISIONS D19): it is "
    "BeyondArena's lending_club_1m, and this project never trains on a TabArena or BeyondArena "
    "dataset"
)


class RetiredSourceError(ValueError):
    """Raised on any attempt to draw a training task from a retired real source."""


#: Columns that identify a row rather than describe it. Never sampled as features.
_BOOKKEEPING_COLUMNS = ("loan_id", "label")

#: Default location a built panel is expected at. Overridable per call so tests can inject a
#: small fixture panel without touching this path, and so a real run can point at wherever
#: `scripts/lendingclub/build_panel.py` wrote its output.
DEFAULT_PANEL_PATH = "data/lendingclub/panel.parquet"


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
    min_positives: int = 0,
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

        min_positives: Positive floor per task, met by swapping in real positive rows, never
            by editing labels. Every task carries at least one real row of each class whatever
            this is; see :func:`fintfm.prior.real_panel.ensure_both_classes`.

    Returns:
        A binary `Task` with `source="real_lendingclub"`, no survival framing (`period=None`).

    Raises:
        RetiredSourceError: Always (AU0.1); the source is retired, and the checks below are
            the ones it made while it was live.
        ValueError: If the panel's feature count is outside `[min_features, max_features]`,
            or if `n_rows` exceeds the panel's own row count (sampling without replacement
            must not silently fall back to sampling the same real loan twice into one task), or if
            the panel cannot supply a real row of each class.
    """
    raise RetiredSourceError(RETIRED)
