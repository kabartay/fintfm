"""Real firm-quarter tasks sampled from SEC EDGAR (FinTFM-R, task A2).

Why this is a *sampler*, not a *generator*
-------------------------------------------

Every other prior in this package (`financial.py`, `scm.py`, `tree.py`) manufactures a fresh
synthetic world on every call: a new coefficient vector, a new sector structure, a new
generative process. That is what gives the synthetic prior its diversity across tasks — and
it is not available here. SEC EDGAR is one fixed, finite real world; there is nothing to
regenerate. What this module does instead is what `FinancialTFMClassifier` already does at
**inference** time on a real panel: draw a random subset of real rows as the in-context
"training" evidence for one task, and a different random subset as the next task's evidence.
Training the model on resampled real panels is training it to do the same kind of in-context
learning it will face when scored on one -- not a new capability, the existing one applied to
pretraining instead of only inference.

What this module does not do (yet)
-----------------------------------

No survival framing (`Task.period`) -- the panel carries a single forward-looking binary
label (did this filer disclose an Item 1.03 bankruptcy/receivership within the label horizon
the panel-building pipeline used), not a horizon grid. `scripts/edgar/build_panel.py` builds
that panel; this module only samples from it. See `openspec/changes/fintfm-r/` for the full
roadmap (task A2) and `docs/design/DECISIONS.md` D17 for why this family exists at all and
what it costs.

The real data file itself never enters this repository or its git history, matching
`CLAUDE.md`'s convention for trained weights -- see `.gitignore`.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from fintfm.prior.base import Task

#: Columns that identify a row rather than describe it. Never sampled as features.
_BOOKKEEPING_COLUMNS = ("cik", "period_end", "label")

#: Default location a built panel is expected at. Overridable per call so tests can inject a
#: small fixture panel without touching this path, and so a real run can point at wherever
#: `scripts/edgar/build_panel.py` wrote its output.
DEFAULT_PANEL_PATH = "data/edgar/panel.parquet"


@functools.lru_cache(maxsize=4)
def _load_panel_cached(path: str) -> pd.DataFrame:
    """Read a panel parquet file once per path and cache it for the process lifetime.

    Args:
        path: Filesystem path to a parquet file built by `scripts/edgar/build_panel.py`.

    Returns:
        The panel, unmodified.

    Raises:
        FileNotFoundError: If no file exists at `path` -- raised here rather than inside
            `sample_real_edgar_task` so a misconfigured `--p-real-edgar` run fails at startup,
            not 500 steps in.
    """
    return pd.read_parquet(path)


def feature_columns(panel: pd.DataFrame) -> list[str]:
    """Every column in `panel` that is a feature rather than bookkeeping.

    Args:
        panel: A panel DataFrame, real or a test fixture, carrying the same schema
            `scripts/edgar/build_panel.py` writes.

    Returns:
        Feature column names, in the panel's own column order -- stable across calls so two
        tasks drawn from the same panel expose the same columns in the same positions, which
        is what column-identity invariance (D4) and consistent normalisation both need.
    """
    return [c for c in panel.columns if c not in _BOOKKEEPING_COLUMNS]


def sample_real_edgar_task(
    rng: np.random.Generator,
    n_rows: int,
    max_features: int = 136,
    min_features: int = 4,
    panel: pd.DataFrame | None = None,
    panel_path: str = DEFAULT_PANEL_PATH,
) -> Task:
    """Sample one task from real SEC EDGAR firm-quarters.

    Args:
        rng: NumPy random generator.
        n_rows: Rows to sample for this task.
        max_features: Upper bound this checkpoint's architecture accepts. The panel's feature
            set is fixed by `scripts/edgar/build_panel.py`, not chosen per task the way the
            synthetic priors choose a random subset -- so this is a hard ceiling checked once,
            never a sampling knob. Raising rather than truncating matches `collate()`'s own
            `n_features <= max_features` invariant: a silent truncation here would train the
            model on a feature set inference cannot reproduce.
        min_features: Lower bound, checked for the same reason -- a panel built with too few
            feature columns is a misconfiguration to catch at startup, not a sampling failure
            to catch per task.
        panel: An already-loaded panel, for tests and for callers that manage their own
            caching. When `None`, `panel_path` is read (and cached) instead.
        panel_path: Where to read the panel from when `panel` is not given directly. See
            `DEFAULT_PANEL_PATH`.

    Returns:
        A binary `Task` with `source="real_edgar"`, no survival framing (`period=None`).

    Raises:
        ValueError: If the panel's feature count is outside `[min_features, max_features]`,
            or if `n_rows` exceeds the panel's own row count (sampling without replacement
            must not silently fall back to sampling the same real row twice into one task).
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
        # A sampled task with only one class present is unscorable (no AUC, no ranking
        # signal) rather than merely hard. Matches the same guarantee the synthetic priors
        # make in `_finish_scm_task` and elsewhere: a task this function returns is always
        # scorable, and a caller never has to discover a degenerate label space mid-batch.
        flip = rng.choice(n_rows, size=max(1, n_rows // 20), replace=False)
        y[flip] = 1 - y[0]
    return Task(
        X=X.astype(np.float32),
        y=y,
        n_classes=2,
        is_categorical=np.zeros(len(cols), dtype=bool),
        source="real_edgar",
    )
