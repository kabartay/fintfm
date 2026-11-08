"""Time-based (out-of-time) train/test splitting that refuses identifier leakage.

``fintfm.experiments.v4_out_of_time.time_split`` only checks that the *year windows* do not
overlap. A firm observed in 2010 and again in 2019 still lands on both sides of such a split,
which is the leakage ``openspec/changes/time-based-evaluation`` exists to remove. This module
adds the identifier check, and is the splitter the benchmark harness uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

OnOverlap = Literal["raise", "drop_from_train"]


@dataclass(frozen=True)
class TimeSplit:
    """An out-of-time split and what it took to make it leak-free.

    Attributes:
        train: Row indices of the training window (after any identifier exclusion).
        test: Row indices of the test window; never altered by an exclusion.
        n_shared_ids: Identifiers present in both windows before any exclusion.
        n_train_dropped: Training rows removed because their identifier is in the test window.
    """

    train: np.ndarray
    test: np.ndarray
    n_shared_ids: int
    n_train_dropped: int


def time_based_split(
    period: np.ndarray,
    groups: np.ndarray,
    train_until: float,
    test_from: float,
    on_overlap: OnOverlap = "raise",
) -> TimeSplit:
    """Split rows into an earlier training window and a later test window.

    Args:
        period: Observation period per row (year, date ordinal, ...), shape ``(n,)``, no NaN.
        groups: Identifier per row (firm, borrower), shape ``(n,)``. Required: without it the
            absence of leakage cannot be checked, so the split is refused.
        train_until: Last period (inclusive) allowed in training.
        test_from: First period (inclusive) allowed in test; must exceed ``train_until``.
        on_overlap: What to do when an identifier appears in both windows. ``"raise"``
            (default) refuses. ``"drop_from_train"`` removes that identifier's training rows,
            keeps the test window intact and reports the count in the result.

    Returns:
        A :class:`TimeSplit`. Its two index sets share no row and no identifier.

    Raises:
        ValueError: If shapes disagree, ``period`` has NaN, the windows overlap, either window
            is empty, ``on_overlap`` is unknown, or an identifier appears in both windows and
            ``on_overlap == "raise"``.
    """
    period = np.asarray(period)
    groups = np.asarray(groups)
    if period.shape != groups.shape or period.ndim != 1:
        raise ValueError(f"period {period.shape} and groups {groups.shape} must be equal 1-D shapes")
    if on_overlap not in ("raise", "drop_from_train"):
        raise ValueError(f"unknown on_overlap {on_overlap!r}")
    if np.issubdtype(period.dtype, np.floating) and np.isnan(period).any():
        raise ValueError("period contains NaN; every row needs an observation period")
    if test_from <= train_until:
        raise ValueError(
            f"windows overlap (train<={train_until}, test>={test_from}); an overlapping "
            "split is not out-of-time validation"
        )
    train = np.flatnonzero(period <= train_until)
    test = np.flatnonzero(period >= test_from)
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"empty window: {len(train)} train rows, {len(test)} test rows")
    shared = np.intersect1d(groups[train], groups[test])
    dropped = 0
    if len(shared):
        if on_overlap == "raise":
            raise ValueError(
                f"{len(shared)} identifier(s) appear in both windows (e.g. {shared[:3].tolist()}); "
                "refusing a split that leaks entities across the boundary. Pass "
                "on_overlap='drop_from_train' to exclude them from training."
            )
        keep = ~np.isin(groups[train], shared)
        dropped = int((~keep).sum())
        train = train[keep]
        if len(train) == 0:
            raise ValueError("every training row shares an identifier with the test window")
    return TimeSplit(train=train, test=test, n_shared_ids=len(shared), n_train_dropped=dropped)
