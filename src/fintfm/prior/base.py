"""Common task container and batching helpers for synthetic priors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Task:
    """One synthetic supervised problem.

    Attributes:
        X: Feature matrix ``(n_rows, n_features)``, float32, may contain NaN.
        y: Integer class labels ``(n_rows,)`` in ``[0, n_classes)``.
        n_classes: Number of classes present in the label space.
        is_categorical: Boolean mask ``(n_features,)`` marking integer-coded
            categorical columns (used only for diagnostics and preprocessing).
        source: Name of the prior that produced the task.
        period: Optional ``(n_rows,)`` zero-based default period for survival training, with
            :data:`fintfm.modeling.hazard.CENSORED` (-1) for a firm that survives the whole
            horizon grid. ``None`` for priors that emit only a binary label. When present,
            ``y`` is exactly ``period != CENSORED``, so a survival task can still train a
            classifier without change.
        n_horizons: Length of the horizon grid ``period`` indexes into, or ``None``.
    """

    X: np.ndarray
    y: np.ndarray
    n_classes: int
    is_categorical: np.ndarray
    source: str = "unknown"
    period: np.ndarray | None = None
    n_horizons: int | None = None

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


@dataclass
class TaskBatch:
    """A padded batch of tasks ready for the model.

    Attributes:
        X: ``(B, N, F_max)`` float32 with NaN for missing *and* padded columns.
        y: ``(B, N)`` int64 labels.
        n_ctx: Number of leading rows in each task that act as labelled context.
            All tasks in a batch share the same split point.
        n_classes: ``(B,)`` int64 number of valid classes per task.
        period: Optional ``(B, N)`` int64 default periods for survival training; ``None``
            when no task in the batch carries them. Mixed batches are refused by
            :func:`collate` rather than silently padded, because a padded period would be
            indistinguishable from a real one and would corrupt the likelihood.
    """

    X: torch.Tensor
    y: torch.Tensor
    n_ctx: int
    n_classes: torch.Tensor
    period: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> TaskBatch:
        return TaskBatch(
            self.X.to(device),
            self.y.to(device),
            self.n_ctx,
            self.n_classes.to(device),
            None if self.period is None else self.period.to(device),
        )


def collate(tasks: list[Task], n_ctx: int, max_features: int) -> TaskBatch:
    """Pad a list of equal-length tasks into a :class:`TaskBatch`.

    Args:
        tasks: Tasks with identical ``n_rows`` and ``n_features <= max_features``.
        n_ctx: Context/query split point.
        max_features: Width to pad feature matrices to.
    """
    n_rows = tasks[0].n_rows
    X = np.full((len(tasks), n_rows, max_features), np.nan, dtype=np.float32)
    y = np.zeros((len(tasks), n_rows), dtype=np.int64)
    n_classes = np.zeros(len(tasks), dtype=np.int64)
    with_period = [t.period is not None for t in tasks]
    if any(with_period) and not all(with_period):
        raise ValueError(
            "batch mixes survival and binary-only tasks; a padded period is "
            "indistinguishable from a real one and would corrupt the likelihood"
        )
    period = np.zeros((len(tasks), n_rows), dtype=np.int64) if all(with_period) else None
    for i, t in enumerate(tasks):
        if t.n_rows != n_rows:
            raise ValueError("all tasks in a batch must share n_rows")
        X[i, :, : t.n_features] = t.X
        y[i] = t.y
        n_classes[i] = t.n_classes
        if period is not None:
            period[i] = t.period
    return TaskBatch(
        torch.from_numpy(X),
        torch.from_numpy(y),
        n_ctx,
        torch.from_numpy(n_classes),
        None if period is None else torch.from_numpy(period),
    )
