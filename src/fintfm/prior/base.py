"""Common task container and batching helpers for synthetic priors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fintfm.inference.binning import QuantileBinner
from fintfm.inference.preprocess import FeatureConditioner


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
        y_continuous: Optional ``(n_rows,)`` float continuous target for a **regression**
            task. When present, ``y`` is a placeholder and :func:`collate` overwrites it by
            binning this array on the **context rows' quantiles** -- see the note there for
            why binning cannot happen in the prior itself.
    """

    X: np.ndarray
    y: np.ndarray
    n_classes: int
    is_categorical: np.ndarray
    source: str = "unknown"
    period: np.ndarray | None = None
    n_horizons: int | None = None
    y_continuous: np.ndarray | None = None

    @property
    def n_rows(self) -> int:
        """Rows in this task."""
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        """Exposed feature columns in this task."""
        return int(self.X.shape[1])


@dataclass
class TaskBatch:
    """A padded batch of tasks ready for the model.

    Attributes:
        X: ``(B, N, F_max)`` float32 with NaN for missing *and* padded columns.
        y: ``(B, N)`` int64 class labels (``head_type="binned"``), or float32 continuous,
            context-normalised targets (``head_type="quantile"``) -- see :func:`collate`.
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
        """Move every tensor in the batch to ``device``.

        Args:
            device: Torch device or device string.

        Returns:
            A new batch; the original is unchanged.
        """
        return TaskBatch(
            self.X.to(device),
            self.y.to(device),
            self.n_ctx,
            self.n_classes.to(device),
            None if self.period is None else self.period.to(device),
        )


def ensure_context_classes(
    rng: np.random.Generator, task: Task, n_ctx: int, min_per_class: int
) -> Task:
    """Reorder a task's rows so every class has ``min_per_class`` rows in the context.

    Audit AU1.16: the SCM and tree priors cut a continuous target at random points, which
    leaves a class with two rows or fewer in about 31% of 256-row tasks, and the batch's
    context/query split is drawn afterwards without looking at labels. A class can then sit
    only in the query, and the loss is taken on a label the context never showed -- a
    prediction no in-context learner can make. This swaps whole rows (features, label and
    period together) between the query and the context until each class present has
    ``min(min_per_class, its row count)`` context rows; a class with fewer rows than that ends
    up entirely in the context. Donor context rows come only from classes above their own
    requirement, so a fix for one class never breaks another. Nothing is relabelled and no row
    is added or dropped, so the task's class balance is unchanged; only which rows are context
    moves.

    Applied at batch level (:func:`fintfm.prior.mixture.sample_batch`) rather than inside the
    priors, because only the batch knows ``n_ctx``.

    Args:
        rng: NumPy random generator; consumed only when a swap is needed.
        task: A classification task. A regression task (``y_continuous`` set) is returned
            unchanged, since its labels are binned later from the context.
        n_ctx: The batch's context size.
        min_per_class: Context rows wanted per class. ``0`` returns ``task`` unchanged.

    Returns:
        ``task`` itself when nothing needs to move, otherwise a reordered copy.
    """
    if min_per_class <= 0 or task.y_continuous is not None:
        return task
    y = task.y
    classes, totals = np.unique(y, return_counts=True)
    need = dict(zip(classes.tolist(), np.minimum(totals, min_per_class).tolist()))
    order = np.arange(len(y))
    moved = False
    for c in classes.tolist():
        while True:
            ctx_y = y[order[:n_ctx]]
            if int((ctx_y == c).sum()) >= need[c]:
                break
            surplus = [k for k in need if int((ctx_y == k).sum()) > need[k]]
            donors = np.flatnonzero(np.isin(ctx_y, surplus))
            if len(donors) == 0:
                break  # the context is too small to hold every class's floor
            recipients = n_ctx + np.flatnonzero(y[order[n_ctx:]] == c)
            d, r = int(rng.choice(donors)), int(rng.choice(recipients))
            order[d], order[r] = order[r], order[d]
            moved = True
    if not moved:
        return task
    return Task(
        X=task.X[order],
        y=y[order],
        n_classes=task.n_classes,
        is_categorical=task.is_categorical,
        source=task.source,
        period=None if task.period is None else task.period[order],
        n_horizons=task.n_horizons,
        y_continuous=None,
    )


def permute_task_classes(rng: np.random.Generator, task: Task) -> Task:
    """Relabel a classification task's classes by a random permutation (audit AU5.5).

    The SCM prior cuts an ordinal latent, so class 0 is always its lowest band, and the
    financial prior always emits the rare default as class 1. A model can therefore learn a
    class-*index* prior ("class 1 is rare") that no in-context reading of the labels taught
    it, and which a real table with the minority coded 0 would contradict. Drawing a fresh
    permutation of ``[0, n_classes)`` per task removes that shortcut. Context and query labels
    are mapped by the same permutation, so the task's in-context problem is unchanged; only
    the arbitrary names of its classes move. Labels stay inside ``[0, n_classes)``, so
    :class:`TaskBatch`'s ``n_classes`` masking is untouched.

    Args:
        rng: NumPy random generator; consumed once (one permutation) per classification task.
        task: The task to relabel. A regression task (``y_continuous`` set) is returned
            unchanged and consumes no draw: its bins are ordinal and are decoded as such, so
            permuting them would not remove a shortcut, it would corrupt the target. A survival
            task (``period`` set) is refused, since its ``y`` must equal
            ``period != CENSORED`` exactly.

    Returns:
        A relabelled copy, or ``task`` itself for a regression task.

    Raises:
        ValueError: If the task carries a ``period``.
    """
    if task.y_continuous is not None:
        return task
    if task.period is not None:
        raise ValueError(
            "class permutation would break a survival task's y == (period != CENSORED) "
            "identity; do not combine permute_classes with n_horizons"
        )
    perm = rng.permutation(task.n_classes)
    return Task(
        X=task.X,
        y=perm[task.y].astype(task.y.dtype, copy=False),
        n_classes=task.n_classes,
        is_categorical=task.is_categorical,
        source=task.source,
        period=None,
        n_horizons=task.n_horizons,
        y_continuous=None,
    )


def condition_task_features(task: Task, n_ctx: int, kind: str) -> Task:
    """Apply one inference-time feature transform to a task, fitted on its context rows only.

    Audit AU5.6: pretraining fed the model raw prior features (z-scored inside the model by
    :func:`fintfm.modeling.model.normalize_features`), while inference defaults to
    ``feature_transform="rank"`` before that same z-score -- so the shipped checkpoint is
    always evaluated on an input distribution it never trained on. This mirrors exactly what
    :class:`~fintfm.inference.classifier.FinancialTFMClassifier` does at ``fit``: a
    :class:`~fintfm.inference.preprocess.FeatureConditioner` with its default quantile and
    subsample, fitted on the labelled rows and applied to every row. Here the labelled rows are
    the batch's **context** rows; the query rows never enter the fit, so a query cannot shape
    its own conditioning, which is the same leakage rule :func:`collate` keeps for regression
    bin edges.

    Args:
        task: Task whose ``X`` to condition; every column is transformed, categorical or
            not, as inference does.
        n_ctx: The batch's context size; rows ``[:n_ctx]`` are the fit rows.
        kind: ``"none"``, ``"rank"``, ``"power"`` or ``"winsor"``. ``"none"`` returns
            ``task`` itself.

    Returns:
        ``task`` for ``"none"``, otherwise a copy with transformed ``X`` (NaN preserved).
    """
    if kind == "none":
        return task
    ctx = task.X[:n_ctx]
    # A column with no finite context value has nothing to fit on, and ``"power"`` raises on
    # one (sklearn's Yeo-Johnson unpacks an empty optimisation). Small contexts over sparse
    # columns produce these, so they are left untransformed -- what ``"none"`` would do -- and
    # the rest are fitted exactly as inference fits them.
    fittable = np.isfinite(ctx).any(axis=0)
    if fittable.all():
        X = FeatureConditioner(kind).fit(ctx).transform(task.X)  # type: ignore[arg-type]
    else:
        X = task.X.copy()
        cols = np.flatnonzero(fittable)
        if len(cols):
            X[:, cols] = (
                FeatureConditioner(kind)  # type: ignore[arg-type]
                .fit(ctx[:, cols])
                .transform(task.X[:, cols])
            )
    return Task(
        X=X,
        y=task.y,
        n_classes=task.n_classes,
        is_categorical=task.is_categorical,
        source=task.source,
        period=task.period,
        n_horizons=task.n_horizons,
        y_continuous=task.y_continuous,
    )


def truncate_query(task: Task, n_ctx: int, max_query_rows: int) -> Task:
    """Keep the context and at most ``max_query_rows`` query rows of a task (audit AU5.4).

    Lets a batch carry a long context without paying for a proportionally long query: the
    row encoder's cost grows with the whole task length, context plus query, so a 4,096-row
    context drawn at the default context fraction would otherwise bring up to ~3,000 query
    rows with it. Rows are kept in order, so this assumes what the context/query split itself
    already assumes -- that a prior's rows are exchangeable -- and consumes no random draw.

    Args:
        task: The task, of ``n_rows`` rows with the first ``n_ctx`` as context.
        n_ctx: The batch's context size.
        max_query_rows: Query rows to keep at most.

    Returns:
        ``task`` itself when its query is already short enough, otherwise a truncated copy.
    """
    keep = n_ctx + max_query_rows
    if task.n_rows <= keep:
        return task
    return Task(
        X=task.X[:keep],
        y=task.y[:keep],
        n_classes=task.n_classes,
        is_categorical=task.is_categorical,
        source=task.source,
        period=None if task.period is None else task.period[:keep],
        n_horizons=task.n_horizons,
        y_continuous=None if task.y_continuous is None else task.y_continuous[:keep],
    )


def collate(
    tasks: list[Task], n_ctx: int, max_features: int, head_type: str = "binned"
) -> TaskBatch:
    """Pad a list of equal-length tasks into a :class:`TaskBatch`.

    Args:
        tasks: Tasks with identical ``n_rows`` and ``n_features <= max_features``. Every
            task must carry ``y_continuous`` when ``head_type="quantile"`` -- that head has
            no class-index path, so a classification task has nowhere to go.
        n_ctx: Context/query split point.
        max_features: Width to pad feature matrices to.
        head_type: ``"binned"`` (default) bins a regression task's continuous target into
            ``t.n_classes`` quantile bins, exactly as before this argument existed.
            ``"quantile"`` (task 48.3) instead z-scores it using context-row statistics, so
            ``TaskBatch.y`` carries the normalised continuous target directly rather than a
            bin index -- the model's pinball-loss head is trained and evaluated in that
            normalised space, and a caller inverts with the same mean/std at inference.

    Raises:
        ValueError: If ``head_type`` is unknown, or if ``head_type="quantile"`` meets a task
            with no continuous target.
    """
    if head_type not in ("binned", "quantile"):
        raise ValueError(f"head_type must be 'binned' or 'quantile', got {head_type!r}")
    n_rows = tasks[0].n_rows
    X = np.full((len(tasks), n_rows, max_features), np.nan, dtype=np.float32)
    y = np.zeros((len(tasks), n_rows), dtype=np.float32 if head_type == "quantile" else np.int64)
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
        if t.y_continuous is None and head_type == "quantile":
            raise ValueError(
                f"head_type='quantile' only supports regression tasks (Task.y_continuous), "
                f"got a {t.source!r} task with none -- train with --p-regression 1.0"
            )
        if t.y_continuous is not None and head_type == "quantile":
            # Context-only z-score, the quantile head's analogue of the binned head's
            # context-only bin edges below: scale invariance without query leakage, since
            # inference sees context statistics only and never the query targets.
            ctx_y = t.y_continuous[:n_ctx]
            mean = float(np.mean(ctx_y))
            std = float(np.std(ctx_y)) + 1e-6
            y[i] = (t.y_continuous - mean) / std
            continue
        if t.y_continuous is not None:
            # **Binning belongs here, not in the prior.** Edges come from the context rows'
            # quantiles, which only this function knows the boundary of (`n_ctx`). Binning in
            # the prior would have to use the whole task, leaking query targets into the
            # discretisation the model is trained against -- and would train the model on a
            # boundary that inference, which sees context only, can never reproduce.
            binner = QuantileBinner(n_bins=t.n_classes).fit(t.y_continuous[:n_ctx])
            y[i] = binner.transform(t.y_continuous)
            n_classes[i] = binner.n_bins_
            continue
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
