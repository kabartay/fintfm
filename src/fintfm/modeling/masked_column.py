"""Masked-column task swap: the measurement arm for openspec task 48.15.

The pretraining objective is ``p(y | x, D)``: only the designated label is ever a target. 48.15
asks whether making *every* column a target (LimiX-2's joint-structure objective) helps, and
scopes the answer as a measurement before any rewrite. This module is the smallest change that
can measure it: a **data-level replacement** of a sampled task by a task built from the same
table in which one feature column plays the label. The model, the head, the loss and the
inference API are untouched; only which ``(X, y)`` the loop trains on changes, and the task
count is unchanged, so an arm and its control are matched in tasks.

The head here is binary (``max_classes`` 2 in the published recipe), and the loss and the
inference-time base-rate correction both assume labels are classes of a question like "does
this firm default". A multiclass quantile-bin target would ask the head a different kind of
question, so the swapped target is **binary**: column ``j`` cut at a quantile of its *context*
rows. Nothing about the query rows reaches the cut, so a swapped task has no leakage the
original did not.

Used by :func:`fintfm.modeling.train.train` behind ``--masked-column-prob`` (default 0.0,
which never calls into this module).
"""

from __future__ import annotations

import numpy as np
import torch

from fintfm.prior.base import TaskBatch

#: Second word of the swap generator's seed, ``[seed, _MASK_STREAM, rank, step, 1]``. The
#: evaluation streams use 1 and the fixed evaluation set 2 in that position; the trailing
#: non-zero word keeps the key distinct from the stream loader's ``[seed, i]`` too, since
#: NumPy's ``SeedSequence`` pads short entropy with zeros.
MASK_STREAM = 3

#: Lowest and highest context positive rate the cut aims for. Below 0.1 a 256-row task has
#: too few positives to learn from; the cap at 0.5 with a random direction makes the target's
#: base rate resemble the prior's imbalanced tasks rather than a coin flip.
RATE_RANGE = (0.1, 0.5)

#: Each class must hold at least this many *context* rows, or the task is not swapped.
MIN_CTX_PER_CLASS = 2


def swap_key(seed: int, rank: int, step: int) -> list[int]:
    """The generator seed for one step's swap decisions.

    Derived from the step index rather than carried as state, so a resumed run draws what an
    uninterrupted one would, and no checkpoint format changes.

    Args:
        seed: The run's training seed.
        rank: This process's rank (0 without ``--ddp``).
        step: The step index, counting from 0.

    Returns:
        The word list for :func:`numpy.random.default_rng`.
    """
    return [seed, MASK_STREAM, rank, step, 1]


def _binarise_column(col: np.ndarray, n_ctx: int, rng: np.random.Generator) -> np.ndarray | None:
    """Cut one fully observed column into a binary target from its context rows alone.

    Args:
        col: ``(N,)`` finite values of the column over all rows of the task.
        n_ctx: Number of leading context rows.
        rng: Draws the target rate and the direction of the cut.

    Returns:
        ``(N,)`` int64 0/1 target, or ``None`` when the column is constant in the context or
        the cut leaves fewer than :data:`MIN_CTX_PER_CLASS` context rows in a class.
    """
    ctx = col[:n_ctx]
    if ctx.min() == ctx.max():
        return None
    rate = rng.uniform(*RATE_RANGE)
    sign = 1.0 if rng.random() < 0.5 else -1.0  # a fixed direction would be a global rule
    signed = sign * col
    threshold = np.quantile(signed[:n_ctx], 1.0 - rate)
    target = (signed > threshold).astype(np.int64)
    n_pos = int(target[:n_ctx].sum())
    if min(n_pos, n_ctx - n_pos) < MIN_CTX_PER_CLASS:
        return None
    return target


def swap_task(
    X: np.ndarray, y: np.ndarray, n_ctx: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray] | None:
    """Rebuild one task with a randomly chosen feature column as its target.

    Column ``j`` is replaced, in place in the feature matrix, by the original label, so the
    width and every other column are unchanged; the new target is column ``j`` binarised at a
    context quantile (:func:`_binarise_column`). Only columns observed in every row are
    candidates: a padded column is NaN throughout, and a partly missing one has no target in
    its missing rows. Candidates are tried in a random order and the first valid one is used.

    Args:
        X: ``(N, F)`` float32 features, NaN for missing and padded cells.
        y: ``(N,)`` original integer labels.
        n_ctx: Number of leading context rows.
        rng: Draws the column order and the cut.

    Returns:
        ``(X_new, y_new)`` (copies), or ``None`` when no column qualifies.
    """
    candidates = np.flatnonzero(~np.isnan(X).any(axis=0))
    for j in rng.permutation(candidates):
        target = _binarise_column(X[:, j], n_ctx, rng)
        if target is None:
            continue
        X_new = X.copy()
        X_new[:, j] = y.astype(np.float32)
        return X_new, target
    return None


def swap_batch(batch: TaskBatch, prob: float, rng: np.random.Generator) -> tuple[TaskBatch, int]:
    """Replace each task of a batch by its masked-column version with probability ``prob``.

    Every task draws its coin from ``rng`` whether or not it is eligible, so the draw count
    depends on the batch size alone. A task that is chosen but has no valid column is left as
    sampled, so ``prob=1`` swaps every *eligible* task.

    Args:
        batch: A batch on any device, with one shared ``n_ctx``.
        prob: Per-task swap probability in ``[0, 1]``.
        rng: The step's swap generator (:func:`swap_key`).

    Returns:
        ``(new_batch, n_swapped)`` on the input's device. The input batch is not modified; ``n_classes`` of a
        swapped task is 2.

    Raises:
        ValueError: If the batch carries default periods (a survival batch has no binary
            relabelling that keeps them meaningful).
    """
    if batch.period is not None:
        raise ValueError("masked-column swap cannot relabel a batch that carries periods")
    device = batch.X.device
    batch = batch.to("cpu")  # the stream loader hands over device tensors; NumPy needs host ones
    chosen = rng.random(batch.X.shape[0]) < prob
    X = batch.X.clone()
    y = batch.y.clone()
    n_classes = batch.n_classes.clone()
    swapped = 0
    for b in np.flatnonzero(chosen):
        out = swap_task(batch.X[b].numpy(), batch.y[b].numpy(), batch.n_ctx, rng)
        if out is None:
            continue
        X[b] = torch.from_numpy(out[0])
        y[b] = torch.from_numpy(out[1]).to(y.dtype)
        n_classes[b] = 2
        swapped += 1
    return TaskBatch(X, y, batch.n_ctx, n_classes, None).to(device), swapped
