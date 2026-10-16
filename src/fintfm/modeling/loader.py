"""Streaming pretraining batches: batch ``i`` from its own generator, in worker processes.

Why this exists
---------------
Design review DR5.2. The default loop (``--loader sync``) samples every batch on the training
process from one generator, ``np.random.default_rng(seed)``, so batch ``i`` depends on every
batch before it. That has three costs at the volumes DR5.3 needs: the GPU waits on CPU prior
sampling, a resume has to carry the generator's state, and two GPUs cannot split the stream
without changing it.

``--loader stream`` draws batch ``i`` from ``np.random.default_rng([seed, i])`` instead, which
depends on nothing but ``(seed, i)``. So:

- batches can be generated in any order, by any number of worker processes, with prefetch,
  and the stream is the same whatever the worker count (:class:`StreamingTaskDataset`);
- a resumed run starts at its step index and needs no generator state;
- a distributed run's rank ``r`` of ``W`` takes the indices ``i ≡ r (mod W)``
  (:func:`rank_indices`), so ranks see disjoint batches and their union is the stream.

It is a different (equally valid) draw of the prior from the ``sync`` stream, not the same
one, except that batch 0 coincides: ``default_rng([seed, 0])`` and ``default_rng(seed)`` seed
alike, because NumPy's ``SeedSequence`` pads its entropy with zeros.

The task-dump equivalent
------------------------
A :mod:`fintfm.prior.dump` dump written with ``stream="per_shard"`` and ``shard_size=1`` draws
shard ``s`` -- one batch -- from ``default_rng([seed, s])``, which is exactly this stream's
batch ``s``. Dumps remain for byte-identity checks of the stream against stored batches;
``tests/test_loader.py`` holds the two equal.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from fintfm.prior import PriorConfig
from fintfm.prior.base import TaskBatch
from fintfm.prior.mixture import sample_batch

#: Accepted values of :attr:`fintfm.modeling.train.TrainConfig.loader`: ``"sync"`` samples on
#: the training process from one sequential generator (every run before DR5.2), ``"stream"``
#: is this module's per-index stream.
LOADERS = ("sync", "stream")


def stream_batch(prior_cfg: PriorConfig, batch_size: int, seed: int, index: int) -> TaskBatch:
    """Batch ``index`` of the stream: a function of ``(seed, index)`` alone.

    Args:
        prior_cfg: Prior to sample from.
        batch_size: Tasks per batch.
        seed: The run's seed.
        index: Zero-based batch index.

    Returns:
        The batch, on the CPU.
    """
    return sample_batch(np.random.default_rng([seed, index]), prior_cfg, batch_size)


def rank_indices(start: int, stop: int, rank: int, world_size: int) -> range:
    """The batch indices in ``[start, stop)`` that rank ``rank`` of ``world_size`` takes.

    Rank ``r`` takes every ``i`` with ``i ≡ r (mod world_size)``, so the ranks' sets are
    disjoint and together cover the interval.

    Args:
        start: First index of the interval.
        stop: One past its last index.
        rank: This process's rank, ``0 <= rank < world_size``.
        world_size: Number of processes.

    Returns:
        The rank's indices in increasing order.

    Raises:
        ValueError: For ``world_size < 1`` or a rank outside ``[0, world_size)``.
    """
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} is not in [0, {world_size})")
    return range(start + (rank - start) % world_size, stop, world_size)


def _pack(index: int, batch: TaskBatch) -> dict[str, Any]:
    """A batch as a plain dict, which the DataLoader can pin and send between processes.

    ``TaskBatch`` is a dataclass, which ``pin_memory`` passes through unpinned; a mapping of
    tensors it pins one by one.

    Args:
        index: The batch's stream index, carried so the consumer can check the order.
        batch: The batch.

    Returns:
        ``{"index", "X", "y", "n_ctx", "n_classes", "period"}``.
    """
    return {
        "index": index, "X": batch.X, "y": batch.y, "n_ctx": batch.n_ctx,
        "n_classes": batch.n_classes, "period": batch.period,
    }


class StreamingTaskDataset(IterableDataset):
    """The stream's batches for one rank, over an index interval, split across workers.

    Each DataLoader worker ``k`` of ``K`` takes every ``K``-th of the rank's indices starting
    at its ``k``-th; the DataLoader returns workers' items round-robin in worker order, which
    restores the rank's order exactly. Each item carries its index, and :func:`stream_batches`
    refuses an out-of-order one rather than trusting that guarantee silently.

    Args:
        prior_cfg: Prior to sample from.
        batch_size: Tasks per batch.
        seed: The run's seed.
        start: First stream index of the interval.
        stop: One past its last index.
        rank: This process's rank.
        world_size: Number of processes.
    """

    def __init__(
        self,
        prior_cfg: PriorConfig,
        batch_size: int,
        seed: int,
        start: int,
        stop: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.prior_cfg = prior_cfg
        self.batch_size = batch_size
        self.seed = seed
        self.indices = rank_indices(start, stop, rank, world_size)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield this worker's share of the rank's batches, each packed with its index."""
        info = get_worker_info()
        mine = self.indices if info is None else self.indices[info.id :: info.num_workers]
        for index in mine:
            yield _pack(index, stream_batch(self.prior_cfg, self.batch_size, self.seed, index))


def stream_batches(
    prior_cfg: PriorConfig,
    batch_size: int,
    seed: int,
    start: int,
    stop: int,
    rank: int = 0,
    world_size: int = 1,
    workers: int = 0,
    prefetch: int = 2,
    device: torch.device | str = "cpu",
) -> Iterator[tuple[int, TaskBatch]]:
    """Iterate a rank's stream batches, generated by worker processes and moved to ``device``.

    Args:
        prior_cfg: Prior to sample from.
        batch_size: Tasks per batch.
        seed: The run's seed.
        start: First stream index of the interval.
        stop: One past its last index.
        rank: This process's rank.
        world_size: Number of processes.
        workers: DataLoader worker processes; 0 generates on this process, which gives the
            same batches.
        prefetch: Batches each worker keeps ready ahead of the consumer (``prefetch_factor``);
            read only when ``workers > 0``.
        device: Where the batches go. On CUDA they are pinned by the DataLoader and copied
            without blocking.

    Yields:
        ``(index, batch)`` in increasing index order.

    Raises:
        RuntimeError: If the DataLoader delivers a batch out of order.
    """
    device = torch.device(device)
    pin = device.type == "cuda"
    extra: dict[str, Any] = {"prefetch_factor": prefetch} if workers > 0 else {}
    loader = DataLoader(
        StreamingTaskDataset(prior_cfg, batch_size, seed, start, stop, rank, world_size),
        batch_size=None,  # the dataset yields whole batches; nothing to collate
        num_workers=workers,
        pin_memory=pin,
        # A DataLoader draws its workers' base seed from this generator, and from torch's
        # global one when none is given -- which the model's per-task column identities
        # also draw from, so a default DataLoader would change training. The batches use
        # NumPy alone, so the base seed itself reaches nothing.
        generator=torch.Generator(),
        **extra,
    )
    expected = iter(rank_indices(start, stop, rank, world_size))
    for item in loader:
        index = int(item["index"])
        want = next(expected)
        if index != want:
            raise RuntimeError(f"stream loader delivered batch {index} where {want} was due")
        period = item["period"]
        yield index, TaskBatch(
            item["X"].to(device, non_blocking=pin),
            item["y"].to(device, non_blocking=pin),
            int(item["n_ctx"]),
            item["n_classes"].to(device, non_blocking=pin),
            None if period is None else period.to(device, non_blocking=pin),
        )
