"""Pre-generated pretraining batches on disk, so a GPU run is not bound by CPU prior sampling.

Why this exists
---------------
Audit AU5.15 asks for at least 2M pretraining tasks against the 48,000 every checkpoint so far
has seen. Prior sampling runs on the CPU inside the training loop, one batch per step, so at
that volume a GPU waits on Python. A dump moves generation off the critical path: many CPU
workers (or a cheap CPU machine) write batches once, and the training run only reads them.

Two streams, recorded in the manifest
-------------------------------------
``"sequential"``: batch ``i`` is the ``i``-th call of
:func:`fintfm.prior.mixture.sample_batch` on ``np.random.default_rng(seed)``. Since audit
AU1.18 the training loop's generator feeds nothing but ``sample_batch``, so a sequential dump
written with a run's ``--seed`` holds exactly the batches that run would have drawn live, and
training from it is byte-identical to training without it. Generation is one process.

``"per_shard"``: shard ``s`` is drawn from ``np.random.default_rng([seed, s])``, so shards are
independent and can be written in parallel. The stream is just as reproducible but is not the
live run's stream; a run trained from it is a different (equally valid) draw of the prior.

The prior configuration, batch size, seed and stream are written to ``manifest.json``, and
:func:`open_task_dump` refuses a dump whose prior configuration or batch size differs from the
run consuming it -- a dump generated under one mixture and trained under another would
otherwise report the other mixture's name on every result.

Storage
-------
One ``.npz`` per shard holding each batch's arrays under ``b<k>_X``, ``b<k>_y``,
``b<k>_n_classes``, ``b<k>_n_ctx`` and, for survival batches, ``b<k>_period``. ``X`` is stored
cropped to the batch's widest column that holds any value; the columns cut are NaN in every
task of the batch (feature padding), and loading re-pads them with NaN to ``max_features``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fintfm.prior.base import TaskBatch
from fintfm.prior.mixture import PriorConfig, sample_batch

#: Version of the on-disk layout, bumped on any incompatible change.
DUMP_FORMAT = 1

#: Streams :func:`write_task_dump` can write; see the module docstring.
STREAMS: tuple[str, ...] = ("sequential", "per_shard")


def _prior_record(cfg: PriorConfig) -> dict[str, Any]:
    """A JSON-stable record of a prior configuration, for the manifest and for comparison.

    Args:
        cfg: Prior configuration.

    Returns:
        ``asdict(cfg)`` passed through a JSON round trip, so tuples and lists compare equal.
    """
    return json.loads(json.dumps(asdict(cfg)))


def _shard_batches(
    cfg: PriorConfig, rng: np.random.Generator, batch_size: int, n: int
) -> dict[str, np.ndarray]:
    """Draw ``n`` consecutive batches from ``rng`` and pack them into ``.npz`` arrays.

    Args:
        cfg: Prior configuration.
        rng: Generator to draw from; advanced by ``n`` batches.
        batch_size: Tasks per batch.
        n: Batches to draw.

    Returns:
        A mapping from array name to array, ready for :func:`numpy.savez`.
    """
    arrays: dict[str, np.ndarray] = {}
    for k in range(n):
        batch = sample_batch(rng, cfg, batch_size)
        X = batch.X.numpy()
        filled = np.flatnonzero(~np.isnan(X).all(axis=(0, 1)))
        width = int(filled[-1]) + 1 if len(filled) else 0
        arrays[f"b{k}_X"] = np.ascontiguousarray(X[:, :, :width])
        arrays[f"b{k}_y"] = batch.y.numpy()
        arrays[f"b{k}_n_classes"] = batch.n_classes.numpy()
        arrays[f"b{k}_n_ctx"] = np.asarray(batch.n_ctx, dtype=np.int64)
        if batch.period is not None:
            arrays[f"b{k}_period"] = batch.period.numpy()
    return arrays


def _write_shard(
    out_dir: str,
    cfg: PriorConfig,
    seed: int,
    batch_size: int,
    shard: int,
    n: int,
    compress: bool = False,
) -> str:
    """Write one ``per_shard`` shard from its own generator; a worker-process entry point.

    Args:
        out_dir: Dump directory.
        cfg: Prior configuration.
        seed: The dump's seed; the shard draws from ``default_rng([seed, shard])``.
        batch_size: Tasks per batch.
        shard: Shard index.
        n: Batches in this shard.
        compress: Write with :func:`numpy.savez_compressed`.

    Returns:
        The shard's file name.
    """
    name = f"shard_{shard:06d}.npz"
    rng = np.random.default_rng([seed, shard])
    save = np.savez_compressed if compress else np.savez
    save(Path(out_dir) / name, **_shard_batches(cfg, rng, batch_size, n))
    return name


def write_task_dump(
    out_dir: str | Path,
    cfg: PriorConfig,
    seed: int,
    batch_size: int,
    n_batches: int,
    shard_size: int = 256,
    stream: str = "sequential",
    workers: int = 1,
    compress: bool = False,
) -> Path:
    """Generate ``n_batches`` pretraining batches and write them as a sharded dump.

    Args:
        out_dir: Directory to create; must not already hold a manifest.
        cfg: Prior configuration, recorded in the manifest.
        seed: Generator seed. For ``"sequential"`` use the training run's ``--seed`` to
            reproduce its live batches exactly.
        batch_size: Tasks per batch; must equal the consuming run's ``--batch-size``.
        n_batches: Batches to write; a run needs at least ``--steps`` of them.
        shard_size: Batches per ``.npz`` file.
        stream: ``"sequential"`` or ``"per_shard"``; see the module docstring.
        workers: Processes for ``"per_shard"``; ``"sequential"`` is always one process.
        compress: Deflate each shard (:func:`numpy.savez_compressed`). Lossless; the reader
            needs no flag, since :func:`numpy.load` reads either.

    Returns:
        The dump directory.

    Raises:
        ValueError: On an unknown stream, non-positive sizes, or ``workers > 1`` with the
            sequential stream (which cannot be split without changing it).
        FileExistsError: If ``out_dir`` already holds a dump.
    """
    if stream not in STREAMS:
        raise ValueError(f"stream must be one of {STREAMS}, got {stream!r}")
    if n_batches < 1 or shard_size < 1 or batch_size < 1:
        raise ValueError("n_batches, shard_size and batch_size must all be at least 1")
    if stream == "sequential" and workers > 1:
        raise ValueError("the sequential stream is one generator and cannot use workers > 1")
    out = Path(out_dir)
    if (out / "manifest.json").exists():
        raise FileExistsError(f"{out} already holds a task dump")
    out.mkdir(parents=True, exist_ok=True)
    sizes = [min(shard_size, n_batches - s) for s in range(0, n_batches, shard_size)]
    names: list[str] = []
    save = np.savez_compressed if compress else np.savez
    if stream == "sequential":
        rng = np.random.default_rng(seed)
        for shard, n in enumerate(sizes):
            name = f"shard_{shard:06d}.npz"
            save(out / name, **_shard_batches(cfg, rng, batch_size, n))
            names.append(name)
    elif workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_write_shard, str(out), cfg, seed, batch_size, shard, n, compress)
                for shard, n in enumerate(sizes)
            ]
            names = [f.result() for f in futures]
    else:
        names = [
            _write_shard(str(out), cfg, seed, batch_size, shard, n, compress)
            for shard, n in enumerate(sizes)
        ]
    manifest = {
        "format": DUMP_FORMAT,
        "stream": stream,
        "seed": seed,
        "batch_size": batch_size,
        "n_batches": n_batches,
        "shard_size": shard_size,
        "compressed": compress,
        "shards": names,
        "prior": _prior_record(cfg),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


@dataclass
class TaskDump:
    """Read a dump written by :func:`write_task_dump`, one :class:`TaskBatch` at a time.

    Indexed by batch number, so a resumed run reads from its step without replaying anything.
    One shard file is open at a time and each batch's arrays are read from it on demand, so
    memory holds one batch, not one shard (a 256-batch shard of the reference recipe is about
    600 MB).

    Attributes:
        path: Dump directory.
        manifest: Parsed ``manifest.json``.
    """

    path: Path
    manifest: dict[str, Any]
    _open: tuple[int, Any] = field(default=(-1, None), repr=False)

    @classmethod
    def open(cls, path: str | Path) -> TaskDump:
        """Open a dump directory.

        Args:
            path: Directory holding ``manifest.json``.

        Returns:
            The reader.

        Raises:
            ValueError: If the dump's format version is not this module's.
        """
        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        if manifest.get("format") != DUMP_FORMAT:
            raise ValueError(
                f"{p}: dump format {manifest.get('format')!r}, this reader expects {DUMP_FORMAT}"
            )
        return cls(p, manifest)

    def __len__(self) -> int:
        """Batches in the dump."""
        return int(self.manifest["n_batches"])

    @property
    def max_features(self) -> int:
        """Feature width every batch is re-padded to: the prior's ``max_features``."""
        return int(self.manifest["prior"]["max_features"])

    def batch(self, index: int) -> TaskBatch:
        """Return batch ``index`` exactly as it was sampled.

        Args:
            index: Zero-based batch number, ``0 <= index < len(self)``.

        Returns:
            The batch, with ``X`` re-padded to ``max_features``.

        Raises:
            IndexError: If ``index`` is out of range -- a run longer than its dump fails
                rather than wrapping round and silently repeating tasks.
        """
        if not 0 <= index < len(self):
            raise IndexError(f"batch {index} is outside this {len(self)}-batch dump")
        shard, k = divmod(index, int(self.manifest["shard_size"]))
        open_shard, f = self._open
        if open_shard != shard or f is None:
            if f is not None:
                f.close()
            f = np.load(self.path / self.manifest["shards"][shard])
            self._open = (shard, f)
        X_stored = f[f"b{k}_X"]
        B, N, width = X_stored.shape
        X = np.full((B, N, self.max_features), np.nan, dtype=np.float32)
        X[:, :, :width] = X_stored
        period_key = f"b{k}_period"
        return TaskBatch(
            torch.from_numpy(X),
            torch.from_numpy(f[f"b{k}_y"]),
            int(f[f"b{k}_n_ctx"]),
            torch.from_numpy(f[f"b{k}_n_classes"]),
            torch.from_numpy(f[period_key]) if period_key in f.files else None,
        )

    def close(self) -> None:
        """Close the open shard file, if any."""
        _, f = self._open
        if f is not None:
            f.close()
        self._open = (-1, None)

    def check_matches(self, cfg: PriorConfig, batch_size: int) -> None:
        """Refuse a dump generated under a different prior or batch size than a run's.

        Args:
            cfg: The consuming run's prior configuration.
            batch_size: The consuming run's batch size.

        Raises:
            ValueError: Naming every differing prior field, or the batch-size mismatch.
        """
        if int(self.manifest["batch_size"]) != batch_size:
            raise ValueError(
                f"{self.path} holds batches of {self.manifest['batch_size']} tasks, but the run "
                f"uses --batch-size {batch_size}"
            )
        want = _prior_record(cfg)
        have = self.manifest["prior"]
        diff = {k: (have.get(k), v) for k, v in want.items() if have.get(k) != v}
        diff.update({k: (v, None) for k, v in have.items() if k not in want})
        if diff:
            raise ValueError(
                f"{self.path} was generated under a different prior (dump, this run): {diff}"
            )


def open_task_dump(path: str | Path, cfg: PriorConfig, batch_size: int, steps: int) -> TaskDump:
    """Open a dump for a training run, checking it can serve every step.

    Args:
        path: Dump directory.
        cfg: The run's prior configuration; must equal the dump's.
        batch_size: The run's batch size; must equal the dump's.
        steps: The run's total step count; the dump must hold at least this many batches.

    Returns:
        The reader.

    Raises:
        ValueError: On a prior or batch-size mismatch, or a dump shorter than ``steps``.
    """
    dump = TaskDump.open(path)
    dump.check_matches(cfg, batch_size)
    if len(dump) < steps:
        raise ValueError(f"{path} holds {len(dump)} batches, fewer than --steps {steps}")
    return dump
