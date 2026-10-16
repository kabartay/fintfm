"""Pre-generate pretraining batches into a sharded dump (audit AU5.15 prerequisite).

Why a dump
----------
Prior sampling runs on the CPU, one batch per training step, so a GPU run at the 2M-task
volume AU5.15 asks for waits on Python. This writes the batches once, on whatever CPUs are
cheap, and ``fintfm-train --task-dump DIR`` reads them back. See :mod:`fintfm.prior.dump` for
the format and the two streams.

The prior flags below carry ``fintfm-train``'s names, default to ``None`` as its flags do,
and are resolved by its own :func:`fintfm.modeling.train.build_prior_config` against the same
configuration, so the command that trains is the command that dumps with the training flags
copied across, and a ``--config`` file serves both. ``fintfm-train`` refuses a dump whose recorded prior differs from its own
in any field, so a flag forgotten here fails at the first step rather than training on a
different mixture.

Usage::

    uv run python scripts/dump_prior_tasks.py --out dumps/ref-s0 --n-batches 6000 \\
        --seed 0 --batch-size 8 --n-rows-choices 256,512,1024 --max-features 136 \\
        --p-financial 1.0
    uv run fintfm-train ... --seed 0 --batch-size 8 --task-dump dumps/ref-s0

``--stream per_shard --workers N`` writes shards in parallel from independent generators;
that dump is reproducible but is not the live stream of any ``--seed``.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from fintfm.config import load_config
from fintfm.modeling.train import build_prior_config as train_build_prior_config
from fintfm.prior.dump import STREAMS, write_task_dump
from fintfm.prior.levers import add_prior_lever_arguments
from fintfm.prior.mixture import PriorConfig


def build_prior_config(args: argparse.Namespace) -> PriorConfig:
    """Build the prior with ``fintfm-train``'s own builder, from the same flags and config.

    Args:
        args: Parsed arguments.

    Returns:
        The prior configuration.
    """
    return train_build_prior_config(args, load_config(args.config))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments, or ``None`` for ``sys.argv``.

    Returns:
        The parsed namespace.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True, help="dump directory to create")
    p.add_argument("--n-batches", type=int, required=True, help="batches to write (>= --steps)")
    # None takes the config's train section, as fintfm-train does
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--shard-size", type=int, default=256, help="batches per .npz shard")
    p.add_argument("--stream", choices=STREAMS, default="sequential")
    p.add_argument("--workers", type=int, default=1, help="processes; per_shard stream only")
    p.add_argument(
        "--compress", action="store_true", help="deflate shards (lossless, slower to write)"
    )
    p.add_argument("--config", type=str, default=None, help="YAML overriding the defaults")
    # fintfm-train's prior flags, same names; None takes the config's prior and model sections
    p.add_argument("--n-rows", type=int, default=None)
    p.add_argument("--n-rows-choices", type=str, default=None)
    p.add_argument("--max-features", type=int, default=None)
    p.add_argument("--max-classes", type=int, default=None)
    p.add_argument("--p-financial", type=float, default=None)
    p.add_argument("--p-tree", type=float, default=None)
    p.add_argument("--p-trivial", type=float, default=None)
    p.add_argument("--p-crossed", type=float, default=None)
    p.add_argument("--p-task-family", type=float, default=None)
    p.add_argument("--p-regression", type=float, default=None)
    p.add_argument("--p-learnability-filter", type=float, default=None)
    p.add_argument("--scm-legacy", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--scm-reuse-graph", type=int, default=None)
    p.add_argument("--min-ctx-per-class", type=int, default=None)
    p.add_argument("--n-horizons", type=int, default=None)
    p.add_argument("--head-type", choices=["binned", "quantile"], default=None)
    p.add_argument("--identity-shuffle", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--discretize-frac", type=float, default=None)
    p.add_argument("--n-noise-features", type=int, default=None)
    p.add_argument("--n-correlated-block-features", type=int, default=None)
    p.add_argument("--label-noise-rate", type=float, default=None)
    add_prior_lever_arguments(p)
    args = p.parse_args(argv)
    if args.task_dump is not None:
        p.error("--task-dump reads a dump; this script writes one with --out")
    if args.workers < 1:
        p.error(f"--workers must be at least 1, got {args.workers}")
    if args.workers > 1 and args.stream != "per_shard":
        p.error("--workers > 1 needs --stream per_shard: the sequential stream is one generator")
    return args


def main(argv: list[str] | None = None) -> None:
    """Write the dump and report generation throughput.

    Args:
        argv: Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    prior = build_prior_config(args)
    train = load_config(args.config).train
    args.seed = train.seed if args.seed is None else args.seed
    args.batch_size = train.batch_size if args.batch_size is None else args.batch_size
    t0 = time.perf_counter()
    out = write_task_dump(
        args.out,
        prior,
        seed=args.seed,
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        shard_size=args.shard_size,
        stream=args.stream,
        workers=args.workers,
        compress=args.compress,
    )
    elapsed = time.perf_counter() - t0
    size = sum(f.stat().st_size for f in Path(out).iterdir())
    print(
        f"wrote {args.n_batches} batches x {args.batch_size} tasks to {out} in {elapsed:.1f}s: "
        f"{args.n_batches / elapsed:.2f} batches/s, {args.n_batches * args.batch_size / elapsed:.1f} "
        f"tasks/s, {size / 1e6:.1f} MB ({size / args.n_batches / 1e6:.2f} MB/batch); "
        f"stream {args.stream}, workers {args.workers}, compress {args.compress}, load average "
        f"{' '.join(f'{x:.1f}' for x in os.getloadavg())}"
    )


if __name__ == "__main__":
    main()
