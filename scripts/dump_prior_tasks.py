"""Pre-generate pretraining batches into a sharded dump (audit AU5.15 prerequisite).

Why a dump
----------
Prior sampling runs on the CPU, one batch per training step, so a GPU run at the 2M-task
volume AU5.15 asks for waits on Python. This writes the batches once, on whatever CPUs are
cheap, and ``fintfm-train --task-dump DIR`` reads them back. See :mod:`fintfm.prior.dump` for
the format and the two streams.

The prior flags below carry ``fintfm-train``'s names and defaults, and the configuration
layering is the same, so the command that trains is the command that dumps with the training
flags copied across. ``fintfm-train`` refuses a dump whose recorded prior differs from its own
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
from fintfm.prior.dump import STREAMS, write_task_dump
from fintfm.prior.levers import add_prior_lever_arguments, prior_lever_values
from fintfm.prior.mixture import PriorConfig


def build_prior_config(args: argparse.Namespace) -> PriorConfig:
    """Build the prior exactly as ``fintfm-train``'s ``main`` builds it from the same flags.

    Args:
        args: Parsed arguments.

    Returns:
        The prior configuration.
    """
    cfg = load_config(args.config)
    return PriorConfig(
        max_features=args.max_features,
        max_classes=args.max_classes,
        n_rows=args.n_rows,
        n_rows_choices=(
            tuple(int(v) for v in args.n_rows_choices.split(",")) if args.n_rows_choices else None
        ),
        scm_legacy=args.scm_legacy,
        scm_reuse_graph=(
            PriorConfig.scm_reuse_graph if args.scm_reuse_graph is None else args.scm_reuse_graph
        ),
        p_tree=PriorConfig.p_tree if args.p_tree is None else args.p_tree,
        p_learnability_filter=(
            PriorConfig.p_learnability_filter
            if args.p_learnability_filter is None
            else args.p_learnability_filter
        ),
        p_financial=(
            1.0
            if args.n_horizons
            else (PriorConfig.p_financial if args.p_financial is None else args.p_financial)
        ),
        n_horizons=args.n_horizons,
        p_trivial=args.p_trivial,
        p_crossed=args.p_crossed,
        p_task_family=args.p_task_family,
        p_regression=args.p_regression,
        identity_shuffle=args.identity_shuffle,
        discretize_frac=args.discretize_frac,
        n_noise_features=args.n_noise_features,
        n_correlated_block_features=args.n_correlated_block_features,
        label_noise_rate=args.label_noise_rate,
        head_type=args.head_type,
        sharpness_min=cfg.prior.sharpness_min,
        sharpness_max=cfg.prior.sharpness_max,
        min_expected_positives=cfg.prior.min_expected_positives,
        absolute_rate_floor=cfg.prior.absolute_rate_floor,
        rate_ceiling=cfg.prior.rate_ceiling,
        n_sectors_max=cfg.prior.n_sectors,
        min_ctx_per_class=(
            cfg.prior.min_ctx_per_class
            if args.min_ctx_per_class is None
            else args.min_ctx_per_class
        ),
        **prior_lever_values(args, cfg.prior),
    )


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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--shard-size", type=int, default=256, help="batches per .npz shard")
    p.add_argument("--stream", choices=STREAMS, default="sequential")
    p.add_argument("--workers", type=int, default=1, help="processes; per_shard stream only")
    p.add_argument(
        "--compress", action="store_true", help="deflate shards (lossless, slower to write)"
    )
    p.add_argument("--config", type=str, default=None, help="YAML overriding the defaults")
    # fintfm-train's prior flags, same names and defaults
    p.add_argument("--n-rows", type=int, default=256)
    p.add_argument("--n-rows-choices", type=str, default=None)
    p.add_argument("--max-features", type=int, default=24)
    p.add_argument("--max-classes", type=int, default=10)
    p.add_argument("--p-financial", type=float, default=None)
    p.add_argument("--p-tree", type=float, default=None)
    p.add_argument("--p-trivial", type=float, default=0.0)
    p.add_argument("--p-crossed", type=float, default=0.0)
    p.add_argument("--p-task-family", type=float, default=0.0)
    p.add_argument("--p-regression", type=float, default=0.0)
    p.add_argument("--p-learnability-filter", type=float, default=None)
    p.add_argument("--scm-legacy", action="store_true")
    p.add_argument("--scm-reuse-graph", type=int, default=None)
    p.add_argument("--min-ctx-per-class", type=int, default=None)
    p.add_argument("--n-horizons", type=int, default=None)
    p.add_argument("--head-type", choices=["binned", "quantile"], default="binned")
    p.add_argument("--identity-shuffle", action="store_true")
    p.add_argument("--discretize-frac", type=float, default=0.0)
    p.add_argument("--n-noise-features", type=int, default=0)
    p.add_argument("--n-correlated-block-features", type=int, default=0)
    p.add_argument("--label-noise-rate", type=float, default=0.0)
    add_prior_lever_arguments(p)
    args = p.parse_args(argv)
    if args.task_dump is not None:
        p.error("--task-dump reads a dump; this script writes one with --out")
    return args


def main(argv: list[str] | None = None) -> None:
    """Write the dump and report generation throughput.

    Args:
        argv: Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    prior = build_prior_config(args)
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
