"""The internal arena: one proxy-suite Elo table over every run CSV given.

Combines `python -m fintfm.evaluation.proxy_suite run` outputs (one with the baselines, any number
with `--no-baselines`) for one problem type and fits a single Bradley-Terry Elo over all of them,
RF (default) at 1,000, so every checkpoint is rated in the same pool. Ratings move as methods
join the pool; compare checkpoints within one table, never across tables.

Usage:
    uv run --frozen --extra bench python scripts/proxy_arena.py --kind binary \
        --out runs/dr73/arena.csv runs/dr73/proxy_binary.csv runs/dr73/proxy_e1_*.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fintfm.evaluation.proxy_suite import elo_table


def main(argv: list[str] | None = None) -> int:
    """Print and write the combined Elo table; returns the exit status."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="+", type=Path, help="proxy_suite run CSVs")
    p.add_argument("--kind", default="binary", choices=("binary", "multiclass", "regression"))
    p.add_argument("--anchor", default="RF")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", type=Path, default=None, help="CSV to write the table to")
    args = p.parse_args(argv)
    res = pd.concat([pd.read_csv(f) for f in args.runs], ignore_index=True)
    table = elo_table(res[res["kind"] == args.kind], anchor=args.anchor, n_boot=args.n_boot)
    with pd.option_context("display.width", 160):
        print(table.to_string(float_format=lambda x: f"{x:.0f}"))
    if args.out is not None:
        table.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
