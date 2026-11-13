"""Print FinTFM's position by Elo in every stored TabArena leaderboard of the fork.

TabArena's printed leaderboard numbers its rows from 0, so reading the printed index as a rank
puts a method one place higher than it is (FINDINGS §225). This script derives the position
from ``tabarena_leaderboard.csv`` instead: one plus the number of methods with a strictly higher
Elo, out of every row in that leaderboard.

    uv run python scripts/lite_positions.py ~/proj/tabarena/eval
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def positions(eval_root: Path) -> list[tuple[str, str, float, int, int]]:
    """Return FinTFM's Elo and position in each leaderboard under ``eval_root``.

    Args:
        eval_root: The fork's ``eval/`` directory; each subdirectory may hold a
            ``tabarena_leaderboard.csv``.

    Returns:
        ``(run, method, elo, position, n_methods)`` per FinTFM row, sorted by run name.
    """
    out = []
    for path in sorted(eval_root.glob("*/tabarena_leaderboard.csv")):
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        elos = [float(r["elo"]) for r in rows]
        for r in rows:
            if "FINTFM" in r["method"].upper():
                elo = float(r["elo"])
                out.append(
                    (path.parent.name, r["method"], elo, 1 + sum(e > elo for e in elos), len(rows))
                )
    return out


def main(argv: list[str] | None = None) -> None:
    """Print one Markdown row per FinTFM entry.

    Args:
        argv: Arguments; ``None`` reads ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("eval_root", type=Path, help="the fork's eval/ directory")
    args = parser.parse_args(argv)
    print("| run | method | Elo | position |")
    print("| --- | --- | --- | --- |")
    for run, method, elo, pos, n in positions(args.eval_root.expanduser()):
        print(f"| `{run}` | {method} | {elo:.1f} | {pos} of {n} |")


if __name__ == "__main__":
    main()
