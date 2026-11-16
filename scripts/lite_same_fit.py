r"""Score several merged TabArena-Lite entries in one Elo fit, beside the shared field.

Each merged entry (``merge_fintfm_lite.py``) is scored in its own fit, so two entries' Elo values
come from two fits and differ by the fit's own noise as well as by the entries (TABFLEX_GPU
(default) moved between 977.7 and 978.0 across the 2026-11-16 merges). The adoption rules of
FINDINGS §256, §257 and §265 compare entries "in the same fit"; this script builds that fit. It
reads each entry's ``results_per_split.csv`` from the fork's ``eval/`` directory, keeps the field
once (from the first entry), relabels each entry's ``[New]`` rows as ``[New] E<i>``, and fits
TabArena's leaderboard once over all of them. It fits no model.

Positions printed here are within a field that holds every entry passed, so the field is larger
than a single entry's (state the size beside the position, as everywhere else).

Usage, with the fork's interpreter from the fork's root::

    cd ~/proj/tabarena && .venv/bin/python -P ~/proj/fintfm/scripts/lite_same_fit.py \
        fintfm_linprobe_all51_1116b fintfm_linprobe_all51_1116f_floor8
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

#: The fork's evaluation directory, where each merged entry has ``<run>/results_per_split.csv``.
EVAL = Path.home() / "proj" / "tabarena" / "eval"
#: Reference rows printed beside the entries.
REFERENCES = ("TABFLEX_GPU (default)", "RF (default)", "RandomForest (default)")


def combined_frame(runs: Sequence[str], eval_dir: Path = EVAL) -> pd.DataFrame:
    """The field once plus every entry's ``[New]`` rows under a distinct method identity.

    Args:
        runs: Merged entry names; the field is taken from the first.
        eval_dir: Directory holding ``<run>/results_per_split.csv``.

    Returns:
        One ``results_per_split`` frame. Entry ``i`` is relabelled ``[New] E<i>`` in
        ``config_type``, ``ta_name`` and ``ta_suite``, and ``[New] E<i> (default)`` in ``method``,
        because the leaderboard groups rows by those columns together.

    Raises:
        ValueError: An entry has no ``[New]`` rows.
    """
    frames = []
    for i, run in enumerate(runs):
        frame = pd.read_csv(eval_dir / run / "results_per_split.csv")
        new = frame["method"].astype(str).str.startswith("[New]")
        if not new.any():
            raise ValueError(f"{run}: no [New] rows in results_per_split.csv")
        if i == 0:
            frames.append(frame[~new])
        part = frame[new].copy()
        tag = f"[New] E{i}"
        part["config_type"] = tag
        part["method"] = f"{tag} (default)"
        part["ta_name"] = tag
        part["ta_suite"] = tag
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def same_fit(runs: Sequence[str], eval_dir: Path = EVAL) -> pd.DataFrame:
    """The leaderboard of one fit over the field and every entry.

    Args:
        runs: Merged entry names.
        eval_dir: Directory holding ``<run>/results_per_split.csv``.

    Returns:
        The leaderboard with ``position`` (1 = best Elo, ties share the better position) added.
    """
    from tabarena.contexts import TabArenaContext

    lb = (
        TabArenaContext()
        .compare(
            output_dir=None, ta_results=combined_frame(runs, eval_dir), fillna=None, plot=False
        )
        .reset_index(drop=True)
    )
    lb["position"] = lb["elo"].rank(ascending=False, method="min").astype(int)
    return lb


def main(argv: Sequence[str] | None = None) -> None:
    """Print each entry's and the reference rows' Elo, interval and position in one fit."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "runs", nargs="+", help="merged entry names; the field comes from the first"
    )
    parser.add_argument("--eval-dir", type=Path, default=EVAL)
    args = parser.parse_args(argv)
    lb = same_fit(args.runs, args.eval_dir)
    n = len(lb)
    for i, run in enumerate(args.runs):
        print(f"E{i} = {run}")
    print(f"field: {n} methods")
    shown = lb["method"].astype(str).str.startswith("[New]") | lb["method"].isin(REFERENCES)
    for _, row in lb[shown].iterrows():
        print(
            f"{row['method']}: Elo {row['elo']:.1f} (+{row['elo+']:.0f}/-{row['elo-']:.0f}), "
            f"{row['position']} of {n}"
        )


if __name__ == "__main__":
    main()
