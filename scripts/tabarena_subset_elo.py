"""Re-aggregate a held TabArena-Lite comparison on one TabArena-v0.1 subset, with no new fits.

Task DR3.6 (design review 2026-10-16). ``context.compare`` writes ``results_per_split.csv`` next
to every leaderboard, holding every method's error on every (dataset, split) it scored. Scoring
that frame again on fewer datasets gives the subset Elo against exactly the same pool, which
``docs/results/TABARENA.md`` records for the ``imbalanced`` subset.

Runs in the TabArena fork's environment, not this one (``tabarena`` is not a fintfm
dependency), read-only::

    ~/proj/tabarena/.venv/bin/python -P scripts/tabarena_subset_elo.py \\
        ~/proj/tabarena/eval/fintfm_lite_tafintfm/results_per_split.csv --subset imbalanced

It first re-scores the whole frame, so a reader can check the reproduction against the
original leaderboard before trusting the subset number.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import pandas as pd

#: The method whose row is reported; ``[New] `` is the prefix ``run_fintfm_lite.py`` gives it.
DEFAULT_METHOD = "[New] TA-FINTFM (default)"


def subset_datasets(subset: str) -> list[str]:
    """Dataset names of one TabArena-v0.1 subset, from the fork's own predicates.

    Args:
        subset: A subset name of ``TabArenaContext.SUBSET_PREDICATES``, e.g. ``"imbalanced"``.

    Returns:
        Sorted dataset names across all 51 datasets, before any coverage restriction.
    """
    from tabarena.contexts import TabArenaContext

    ctx = TabArenaContext()
    grid = ctx.task_metadata_collection.task_grid()
    keep = ctx.subset_predicates[subset].evaluate(grid, name=subset)
    return sorted(grid.loc[keep, "dataset"].astype(str).unique())


def score(frame: pd.DataFrame, datasets: list[str] | None, method: str) -> pd.Series:
    """One method's leaderboard row over ``datasets`` of ``frame`` (all of it when ``None``).

    Elo is anchored at RF (default) = 1000 by the context's default, and ``fillna=None`` because
    the frame already carries the imputation applied when it was first scored.

    Args:
        frame: A ``results_per_split.csv`` frame.
        datasets: Datasets to keep, or ``None`` for every dataset in the frame.
        method: The method whose row to return.

    Returns:
        The row, with ``position`` (1 = best Elo, ties share the better position) and
        ``n_methods`` added.
    """
    from tabarena.contexts import TabArenaContext

    lb = TabArenaContext().compare(
        output_dir=None, ta_results=frame, datasets=datasets, fillna=None, plot=False
    ).reset_index(drop=True)
    lb["position"] = lb["elo"].rank(ascending=False, method="min").astype(int)
    row = lb[lb["method"] == method].iloc[0].copy()
    row["n_methods"] = len(lb)
    return row


def main(argv: Sequence[str] | None = None) -> None:
    """Print the full-frame reproduction, the subset's coverage, and the subset row."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", help="a results_per_split.csv written by context.compare")
    parser.add_argument("--subset", default="imbalanced")
    parser.add_argument("--method", default=DEFAULT_METHOD)
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.results)
    wanted = subset_datasets(args.subset)
    covered = sorted(set(wanted) & set(frame["dataset"]))
    print(f"frame: {frame['method'].nunique()} methods x {frame['dataset'].nunique()} datasets, "
          f"splits {sorted(frame['fold'].unique().tolist())}")
    print(f"{args.subset}: {len(wanted)} datasets; {len(covered)} in the frame: {covered}")
    print(f"not in the frame: {sorted(set(wanted) - set(covered))}")
    sub = frame[frame["dataset"].isin(covered)]
    print(f"imputed rows in the subset frame: {int(sub['imputed'].sum())} of {len(sub)}, "
          f"{args.method}: {int(sub.loc[sub['method'] == args.method, 'imputed'].sum())}")
    for label, datasets in (("all datasets", None), (args.subset, covered)):
        r = score(frame, datasets, args.method)
        print(f"{label}: Elo {r['elo']:.1f} (+{r['elo+']:.1f}/-{r['elo-']:.1f}), "
              f"position {r['position']} of {r['n_methods']}, win rate {r['winrate']:.3f}, "
              f"mean rank {r['rank']:.2f}")


if __name__ == "__main__":
    main()
