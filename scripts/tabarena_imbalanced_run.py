"""Score FinTFM on every split of TabArena-v0.1.9's ``imbalanced`` subset (audit AU5.18).

``docs/results/TABARENA.md`` holds an imbalanced-subset Elo re-aggregated from split 0 alone
(DR3.6). The published imbalanced board scores every split, so this runs the default config on
all of them: the 7 binary imbalanced datasets FinTFM covers, 9 splits each, 63 fits. The other
8 datasets of the 15 are outside the declared problem types or over the 136-feature cap; the
script derives that from the fork's own predicates and the wrapper's declaration, prints it,
and never imputes FinTFM on them, which would score the imputing method instead.

Runs in the TabArena fork's environment (``tabarena`` is not a fintfm dependency), from the
fork's root, because Ray puts the driver's working directory on every worker's path::

    cd ~/proj/tabarena
    PYTHONUNBUFFERED=1 .venv/bin/python -P <fintfm>/scripts/tabarena_imbalanced_run.py \\
        --run-name fintfm_imbalanced_<MMDD> > <fintfm>/runs/au518_imbalanced.log 2>&1; echo "EXIT=$?"

``--dry-run`` prints the coverage and the task count and fits nothing. ``--datasets`` and
``--splits`` narrow a smoke run; a narrowed run prints that it is not the subset's number.
Results land in the fork's ``experiments/<run-name>/`` and the leaderboard CSVs in
``eval/<run-name>/``. A new run name is needed for any changed configuration, because results
are cached per (config, dataset, split) (``TABARENA.md``, "Four things that cost a run").
"""

from __future__ import annotations

import argparse
import importlib.util
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

#: The checkpoint's feature budget; the wrapper raises above it under the default config.
MAX_FEATURES = 136


def _subset_elo_module() -> ModuleType:
    """Load the sibling ``tabarena_subset_elo.py``, whose subset resolution this script reuses.

    Loaded by path because ``python -P`` keeps the script's directory off ``sys.path``.
    """
    path = Path(__file__).with_name("tabarena_subset_elo.py")
    spec = importlib.util.spec_from_file_location("tabarena_subset_elo", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def covered_datasets(subset: str) -> tuple[list[str], dict[str, str]]:
    """Split a subset's datasets into those FinTFM's default config can score and the rest.

    The problem types come from the wrapper's ``_supported_problem_types``, never restated here
    (``run_fintfm_lite.py`` reported 90% coverage for a binary-only model when it did).

    Args:
        subset: A ``TabArenaContext.subset_predicates`` name, e.g. ``"imbalanced"``.

    Returns:
        The sorted covered dataset names, and every excluded dataset mapped to its reason.
    """
    from tabarena.contexts import TabArenaContext
    from tabarena.models.fintfm.model import FinTFMModel

    wanted = _subset_elo_module().subset_datasets(subset)
    tm = TabArenaContext().task_metadata.set_index("dataset_name")
    kinds = tuple(FinTFMModel._supported_problem_types)
    covered, excluded = [], {}
    for name in wanted:
        row = tm.loc[name]
        if row["problem_type"] not in kinds:
            excluded[name] = f"{row['problem_type']}, not declared"
        elif row["n_features"] > MAX_FEATURES:
            excluded[name] = f"{int(row['n_features'])} features, over {MAX_FEATURES}"
        else:
            covered.append(name)
    return sorted(covered), excluded


def main(argv: Sequence[str] | None = None) -> None:
    """Resolve the covered datasets, fit the default config on their splits, print the row."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", default="fintfm_imbalanced", help="results directory name")
    parser.add_argument("--subset", default="imbalanced")
    parser.add_argument(
        "--datasets", default=None, help="comma-separated smoke subset of the covered ones"
    )
    parser.add_argument("--splits", default=None, help='comma-separated split ids, e.g. "r0f0"')
    parser.add_argument("--fork", type=Path, default=Path.cwd(), help="the TabArena fork's root")
    parser.add_argument(
        "--dry-run", action="store_true", help="print coverage and task count; fit nothing"
    )
    args = parser.parse_args(argv)

    from tabarena.benchmark.task.metadata.collection import TaskSubset
    from tabarena.contexts import TabArenaContext

    covered, excluded = covered_datasets(args.subset)
    n_total = len(covered) + len(excluded)
    print(f"{args.subset}: {n_total} datasets; FinTFM covers {len(covered)}: {covered}")
    for name, reason in sorted(excluded.items()):
        print(f"  excluded {name}: {reason}")
    datasets = covered
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        missing = sorted(set(datasets) - set(covered))
        if missing:
            raise SystemExit(f"not covered or not in the subset: {missing}")
    splits = [s.strip() for s in args.splits.split(",")] if args.splits else None
    narrowed = datasets != covered or splits is not None
    if narrowed:
        print("NARROWED RUN: not the subset's number")

    context = TabArenaContext()
    task_subset = TaskSubset(dataset_names=datasets, split_indices=splits)
    # counted from the same filter build_and_run_jobs applies, so the two cannot disagree
    grid = context.task_metadata_collection.subset_tasks(**task_subset.as_kwargs()).task_grid()
    print(f"fits: {len(grid)} (dataset, split) pairs, default config only")
    if args.dry_run:
        return

    from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle

    experiments = TabArenaV0pt1ExperimentBundle(models=[("FinTFM", 0)]).build_experiments()
    context.build_and_run_jobs(
        experiments,
        expname=str(args.fork / "experiments" / args.run_name),
        task_subset=task_subset,
        new_result_prefix="[New] ",
        debug_mode=True,
    )
    leaderboard = context.compare(output_dir=args.fork / "eval" / args.run_name, datasets=datasets)
    leaderboard = leaderboard.reset_index(drop=True)
    leaderboard["position"] = leaderboard["elo"].rank(ascending=False, method="min").astype(int)
    row = leaderboard[leaderboard["method"].str.startswith("[New] ")].iloc[0]
    print(
        f"{row['method']} on {len(datasets)} of {n_total} {args.subset} datasets: "
        f"Elo {row['elo']:.1f} (+{row['elo+']:.1f}/-{row['elo-']:.1f}), position {row['position']} "
        f"of {len(leaderboard)}, win rate {row['winrate']:.3f}, mean rank {row['rank']:.2f}"
    )


if __name__ == "__main__":
    main()
