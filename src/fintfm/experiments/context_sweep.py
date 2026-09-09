"""Which context construction wins, measured out of time rather than cited.

``docs/FINDINGS.md`` §29. The default ``context_strategy="balanced"`` was adopted from
published evidence (Tanna et al. 2026: balanced worth 3-4 AUC points over uniform on
credit-risk TFMs, ``docs/FINDINGS.md`` §5, decision D5). On the V4FinBench out-of-time split
the ordering is **reversed and three times larger**: uniform beats balanced by 10-12 mean AUC
points at every context size tested.

This module exists so that result is reproducible on demand rather than resting on a
throwaway script, because it reverses a decision (D9) and a decision reversal has to be
re-runnable by whoever doubts it.

Two properties of the design matter for reading the output:

**Base-rate correction is on in every arm.** Without it the comparison is meaningless — an
uncorrected balanced context states a default rate near 50%, which was §28's bug and hid this
effect entirely.

**Context size varies alongside strategy**, because the obvious objection to §29 is that
uniform simply gets a more representative sample and balanced would catch up with a bigger
budget. It does not: uniform at 1,000 rows with 12 in-context defaults beats balanced at
4,000 rows with 1,122.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fintfm.evaluation.datasets import load_v4finbench
from fintfm.evaluation.metrics import evaluate_binary
from fintfm.experiments.v4_out_of_time import _curve_truth, time_split
from fintfm.inference.classifier import ContextStrategy, FinancialTFMClassifier

#: Strategies compared. The first three are blind — they never look at the query. §29
#: reports them in this order; ``retrieval`` was added for §31, which showed *which*
#: rows is the only remaining lever on the dominant term of the gap.
STRATEGIES: tuple[ContextStrategy, ...] = ("balanced", "hybrid", "uniform", "retrieval")

#: Context budgets. Three sizes give three replications of the strategy ordering, which is
#: the only reason a single-seed result is reported at all.
CONTEXT_SIZES: tuple[int, ...] = (1000, 2000, 4000)


@dataclass
class SweepCell:
    """One (strategy, context size) cell of the sweep."""

    strategy: str
    max_context: int
    seed: int
    context_rate: float
    population_rate: float
    n_context_positive: int
    mean_auc: float
    mean_ece: float
    auc_by_horizon: list[float]
    seconds: float


def run(
    model_path: str,
    out_dir: Path,
    max_rows: int = 120_000,
    train_until: int = 2016,
    test_from: int = 2017,
    seeds: tuple[int, ...] = (0,),
    retrieval_groups: int = 64,
) -> dict:
    """Sweep context strategy against context size on the out-of-time survival split.

    Args:
        model_path: Checkpoint carrying a hazard head.
        out_dir: Directory for ``context_sweep.json``.
        max_rows: Row cap passed to the V4FinBench loader.
        train_until: Last training year, inclusive.
        test_from: First test year, inclusive.
        seeds: Seeds per cell. §29 used one; three are needed before the result is promoted
            (``openspec/changes/revisit-context-strategy`` task 32.1).
        retrieval_groups: Query groups sharing a retrieved context, for the ``retrieval``
            strategy only. Ignored by the blind strategies.

    Returns:
        The recorded result dictionary.
    """
    ds = load_v4finbench(max_rows=max_rows)
    tr, te = time_split(ds, train_until, test_from)
    truth, seen = _curve_truth(ds, te)
    print(f"{ds.name}: train {len(tr):,} (<= {train_until})  test {len(te):,} (>= {test_from})")
    print(f"  population default rate {ds.y[tr].mean():.3%}")

    cells: list[SweepCell] = []
    for max_context in CONTEXT_SIZES:
        for strategy in STRATEGIES:
            for seed in seeds:
                started = time.perf_counter()
                clf = FinancialTFMClassifier(
                    model_path,
                    max_context=max_context,
                    context_strategy=strategy,
                    correct_prior=True,  # never off; see the module docstring
                    random_state=seed,
                    retrieval_groups=retrieval_groups,
                ).fit(ds.X[tr], ds.y[tr])
                curve = clf.predict_term_structure(ds.X[te])
                elapsed = time.perf_counter() - started
                aucs, eces = [], []
                for k in range(curve.shape[1]):
                    m = seen[:, k]
                    if m.sum() < 100 or len(np.unique(truth[m, k])) < 2:
                        continue
                    met = evaluate_binary(truth[m, k], curve[m, k])
                    aucs.append(met.roc_auc)
                    eces.append(met.ece)
                # for retrieval, _ctx_rate is the whole pool's rate and says nothing about
                # what was actually retrieved; report the pooled context rate instead
                if strategy == "retrieval":
                    rate = clf.pooled_context_rate_
                    n_pos = round(rate * max_context)
                else:
                    rate, n_pos = clf._ctx_rate, int(clf._ctx_y.sum())
                cell = SweepCell(
                    strategy=strategy,
                    max_context=max_context,
                    seed=seed,
                    context_rate=rate,
                    population_rate=clf._full_rate,
                    n_context_positive=n_pos,
                    mean_auc=float(np.mean(aucs)),
                    mean_ece=float(np.mean(eces)),
                    auc_by_horizon=[float(a) for a in aucs],
                    seconds=float(elapsed),
                )
                cells.append(cell)
                print(
                    f"  {strategy:>9} ctx={max_context:<5} rate={cell.context_rate:>7.2%} "
                    f"pos={cell.n_context_positive:<5} AUC={cell.mean_auc:.4f} "
                    f"ECE={cell.mean_ece:.4f} {cell.seconds:>6.0f}s",
                    flush=True,
                )

    record = {
        "dataset": ds.name,
        "attribution": ds.attribution,
        "config": {
            "model": model_path,
            "max_rows": max_rows,
            "train_until": train_until,
            "test_from": test_from,
            "seeds": list(seeds),
            "retrieval_groups": retrieval_groups,
        },
        "cells": [asdict(c) for c in cells],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "context_sweep.json").write_text(json.dumps(record, indent=2))
    return record


def summarise(record: dict) -> str:
    """Render the sweep, with the context's default rate beside the score.

    The context rate is the column that carries §29's mechanism: mean AUC tracks it
    monotonically and ignores the positive count, which is what rules out "balanced supplies
    more signal about the rare class".
    """
    header = (
        f"{'ctx':>6} {'strategy':>9} {'seed':>5} {'ctx rate':>9} {'n_pos':>6} "
        f"{'meanAUC':>8} {'meanECE':>8} {'secs':>7}"
    )
    lines = [header]
    for c in record["cells"]:
        lines.append(
            f"{c['max_context']:>6} {c['strategy']:>9} {c['seed']:>5} "
            f"{c['context_rate']:>8.2%} {c['n_context_positive']:>6} "
            f"{c['mean_auc']:>8.4f} {c['mean_ece']:>8.4f} {c['seconds']:>7.0f}"
        )
    best = max(record["cells"], key=lambda c: c["mean_auc"])
    verdict = (
        f"best mean AUC: {best['strategy']} at {best['max_context']} rows "
        f"({best['mean_auc']:.4f}, ECE {best['mean_ece']:.4f})"
    )
    lines += ["", verdict]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=str, default="runs/context-sweep")
    p.add_argument("--max-rows", type=int, default=120_000)
    p.add_argument("--train-until", type=int, default=2016)
    p.add_argument("--test-from", type=int, default=2017)
    p.add_argument("--seeds", type=str, default="0", help="comma-separated")
    p.add_argument("--retrieval-groups", type=int, default=64)
    args = p.parse_args()
    record = run(
        args.model,
        Path(args.out),
        max_rows=args.max_rows,
        train_until=args.train_until,
        test_from=args.test_from,
        seeds=tuple(int(s) for s in args.seeds.split(",")),
        retrieval_groups=args.retrieval_groups,
    )
    print("\n" + summarise(record))
    print(f"\n{record['attribution']}")


if __name__ == "__main__":
    main()
