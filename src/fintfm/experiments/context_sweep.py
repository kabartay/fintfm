"""Which context construction wins, measured out of time rather than cited.

``docs/results/FINDINGS.md`` §29. The default ``context_strategy="balanced"`` was adopted from
published evidence (Tanna et al. 2026: balanced worth 3-4 AUC points over uniform on
credit-risk TFMs, ``docs/results/FINDINGS.md`` §5, decision D5). On the V4FinBench out-of-time split
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
import itertools
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from fintfm.config import Config, load_config
from fintfm.evaluation.datasets import load_v4finbench
from fintfm.evaluation.metrics import evaluate_binary, holm_adjusted_p, paired_auc_difference
from fintfm.experiments.v4_out_of_time import SplitMode, _curve_truth, split_dataset
from fintfm.inference.classifier import FinancialTFMClassifier

# Which strategies and context sizes are compared now comes from ``context_sweep`` in the
# configuration (``fintfm/configs/default.yaml``), not from constants here. The first three
# strategies are blind — they never look at the query; ``retrieval`` was added for §31, which
# showed *which* rows is the only remaining lever on the dominant term of the gap. Three
# context sizes give three replications of the strategy ordering, which is what made §29
# reportable from one seed per cell.


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
    mean_pred_sd: float = float("nan")
    mean_pred_p99_over_p50: float = float("nan")
    # the level beside the score (CLAUDE.md, "Print the level, not only the score"):
    # mean predicted PD and observed rate over the same scored horizons
    mean_predicted: float = float("nan")
    observed_rate: float = float("nan")
    mean_predicted_by_horizon: list[float] = field(default_factory=list)
    observed_by_horizon: list[float] = field(default_factory=list)
    n_horizons_scored: int = 0


def _nanmean(values: list[float]) -> float:
    """Mean of the finite values, NaN when there are none (no warning, no zero).

    Args:
        values: Values that may include NaN.

    Returns:
        The mean of the finite entries, or NaN.
    """
    vals = [v for v in values if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def paired_strategy_comparisons(
    curves: dict[tuple[int, str, int], np.ndarray],
    truth: np.ndarray,
    seen: np.ndarray,
    groups: np.ndarray | None,
    min_rows: int,
    n_boot: int = 1000,
    seed: int = 0,
) -> list[dict]:
    """Holm-corrected paired bootstrap over every strategy comparison (task 32.3).

    For each context size and seed, every pair of strategies is compared on every scored
    horizon with :func:`fintfm.evaluation.metrics.paired_auc_difference` (the shared paired
    bootstrap, resampling whole companies when ``groups`` is given). All the resulting
    p-values form one family, adjusted together with
    :func:`fintfm.evaluation.metrics.holm_adjusted_p`, so an AUC difference quoted from this
    record always has its family-wise-adjusted p-value beside it.

    Args:
        curves: ``(max_context, strategy, seed) -> (n_test, n_horizons)`` predicted curve on
            the shared test rows.
        truth: ``(n_test, n_horizons)`` cumulative-default labels.
        seen: ``(n_test, n_horizons)`` boolean, True where the horizon is observed.
        groups: Company label per test row, or ``None`` for a row bootstrap.
        min_rows: Fewest observed rows a horizon needs to be scored.
        n_boot: Bootstrap resamples per comparison.
        seed: Bootstrap seed.

    Returns:
        One dict per (context size, seed, strategy pair, horizon) with ``delta`` (AUC of
        ``a`` minus AUC of ``b``), ``ci_lo``, ``ci_hi``, ``p_raw`` and ``p_holm``. Empty
        when fewer than two strategies share a (context size, seed).
    """
    rows: list[dict] = []
    for ctx, sd in sorted({(c, s) for c, _, s in curves}):
        strategies = sorted(st for c, st, s in curves if (c, s) == (ctx, sd))
        for a, b in itertools.combinations(strategies, 2):
            for k in range(truth.shape[1]):
                m = seen[:, k]
                if m.sum() < min_rows or len(np.unique(truth[m, k])) < 2:
                    continue
                d, (lo, hi), p = paired_auc_difference(
                    truth[m, k], curves[(ctx, a, sd)][m, k], curves[(ctx, b, sd)][m, k],
                    n_boot=n_boot, seed=seed, groups=None if groups is None else groups[m],
                )
                rows.append({
                    "max_context": ctx, "seed": sd, "a": a, "b": b, "horizon": k,
                    "delta": d, "ci_lo": lo, "ci_hi": hi, "p_raw": p,
                })
    for r, adj in zip(rows, holm_adjusted_p([r["p_raw"] for r in rows]), strict=True):
        r["p_holm"] = adj
    return rows


def run(
    model_path: str,
    out_dir: Path,
    max_rows: int | None = None,
    train_until: int | None = None,
    test_from: int | None = None,
    seeds: tuple[int, ...] | None = None,
    retrieval_groups: int | None = None,
    cfg: Config | None = None,
    split: SplitMode = "embargoed",
    n_boot: int = 1000,
) -> dict:
    """Sweep context strategy against context size on the out-of-time survival split.

    Args:
        model_path: Checkpoint carrying a hazard head.
        out_dir: Directory for ``context_sweep.json``.
        max_rows: Row cap passed to the V4FinBench loader.
        train_until: Last training year, inclusive.
        test_from: First test year, inclusive.
        seeds: Seeds per cell. Defaults to ``context_sweep.seeds``. §29 used one; §33 used
            three, which is what a comparison between blind strategies needs — hybrid at
            1,000 rows spreads ±0.046 across seeds.
        retrieval_groups: Query groups sharing a retrieved context, for the ``retrieval``
            strategy only. Ignored by the blind strategies. Defaults to
            ``context_sweep.retrieval_groups``.
        cfg: Configuration; loaded from the packaged default when omitted.
        split: ``"embargoed"`` (default: no company on both sides, label-window embargo) or
            ``"legacy"`` (the leaking split FINDINGS §29-§35 used); see
            :func:`fintfm.experiments.v4_out_of_time.split_dataset`.
        n_boot: Bootstrap resamples per strategy comparison (32.3); ``0`` skips it.

    Returns:
        The recorded result dictionary.
    """
    cfg = cfg or load_config()
    max_rows = cfg.v4finbench.max_rows if max_rows is None else max_rows
    train_until = cfg.v4finbench.train_until if train_until is None else train_until
    test_from = cfg.v4finbench.test_from if test_from is None else test_from
    seeds = tuple(cfg.context_sweep.seeds) if seeds is None else seeds
    retrieval_groups = (
        cfg.context_sweep.retrieval_groups if retrieval_groups is None else retrieval_groups
    )
    ds = load_v4finbench(max_rows=max_rows)
    sp = split_dataset(ds, train_until, test_from, split)
    tr, te = sp.train, sp.test
    truth, seen = _curve_truth(ds, te)
    print(
        f"{ds.name}: split={sp.mode} train {len(tr):,} (<= {sp.effective_train_until})  "
        f"test {len(te):,} (>= {test_from})  companies on both sides {sp.n_shared_ids_after}"
    )
    print(f"  population default rate {ds.y[tr].mean():.3%}")

    cells: list[SweepCell] = []
    curves: dict[tuple[int, str, int], np.ndarray] = {}
    for max_context in cfg.context_sweep.context_sizes:
        for strategy in cfg.context_sweep.strategies:
            for seed in seeds:
                started = time.perf_counter()
                # every setting the sweep does not vary comes from cfg.inference (DR1.3)
                clf = FinancialTFMClassifier.from_config(
                    model_path,
                    cfg.inference,
                    max_context=max_context,
                    context_strategy=strategy,
                    correct_prior=True,  # never off; see the module docstring
                    random_state=seed,
                    retrieval_groups=retrieval_groups,
                ).fit(ds.X[tr], ds.y[tr], periods=ds.period[tr])
                curve = clf.predict_term_structure(ds.X[te])
                elapsed = time.perf_counter() - started
                curves[(max_context, strategy, seed)] = np.asarray(curve)
                aucs, eces, sds, ratios, preds, obs = [], [], [], [], [], []
                for k in range(curve.shape[1]):
                    m = seen[:, k]
                    if (
                        m.sum() < cfg.evaluation.min_rows_per_horizon
                        or len(np.unique(truth[m, k])) < 2
                    ):
                        continue
                    met = evaluate_binary(truth[m, k], curve[m, k])
                    aucs.append(met.roc_auc)
                    eces.append(met.ece)
                    sds.append(met.pred_sd)
                    ratios.append(met.pred_p99_over_p50)
                    preds.append(met.mean_predicted)
                    obs.append(met.base_rate)
                # for retrieval, the stored context is the whole pool and its rate says
                # nothing about what was actually retrieved; report the pooled rate instead
                summary = clf.context_summary_
                if strategy == "retrieval":
                    assert summary.pooled_context_rate is not None
                    rate = summary.pooled_context_rate
                    n_pos = round(rate * max_context)
                else:
                    rate, n_pos = summary.context_rate, summary.n_positive
                cell = SweepCell(
                    strategy=strategy,
                    max_context=max_context,
                    seed=seed,
                    context_rate=rate,
                    population_rate=summary.population_rate,
                    n_context_positive=n_pos,
                    mean_auc=_nanmean(aucs),
                    mean_ece=_nanmean(eces),
                    mean_pred_sd=_nanmean(sds),
                    mean_pred_p99_over_p50=_nanmean(ratios),
                    mean_predicted=_nanmean(preds),
                    observed_rate=_nanmean(obs),
                    mean_predicted_by_horizon=[float(v) for v in preds],
                    observed_by_horizon=[float(v) for v in obs],
                    n_horizons_scored=len(aucs),
                    auc_by_horizon=[float(a) for a in aucs],
                    seconds=float(elapsed),
                )
                cells.append(cell)
                print(
                    f"  {strategy:>9} ctx={max_context:<5} rate={cell.context_rate:>7.2%} "
                    f"pos={cell.n_context_positive:<5} AUC={cell.mean_auc:.4f} "
                    f"ECE={cell.mean_ece:.4f} sd={cell.mean_pred_sd:.4f} "
                    f"p99/p50={cell.mean_pred_p99_over_p50:.2f} "
                    f"meanPD={cell.mean_predicted:.4f} obs={cell.observed_rate:.4f} "
                    f"{cell.seconds:>6.0f}s",
                    flush=True,
                )

    comparisons = (
        paired_strategy_comparisons(
            curves, truth, seen, None if ds.groups is None else ds.groups[te],
            cfg.evaluation.min_rows_per_horizon, n_boot=n_boot,
        )
        if n_boot > 0
        else []
    )
    record = {
        "dataset": ds.name,
        "attribution": ds.attribution,
        "config": {
            "model": model_path,
            "max_rows": max_rows,
            "train_until": train_until,
            "test_from": test_from,
            "split": split,
            "seeds": list(seeds),
            "retrieval_groups": retrieval_groups,
            "feature_transform": cfg.inference.feature_transform,
            "config_sources": list(cfg.sources),
        },
        "split": sp.record() | {"n_train": len(tr), "n_test": len(te)},
        "cells": [asdict(c) for c in cells],
        "comparisons": comparisons,
        "comparison_method": (
            "paired bootstrap per horizon, whole companies resampled, n_boot="
            f"{n_boot}; Holm-adjusted across all comparisons in this record"
        ),
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
        f"{'meanAUC':>8} {'meanECE':>8} {'meanPD':>8} {'obs':>8} {'secs':>7}"
    )
    lines = [header]
    for c in record["cells"]:
        lines.append(
            f"{c['max_context']:>6} {c['strategy']:>9} {c['seed']:>5} "
            f"{c['context_rate']:>8.2%} {c['n_context_positive']:>6} "
            f"{c['mean_auc']:>8.4f} {c['mean_ece']:>8.4f} "
            f"{c.get('mean_predicted', float('nan')):>8.4f} "
            f"{c.get('observed_rate', float('nan')):>8.4f} {c['seconds']:>7.0f}"
        )
    # NaN-aware: max() with a NaN key returns whichever cell came first, not the best
    scored = [c for c in record["cells"] if np.isfinite(c["mean_auc"])]
    n_nan = len(record["cells"]) - len(scored)
    if scored:
        best = max(scored, key=lambda c: c["mean_auc"])
        verdict = (
            f"best mean AUC: {best['strategy']} at {best['max_context']} rows "
            f"({best['mean_auc']:.4f}, ECE {best['mean_ece']:.4f}); "
            f"{n_nan} cell(s) with no scorable horizon excluded"
        )
    else:
        verdict = f"no cell had a scorable horizon ({n_nan} cells)"
    lines += ["", verdict]
    cmps = record.get("comparisons") or []
    if cmps:
        lines += ["", f"paired strategy comparisons ({len(cmps)} in the Holm family):"]
        lines.append(
            f"{'ctx':>6} {'seed':>5} {'a - b':>22} {'horizons':>9} {'mean dAUC':>10} "
            f"{'min Holm p':>11} {'Holm<.05':>9}"
        )
        keys = sorted({(c["max_context"], c["seed"], c["a"], c["b"]) for c in cmps})
        for ctx, sd, a, b in keys:
            g = [c for c in cmps if (c["max_context"], c["seed"], c["a"], c["b"]) == (ctx, sd, a, b)]
            ps = [c["p_holm"] for c in g if np.isfinite(c["p_holm"])]
            lines.append(
                f"{ctx:>6} {sd:>5} {a + ' - ' + b:>22} {len(g):>9} "
                f"{_nanmean([c['delta'] for c in g]):>+10.4f} "
                f"{(min(ps) if ps else float('nan')):>11.4f} {sum(p < 0.05 for p in ps):>9}"
            )
    return "\n".join(lines)


def main() -> None:
    """Sweep context size and construction strategy.

    Entry point for the ``context-sweep`` console script; see
    ``--help`` for the flags. Writes its record as JSON under ``--out``.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=str, default="runs/context-sweep")
    p.add_argument("--config", type=str, default=None, help="YAML overriding the defaults")
    # None means "take it from the configuration"; an explicit flag wins over both layers
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--train-until", type=int, default=None)
    p.add_argument("--test-from", type=int, default=None)
    p.add_argument("--seeds", type=str, default=None, help="comma-separated")
    p.add_argument("--retrieval-groups", type=int, default=None)
    p.add_argument("--n-boot", type=int, default=1000,
                   help="bootstrap resamples per strategy comparison; 0 skips the Holm table")
    p.add_argument(
        "--split", choices=("embargoed", "legacy"), default="embargoed",
        help="'legacy' reproduces the leaking split of FINDINGS 29-35",
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    record = run(
        args.model,
        Path(args.out),
        max_rows=args.max_rows,
        train_until=args.train_until,
        test_from=args.test_from,
        seeds=(
            tuple(int(s) for s in args.seeds.split(",")) if args.seeds else None
        ),
        retrieval_groups=args.retrieval_groups,
        cfg=cfg,
        split=args.split,
        n_boot=args.n_boot,
    )
    print("\n" + summarise(record))
    print(f"\nconfig: {cfg.provenance()}")
    print(f"\n{record['attribution']}")


if __name__ == "__main__":
    main()
