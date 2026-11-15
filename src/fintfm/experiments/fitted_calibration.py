"""Analytic, fitted and uncorrected calibration, compared on every horizon and seed (4.3).

``openspec/changes/fitted-calibration`` task 4.3 asks whether a temperature fitted on a held-out
slice of the training rows (4.1, 4.2) beats the analytic label-shift correction of decision D5
on expected calibration error, across all horizons and several seeds. ``FINDINGS.md`` §6 is the
motivating case: at the Polish 1-year horizon the analytic shift overshot (ECE 0.0151 to 0.0173).

Four arms are scored on the same held-out test rows of each panel, per seed:

- ``uncorrected``: ``correct_prior=False``. The broken configuration, kept as a permanent arm.
- ``analytic``: ``correct_prior=True``, no fitted slice. The current default.
- ``fitted``: ``correct_prior=False`` and ``calibration_val_frac > 0``, so the temperature is
  fitted on the uncorrected probabilities alone.
- ``analytic_fitted``: both, as ``FinancialTFMClassifier`` applies them together.

The two fitted arms hold ``calibration_val_frac`` of the training rows out of the context, so
they draw a different (smaller) context from the other two; the record states this beside the
numbers rather than hiding it. Every arm reports the mean predicted probability beside the
observed rate, ECE, Brier, AUC, average precision and the prediction-spread fields
(:func:`fintfm.evaluation.metrics.spread_fields`), because a low ECE reached by predicting the
base rate for everyone is not skill.

Seed means are reported per (panel, context strategy, arm). The paired comparison takes, for
each seed, the difference of two arms' metric on the **same** test rows with a row bootstrap of
that difference, then reports the mean difference over seeds, the number of seeds where the
first arm is better, and the number whose bootstrap interval excludes zero. With few seeds the
seed count is the weak link; nothing here converts three seeds into a p-value.

Usage (MEASURED only on the published checkpoint; the tests use a tiny untrained model)::

    uv run fintfm-fitcal --model runs/v4-cellattn-labels.pt --out runs/fitted-calibration
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from fintfm.evaluation.datasets import CreditDataset, load_polish_bankruptcy
from fintfm.evaluation.metrics import evaluate_binary, expected_calibration_error, spread_fields
from fintfm.inference.classifier import FinancialTFMClassifier

#: Arm name to the classifier settings that define it (``calibration_val_frac`` is added to the
#: fitted arms at run time).
ARMS: dict[str, dict[str, bool]] = {
    "uncorrected": {"correct_prior": False, "fitted": False},
    "analytic": {"correct_prior": True, "fitted": False},
    "fitted": {"correct_prior": False, "fitted": True},
    "analytic_fitted": {"correct_prior": True, "fitted": True},
}

#: Paired comparisons, as ``(a, b)``: a metric difference ``a - b`` below zero means ``a`` wins.
PAIRS: tuple[tuple[str, str], ...] = (
    ("fitted", "analytic"),
    ("analytic_fitted", "analytic"),
    ("analytic", "uncorrected"),
)

#: Metrics the paired comparison bootstraps; lower is better for both.
PAIRED_METRICS: tuple[str, ...] = ("ece", "brier")


def _ece(y: np.ndarray, p: np.ndarray) -> float:
    """Expected calibration error, in the form the bootstrap takes."""
    return float(expected_calibration_error(y, p)[0])


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of the probabilities."""
    return float(np.mean((p - y) ** 2))


_METRIC_FN: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {"ece": _ece, "brier": _brier}


def paired_metric_delta(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    metric: str,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, tuple[float, float]]:
    """Difference of a calibration metric between two arms on the same rows, with a bootstrap CI.

    Rows are resampled with replacement and both arms rescored on each resample, which keeps
    the pairing. The panels carry no company identifier (``FINDINGS.md`` §7), so a row
    bootstrap is the only resampling available.

    Args:
        y: ``(n,)`` binary outcomes.
        p_a: ``(n,)`` arm A's predicted positive-class probability.
        p_b: ``(n,)`` arm B's, on the same rows in the same order.
        metric: ``"ece"`` or ``"brier"``.
        n_boot: Resamples; ``0`` returns a NaN interval.
        seed: Bootstrap seed.

    Returns:
        ``(delta, (lo, hi))``: ``metric(A) - metric(B)`` on the full sample and the 95%
        percentile interval. Negative means A is better calibrated or more accurate.

    Raises:
        ValueError: If the arrays disagree in length or ``metric`` is unknown.
    """
    if metric not in _METRIC_FN:
        raise ValueError(f"metric must be one of {sorted(_METRIC_FN)}, got {metric!r}")
    y, p_a, p_b = np.asarray(y), np.asarray(p_a, float), np.asarray(p_b, float)
    if not (y.shape == p_a.shape == p_b.shape):
        raise ValueError(f"shape mismatch: y {y.shape}, p_a {p_a.shape}, p_b {p_b.shape}")
    fn = _METRIC_FN[metric]
    delta = fn(y, p_a) - fn(y, p_b)
    if n_boot <= 0:
        return delta, (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        draws[i] = fn(y[idx], p_a[idx]) - fn(y[idx], p_b[idx])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(delta), (float(lo), float(hi))


def _split(y: np.ndarray, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/test row indices, deterministic in ``seed``.

    Args:
        y: ``(n,)`` binary labels.
        test_frac: Share of rows (per class) held out for scoring.
        seed: Split seed.

    Returns:
        ``(train_rows, test_rows)``, sorted and disjoint.
    """
    rng = np.random.default_rng([int(seed), 7])
    test: list[np.ndarray] = []
    for c in np.unique(y):
        rows = rng.permutation(np.flatnonzero(y == c))
        test.append(rows[: max(1, round(test_frac * len(rows)))])
    te = np.sort(np.concatenate(test))
    return np.setdiff1d(np.arange(len(y)), te), te


def score_cell(
    model: Any,
    ds: CreditDataset,
    strategy: str,
    seed: int,
    max_context: int,
    val_frac: float,
    test_frac: float,
    arms: Sequence[str] = tuple(ARMS),
) -> tuple[dict[str, dict], dict[str, np.ndarray], np.ndarray]:
    """Fit and score every arm for one panel, context strategy and seed on one shared split.

    Args:
        model: A checkpoint path or a loaded :class:`~fintfm.modeling.model.FinancialTFM`.
        ds: The panel.
        strategy: Context strategy (``"balanced"`` or ``"uniform"`` for the §6 comparison).
        seed: Seeds both the split and the classifier.
        max_context: Context cap.
        val_frac: ``calibration_val_frac`` of the two fitted arms.
        test_frac: Share of rows held out for scoring.
        arms: Arms to run, keys of :data:`ARMS`.

    Returns:
        ``(rows, predictions, y_test)``: per-arm metric dicts, per-arm test probabilities and
        the test labels.
    """
    tr, te = _split(ds.y, test_frac, seed)
    rows: dict[str, dict] = {}
    preds: dict[str, np.ndarray] = {}
    for name in arms:
        spec = ARMS[name]
        started = time.perf_counter()
        clf = FinancialTFMClassifier(
            model,
            max_context=max_context,
            context_strategy=strategy,  # type: ignore[arg-type]
            correct_prior=spec["correct_prior"],
            random_state=seed,
            calibration_val_frac=val_frac if spec["fitted"] else 0.0,
        ).fit(ds.X[tr], ds.y[tr])
        p = np.asarray(clf.predict_proba(ds.X[te]))[:, -1].astype(float)
        met = evaluate_binary(ds.y[te], p)
        summary = clf.context_summary_
        rows[name] = {
            "mean_predicted": met.mean_predicted,
            "observed_rate": met.base_rate,
            "ece": met.ece,
            "brier": met.brier,
            "roc_auc": met.roc_auc,
            "average_precision": met.average_precision,
            **spread_fields(p),
            "temperature": clf.temperature_,
            "context_rate": summary.context_rate,
            "population_rate": summary.population_rate,
            "n_context": summary.n_context,
            "n_train_rows": len(clf.train_rows_),
            "n_test": len(te),
            "seconds": float(time.perf_counter() - started),
        }
        preds[name] = p
    return rows, preds, ds.y[te]


def _mean(values: list[float]) -> float:
    """NaN-aware mean; NaN for an empty or all-NaN list."""
    v = np.array(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")


def run(
    model: Any,
    out_dir: Path,
    horizons: Sequence[int] = (1, 3, 5),
    strategies: Sequence[str] = ("balanced", "uniform"),
    seeds: Sequence[int] = (0, 1, 2),
    max_context: int = 1000,
    val_frac: float = 0.25,
    test_frac: float = 0.3,
    n_boot: int = 1000,
    panels: Sequence[CreditDataset] | None = None,
    model_label: str | None = None,
) -> dict:
    """Score every arm on every panel, strategy and seed, then pool and pair them.

    Args:
        model: Checkpoint path or loaded model.
        out_dir: Directory for ``fitted_calibration.json``.
        horizons: Polish bankruptcy horizons (years) when ``panels`` is not given.
        strategies: Context strategies to cross with the arms.
        seeds: Seeds; the comparison needs at least three to be read at all.
        max_context: Context cap.
        val_frac: ``calibration_val_frac`` of the fitted arms.
        test_frac: Share of each panel held out for scoring.
        n_boot: Bootstrap resamples per paired difference; ``0`` skips the intervals.
        panels: Panels to use instead of the Polish horizons (tests).
        model_label: What to record as the model; defaults to ``model`` when it is a path.

    Returns:
        The record, also written as JSON: ``cells`` (one per panel, strategy, seed, arm),
        ``seed_means`` (one per panel, strategy, arm) and ``paired`` (one per panel, strategy,
        pair, metric).
    """
    data = list(panels) if panels is not None else [load_polish_bankruptcy(h) for h in horizons]
    cells: list[dict] = []
    paired: list[dict] = []
    for ds, strategy in product(data, strategies):
        per_seed: dict[int, tuple[dict, dict, np.ndarray]] = {}
        for seed in seeds:
            rows, preds, y_te = score_cell(
                model, ds, strategy, seed, max_context, val_frac, test_frac
            )
            per_seed[seed] = (rows, preds, y_te)
            for arm, r in rows.items():
                cells.append(
                    {"panel": ds.name, "strategy": strategy, "seed": seed, "arm": arm, **r}
                )
                print(
                    f"  {ds.name} {strategy:>8} seed={seed} {arm:>15} "
                    f"pred={r['mean_predicted']:.4f} obs={r['observed_rate']:.4f} "
                    f"ECE={r['ece']:.4f} Brier={r['brier']:.4f} T={r['temperature']:.3f} "
                    f"sd={r['pred_sd']:.4f} {r['seconds']:.0f}s",
                    flush=True,
                )
        for a, b in PAIRS:
            for metric in PAIRED_METRICS:
                deltas, los, his = [], [], []
                for seed in seeds:
                    rows, preds, y_te = per_seed[seed]
                    d, (lo, hi) = paired_metric_delta(
                        y_te, preds[a], preds[b], metric, n_boot=n_boot, seed=seed
                    )
                    deltas.append(d)
                    los.append(lo)
                    his.append(hi)
                excl = [lo > 0 or hi < 0 for lo, hi in zip(los, his, strict=True)]
                paired.append(
                    {
                        "panel": ds.name,
                        "strategy": strategy,
                        "a": a,
                        "b": b,
                        "metric": metric,
                        "delta_by_seed": deltas,
                        "mean_delta": _mean(deltas),
                        "n_seeds": len(deltas),
                        "n_seeds_a_better": int(sum(d < 0 for d in deltas)),
                        "n_seeds_ci_excludes_zero": int(sum(excl)) if n_boot > 0 else None,
                    }
                )
    seed_means: list[dict] = []
    keys = sorted({(c["panel"], c["strategy"], c["arm"]) for c in cells})
    fields = (
        "mean_predicted",
        "observed_rate",
        "ece",
        "brier",
        "roc_auc",
        "average_precision",
        "pred_sd",
        "pred_p99_over_p50",
        "temperature",
        "context_rate",
        "n_context",
    )
    for panel, strategy, arm in keys:
        grp = [c for c in cells if (c["panel"], c["strategy"], c["arm"]) == (panel, strategy, arm)]
        row: dict[str, Any] = {
            "panel": panel,
            "strategy": strategy,
            "arm": arm,
            "n_seeds": len(grp),
        }
        for f in fields:
            vals = [c[f] for c in grp]
            row[f] = _mean(vals)
            row[f + "_sd"] = float(np.nanstd(vals)) if len(vals) > 1 else float("nan")
        row["n_near_constant"] = int(sum(bool(c["is_near_constant"]) for c in grp))
        seed_means.append(row)
    record = {
        "model": model_label or (model if isinstance(model, str) else "<in-memory model>"),
        "config": {
            "horizons": [d.horizon_years for d in data],
            "panels": [d.name for d in data],
            "strategies": list(strategies),
            "seeds": list(seeds),
            "max_context": max_context,
            "calibration_val_frac": val_frac,
            "test_frac": test_frac,
            "n_boot": n_boot,
        },
        "note": (
            "Verification against held-out rows of the same panel, one row-level split per "
            "seed. The fitted arms train on (1 - calibration_val_frac) of the training rows "
            "and so draw a different context from the uncorrected and analytic arms."
        ),
        "cells": cells,
        "seed_means": seed_means,
        "paired": paired,
        "attribution": [d.attribution for d in data],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fitted_calibration.json").write_text(json.dumps(record, indent=2))
    return record


def summarise(record: dict) -> str:
    """Render seed means and the paired comparison, the level beside the score.

    Args:
        record: A record from :func:`run`.

    Returns:
        A multi-line table: per (panel, strategy, arm) seed means with the mean predicted rate
        beside the observed rate and the spread fields, then each paired difference, then the
        falsification count (cells where fitted ECE is below analytic ECE).
    """
    lines = [
        (
            f"{'panel':<22} {'strategy':>8} {'arm':>15} {'pred':>8} {'obs':>8} {'ECE':>8} "
            f"{'Brier':>8} {'AUC':>7} {'AP':>7} {'sd':>7} {'p99/p50':>8} {'T':>6} {'n':>3}"
        )
    ]
    for r in record["seed_means"]:
        lines.append(
            f"{r['panel']:<22} {r['strategy']:>8} {r['arm']:>15} {r['mean_predicted']:>8.4f} "
            f"{r['observed_rate']:>8.4f} {r['ece']:>8.4f} {r['brier']:>8.4f} {r['roc_auc']:>7.4f} "
            f"{r['average_precision']:>7.4f} {r['pred_sd']:>7.4f} "
            f"{r['pred_p99_over_p50']:>8.2f} {r['temperature']:>6.3f} {r['n_seeds']:>3}"
        )
    lines += ["", "paired differences, a - b, mean over seeds (negative: a is better):"]
    lines.append(
        f"{'panel':<22} {'strategy':>8} {'a - b':>26} {'metric':>7} {'mean d':>9} "
        f"{'a better':>9} {'CI!=0':>6}"
    )
    for r in record["paired"]:
        ci = "-" if r["n_seeds_ci_excludes_zero"] is None else str(r["n_seeds_ci_excludes_zero"])
        lines.append(
            f"{r['panel']:<22} {r['strategy']:>8} {r['a'] + ' - ' + r['b']:>26} "
            f"{r['metric']:>7} {r['mean_delta']:>+9.4f} "
            f"{r['n_seeds_a_better']:>4}/{r['n_seeds']:<4} {ci:>6}"
        )
    ece = [
        r
        for r in record["paired"]
        if (r["a"], r["b"], r["metric"]) == ("fitted", "analytic", "ece")
    ]
    wins = sum(r["mean_delta"] < 0 for r in ece)
    lines += [
        "",
        (
            f"fitted ECE below analytic ECE (seed mean) in {wins} of {len(ece)} "
            "panel/strategy cells; the change is falsified if this is not all of them"
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    """Command-line entry for ``fintfm-fitcal``: run the 4.3 comparison and print it."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="runs/fitted-calibration")
    p.add_argument("--horizons", default="1,3,5", help="Polish bankruptcy horizons, years")
    p.add_argument("--strategies", default="balanced,uniform")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--max-context", type=int, default=1000)
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--test-frac", type=float, default=0.3)
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args()
    record = run(
        a.model,
        Path(a.out),
        horizons=tuple(int(h) for h in a.horizons.split(",")),
        strategies=tuple(s for s in a.strategies.split(",") if s),
        seeds=tuple(int(s) for s in a.seeds.split(",")),
        max_context=a.max_context,
        val_frac=a.val_frac,
        test_frac=a.test_frac,
        n_boot=a.n_boot,
    )
    print("\n" + summarise(record))
    print("\n" + "\n".join(record["attribution"][:1]))


if __name__ == "__main__":
    main()
