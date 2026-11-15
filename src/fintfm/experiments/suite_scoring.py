"""Score every arm across the ten licence-cleared tabular datasets (openspec 37.3 and 37.4).

``diverse-eval-suite`` replaces "does V4FinBench go up" (a benchmark one column solves, §42)
with a suite of ten datasets from ``inria-soda/tabular-benchmark``
(:mod:`fintfm.evaluation.tabular_benchmark`). This harness scores, on every dataset:

- each supplied fintfm checkpoint, with its usual inference settings (``max_context``,
  ``n_ensemble``, ``feature_chunk``; flags below, defaults as ``openml_breadth``);
- a seeded **untrained control** of the first checkpoint's architecture, the floor (CLAUDE.md
  "Every evaluation needs a floor"; a trained arm below it means training steered it wrong);
- logistic regression, and LightGBM, CatBoost and XGBoost with library defaults plus the fold
  seed, fitted out-of-process (:mod:`fintfm.evaluation.boosting`: torch and LightGBM segfault
  together on macOS).

Protocol
--------
- **Stratified k folds** (default 3, ``--folds``); the fold seed also seeds every arm.
- **A stated row cap** (``--max-rows``, default 30,000): a dataset larger than the cap is
  stratified-subsampled to it once, with a fixed seed, before the folds are cut. A fold then
  trains on ``(k-1)/k`` of the capped rows and scores the remaining ``1/k``, so a cap of
  30,000 at 3 folds trains on 20,000 rows. The cap and the rows actually used per dataset are
  recorded in each dataset's JSON and printed beside its table row. Higgs, covertype
  (both), MiniBooNE, Diabetes130US and road-safety exceed 30,000 rows and are capped.
  **A capped result is not a full-data result**: a context conclusion is a statement about its
  pool (CLAUDE.md, "``max_context`` is a fraction of the pool it samples from"), and
  ``max_context`` here is 1,000 of a 20,000-row training pool.
- **Average precision first, then ROC-AUC**, per dataset and fold, with the dataset's base rate
  beside them (the AP of a constant predictor), because several datasets are balanced by
  construction and AP then reads high for every arm.
- **Native coverage** (:class:`fintfm.evaluation.coverage.Coverage`): every (dataset, fold,
  arm) is a unit. A checkpoint is skipped, with its reason, on a dataset wider than its
  ``max_features``; a booster that cannot be fitted is skipped, never silently absent.
- **Spread, not only the mean** (task 37.4): per arm, the mean over datasets of the per-dataset
  fold mean, the standard deviation across datasets (``ddof=1``), and the per-dataset minimum
  and maximum with the datasets that set them. A dataset an arm skipped is excluded from that
  arm's aggregate, so a second aggregate over the datasets **every arm** scored is also given;
  compare arms on that one.
- **A single fold is not a result** (CLAUDE.md): the per-dataset figure is the fold mean, and
  the per-fold values and the fold standard deviation are kept beside it.
- **Resumable per dataset**: ``<out>/<dataset>.json`` is written when a dataset finishes and
  re-used on the next run if its configuration (folds, cap, seeds, arms, inference settings)
  is unchanged; a changed configuration refuses to mix and asks for a new ``--out``.

Numbers are MEASURED only when produced by the command in the openspec Progress note on the
real checkpoint; the unit tests are SMOKE-TEST only and run on synthetic data with stub models.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from fintfm.evaluation.coverage import Coverage

#: Default cap on rows per dataset before the folds are cut.
DEFAULT_MAX_ROWS = 30_000

#: Seed of the one-off row cap, fixed so the same rows are used for every arm and every run.
CAP_SEED = 20261115

#: Seed for the untrained control's weights; shared with the other experiment floors.
UNTRAINED_SEED = 20260910

#: Name of the untrained-control arm.
CONTROL_ARM = "untrained_control"

#: Non-fintfm arms, in report order.
BASELINE_ARMS: tuple[str, ...] = ("logreg", "lightgbm", "catboost", "xgboost")


class ArmSkip(Exception):
    """Raised by a predictor that cannot score a task; the message is the reason recorded."""


#: A predictor fits on ``(X_train, y_train, X_test, seed)`` and returns positive-class scores.
#: It raises :class:`ArmSkip` (or returns ``None``) when it cannot score the task.
Predictor = Callable[[np.ndarray, np.ndarray, np.ndarray, int], "np.ndarray | None"]


class DatasetLike(Protocol):
    """What the runner needs from a loaded dataset (satisfied by ``TabularDataset``)."""

    X: pd.DataFrame
    y: np.ndarray


def to_float_matrix(X: pd.DataFrame) -> np.ndarray:
    """Convert a feature frame to ``float32``, mapping categorical columns to their codes.

    Args:
        X: Feature frame; non-numeric columns are ``category`` as the loader returns them.

    Returns:
        ``(n, d)`` ``float32`` array; missing categories become NaN.
    """
    cols = []
    for col in X.columns:
        s = X[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            codes = s.cat.codes.to_numpy(dtype=np.float32)
            codes[codes < 0] = np.nan
            cols.append(codes)
        else:
            cols.append(s.to_numpy(dtype=np.float32))
    return np.column_stack(cols) if cols else np.empty((len(X), 0), dtype=np.float32)


def cap_rows(
    X: np.ndarray, y: np.ndarray, max_rows: int, seed: int = CAP_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified-subsample a dataset to at most ``max_rows`` rows, class shares preserved.

    Args:
        X: Features ``(n, d)``.
        y: Integer labels ``(n,)``.
        max_rows: Row cap; a dataset at or under it is returned unchanged.
        seed: Seed of the draw.

    Returns:
        ``(X, y)`` of at most ``max_rows`` rows, in original row order.
    """
    n = len(y)
    if n <= max_rows:
        return X, y
    from sklearn.model_selection import train_test_split

    idx, _ = train_test_split(
        np.arange(n), train_size=max_rows, random_state=seed, stratify=y, shuffle=True
    )
    idx = np.sort(idx)
    return X[idx], y[idx]


def _scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Average precision and ROC-AUC of ``p`` against ``y`` (NaN with a single class)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    if len(np.unique(y)) < 2:
        return {"average_precision": float("nan"), "roc_auc": float("nan")}
    p = np.nan_to_num(np.asarray(p, dtype=float), nan=0.5)
    return {
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
    }


def score_dataset(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    predictors: Mapping[str, Predictor],
    folds: int,
    seed: int,
    max_rows: int,
    n_rows_full: int | None = None,
) -> dict[str, Any]:
    """Score every arm on one dataset over stratified folds.

    Args:
        name: Dataset key, for unit names and the record.
        X: Features ``(n, d)`` (already capped).
        y: Binary labels ``(n,)``.
        predictors: ``{arm: predictor}``.
        folds: Number of stratified folds (at least 2).
        seed: Fold seed; fold ``f`` seeds every arm with ``seed + f``.
        max_rows: The cap that was applied, recorded.
        n_rows_full: Rows before the cap, recorded; ``None`` means no cap was needed.

    Returns:
        A record with ``cells[arm][metric]`` (per-fold list, NaN where skipped), ``fold_seconds``
        per arm, the dataset's ``base_rate``, ``coverage`` and the row counts.

    Raises:
        ValueError: If ``y`` is not binary or ``folds`` is below 2.
    """
    from sklearn.model_selection import StratifiedKFold

    if folds < 2:
        raise ValueError(f"folds must be at least 2, got {folds}")
    y = np.asarray(y).astype(int)
    if set(np.unique(y)) - {0, 1}:
        raise ValueError(f"{name}: labels are not binary: {sorted(set(np.unique(y)))}")
    cov = Coverage()
    cells: dict[str, dict[str, list[float]]] = {
        arm: {"average_precision": [], "roc_auc": []} for arm in predictors
    }
    seconds: dict[str, list[float]] = {arm: [] for arm in predictors}
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for f, (tr, te) in enumerate(splitter.split(X, y)):
        for arm, predict in predictors.items():
            unit = f"{name}/fold{f}/{arm}"
            cov.attempt(unit)
            t0 = time.time()
            try:
                p = predict(X[tr], y[tr], X[te], seed + f)
                reason = "predictor returned no scores (unavailable or failed)"
            except ArmSkip as exc:
                p, reason = None, str(exc)
            seconds[arm].append(time.time() - t0)
            if p is None:
                cov.skip(unit, reason)
                s = {"average_precision": float("nan"), "roc_auc": float("nan")}
            else:
                cov.scored(unit)
                s = _scores(y[te], p)
            for metric, value in s.items():
                cells[arm][metric].append(value)
    return {
        "dataset": name,
        "n_rows": len(y),
        "n_rows_before_cap": n_rows_full if n_rows_full is not None else len(y),
        "max_rows": max_rows,
        "n_features": int(X.shape[1]),
        "base_rate": float(y.mean()),
        "folds": folds,
        "seed": seed,
        "cells": cells,
        "fold_seconds": seconds,
        "coverage": cov.to_dict(),
    }


def _safe(name: str) -> str:
    """File-name form of a dataset key (``clf_num/Higgs`` to ``clf_num__Higgs``)."""
    return name.replace("/", "__")


def run_suite(
    loaders: Mapping[str, Callable[[], DatasetLike]],
    predictors: Mapping[str, Predictor],
    out_dir: Path,
    folds: int = 3,
    seed: int = 0,
    max_rows: int = DEFAULT_MAX_ROWS,
    settings: Mapping[str, Any] | None = None,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Run every dataset, resuming those already finished, and return their records.

    Args:
        loaders: ``{dataset key: callable returning an object with X (DataFrame) and y}``.
        predictors: ``{arm: predictor}``.
        out_dir: Directory of the per-dataset JSON files.
        folds: Stratified folds.
        seed: Fold seed.
        max_rows: Row cap per dataset.
        settings: Anything else that defines the configuration (checkpoints, inference
            settings); stored and compared on resume so two configurations never mix.
        log: Progress sink.

    Returns:
        One record per dataset, in ``loaders`` order.

    Raises:
        RuntimeError: If a finished dataset in ``out_dir`` was produced by a different
            configuration.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "folds": folds,
        "seed": seed,
        "max_rows": max_rows,
        "arms": list(predictors),
        "settings": dict(settings or {}),
    }
    records = []
    for name, load in loaders.items():
        path = out_dir / f"{_safe(name)}.json"
        if path.exists():
            rec = json.loads(path.read_text())
            if rec.get("config") != config:
                raise RuntimeError(
                    f"{path} was produced by a different configuration {rec.get('config')}; "
                    f"this run is {config}. Use a new --out."
                )
            log(f"resume: {name} already scored ({path.name})")
            records.append(rec)
            continue
        ds = load()
        X_full = to_float_matrix(ds.X)
        y_full = np.asarray(ds.y)
        X, y = cap_rows(X_full, y_full, max_rows)
        log(
            f"=== {name}: {len(y_full)} rows"
            + (f" capped to {len(y)}" if len(y) < len(y_full) else "")
            + f", {X.shape[1]} features, base rate {y.mean():.3f}"
        )
        t0 = time.time()
        rec = score_dataset(name, X, y, predictors, folds, seed, max_rows, len(y_full))
        rec["config"] = config
        rec["wall_seconds"] = time.time() - t0
        path.write_text(json.dumps(rec, indent=2))
        for arm in predictors:
            ap = rec["cells"][arm]["average_precision"]
            log(
                f"  {arm:<18} AP {np.nanmean(ap) if np.isfinite(ap).any() else float('nan'):.4f}"
                f"  {sum(rec['fold_seconds'][arm]):.1f}s"
            )
        records.append(rec)
    return records


def _fold_mean(values: Sequence[float]) -> float:
    """Mean of the finite fold values; NaN when none (the arm was skipped on every fold)."""
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()) if np.isfinite(arr).all() and len(arr) else float("nan")


def _spread(per_dataset: Mapping[str, float]) -> dict[str, Any]:
    """Mean, sample standard deviation and range of per-dataset values (finite ones only).

    Args:
        per_dataset: ``{dataset: value}``; NaN entries are left out.

    Returns:
        ``n``, ``mean``, ``std`` (``ddof=1``, NaN below two datasets), ``min``/``max`` with
        ``min_dataset``/``max_dataset``.
    """
    items = {k: v for k, v in per_dataset.items() if np.isfinite(v)}
    if not items:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"),
                "max": float("nan"), "min_dataset": None, "max_dataset": None}  # fmt: skip
    vals = np.array(list(items.values()))
    lo, hi = min(items, key=items.__getitem__), max(items, key=items.__getitem__)
    return {
        "n": len(items),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
        "min": float(vals[list(items).index(lo)]),
        "max": float(vals[list(items).index(hi)]),
        "min_dataset": lo,
        "max_dataset": hi,
    }


def aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-arm spread across datasets (task 37.4) and the coverage of the whole run.

    Args:
        records: Output of :func:`run_suite`.

    Returns:
        ``arms`` (list), ``datasets`` (list), ``per_dataset[metric][arm][dataset]`` (fold mean),
        ``fold_std[metric][arm][dataset]``, ``spread[metric][arm]`` over every dataset the arm
        scored, ``spread_common[metric][arm]`` over the datasets every arm scored,
        ``common_datasets`` and the merged ``coverage``.
    """
    arms = list(records[0]["cells"]) if records else []
    datasets = [r["dataset"] for r in records]
    metrics = ("average_precision", "roc_auc")
    per: dict[str, dict[str, dict[str, float]]] = {m: {a: {} for a in arms} for m in metrics}
    fstd: dict[str, dict[str, dict[str, float]]] = {m: {a: {} for a in arms} for m in metrics}
    for r in records:
        for m in metrics:
            for a in arms:
                v = r["cells"][a][m]
                per[m][a][r["dataset"]] = _fold_mean(v)
                arr = np.asarray(v, dtype=float)
                ok = np.isfinite(arr).all() and len(arr) > 1
                fstd[m][a][r["dataset"]] = float(arr.std(ddof=1)) if ok else float("nan")
    common = [d for d in datasets if all(np.isfinite(per["average_precision"][a][d]) for a in arms)]
    spread = {m: {a: _spread(per[m][a]) for a in arms} for m in metrics}
    spread_common = {
        m: {a: _spread({d: per[m][a][d] for d in common}) for a in arms} for m in metrics
    }
    skipped: dict[str, str] = {}
    for r in records:
        skipped |= {u: why for u, why in r["coverage"]["skipped"].items()}
    merged = {
        "n_attempted": sum(r["coverage"]["n_attempted"] for r in records),
        "n_scored": sum(r["coverage"]["n_scored"] for r in records),
        "n_skipped": sum(r["coverage"]["n_skipped"] for r in records),
        "skipped": skipped,
    }
    return {
        "arms": arms,
        "datasets": datasets,
        "common_datasets": common,
        "per_dataset": per,
        "fold_std": fstd,
        "spread": spread,
        "spread_common": spread_common,
        "coverage": merged,
    }


def format_report(records: Sequence[Mapping[str, Any]], agg: Mapping[str, Any]) -> str:
    """Render the per-dataset table (37.3) and the spread table (37.4), AP first.

    Args:
        records: Output of :func:`run_suite`.
        agg: Output of :func:`aggregate`.

    Returns:
        Markdown: per metric a datasets-by-arms table of fold mean (``--`` where the arm was
        skipped), then per metric the spread per arm over all scored datasets and over the
        common datasets, then the coverage line and every skip reason.
    """
    arms = agg["arms"]
    titles = {"average_precision": "AVERAGE PRECISION (read first)", "roc_auc": "ROC-AUC"}
    lines = [
        (
            f"datasets: {len(records)}; folds: {records[0]['folds'] if records else 0}; "
            "per-dataset value = mean over folds"
        ),
    ]

    def cell(v: float) -> str:
        """Four-decimal value, or ``--`` for a skipped arm."""
        return f"{v:.4f}" if np.isfinite(v) else "--"

    for m, title in titles.items():
        lines += [f"\n### {title}: per dataset", ""]
        lines.append("| dataset | rows used (of) | d | base rate | " + " | ".join(arms) + " |")
        lines.append("|" + "---|" * (4 + len(arms)))
        for r in records:
            d = r["dataset"]
            vals = " | ".join(cell(agg["per_dataset"][m][a][d]) for a in arms)
            lines.append(
                f"| {d} | {r['n_rows']} ({r['n_rows_before_cap']}) | {r['n_features']} | "
                f"{r['base_rate']:.3f} | {vals} |"
            )
    for m, title in titles.items():
        for key, label in (("spread", "all datasets the arm scored"),
                           ("spread_common", "datasets every arm scored")):  # fmt: skip
            lines += [f"\n### {title}: spread across datasets ({label})", ""]
            lines.append("| arm | n | mean | std (ddof=1) | min (dataset) | max (dataset) |")
            lines.append("|---|---|---|---|---|---|")
            for a in arms:
                s = agg[key][m][a]
                lines.append(
                    f"| {a} | {s['n']} | {cell(s['mean'])} | {cell(s['std'])} | "
                    f"{cell(s['min'])} ({s['min_dataset']}) | {cell(s['max'])} "
                    f"({s['max_dataset']}) |"
                )
    c = agg["coverage"]
    lines.append(
        f"\ncoverage: {c['n_scored']}/{c['n_attempted']} (dataset, fold, arm) units scored, "
        f"{c['n_skipped']} skipped"
    )
    by_reason: dict[str, list[str]] = {}
    for unit, why in c["skipped"].items():
        by_reason.setdefault(why, []).append(unit)
    for why, units in by_reason.items():
        lines.append(f"  skipped {len(units)} units, e.g. {units[0]}: {why}")
    secs: dict[str, float] = {a: 0.0 for a in arms}
    for r in records:
        for a in arms:
            secs[a] += sum(r["fold_seconds"][a])
    lines.append(
        "wall seconds per arm (sum over datasets and folds): "
        + ", ".join(f"{a} {s:.0f}" for a, s in secs.items())
    )
    return "\n".join(lines)


def build_predictors(
    checkpoints: Mapping[str, str],
    boosters: Sequence[str] = ("lightgbm", "catboost", "xgboost"),
    max_context: int = 1000,
    n_ensemble: int = 8,
    feature_chunk: int = 16,
    include: Sequence[str] | None = None,
) -> dict[str, Predictor]:
    """Build the checkpoint arms, the untrained control, logistic regression and the boosters.

    The fintfm arms use the ``openml_breadth`` inference settings (``max_context`` rows of
    context, ``n_ensemble`` members, ``feature_chunk``), median-imputing nothing: the model
    takes NaN natively. A table wider than the checkpoint's ``max_features`` raises
    :class:`ArmSkip` with that reason.

    Args:
        checkpoints: ``{arm name: checkpoint path}``; may be empty.
        boosters: Booster arms to include.
        max_context: Context rows per fintfm member.
        n_ensemble: fintfm ensemble members.
        feature_chunk: fintfm feature chunk.
        include: If given, keep only these arm names (for a smoke test).

    Returns:
        ``{arm: predictor}`` in report order: checkpoints, control, logreg, boosters.
    """
    arms: dict[str, Predictor] = {}
    if checkpoints:
        import torch

        from fintfm.inference.classifier import FinancialTFMClassifier
        from fintfm.modeling.model import FinancialTFM

        models = {name: FinancialTFM.load(path) for name, path in checkpoints.items()}
        torch.manual_seed(UNTRAINED_SEED)  # seeded: an unseeded floor moves by 0.2 (§42)
        models[CONTROL_ARM] = FinancialTFM(next(iter(models.values())).cfg)

        def make(model: Any) -> Predictor:
            """Predictor for one in-context model."""

            def fn(Xs: np.ndarray, ys: np.ndarray, Xt: np.ndarray, seed: int) -> np.ndarray:
                """Fit the in-context classifier and score ``Xt``."""
                if Xs.shape[1] > model.cfg.max_features:
                    raise ArmSkip(
                        f"{Xs.shape[1]} features exceed the checkpoint's max_features "
                        f"{model.cfg.max_features}"
                    )
                clf = FinancialTFMClassifier(
                    model,
                    max_context=max_context,
                    n_ensemble=n_ensemble,
                    feature_chunk=feature_chunk,
                    random_state=seed,
                ).fit(Xs, ys)
                return clf.predict_proba(Xt)[:, 1]

            return fn

        arms |= {name: make(m) for name, m in models.items()}

    def logreg(Xs: np.ndarray, ys: np.ndarray, Xt: np.ndarray, seed: int) -> np.ndarray:
        """Median-impute, standardise, fit logistic regression, score ``Xt``."""
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed),
        )
        return pipe.fit(Xs, ys).predict_proba(Xt)[:, 1]

    def booster(name: str) -> Predictor:
        """Predictor for one booster, library defaults plus the fold seed."""
        key = "random_seed" if name == "catboost" else "random_state"

        def fn(Xs: np.ndarray, ys: np.ndarray, Xt: np.ndarray, seed: int) -> np.ndarray | None:
            """Fit and score in a subprocess; ``None`` if the booster is unavailable."""
            from fintfm.evaluation.boosting import fit_predict_boosting

            return fit_predict_boosting(name, Xs, ys, Xt, params={key: seed})

        return fn

    arms["logreg"] = logreg
    arms |= {b: booster(b) for b in boosters}
    if include is not None:
        unknown = set(include) - set(arms)
        if unknown:
            raise ValueError(f"unknown arms {sorted(unknown)}; available {list(arms)}")
        arms = {a: p for a, p in arms.items() if a in include}
    return arms


def _parse_checkpoints(items: Sequence[str]) -> dict[str, str]:
    """Parse ``name=path`` or bare ``path`` (name = file stem) into an ordered mapping."""
    out: dict[str, str] = {}
    for item in items:
        name, _, path = item.partition("=")
        if not path:
            name, path = Path(item).stem, item
        out[name] = path
    return out


def main(argv: Sequence[str] | None = None) -> None:
    """Command line entry: score the arms over the cleared datasets, print and write the report.

    Args:
        argv: Arguments; ``None`` reads ``sys.argv``.
    """
    from fintfm.evaluation.tabular_benchmark import CLEARED, load_tabular_benchmark

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", action="append", default=[], metavar="[NAME=]PATH")
    p.add_argument("--datasets", nargs="+", default=None, help="cleared keys; default all ten")
    p.add_argument("--arms", nargs="+", default=None, help="restrict arms (smoke tests)")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    p.add_argument("--max-context", type=int, default=1000)
    p.add_argument("--n-ensemble", type=int, default=8)
    p.add_argument("--feature-chunk", type=int, default=16)
    p.add_argument("--threads", type=int, default=None, help="cap torch CPU threads")
    p.add_argument("--out", default="runs/suite-scoring")
    args = p.parse_args(argv)
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)
    keys = args.datasets or list(CLEARED)
    bad = [k for k in keys if k not in CLEARED]
    if bad:
        p.error(f"not cleared datasets: {bad}; cleared: {sorted(CLEARED)}")
    checkpoints = _parse_checkpoints(args.checkpoint)
    predictors = build_predictors(
        checkpoints,
        max_context=args.max_context,
        n_ensemble=args.n_ensemble,
        feature_chunk=args.feature_chunk,
        include=args.arms,
    )
    loaders = {k: (lambda k=k: load_tabular_benchmark(k)) for k in keys}
    t0 = time.time()
    records = run_suite(
        loaders,
        predictors,
        Path(args.out),
        folds=args.folds,
        seed=args.seed,
        max_rows=args.max_rows,
        settings={
            "checkpoints": checkpoints,
            "max_context": args.max_context,
            "n_ensemble": args.n_ensemble,
            "feature_chunk": args.feature_chunk,
        },
    )
    agg = aggregate(records)
    report = format_report(records, agg)
    out = Path(args.out)
    (out / "suite.json").write_text(json.dumps({"records": records, "aggregate": agg}, indent=2))
    (out / "report.md").write_text(report)
    print("\n" + report)
    print(f"\nwrote {out / 'suite.json'} and report.md ({time.time() - t0:.0f}s this run)")


if __name__ == "__main__":
    main()
