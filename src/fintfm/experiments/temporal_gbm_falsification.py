"""Falsification test for the temporal prior: levels versus levels+deltas+trends (task 2.2).

Why this exists
---------------
``openspec/changes/temporal-financial-prior/proposal.md`` pre-registers the cheapest test of the
whole change: give gradient boosting per-company deltas and trends and check whether it improves
over levels alone. "If the trajectory carries no signal on real data, generating it synthetically
cannot help either." V4FinBench is the company-year panel that makes the test possible (the UCI
panels have no company identifier, ``docs/results/FINDINGS.md`` §7).

Design
------
- **Arms.** ``levels``: the ratio columns of the row's own year. ``levels+traj``: those plus a
  one-year delta and a short trend per column.
- **No leakage from the future.** For a row of company ``c`` in year ``t`` every trajectory
  feature is computed from that company's rows in years ``t-3 .. t`` only (the row's own level and
  strictly earlier years). Rows from later years never enter. A lag counts only if the prior year
  is exactly one year earlier, so a gap in a company's history gives ``NaN``, not a stretched
  delta. A company's first observed year therefore has ``NaN`` deltas and trends.
- **Delta** = ``x_t - x_{t-1}``. **Trend** = ordinary-least-squares slope over the years
  ``t-3 .. t`` that are present, needing at least three of the four, else ``NaN``.
- **Sample.** Whole companies are drawn at random until the row cap is reached, so every
  sampled company keeps its complete history and a company is never split by the subsample.
- **Folds.** Three ``StratifiedGroupKFold`` folds grouped by company, shuffled, seed 0, the same
  construction as ``fintfm.evaluation.credit_suite.stratified_splits`` (``N_SPLITS = 3``,
  ``SPLIT_SEED = 0``; reproduced here because importing that module imports torch).
- **Metrics.** Average precision first (the default rate is 0.19-0.36%, §43), then ROC-AUC, per
  fold. The paired comparison is the per-fold difference (traj minus levels) with a paired
  bootstrap over the same test rows.

**Run as a script, not with ``-m``**: LightGBM and torch segfault together on macOS, and this
module imports neither torch nor the ``fintfm`` package. The dataset module is loaded by file
path, as in ``attribution_baseline.py``.

Nothing here trains a FinTFM model. Test numbers are SMOKE-TEST; a run is MEASURED only with the
command recorded beside it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

#: LightGBM settings, fixed and identical for both arms.
LGBM_PARAMS: dict = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "n_jobs": 2,
    "verbose": -1,
}

N_SPLITS = 3
SPLIT_SEED = 0
#: Trend window in years, ending at the row's own year.
TREND_WINDOW = 4
#: Minimum present points inside the window for a trend.
TREND_MIN_POINTS = 3


def lag_index(groups: np.ndarray, year: np.ndarray, lag: int) -> np.ndarray:
    """Row index of the same company's row exactly ``lag`` years earlier, else ``-1``.

    Args:
        groups: Company identifier per row, ``(n,)``.
        year: Integer reporting year per row, ``(n,)``.
        lag: Positive number of years back.

    Returns:
        ``(n,)`` int array. Only rows of strictly earlier years are ever returned.

    Raises:
        ValueError: If ``lag`` is not positive or a (company, year) pair repeats.
    """
    if lag < 1:
        raise ValueError("lag must be at least 1: only strictly earlier years are allowed")
    _, gid = np.unique(groups, return_inverse=True)
    year = np.asarray(year).astype(np.int64)
    span = int(year.max() - year.min()) + lag + 2
    key = gid.astype(np.int64) * span + (year - year.min())
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    if np.any(np.diff(sorted_key) == 0):
        raise ValueError("duplicate (company, year) rows: lag lookup would be ambiguous")
    want = key - lag
    pos = np.minimum(np.searchsorted(sorted_key, want), len(key) - 1)
    # the key packs company and year, so key equality implies the same company
    hit = (sorted_key[pos] == want) & (year - lag >= year.min())
    return np.where(hit, order[pos], -1)


def trajectory_features(
    X: np.ndarray, groups: np.ndarray, year: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-company one-year deltas and short trends, from the row's own and earlier years only.

    Args:
        X: Levels ``(n, d)``, NaN for missing.
        groups: Company identifier per row ``(n,)``.
        year: Reporting year per row ``(n,)``.

    Returns:
        ``(delta, trend)``, each ``(n, d)`` float32. ``delta`` is ``x_t - x_{t-1}`` and ``NaN``
        when the company has no row exactly one year earlier (always so in its first year).
        ``trend`` is the OLS slope per year over ``t-3 .. t`` and ``NaN`` with fewer than
        :data:`TREND_MIN_POINTS` present points.
    """
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    lags = [np.arange(n)] + [lag_index(groups, year, k) for k in range(1, TREND_WINDOW)]

    def at(lag_rows: np.ndarray) -> np.ndarray:
        out = np.full((n, d), np.nan, dtype=np.float32)
        ok = lag_rows >= 0
        out[ok] = X[lag_rows[ok]]
        return out

    delta = X - at(lags[1])

    # OLS slope over the present points: sum((k - kbar)(x - xbar)) / sum((k - kbar)^2), with
    # k = 0 the oldest year of the window and k = TREND_WINDOW - 1 the row's own year.
    ks = np.arange(TREND_WINDOW, dtype=np.float32)
    stack = np.stack([at(lags[TREND_WINDOW - 1 - j]) for j in range(TREND_WINDOW)], axis=0)
    present = ~np.isnan(stack)
    cnt = present.sum(axis=0)
    safe = np.maximum(cnt, 1)
    kbar = (ks[:, None, None] * present).sum(axis=0) / safe
    xz = np.where(present, stack, 0.0)
    xbar = xz.sum(axis=0) / safe
    dk = (ks[:, None, None] - kbar) * present
    num = (dk * (xz - xbar)).sum(axis=0)
    den = (dk**2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        trend = np.where((cnt >= TREND_MIN_POINTS) & (den > 0), num / den, np.nan)
    return delta.astype(np.float32), trend.astype(np.float32)


def sample_companies(groups: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    """Row indices of a random set of whole companies totalling at most ``max_rows`` rows.

    Args:
        groups: Company identifier per row.
        max_rows: Row cap.
        seed: Seed for the company order.

    Returns:
        Sorted row indices; every company in the sample keeps all of its rows.
    """
    uniq, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(uniq))
    cum = np.cumsum(counts[order])
    keep = np.zeros(len(uniq), dtype=bool)
    keep[order[cum <= max_rows]] = True
    return np.flatnonzero(keep[inverse])


def grouped_folds(y: np.ndarray, groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Three company-grouped stratified folds (the project's ``stratified_splits`` recipe)."""
    from sklearn.model_selection import StratifiedGroupKFold

    folds = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
    return list(folds.split(np.zeros(len(y)), y, groups))


def paired_bootstrap_diff(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, metric, n_boot: int, seed: int
) -> tuple[float, float, float]:
    """Metric(b) minus metric(a) on the same rows, with a paired bootstrap 95% interval.

    Returns:
        ``(difference, lower, upper)``; resamples lacking a class are skipped.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(metric(y[idx], p_b[idx]) - metric(y[idx], p_a[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(metric(y, p_b) - metric(y, p_a)), float(lo), float(hi)


def run(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    year: np.ndarray,
    n_boot: int = 500,
    seed: int = 0,
    params: dict | None = None,
) -> dict:
    """Fit both arms on each company-grouped fold and compare them.

    Args:
        X: Levels ``(n, d)``.
        y: Binary label ``(n,)``.
        groups: Company id ``(n,)``.
        year: Reporting year ``(n,)``.
        n_boot: Paired bootstrap resamples per fold.
        seed: LightGBM and bootstrap seed.
        params: LightGBM overrides.

    Returns:
        Dict with the row counts, per-fold metrics for both arms and the paired differences.
    """
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, roc_auc_score

    delta, trend = trajectory_features(X, groups, year)
    arms = {"levels": X, "levels+traj": np.hstack([X, delta, trend])}
    out_folds = []
    for k, (tr, te) in enumerate(grouped_folds(y, groups)):
        assert not set(groups[tr].tolist()) & set(groups[te].tolist())
        row: dict = {
            "fold": k,
            "n_train": len(tr),
            "n_test": len(te),
            "pos_train": int(y[tr].sum()),
            "pos_test": int(y[te].sum()),
            "test_default_rate": float(y[te].mean()),
        }
        proba = {}
        for name, A in arms.items():
            t0 = time.perf_counter()
            m = lgb.LGBMClassifier(random_state=seed, **{**LGBM_PARAMS, **(params or {})})
            m.fit(A[tr], y[tr])
            p = m.predict_proba(A[te])[:, 1]
            proba[name] = p
            row[name] = {
                "n_features": A.shape[1],
                "average_precision": float(average_precision_score(y[te], p)),
                "roc_auc": float(roc_auc_score(y[te], p)),
                "seconds": time.perf_counter() - t0,
            }
        row["paired"] = {}
        for label, metric in (
            ("average_precision", average_precision_score),
            ("roc_auc", roc_auc_score),
        ):
            d, lo, hi = paired_bootstrap_diff(
                y[te], proba["levels"], proba["levels+traj"], metric, n_boot, seed + k
            )
            row["paired"][label] = {"diff": d, "ci95_low": lo, "ci95_high": hi}
        out_folds.append(row)
    return {
        "n_rows": len(y),
        "n_companies": len(np.unique(groups)),
        "default_rate": float(y.mean()),
        "n_levels": int(X.shape[1]),
        "delta_nan_share": float(np.isnan(delta).mean()),
        "trend_nan_share": float(np.isnan(trend).mean()),
        "folds": out_folds,
    }


def _load_datasets_module():
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "evaluation" / "datasets.py"
    spec = importlib.util.spec_from_file_location("_fintfm_datasets", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_fintfm_datasets"] = mod  # dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=None, help="V4FinBench parquet directory")
    ap.add_argument("--max-rows", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ds = _load_datasets_module().load_v4finbench(root=args.root)
    pool = {"rows": len(ds.y), "companies": len(np.unique(ds.groups))}
    idx = sample_companies(ds.groups, args.max_rows, args.seed)
    result = run(
        ds.X[idx], ds.y[idx], ds.groups[idx], ds.year[idx], n_boot=args.n_boot, seed=args.seed
    )
    result["pool"] = pool
    result["seed"] = args.seed
    result["feature_names"] = list(ds.feature_names)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text)
    for f in result["folds"]:
        print(
            f"fold {f['fold']}: AP levels {f['levels']['average_precision']:.4f} "
            f"traj {f['levels+traj']['average_precision']:.4f} | AUC levels "
            f"{f['levels']['roc_auc']:.4f} traj {f['levels+traj']['roc_auc']:.4f}"
        )
    print(text)


if __name__ == "__main__":
    main()
