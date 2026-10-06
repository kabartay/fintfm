"""Local, CPU-only feasibility benchmark for one real-data panel.

The same questions §151 (EDGAR) and §154 (Freddie Mac) answered by hand, made repeatable so
candidates can be evaluated one at a time before any loader or training compute is spent:

- **Shape**: rows, feature count, overall and per-feature missingness.
- **Label regime**: positive rate, and where it sits against V4FinBench's 0.19-4% band -- a
  source far outside it teaches a different base-rate regime than the one this project is
  scored on, which is worth knowing before mixing it in, not after.
- **Learnability**: stratified 5-fold ROC-AUC and average precision for logistic regression
  and LightGBM. A label no classical model can predict above chance is noise for in-context
  pretraining too (CLAUDE.md's "a prior must span difficulty" -- but it must contain *some*
  learnable signal); one LightGBM solves near-perfectly may be leaking the outcome.

Every number this prints is MEASURED on the panel itself. It says nothing yet about whether the
source helps FinTFM-R -- that is the pilot ablation's job, not this script's.

Usage:
    uv run python scripts/candidates/feasibility.py data/real/ppdai/panel.parquet
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#: Columns that identify a row rather than describe it, across every panel builder here.
BOOKKEEPING = {"label", "loan_id", "cik", "period_end"}

V4_BAND = (0.0019, 0.04)


def _cv_scores(X: np.ndarray, y: np.ndarray, model_name: str, seed: int) -> dict[str, float]:
    """Stratified 5-fold out-of-fold AUC and AP for one model.

    Args:
        X: Feature matrix, NaN for missing.
        y: Binary labels.
        model_name: ``"logreg"`` or ``"lightgbm"``.
        seed: Fold-assignment seed.

    Returns:
        Mean and per-fold ROC-AUC and average precision.
    """
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        if model_name == "logreg":
            model = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(),
                LogisticRegression(max_iter=2000),
            )
        else:
            import lightgbm as lgb

            # n_jobs capped: the machine is shared with an always-on pipeline (CLAUDE.md).
            model = lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.05, verbose=-1, n_jobs=4
            )
        model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
    return {"auc": float(roc_auc_score(y, oof)), "ap": float(average_precision_score(y, oof))}


def feasibility(panel: pd.DataFrame, max_rows: int, seed: int) -> dict:
    """Compute every number this module reports for one panel.

    Args:
        panel: A panel with a binary `label` column and float feature columns.
        max_rows: Stratified subsample cap for the CV models, so a million-row panel takes
            minutes rather than hours on a shared CPU. Shape and label statistics always use
            the full panel.
        seed: Subsample and fold seed.

    Returns:
        A JSON-serialisable dict of results.
    """
    feats = [c for c in panel.columns if c not in BOOKKEEPING]
    y_full = panel["label"].to_numpy(dtype=np.int64)
    rate = float(y_full.mean())
    miss = panel[feats].isna().mean()
    out = {
        "rows": len(panel),
        "features": len(feats),
        "positives": int(y_full.sum()),
        "positive_rate": rate,
        "v4_regime": (
            "inside" if V4_BAND[0] <= rate <= V4_BAND[1]
            else "rarer" if rate < V4_BAND[0] else "more common"
        ),
        "missing_overall": float(miss.mean()),
        "features_over_50pct_missing": int((miss > 0.5).sum()),
        "constant_features": int((panel[feats].nunique(dropna=True) <= 1).sum()),
    }
    sub = panel
    if len(panel) > max_rows:
        rng = np.random.default_rng(seed)
        pos, neg = np.flatnonzero(y_full == 1), np.flatnonzero(y_full == 0)
        k_pos = max(1, round(max_rows * rate))
        idx = np.concatenate([
            rng.choice(pos, size=min(k_pos, len(pos)), replace=False),
            rng.choice(neg, size=min(max_rows - k_pos, len(neg)), replace=False),
        ])
        sub = panel.iloc[idx]
    X = sub[feats].to_numpy(dtype=np.float64)
    y = sub["label"].to_numpy(dtype=np.int64)
    out["cv_rows"] = len(sub)
    out["cv_positives"] = int(y.sum())
    if y.sum() < 10:
        out["cv_skipped"] = "fewer than 10 positives in the CV sample -- not enough to score"
        return out
    for name in ("logreg", "lightgbm"):
        t0 = time.perf_counter()
        out[name] = _cv_scores(X, y, name, seed)
        out[name]["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def main() -> None:
    """Entry point: print and optionally save one panel's feasibility report."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("panel", help="parquet panel with a binary `label` column")
    p.add_argument("--max-rows", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()
    result = feasibility(pd.read_parquet(args.panel), args.max_rows, args.seed)
    result["panel"] = args.panel
    print(json.dumps(result, indent=2))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
