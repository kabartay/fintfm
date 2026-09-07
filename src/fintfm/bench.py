"""Benchmark FinancialTFM against classical baselines.

Two data sources:
  - synthetic: fresh draws from the financial prior, held out from training
    (sanity check that pretraining transfers within-distribution).
  - openml: real datasets fetched via sklearn.datasets.fetch_openml (network
    required); pass --openml-id repeatedly.

Baselines: logistic regression, random forest, gradient boosting (sklearn) and
LightGBM when installed. Metric: ROC-AUC (binary) or macro-OVR AUC (multiclass).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fintfm.classifier import FinancialTFMClassifier
from fintfm.model import FinancialTFM
from fintfm.prior import PriorConfig
from fintfm.prior.mixture import sample_task


def _auc(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    if len(classes) == 2:
        return roc_auc_score(y_true, proba[:, 1])
    return roc_auc_score(y_true, proba, multi_class="ovr", labels=classes)


def _baselines() -> dict[str, object]:
    return {
        "logreg": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, random_state=0)
        ),
        "gboost": make_pipeline(
            SimpleImputer(strategy="median"), GradientBoostingClassifier(random_state=0)
        ),
    }


def _maybe_lightgbm() -> dict[str, object]:
    try:
        from lightgbm import LGBMClassifier
    except (ImportError, OSError):
        # OSError covers e.g. a missing libomp shared library on macOS.
        return {}
    return {"lightgbm": make_pipeline(SimpleImputer(strategy="median"), LGBMClassifier(verbosity=-1))}


def run_one(name: str, X: np.ndarray, y: np.ndarray, model_path: str, seed: int = 0) -> dict[str, float]:
    """Fit every model on a train split and report test AUC."""
    values, counts = np.unique(y, return_counts=True)
    # stratify requires >= 2 members per class; synthetic multiclass tasks can be this rare
    can_stratify = len(values) > 1 and counts.min() >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y if can_stratify else None
    )
    classes = np.unique(y)
    results: dict[str, float] = {}
    models = {**_baselines(), **_maybe_lightgbm()}
    models["fintfm"] = FinancialTFMClassifier(model_path)
    for model_name, clf in models.items():
        t0 = time.time()
        try:
            clf.fit(X_train, y_train)
            proba = clf.predict_proba(X_test)
            auc = _auc(y_test, proba, classes)
        except Exception as exc:  # keep the benchmark going if one model errors
            print(f"  [{name}] {model_name} failed: {exc}")
            continue
        results[model_name] = auc
        print(f"  [{name}] {model_name}: AUC={auc:.4f}  ({time.time() - t0:.1f}s)")
    return results


def run_synthetic(model_path: str, n_tasks: int = 20, n_rows: int = 1000, seed: int = 123) -> None:
    """Held-out synthetic financial tasks with a fixed seed disjoint from pretraining.

    The task's feature/class width is capped at the loaded model's capacity,
    otherwise a model pretrained with a small ``--max-features``/``--max-classes``
    would reject most held-out tasks generated from the library defaults.
    """
    rng = np.random.default_rng(seed)
    model_cfg = FinancialTFM.load(model_path).cfg
    cfg = PriorConfig(max_features=model_cfg.max_features, max_classes=model_cfg.max_classes)
    wins = {"fintfm": 0}
    totals: dict[str, list[float]] = {}
    n_scored = 0
    for i in range(n_tasks):
        task = sample_task(rng, cfg, n_rows=n_rows)
        res = run_one(f"synthetic-{i}", task.X, task.y, model_path)
        # a degenerate test split (single class in y_test) makes AUC undefined; skip it
        # entirely rather than let one NaN silently poison every aggregate.
        if any(np.isnan(v) for v in res.values()):
            print(f"  [synthetic-{i}] skipped: degenerate test split (single class)")
            continue
        n_scored += 1
        best_other = max((v for k, v in res.items() if k != "fintfm"), default=0.0)
        if res.get("fintfm", -1) > best_other:
            wins["fintfm"] += 1
        for k, v in res.items():
            totals.setdefault(k, []).append(v)
    print(f"\n=== mean AUC over {n_scored}/{n_tasks} scorable synthetic held-out tasks ===")
    for k, vs in totals.items():
        print(f"  {k}: {np.mean(vs):.4f}")
    print(f"fintfm beats every baseline on {wins['fintfm']}/{n_scored} scorable tasks")


def run_openml(model_path: str, dataset_ids: list[int]) -> None:
    from sklearn.datasets import fetch_openml

    for did in dataset_ids:
        data = fetch_openml(data_id=did, as_frame=True, parser="auto")
        X = data.data.apply(
            lambda c: c.astype("category").cat.codes if c.dtype.name in ("object", "category") else c
        )
        X = X.to_numpy(dtype=np.float32)
        y = data.target
        y = y.astype("category").cat.codes.to_numpy() if hasattr(y, "cat") else np.asarray(y)
        run_one(f"openml-{did}", X, y, model_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="runs/v0.pt")
    p.add_argument("--synthetic-tasks", type=int, default=20)
    p.add_argument("--openml-id", type=int, action="append", default=[])
    args = p.parse_args()
    run_synthetic(args.model, n_tasks=args.synthetic_tasks)
    if args.openml_id:
        run_openml(args.model, args.openml_id)


if __name__ == "__main__":
    main()
