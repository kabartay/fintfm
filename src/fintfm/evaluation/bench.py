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

from fintfm.evaluation.boosting import BOOSTING_MODELS, fit_predict_boosting
from fintfm.evaluation.datasets import (
    CreditDataset,
    SurvivalDataset,
    load_polish_bankruptcy,
    load_taiwan_bankruptcy,
    load_v4finbench,
)
from fintfm.evaluation.metrics import evaluate_binary
from fintfm.evaluation.splits import time_based_split
from fintfm.inference.classifier import ContextStrategy, FinancialTFMClassifier
from fintfm.modeling.model import FinancialTFM
from fintfm.prior import PriorConfig
from fintfm.prior.mixture import sample_task


def _auc(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    """ROC-AUC, binary or macro one-vs-rest depending on the class count.

    Args:
        y_true: ``(n,)`` labels.
        proba: ``(n, n_classes)`` predicted probabilities.
        classes: The label space, used only to decide binary versus multiclass.

    Returns:
        ROC-AUC. Read average precision beside it at low base rates — ROC-AUC's chance floor
        is 0.5 whatever the prevalence.
    """
    if len(classes) == 2:
        return roc_auc_score(y_true, proba[:, 1])
    return roc_auc_score(y_true, proba, multi_class="ovr", labels=classes)


def _baselines() -> dict[str, object]:
    """The fitted comparators every benchmark run scores alongside the model.

    Returns:
        ``{name: estimator}``. Each is a pipeline with imputation, because a baseline that
        crashes on a NaN is a baseline silently absent from the comparison — which is how
        gradient boosting went missing from every early result (§25).
    """
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


def run_one(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    model_path: str,
    seed: int = 0,
    split: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    """Fit every model on a train split and report test AUC.

    Args:
        name: Label used in the printed rows.
        X: Features ``(n, n_features)``.
        y: Labels ``(n,)``.
        model_path: Checkpoint for the fintfm arm.
        seed: Seed for the default random split.
        split: Optional ``(train_idx, test_idx)`` row indices into ``X``/``y``. ``None``
            (default) keeps the stratified random 70/30 split.

    Returns:
        Test AUC per model name.
    """
    values, counts = np.unique(y, return_counts=True)
    # stratify requires >= 2 members per class; synthetic multiclass tasks can be this rare
    can_stratify = len(values) > 1 and counts.min() >= 2
    if split is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y if can_stratify else None
        )
    else:
        X_train, X_test, y_train, y_test = X[split[0]], X[split[1]], y[split[0]], y[split[1]]
    classes = np.unique(y)
    results: dict[str, float] = {}

    def _report(model_name: str, proba: np.ndarray, seconds: float) -> None:
        """Score one arm and print its row.

        Args:
            model_name: Arm name, used as the results key.
            proba: ``(n, n_classes)`` predictions for the test split.
            seconds: Wall-clock time for that arm.
        """
        auc = _auc(y_test, proba, classes)
        results[model_name] = auc
        if len(classes) == 2:
            detail = evaluate_binary(y_test, proba[:, 1]).summary()
            print(f"  [{name}] {model_name}: {detail}  ({seconds:.1f}s)")
        else:
            print(f"  [{name}] {model_name}: AUC={auc:.4f}  ({seconds:.1f}s)")

    # in-process: classical baselines and our own model
    models: dict[str, object] = dict(_baselines())
    # n_ensemble=8, no retrieval: the lever fix docs/results/FINDINGS.md S70-S71 measured as a real,
    # significant AP gain (+0.031, Holm p<0.001) over the library defaults this call used to
    # pass implicitly. context_strategy is left at its config default (uniform), not
    # retrieval, because S70 found retrieval actively harmful in this base-rate regime and
    # its mechanism is still open (task 38.13).
    models["fintfm"] = FinancialTFMClassifier(model_path, n_ensemble=8)
    for model_name, clf in models.items():
        t0 = time.time()
        try:
            clf.fit(X_train, y_train)
            proba = clf.predict_proba(X_test)
        except Exception as exc:  # noqa: BLE001 - one model failing must not abort the sweep
            print(f"  [{name}] {model_name} failed: {exc}")
            continue
        _report(model_name, proba, time.time() - t0)

    # Out-of-process: the boosting family. Fitting these in a process that has imported
    # torch segfaults, because two OpenMP runtimes collide (docs/results/FINDINGS.md §25). These are
    # the baselines that actually compete, so the process boundary is worth the cost.
    if len(classes) == 2:
        for model_name in BOOSTING_MODELS:
            t0 = time.time()
            p1 = fit_predict_boosting(model_name, X_train, y_train, X_test)
            if p1 is None:
                continue
            _report(model_name, np.column_stack([1.0 - p1, p1]), time.time() - t0)

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
    """Score a checkpoint against the baselines on OpenML datasets.

    Args:
        model_path: Pretrained checkpoint.
        dataset_ids: OpenML dataset ids to fetch and score.
    """
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


def run_credit(model_path: str, horizons: tuple[int, ...] = (1, 3, 5)) -> None:
    """Benchmark on real corporate-default data across context-construction strategies.

    This is the task the project exists for, so it is reported separately from the
    synthetic sanity check. Every strategy is run against identical splits, because
    Tanna et al. (arXiv:2605.18635) find context construction explains more AUC variance
    than the choice of model family — a claim this harness is here to test rather than
    assume.

    Args:
        model_path: Checkpoint to evaluate. Must have been pretrained with
            ``--max-features`` at least as wide as the dataset (64 here).
        horizons: Which bankruptcy forecast horizons (years) to evaluate.
    """
    cfg = FinancialTFM.load(model_path).cfg
    panels = [load_polish_bankruptcy(h) for h in horizons]
    # A second economy and accounting regime, so a result is not an artefact of one panel.
    panels.append(load_taiwan_bankruptcy())
    for ds in panels:
        print(f"\n=== {ds.name}: {ds.X.shape[0]} companies, {ds.X.shape[1]} features, "
              f"default rate {ds.default_rate:.3%} ===")
        print(f"    source: {ds.attribution}")
        if ds.X.shape[1] > cfg.max_features:
            print(f"    SKIPPED for fintfm: model takes {cfg.max_features} features, data has "
                  f"{ds.X.shape[1]}. Pretrain with --max-features {ds.X.shape[1]} to evaluate it.")
        run_one(ds.name, ds.X, ds.y, model_path)
        if ds.X.shape[1] <= cfg.max_features:
            strategies: tuple[ContextStrategy, ...] = ("uniform", "hybrid", "balanced")
            _compare_context_strategies(ds, model_path, strategies)


def run_time_vs_random(
    ds: SurvivalDataset,
    model_path: str,
    train_until: int = 2016,
    test_from: int = 2017,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Report random-split and time-based test AUC side by side on a dated panel.

    The time split trains on ``year <= train_until`` and tests on ``year >= test_from``, with
    any company present in both windows removed from training (the count is printed). The
    random split draws the same number of train and test rows, stratified, from the same row
    pool, so the gap is not confounded by sample size. Firms may sit on both sides of it,
    which is the optimism being measured.

    Args:
        ds: A panel with ``year`` and ``groups`` (V4FinBench).
        model_path: Checkpoint for the fintfm arm; features beyond its ``max_features`` are
            cut (first columns kept) and the cut is printed.
        train_until: Last training year (inclusive).
        test_from: First test year (inclusive).
        seed: Seed for the random split.

    Returns:
        ``{"random": {model: auc}, "time": {model: auc}}``.

    Raises:
        ValueError: If the panel has no period labels or no identifiers, or the split would
            leak (see :func:`fintfm.evaluation.splits.time_based_split`).
    """
    if not ds.has_period_labels or ds.year is None or ds.groups is None:
        raise ValueError(
            f"{ds.name} has no period/identifier columns, so it cannot support a time-based "
            "split; see docs/results/FINDINGS.md §7"
        )
    ts = time_based_split(ds.year, ds.groups, train_until, test_from, on_overlap="drop_from_train")
    cfg = FinancialTFM.load(model_path).cfg
    X = ds.X[:, : cfg.max_features]
    print(f"\n=== {ds.name}: random vs time-based (train<={train_until}, test>={test_from}) ===")
    print(f"    {len(ts.train)} train / {len(ts.test)} test rows; {ts.n_shared_ids} companies in "
          f"both windows, {ts.n_train_dropped} train rows dropped for it")
    if ds.X.shape[1] > cfg.max_features:
        print(f"    NOTE: model takes {cfg.max_features} features, data has {ds.X.shape[1]}; "
              f"first {cfg.max_features} columns used")
    pool = np.concatenate([ts.train, ts.test])
    rtr, rte = train_test_split(
        pool, train_size=len(ts.train), test_size=len(ts.test), random_state=seed,
        stratify=ds.y[pool],
    )
    out = {
        "random": run_one(f"{ds.name}-random", X, ds.y, model_path, split=(rtr, rte)),
        "time": run_one(f"{ds.name}-time", X, ds.y, model_path, split=(ts.train, ts.test)),
    }
    print(f"\n    {'model':<14}{'random AUC':>12}{'time AUC':>12}{'gap':>9}")
    for model_name in out["random"]:
        r, t = out["random"][model_name], out["time"].get(model_name, float("nan"))
        print(f"    {model_name:<14}{r:>12.4f}{t:>12.4f}{r - t:>+9.4f}")
    return out


def _compare_context_strategies(
    ds: CreditDataset, model_path: str, strategies: tuple[ContextStrategy, ...], seed: int = 0
) -> None:
    """Fit the same checkpoint under each context strategy on one fixed split."""
    X_train, X_test, y_train, y_test = train_test_split(
        ds.X, ds.y, test_size=0.3, random_state=seed, stratify=ds.y
    )
    for strategy in strategies:
        clf = FinancialTFMClassifier(model_path, context_strategy=strategy, n_ensemble=8)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        metrics = evaluate_binary(y_test, proba[:, 1])
        summary = clf.context_summary_
        print(f"    [{ds.name}] fintfm context={strategy}: {metrics.summary()}")
        print(f"        {summary.n_positive} defaults in a {summary.n_context}-row context")


def main() -> None:
    """Run the credit-panel and OpenML benchmarks from the command line.

    Entry point for the ``bench`` console script; see
    ``--help`` for the flags. Writes its record as JSON under ``--out``.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="runs/v0.pt")
    p.add_argument("--synthetic-tasks", type=int, default=20)
    p.add_argument("--openml-id", type=int, action="append", default=[])
    p.add_argument("--credit", action="store_true", help="benchmark on real corporate-default data")
    p.add_argument("--time-split", action="store_true",
                   help="V4FinBench random vs time-based AUC side by side (needs the cached data)")
    p.add_argument("--max-rows", type=int, default=None,
                   help="with --time-split: subsample this many company-years")
    p.add_argument("--v4-root", type=str, default=None, help="V4FinBench parquet directory")
    args = p.parse_args()
    if args.time_split:
        run_time_vs_random(load_v4finbench(args.v4_root, max_rows=args.max_rows), args.model)
        return
    if args.credit:
        run_credit(args.model)
        return
    run_synthetic(args.model, n_tasks=args.synthetic_tasks)
    if args.openml_id:
        run_openml(args.model, args.openml_id)


if __name__ == "__main__":
    main()
