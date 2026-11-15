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
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fintfm.evaluation.boosting import BOOSTING_MODELS, fit_predict_boosting
from fintfm.evaluation.coverage import Coverage
from fintfm.evaluation.datasets import (
    CreditDataset,
    SurvivalDataset,
    load_polish_bankruptcy,
    load_taiwan_bankruptcy,
    load_v4finbench,
)
from fintfm.evaluation.metrics import evaluate_binary, holm_adjusted_p, paired_auc_difference
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
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=300, random_state=0),
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
    details: dict[str, dict[str, float]] | None = None,
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
        details: Optional dict filled in place with ``{model: {"average_precision", "roc_auc",
            "n_test", "n_positive"}}`` for binary tasks, so a caller can report AP first.

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
            metrics = evaluate_binary(y_test, proba[:, 1])
            detail = metrics.summary()
            if details is not None:
                details[model_name] = {
                    "average_precision": float(metrics.average_precision),
                    "roc_auc": float(metrics.roc_auc),
                    "n_test": int(metrics.n),
                    "n_positive": int(metrics.n_positive),
                }
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


def run_synthetic(model_path: str, n_tasks: int = 20, n_rows: int = 1000, seed: int = 123) -> dict:
    """Held-out synthetic financial tasks with a fixed seed disjoint from pretraining.

    The task's feature/class width is capped at the loaded model's capacity,
    otherwise a model pretrained with a small ``--max-features``/``--max-classes``
    would reject most held-out tasks generated from the library defaults.

    Returns:
        ``{"mean_auc": {model: mean}, "wins": {...}, **coverage}`` where the coverage fields
        are ``n_attempted``, ``n_scored``, ``n_skipped``, ``coverage`` and ``skipped``
        (see :class:`fintfm.evaluation.coverage.Coverage`).
    """
    rng = np.random.default_rng(seed)
    model_cfg = FinancialTFM.load(model_path).cfg
    cfg = PriorConfig(max_features=model_cfg.max_features, max_classes=model_cfg.max_classes)
    wins = {"fintfm": 0}
    totals: dict[str, list[float]] = {}
    cov = Coverage()
    for i in range(n_tasks):
        cov.attempt(f"synthetic-{i}")
        task = sample_task(rng, cfg, n_rows=n_rows)
        res = run_one(f"synthetic-{i}", task.X, task.y, model_path)
        # a degenerate test split (single class in y_test) makes AUC undefined; skip it
        # entirely rather than let one NaN silently poison every aggregate.
        if any(np.isnan(v) for v in res.values()):
            print(f"  [synthetic-{i}] skipped: degenerate test split (single class)")
            cov.skip(f"synthetic-{i}", "degenerate test split (single class)")
            continue
        cov.scored(f"synthetic-{i}")
        best_other = max((v for k, v in res.items() if k != "fintfm"), default=0.0)
        if res.get("fintfm", -1) > best_other:
            wins["fintfm"] += 1
        for k, v in res.items():
            totals.setdefault(k, []).append(v)
    n_scored = cov.n_scored
    print(f"\n=== mean AUC over {n_scored}/{n_tasks} scorable synthetic held-out tasks ===")
    for k, vs in totals.items():
        print(f"  {k}: {np.mean(vs):.4f}")
    print(f"fintfm beats every baseline on {wins['fintfm']}/{n_scored} scorable tasks")
    print(cov.summary("synthetic tasks"))
    return {
        "mean_auc": {k: float(np.mean(vs)) for k, vs in totals.items()},
        "wins": dict(wins),
        **cov.to_dict(),
    }


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
            lambda c: (
                c.astype("category").cat.codes if c.dtype.name in ("object", "category") else c
            )
        )
        X = X.to_numpy(dtype=np.float32)
        y = data.target
        y = y.astype("category").cat.codes.to_numpy() if hasattr(y, "cat") else np.asarray(y)
        run_one(f"openml-{did}", X, y, model_path)


#: Training sizes the default credit run probes. Small on purpose: the full sweep
#: (100-4000) is ``fintfm-ablate --sample-efficiency``. These are the sizes below the
#: roughly 1,000-row mark where Baesens et al. (arXiv:2605.18147) put the TFM advantage.
DEFAULT_PROBE_SIZES: tuple[int, ...] = (100, 250, 500, 1000)


def run_credit(
    model_path: str,
    horizons: tuple[int, ...] = (1, 3, 5),
    sample_efficiency: bool = True,
    probe_sizes: tuple[int, ...] = DEFAULT_PROBE_SIZES,
    time_panel: SurvivalDataset | None = None,
) -> dict:
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
        sample_efficiency: Also run the small-n sample-efficiency probe (openspec
            ``sample-efficiency-regime`` task 13.6), so a full-panel-only number is never
            the whole credit output. ``--no-sample-efficiency`` turns it off.
        probe_sizes: Training sizes for that probe, ascending.
        time_panel: A dated panel (V4FinBench). When given, random and time-based AUC are
            printed side by side for it after the credit panels (openspec
            ``time-based-evaluation`` task 3.3). When ``None`` the output says, per panel,
            that no time-based number exists, so its absence is never silent.

    Returns:
        ``{"sample_efficiency": <probe record>}`` when the probe ran, plus
        ``{"time_vs_random": {"random": {...}, "time": {...}}}`` when ``time_panel`` was
        given; else ``{}``. The probe record carries per-size metrics, ``crossover_size``
        and ``leads_at`` (see
        :func:`fintfm.experiments.prior_ablation.sample_efficiency_probe`).
    """
    cfg = FinancialTFM.load(model_path).cfg
    panels = [load_polish_bankruptcy(h) for h in horizons]
    # A second economy and accounting regime, so a result is not an artefact of one panel.
    panels.append(load_taiwan_bankruptcy())
    for ds in panels:
        print(
            f"\n=== {ds.name}: {ds.X.shape[0]} companies, {ds.X.shape[1]} features, "
            f"default rate {ds.default_rate:.3%} ==="
        )
        print(f"    source: {ds.attribution}")
        print(f"    {time_split_note(ds)}")
        if ds.X.shape[1] > cfg.max_features:
            print(
                f"    SKIPPED for fintfm: model takes {cfg.max_features} features, data has "
                f"{ds.X.shape[1]}. Pretrain with --max-features {ds.X.shape[1]} to evaluate it."
            )
        run_one(ds.name, ds.X, ds.y, model_path)
        if ds.X.shape[1] <= cfg.max_features:
            strategies: tuple[ContextStrategy, ...] = ("uniform", "hybrid", "balanced")
            _compare_context_strategies(ds, model_path, strategies)
    record: dict = {}
    if time_panel is not None:
        record["time_vs_random"] = run_time_vs_random(time_panel, model_path)
    else:
        print(
            "\nTIME-BASED SPLIT: NOT RUN. The panels above carry no period column, and no "
            "dated panel was supplied (use --time-split with --credit and the V4FinBench "
            "data). Every number above is a RANDOM-split number."
        )
    if sample_efficiency:
        record["sample_efficiency"] = _run_sample_efficiency(
            panels, cfg.max_features, model_path, probe_sizes
        )
    return record


def _run_sample_efficiency(
    panels: list[CreditDataset], max_features: int, model_path: str, sizes: tuple[int, ...]
) -> dict:
    """Run the small-n probe on the 3-year Polish panel (or the first one that fits).

    Args:
        panels: The panels ``run_credit`` loaded.
        max_features: The checkpoint's feature capacity.
        model_path: Checkpoint to probe.
        sizes: Training sizes to sweep.

    Returns:
        The probe record, with ``crossover_size`` and ``leads_at``; prints a summary line.
        Single seed (0), so read the crossover as indicative, not a range.
    """
    from fintfm.experiments.prior_ablation import sample_efficiency_probe

    fitting = [p for p in panels if p.X.shape[1] <= max_features]
    ds = next((p for p in fitting if p.horizon_years == 3), fitting[0] if fitting else None)
    print("\n=== sample-efficiency probe (small n, fixed test set, seed 0) ===")
    if ds is None:
        print("    SKIPPED: no panel fits the checkpoint's feature capacity")
        return {"skipped": "no panel fits the checkpoint"}
    record = sample_efficiency_probe(model_path, train_sizes=sizes, seed=0, ds=ds)
    print(
        f"    crossover_size={record['crossover_size']} leads_at={record['leads_at']} (single seed)"
    )
    return record


def time_split_note(ds: CreditDataset | SurvivalDataset) -> str:
    """Say whether a panel supports a time-based split, and why not if it does not.

    Args:
        ds: Any benchmark panel.

    Returns:
        A one-line statement for the bench output, so a missing time-based number is stated
        rather than silently omitted.
    """
    if ds.has_period_labels:
        return "time-based split: available (period column present)"
    return (
        "time-based split: NOT AVAILABLE (no period column; scores on this panel are "
        "random-split only)"
    )


def run_time_vs_random(
    ds: SurvivalDataset,
    model_path: str,
    train_until: int = 2016,
    test_from: int = 2017,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Report random-split and time-based test AP and AUC side by side on a dated panel.

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
        ``{"random": {model: auc}, "time": {model: auc}, "details": {...}}``. ``details`` holds
        per-window ``average_precision`` and ``roc_auc`` per model, plus the row counts, base
        rates, companies in both windows and the cut-off years, for the ``--out`` JSON.

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
    print(
        f"    {len(ts.train)} train / {len(ts.test)} test rows; {ts.n_shared_ids} companies in "
        f"both windows, {ts.n_train_dropped} train rows dropped for it"
    )
    if ds.X.shape[1] > cfg.max_features:
        print(
            f"    NOTE: model takes {cfg.max_features} features, data has {ds.X.shape[1]}; "
            f"first {cfg.max_features} columns used"
        )
    pool = np.concatenate([ts.train, ts.test])
    rtr, rte = train_test_split(
        pool,
        train_size=len(ts.train),
        test_size=len(ts.test),
        random_state=seed,
        stratify=ds.y[pool],
    )
    det_random: dict[str, dict[str, float]] = {}
    det_time: dict[str, dict[str, float]] = {}
    out: dict = {
        "random": run_one(
            f"{ds.name}-random", X, ds.y, model_path, split=(rtr, rte), details=det_random
        ),
        "time": run_one(
            f"{ds.name}-time", X, ds.y, model_path, split=(ts.train, ts.test), details=det_time
        ),
    }
    rate_time = float(np.mean(ds.y[ts.test]))
    rate_random = float(np.mean(ds.y[rte]))
    print(
        f"    test base rate: time {rate_time:.3%} ({int(ds.y[ts.test].sum())}/{len(ts.test)}), "
        f"random {rate_random:.3%} ({int(ds.y[rte].sum())}/{len(rte)}); train base rate "
        f"time {float(np.mean(ds.y[ts.train])):.3%}"
    )
    print(f"\n    {'model':<14}{'random AP':>12}{'time AP':>12}{'gap':>9}   (read AP first)")
    for model_name, r in det_random.items():
        t = det_time.get(model_name, {}).get("average_precision", float("nan"))
        ap_r = r["average_precision"]
        print(f"    {model_name:<14}{ap_r:>12.4f}{t:>12.4f}{ap_r - t:>+9.4f}")
    print(f"\n    {'model':<14}{'random AUC':>12}{'time AUC':>12}{'gap':>9}")
    for model_name in out["random"]:
        r, t = out["random"][model_name], out["time"].get(model_name, float("nan"))
        print(f"    {model_name:<14}{r:>12.4f}{t:>12.4f}{r - t:>+9.4f}")
    out["details"] = {
        "panel": ds.name,
        "checkpoint": model_path,
        "train_until": train_until,
        "test_from": test_from,
        "seed": seed,
        "n_rows_panel": len(ds.y),
        "n_train": len(ts.train),
        "n_test": len(ts.test),
        "n_train_dropped": int(ts.n_train_dropped),
        "n_companies_in_both_windows": int(ts.n_shared_ids),
        "test_base_rate_time": rate_time,
        "test_base_rate_random": rate_random,
        "random": det_random,
        "time": det_time,
    }
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


TEMPORAL_GBM_SCRIPT = (
    Path(__file__).resolve().parents[1] / "experiments" / "temporal_gbm_falsification.py"
)


def temporal_gbm_command(
    v4_root: str | None = None, max_rows: int | None = None, seed: int = 0, out: str | None = None
) -> list[str]:
    """Build the subprocess command for the temporal-GBM falsification script.

    The script runs in its own interpreter because LightGBM and torch segfault together on
    macOS and this module imports torch; the script itself never imports it.

    Args:
        v4_root: V4FinBench parquet directory, or ``None`` for the default cache.
        max_rows: Company-grouped subsample size, or ``None`` for the script default.
        seed: LightGBM and bootstrap seed.
        out: Where the script writes its JSON record.

    Returns:
        The argument vector, starting with ``sys.executable``.
    """
    cmd = [sys.executable, str(TEMPORAL_GBM_SCRIPT), "--seed", str(seed)]
    if v4_root is not None:
        cmd += ["--root", v4_root]
    if max_rows is not None:
        cmd += ["--max-rows", str(max_rows)]
    if out is not None:
        cmd += ["--out", out]
    return cmd


def format_temporal_gbm(result: dict) -> str:
    """Render both arms side by side, one row per fold, with AP and ROC-AUC."""
    lines = [
        (
            f"TEMPORAL GBM (levels vs levels+deltas+trends): {result['n_rows']} rows, "
            f"{result['n_companies']} companies, default rate {result['default_rate']:.4f}"
        ),
        (
            f"{'fold':>4} {'n_test':>8} {'AP levels':>10} {'AP +traj':>10} "
            f"{'AUC levels':>11} {'AUC +traj':>10}"
        ),
    ]
    for f in result["folds"]:
        a, b = f["levels"], f["levels+traj"]
        lines.append(
            f"{f['fold']:>4} {f['n_test']:>8} {a['average_precision']:>10.4f} "
            f"{b['average_precision']:>10.4f} {a['roc_auc']:>11.4f} {b['roc_auc']:>10.4f}"
        )
    return "\n".join(lines)


def run_temporal_gbm(v4_root: str | None = None, max_rows: int | None = None, seed: int = 0) -> int:
    """Run the temporal-GBM comparison in a subprocess and print both arms side by side.

    Returns:
        The subprocess exit status; the table is printed only on success.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "temporal_gbm.json")
        proc = subprocess.Popen(
            temporal_gbm_command(v4_root, max_rows, seed, out),
            stdout=subprocess.PIPE,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:  # stream the per-fold lines, skip the trailing JSON dump
            if line.startswith("fold "):
                print(line, end="", flush=True)
        code = proc.wait()
        if code != 0:
            print(f"TEMPORAL GBM: FAILED with exit status {code}")
            return code
        print(format_temporal_gbm(json.loads(Path(out).read_text())))
    return 0


def assign_folds(y: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Give every row one stratified fold id, so the k test folds are disjoint and cover all rows.

    Args:
        y: Binary labels ``(n,)``.
        k: Number of folds, at least 2.
        seed: Shuffle seed.

    Returns:
        ``(n,)`` integer fold ids in ``0..k-1``; each class is spread evenly across folds.

    Raises:
        ValueError: If ``k < 2`` or a class has fewer than ``k`` members.
    """
    if k < 2:
        raise ValueError(f"need at least 2 folds, got {k}")
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < k:
        raise ValueError(f"smallest class has {counts.min()} rows, fewer than {k} folds")
    fold = np.empty(len(y), dtype=int)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for i, (_, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold[test_idx] = i
    return fold


def run_credit_compare(
    model_a: str,
    model_b: str,
    folds: int = 5,
    horizons: tuple[int, ...] = (1, 3, 5),
    n_boot: int = 2000,
    seed: int = 0,
    panels: list[CreditDataset] | None = None,
    out: str | None = None,
) -> dict:
    """Compare two checkpoints on the credit panels at ``folds`` stratified folds.

    This is the multi-split upgrade of the single 70/30 split behind FINDINGS §73/§75
    (openspec ``cross-domain-coverage`` task 38.16). Each panel is split once into ``folds``
    disjoint stratified test folds; both checkpoints are fitted on each training part and score
    the same held-out rows. Average precision is reported first, per fold and as the mean,
    then ROC-AUC. The difference ``A - B`` is bootstrapped paired on the pooled out-of-fold
    predictions (every row scored once by each checkpoint, so rows are matched), per panel,
    and the p-values are Holm-adjusted across panels, separately for AP and ROC-AUC.

    The pooled bootstrap mixes folds fitted on different training parts, so it is a paired
    interval on out-of-fold predictions, not on one model's fit. The panels are observed
    bankruptcies, so this is a comparison on real data, not coverage against a solver.

    Args:
        model_a: First checkpoint path.
        model_b: Second checkpoint path; the difference is ``A - B``.
        folds: Number of stratified folds, at least 2.
        horizons: Polish bankruptcy horizons (years); Taiwan is always added.
        n_boot: Bootstrap resamples per panel and metric.
        seed: Seed for fold assignment and the bootstrap.
        panels: Datasets to use instead of the default Polish and Taiwan panels (tests).
        out: Optional path to write the record to as JSON.

    Returns:
        ``{"models", "folds", "panels": {name: {...}}, "coverage"}``. Each scored panel holds
        ``n_rows``, ``base_rate``, ``per_fold`` and ``mean`` (AP then ROC-AUC per checkpoint)
        and ``difference`` (delta, interval, raw and Holm-adjusted p, for ``ap`` and
        ``roc_auc``). ``coverage`` is a :class:`~fintfm.evaluation.coverage.Coverage` record
        over panels, with the reason for each skip.
    """
    cfgs = {m: FinancialTFM.load(m).cfg for m in (model_a, model_b)}
    max_features = min(c.max_features for c in cfgs.values())
    if panels is None:
        panels = [load_polish_bankruptcy(h) for h in horizons]
        panels.append(load_taiwan_bankruptcy())
    cov = Coverage()
    rec: dict = {}
    pooled: dict[str, dict] = {}
    for ds in panels:
        cov.attempt(ds.name)
        n_pos = int(np.sum(ds.y))
        print(
            f"\n=== {ds.name}: {len(ds.y)} companies, {ds.X.shape[1]} features, "
            f"default rate {ds.default_rate:.3%} ==="
        )
        if ds.X.shape[1] > max_features:
            reason = f"data has {ds.X.shape[1]} features, a checkpoint takes {max_features}"
            cov.skip(ds.name, reason)
            print(f"    SKIPPED: {reason}")
            continue
        if n_pos < folds or len(ds.y) - n_pos < folds:
            reason = f"fewer than {folds} rows in a class"
            cov.skip(ds.name, reason)
            print(f"    SKIPPED: {reason}")
            continue
        fold_of = assign_folds(ds.y, folds, seed)
        oof = {m: np.full(len(ds.y), np.nan) for m in (model_a, model_b)}
        per_fold: list[dict] = []
        failed = ""
        for f in range(folds):
            tr, te = np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)
            row: dict = {"fold": f, "n_test": len(te), "n_positive": int(ds.y[te].sum())}
            for m in (model_a, model_b):
                try:
                    clf = FinancialTFMClassifier(m, n_ensemble=8)
                    clf.fit(ds.X[tr], ds.y[tr])
                    oof[m][te] = clf.predict_proba(ds.X[te])[:, 1]
                except Exception as exc:  # noqa: BLE001 - a failed panel is a skip, not an abort
                    failed = f"fold {f}, {m}: {exc}"
                    break
                row[m] = {
                    "ap": float(average_precision_score(ds.y[te], oof[m][te])),
                    "roc_auc": float(roc_auc_score(ds.y[te], oof[m][te])),
                }
            if failed:
                break
            per_fold.append(row)
            print(
                f"    fold {f}: AP {row[model_a]['ap']:.4f} vs {row[model_b]['ap']:.4f}, "
                f"AUC {row[model_a]['roc_auc']:.4f} vs {row[model_b]['roc_auc']:.4f}"
            )
        if failed:
            cov.skip(ds.name, f"failed at {failed}")
            print(f"    SKIPPED: {failed}")
            continue
        cov.scored(ds.name)
        rec[ds.name] = {
            "n_rows": len(ds.y),
            "base_rate": float(np.mean(ds.y)),
            "per_fold": per_fold,
            "mean": {
                m: {k: float(np.mean([r[m][k] for r in per_fold])) for k in ("ap", "roc_auc")}
                for m in (model_a, model_b)
            },
            "difference": {},
        }
        pooled[ds.name] = {"y": ds.y, "a": oof[model_a], "b": oof[model_b]}

    for key, metric in (("ap", average_precision_score), ("roc_auc", None)):
        names = list(pooled)
        stats = [
            paired_auc_difference(
                pooled[n]["y"], pooled[n]["a"], pooled[n]["b"], n_boot, seed, metric=metric
            )
            for n in names
        ]
        adj = holm_adjusted_p([s[2] for s in stats])
        for n, (delta, (lo, hi), p), p_holm in zip(names, stats, adj, strict=True):
            rec[n]["difference"][key] = {
                "delta": delta,
                "ci95": [lo, hi],
                "p": p,
                "p_holm": p_holm,
                "n_panels_in_family": len(names),
            }
    for n, r in rec.items():
        d_ap, d_auc = r["difference"]["ap"], r["difference"]["roc_auc"]
        print(
            f"[{n}] mean AP {r['mean'][model_a]['ap']:.4f} vs {r['mean'][model_b]['ap']:.4f}; "
            f"AP diff {d_ap['delta']:+.4f} [{d_ap['ci95'][0]:+.4f}, {d_ap['ci95'][1]:+.4f}] "
            f"Holm p={d_ap['p_holm']:.4f}; mean AUC {r['mean'][model_a]['roc_auc']:.4f} vs "
            f"{r['mean'][model_b]['roc_auc']:.4f}; AUC diff {d_auc['delta']:+.4f} "
            f"Holm p={d_auc['p_holm']:.4f}"
        )
    print(cov.summary("panels"))
    record = {
        "models": {"a": model_a, "b": model_b},
        "folds": folds,
        "panels": rec,
        "coverage": cov.to_dict(),
    }
    if out:
        Path(out).write_text(json.dumps(record, indent=2))
    return record


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
    p.add_argument(
        "--no-sample-efficiency",
        action="store_true",
        help="with --credit: skip the small-n sample-efficiency probe (on by default)",
    )
    p.add_argument(
        "--time-split",
        action="store_true",
        help="V4FinBench random vs time-based AUC side by side (needs the cached data); "
        "with --credit it is printed after the credit panels",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="with --time-split: subsample this many company-years",
    )
    p.add_argument(
        "--folds", type=int, default=None, help="with --credit --compare: stratified folds (5)"
    )
    p.add_argument(
        "--compare",
        type=str,
        default=None,
        help="with --credit: two checkpoints 'A,B' scored at --folds stratified folds with a "
        "paired bootstrap of A - B per panel, Holm-adjusted across panels",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="with --compare or --time-split: write the record as JSON here",
    )
    p.add_argument("--v4-root", type=str, default=None, help="V4FinBench parquet directory")
    p.add_argument(
        "--temporal-gbm",
        action="store_true",
        help="LightGBM on levels vs levels+deltas+trends, run in a subprocess "
        "(uses --v4-root, --max-rows, --seed); no checkpoint needed",
    )
    p.add_argument("--seed", type=int, default=0, help="with --temporal-gbm: LightGBM seed")
    args = p.parse_args()
    if args.temporal_gbm:
        raise SystemExit(run_temporal_gbm(args.v4_root, args.max_rows, args.seed))
    if args.credit and args.compare:
        pair = [m.strip() for m in args.compare.split(",") if m.strip()]
        if len(pair) != 2:
            p.error("--compare takes exactly two checkpoints, 'A,B'")
        run_credit_compare(pair[0], pair[1], folds=args.folds or 5, out=args.out)
        return
    if args.credit:
        kwargs: dict = {"sample_efficiency": not args.no_sample_efficiency}
        if args.time_split:
            try:
                kwargs["time_panel"] = load_v4finbench(args.v4_root, max_rows=args.max_rows)
            except FileNotFoundError as exc:
                print(f"TIME-BASED SPLIT: NOT RUN, V4FinBench data not found ({exc})")
        record = run_credit(args.model, **kwargs)
        if args.out and "time_vs_random" in record:
            Path(args.out).write_text(json.dumps(record["time_vs_random"], indent=2))
        return
    if args.time_split:
        record = run_time_vs_random(
            load_v4finbench(args.v4_root, max_rows=args.max_rows), args.model
        )
        if args.out:
            Path(args.out).write_text(json.dumps(record, indent=2))
        return
    run_synthetic(args.model, n_tasks=args.synthetic_tasks)
    if args.openml_id:
        run_openml(args.model, args.openml_id)


if __name__ == "__main__":
    main()
