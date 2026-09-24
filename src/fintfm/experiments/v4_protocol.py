"""V4FinBench's own evaluation protocol, so a comparable number can exist at all.

Why this exists
---------------
Every number this project has produced on V4FinBench uses an out-of-time split of our own
design. The benchmark's published protocol is different in four ways that each break
comparability (``docs/results/FINDINGS.md`` §36):

1. **5-fold company-grouped cross-validation, not out-of-time.** Grouping is by company, not
   by date, so a fold may contain 2019 observations while predicting a 2008 one.
2. **The horizon tasks are built on different rows.** For horizon *h* a distressed company has
   its final *h* reports removed and the remaining latest report is labelled positive. Our
   ``_curve_truth`` derives cumulative labels for a *fixed* row from its ``period``.
3. **Their inference context is 10,000 rows**; ours is 2,000.
4. **Their TabPFN is fine-tuned on this data**; ours never touches real data (decision D2).

So "where would we place on their table" has had no answer. This module implements their
protocol as specified in their repository's ``docs/benchmark_protocol.md`` and
``src/v4finbench/data/folds.py`` (github.com/genwro-ai/V4FinBench, MIT). **The algorithm is
reimplemented from that specification rather than copied**, which is what ``CLAUDE.md``'s
licensing boundary asks for; their published numbers are cited, their code is not vendored.

What is reproduced exactly
--------------------------
Fold assignment is deterministic given the same parquet row order and seed: one
``RandomState(42)`` consumed across countries in sorted order, companies shuffled within each
country and dealt round-robin to five folds. For run ``fold``, validation is ``fold``, test is
``(fold + 1) % 5``, and training is the remaining three. Preprocessing — median imputation
then standardisation — is fitted on the training split alone and applied to the other two.
Decision thresholds are calibrated on validation by maximising F₁ on the precision-recall
curve, then applied unchanged to test.

What is still not comparable, and is reported rather than hidden
----------------------------------------------------------------
Their TabPFN arm is fine-tuned on this data. Ours cannot be, so the honest comparison is
against their *classical* baselines and their vanilla TabPFN, not their fine-tuned one.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from fintfm.config import Config, load_config

#: Paper horizon to Kaggle file, from their ``docs/benchmark_protocol.md``. The off-by-one is
#: theirs and is a trap: ``h=0`` is ``company_years_h1.parquet``.
HORIZON_FILES: dict[int, str] = {
    0: "company_years_h1.parquet",
    1: "company_years_h2.parquet",
    2: "company_years_h3.parquet",
    3: "company_years_h4.parquet",
    4: "company_years_h5.parquet",
    5: "company_years_h6.parquet",
}

#: Columns their protocol drops before feature selection. Identifiers plus fields excluded
#: from the released schema.
DROP_COLUMNS: tuple[str, ...] = (
    "company", "industry", "link", "num", "emis_id", "sector_2", "sector_3", "sector_4",
    "Revenue/employee", "Fixed_assets/employee", "EBITDA/cash_flow",
)

#: Their target column.
TARGET = "main_label"

#: Hyperparameter grids from V4FinBench's Table 5, searched on the **validation fold** as
#: their protocol specifies. Reproducing their baselines means reproducing their tuning:
#: measured here, default LightGBM and XGBoost score *below* logistic regression on ROC-AUC
#: (0.95 and 0.81-0.96 against 0.98), which would understate the field in a comparison we
#: intend to publish — the same defect as ``docs/results/FINDINGS.md`` §25, pointed the other way.
#:
#: Searching this is expensive: roughly 76 fits per fold on ~600,000 rows, and the boosters
#: fit out-of-process one configuration at a time. It is therefore opt-in via ``--tune``,
#: and an untuned run says so in its output rather than presenting defaults as baselines.
CLASSICAL_GRIDS: dict[str, dict[str, list]] = {
    "logistic_regression": {"C": [1e-3, 1e-2, 1e-1, 1.0]},
    "mlp": {
        "hidden_layer_sizes": [(64, 64), (128, 128), (256, 256)],
        "alpha": [1e-4, 1e-3],
        "learning_rate_init": [1e-3, 1e-2],
    },
    "random_forest": {"n_estimators": [100, 300], "max_depth": [5, 10, None]},
}

#: Boosting grids from the same table. Applied through the out-of-process worker.
BOOSTING_GRIDS: dict[str, dict[str, list]] = {
    "xgboost": {
        "n_estimators": [100, 200], "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
    },
    "catboost": {
        "iterations": [100, 200], "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
    },
    "lightgbm": {
        "n_estimators": [100, 200], "max_depth": [-1, 5, 10],
        "learning_rate": [0.05, 0.1, 0.2],
    },
}


def _grid(space: dict[str, list]) -> list[dict]:
    """Every combination in a hyperparameter grid, as a list of keyword dicts."""
    from itertools import product

    keys = sorted(space)
    return [dict(zip(keys, combo)) for combo in product(*(space[k] for k in keys))]


def _classical_estimator(name: str, params: dict):
    """Build one of the paper's non-boosting baselines."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, **params)
    if name == "mlp":
        return MLPClassifier(max_iter=300, early_stopping=True, **params)
    if name == "random_forest":
        return RandomForestClassifier(n_jobs=-1, random_state=0, **params)
    raise ValueError(f"unknown classical baseline {name!r}")

#: Fold-assignment parameters, fixed by their protocol.
N_SPLITS = 5
FOLD_SEED = 42


def build_fold_assignments(
    country: np.ndarray, company: np.ndarray, n_splits: int = N_SPLITS, seed: int = FOLD_SEED
) -> np.ndarray:
    """Assign every row to a company-grouped, country-preserving fold.

    Reimplemented from V4FinBench's ``build_fold_assignments`` specification: a single
    ``RandomState`` is consumed across countries **in sorted order**, the unique companies
    within each country are shuffled, and they are dealt round-robin to folds. Every
    observation of a company therefore lands in the same fold, and each country's companies
    are spread evenly across folds.

    The single shared generator matters: consuming it per country in a different order gives
    a different, equally valid assignment that will not match the published one.

    Args:
        country: ``(n,)`` country label per row.
        company: ``(n,)`` company identifier per row.
        n_splits: Number of folds.
        seed: Random seed.

    Returns:
        ``(n,)`` fold index in ``[0, n_splits)``.

    Raises:
        ValueError: If a row carries a null company, which cannot be grouped.
    """
    if any(c is None or (isinstance(c, float) and np.isnan(c)) for c in company):
        raise ValueError("rows with a null company cannot be assigned to a group-held-out fold")
    rng = np.random.RandomState(seed)
    assignment: dict[tuple[object, object], int] = {}
    for c in sorted(np.unique(country).tolist()):
        members = company[country == c]
        # first-appearance order, matching pandas' unique(), because the shuffle is seeded and
        # therefore sensitive to the order it is handed
        _, first = np.unique(members, return_index=True)
        companies = members[np.sort(first)].copy()
        rng.shuffle(companies)
        for i, name in enumerate(companies):
            assignment[(c, name)] = i % n_splits
    return np.array([assignment[(a, b)] for a, b in zip(country, company)], dtype=np.int64)


def split_indices_for_fold(
    folds: np.ndarray, fold: int, n_splits: int = N_SPLITS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train, validation and test indices for one rotation of their protocol.

    Args:
        folds: Fold assignment per row.
        fold: Which rotation, in ``[0, n_splits)``.
        n_splits: Number of folds.

    Returns:
        ``(train_idx, val_idx, test_idx)``. Validation is ``fold``, test is
        ``(fold + 1) % n_splits``, training is the remaining three — roughly 60/20/20.

    Raises:
        ValueError: If ``fold`` is out of range.
    """
    if not 0 <= fold < n_splits:
        raise ValueError(f"fold must lie in [0, {n_splits}), got {fold}")
    val_fold, test_fold = fold, (fold + 1) % n_splits
    return (
        np.flatnonzero((folds != val_fold) & (folds != test_fold)),
        np.flatnonzero(folds == val_fold),
        np.flatnonzero(folds == test_fold),
    )


def best_f1_threshold(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Threshold maximising F₁ on the precision-recall curve, as their protocol specifies.

    Args:
        y_true: Binary outcomes.
        score: Predicted probability of the positive class.

    Returns:
        ``(threshold, f1_at_that_threshold)``. Returns ``(0.5, 0.0)`` when no positive class
        is present, which is a degenerate split rather than a result.
    """
    from sklearn.metrics import precision_recall_curve

    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    # precision_recall_curve returns one more point than thresholds
    best = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    return (float(thresholds[best]) if len(thresholds) else 0.5), float(f1[best])


@dataclass
class FoldResult:
    """One arm scored on one fold of their protocol."""

    arm: str
    fold: int
    n_train: int
    n_test: int
    test_positives: int
    roc_auc: float
    f1: float
    threshold: float
    #: Average precision (area under the precision-recall curve). **Our addition, not part of
    #: their published protocol** -- kept separate so the comparable columns stay comparable.
    #: At a 0.38% base rate ROC-AUC is dominated by the negative majority and can read 0.98
    #: while precision in the decision region is poor; AP is the metric that notices
    #: (``docs/results/FINDINGS.md`` §59).
    avg_precision: float = float("nan")
    #: Best F1 achievable on *test* by any threshold. Compared against ``f1``, which uses the
    #: threshold chosen on validation, this separates a ranking that is weak near the decision
    #: boundary from a threshold that simply failed to transfer.
    f1_oracle: float = float("nan")


@dataclass
class ArmSummary:
    """An arm's fold-averaged scores, reported the way their table is."""

    arm: str
    roc_auc_mean: float
    roc_auc_std: float
    f1_mean: float
    f1_std: float
    avg_precision_mean: float = float("nan")
    f1_oracle_mean: float = float("nan")
    folds: list[FoldResult] = field(default_factory=list)


def _load_horizon(root: Path, horizon: int, max_rows: int | None):
    """Read one horizon file, returning features, target, and the fold-grouping columns."""
    import pyarrow.parquet as pq

    path = root / HORIZON_FILES[horizon]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Fetch with:  uv run fintfm-fetch v4finbench"
        )
    table = pq.read_table(path)
    df = table.to_pandas()
    if max_rows is not None and len(df) > max_rows:
        # take whole companies, never a partial company, or the grouping guarantee breaks
        keep = df["company"].drop_duplicates()
        rng = np.random.RandomState(0)
        chosen = set(rng.permutation(keep.to_numpy())[: max(1, max_rows // 6)].tolist())
        df = df[df["company"].isin(chosen)].reset_index(drop=True)
    y = df[TARGET].to_numpy().astype(np.int64)
    country = df["country"].to_numpy()
    company = df["company"].to_numpy()
    features = df.drop(columns=[c for c in (*DROP_COLUMNS, TARGET) if c in df.columns])
    features = features.select_dtypes(include=[np.number])
    return features.to_numpy(dtype=np.float32), y, country, company, list(features.columns)


def run(
    model_path: str,
    out_dir: Path,
    horizon: int = 0,
    folds: tuple[int, ...] = (0, 1, 2, 3, 4),
    max_rows: int | None = None,
    root: Path | None = None,
    with_boosting: bool = True,
    tune: bool = False,
    classical: tuple[str, ...] = ("logistic_regression",),
    cfg: Config | None = None,
    device: str = "cpu",
) -> dict:
    """Score arms under V4FinBench's published protocol.

    Args:
        model_path: Checkpoint. Its classification head is used, so it must have been trained
            for classification (``docs/results/FINDINGS.md`` §34 — a hazard-only checkpoint is
            refused rather than silently served).
        out_dir: Directory for ``v4_protocol.json``.
        horizon: Paper horizon, 0-5. Note the file mapping is off by one; see
            :data:`HORIZON_FILES`.
        folds: Which rotations to run.
        max_rows: Development cap. Companies are kept whole, so the grouping guarantee holds.
        root: Directory holding the parquet files.
        with_boosting: Run LightGBM, CatBoost and XGBoost as baselines. They fit
            out-of-process because of the macOS OpenMP conflict (``docs/infra/COMPUTE.md``), and
            they are the baselines that actually compete — the paper reports gradient-boosted
            trees as its strongest classical cluster.
        tune: Grid-search each baseline on the validation fold, per their protocol and
            Table 5. **Off by default because it is expensive** — roughly 76 fits per fold on
            600,000 rows — and an untuned run labels itself as such.
        classical: Which non-boosting baselines to run, from :data:`CLASSICAL_GRIDS`.
        cfg: Configuration; loaded from the packaged default when omitted.
        device: Torch device for the fintfm arm.

    Returns:
        The recorded result dictionary.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    out_dir.mkdir(parents=True, exist_ok=True)

    from fintfm.evaluation.boosting import available_boosting, fit_predict_boosting
    from fintfm.evaluation.datasets import CACHE_DIR
    from fintfm.inference.classifier import FinancialTFMClassifier
    from fintfm.modeling.model import FinancialTFM

    cfg = cfg or load_config()
    root = root or (CACHE_DIR / "v4finbench")
    X, y, country, company, _names = _load_horizon(root, horizon, max_rows)
    print(
        f"horizon {horizon} ({HORIZON_FILES[horizon]}): {len(y):,} rows, {X.shape[1]} features, "
        f"{y.sum():,} positive ({y.mean():.3%})"
    )
    assignments = build_fold_assignments(country, company)
    print(f"folds: {np.bincount(assignments).tolist()} rows per fold\n")

    model = FinancialTFM.load(model_path)
    # §25: every gradient-boosting comparison in this project once used sklearn's weakest
    # implementation because the strong ones were silently absent. available_boosting()
    # announces what is missing rather than skipping it quietly.
    boosters = available_boosting() if with_boosting else []
    if with_boosting and not boosters:
        print("  no boosting baselines available; install with `uv sync --extra bench`")
    results: list[FoldResult] = []
    for fold in folds:
        tr, va, te = split_indices_for_fold(assignments, fold)
        # preprocessing fitted on the training split alone, per their protocol
        imputer = SimpleImputer(strategy="median").fit(X[tr])
        scaler = StandardScaler().fit(imputer.transform(X[tr]))
        Xtr, Xva, Xte = (scaler.transform(imputer.transform(X[i])) for i in (tr, va, te))

        arms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in classical:
            configs = _grid(CLASSICAL_GRIDS[name]) if tune else [{}]
            best, best_auc = None, -np.inf
            for params in configs:
                est = _classical_estimator(name, params).fit(Xtr, y[tr])
                p_val = est.predict_proba(Xva)[:, 1]
                # selected on the validation fold, as their protocol specifies
                auc = roc_auc_score(y[va], p_val) if len(np.unique(y[va])) > 1 else 0.0
                if auc > best_auc:
                    best, best_auc = (est, params), auc
            est, params = best
            arms[name] = (est.predict_proba(Xva)[:, 1], est.predict_proba(Xte)[:, 1])
            if tune:
                print(f"    {name} best params: {params} (val AUC {best_auc:.4f})")

        if model.cfg.max_features >= Xtr.shape[1]:
            clf = FinancialTFMClassifier(
                model,
                max_context=cfg.inference.max_context,
                context_strategy=cfg.inference.context_strategy,
                feature_transform=cfg.inference.feature_transform,
                n_ensemble=cfg.inference.n_ensemble,
                query_chunk=cfg.inference.query_chunk,
                feature_chunk=cfg.inference.feature_chunk,
                device=device,
                random_state=fold,
            ).fit(Xtr.astype(np.float32), y[tr])
            arms["fintfm"] = (
                clf.predict_proba(Xva.astype(np.float32))[:, 1],
                clf.predict_proba(Xte.astype(np.float32))[:, 1],
            )
        else:
            print(
                f"  fintfm skipped: checkpoint takes {model.cfg.max_features} features, "
                f"data has {Xtr.shape[1]}"
            )

        # the boosters score validation and test in one fit: the threshold is calibrated on
        # validation and applied to test, so both are needed from the same fitted model
        if boosters:
            both = np.vstack([Xva, Xte])
            for name in boosters:
                configs = _grid(BOOSTING_GRIDS[name]) if tune else [None]
                best_pred, best_auc, best_params = None, -np.inf, None
                for params in configs:
                    p_both = fit_predict_boosting(name, Xtr, y[tr], both, params=params)
                    if p_both is None:
                        continue
                    p_val = p_both[: len(va)]
                    auc = roc_auc_score(y[va], p_val) if len(np.unique(y[va])) > 1 else 0.0
                    if auc > best_auc:
                        best_pred, best_auc, best_params = p_both, auc, params
                if best_pred is None:
                    print(f"  {name} failed on fold {fold}; excluded from this fold")
                    continue
                arms[name] = (best_pred[: len(va)], best_pred[len(va) :])
                if tune:
                    print(f"    {name} best params: {best_params} (val AUC {best_auc:.4f})")

        # Per-row test predictions, kept so arms can be compared with a **paired** bootstrap.
        # Two models scored on the same rows have correlated errors, so an unpaired interval
        # overstates the uncertainty of their difference; without the raw vectors the only
        # available comparison is point estimates, which is how a 79-positive fold turns into
        # a league table nobody can check (docs/results/FINDINGS.md §60).
        np.savez_compressed(
            out_dir / f"predictions_fold{fold}.npz",
            y_true=y[te],
            **{f"pred_{a}": v[1] for a, v in arms.items()},
        )

        for arm, (p_val, p_test) in arms.items():
            thr, _ = best_f1_threshold(y[va], p_val)
            both = len(np.unique(y[te])) > 1
            auc = roc_auc_score(y[te], p_test) if both else float("nan")
            f1 = f1_score(y[te], (p_test >= thr).astype(int), zero_division=0)
            ap = average_precision_score(y[te], p_test) if both else float("nan")
            # the same threshold search, but fitted on test: an upper bound no honest
            # procedure reaches, and useful only as the gap against `f1`
            _, f1_orc = best_f1_threshold(y[te], p_test) if both else (0.0, float("nan"))
            results.append(
                FoldResult(arm, fold, len(tr), len(te), int(y[te].sum()), float(auc),
                           float(f1), thr, float(ap), float(f1_orc))
            )
            print(
                f"  fold {fold} {arm:>20}: ROC-AUC {auc:.4f}  AP {ap:.4f}  F1 {f1:.4f}  "
                f"(oracle {f1_orc:.4f})  thr {thr:.4g}  "
                f"({y[te].sum():,} positives in {len(te):,})",
                flush=True,
            )

    summaries: list[ArmSummary] = []
    for arm in sorted({r.arm for r in results}):
        rows = [r for r in results if r.arm == arm]
        aucs = np.array([r.roc_auc for r in rows])
        f1s = np.array([r.f1 for r in rows])
        summaries.append(
            ArmSummary(
                arm=arm,
                roc_auc_mean=float(np.nanmean(aucs)),
                roc_auc_std=float(np.nanstd(aucs, ddof=1)) if len(aucs) > 1 else float("nan"),
                f1_mean=float(np.mean(f1s)),
                f1_std=float(np.std(f1s, ddof=1)) if len(f1s) > 1 else float("nan"),
                avg_precision_mean=float(np.nanmean([r.avg_precision for r in rows])),
                f1_oracle_mean=float(np.nanmean([r.f1_oracle for r in rows])),
                folds=rows,
            )
        )

    record = {
        "dataset": "v4finbench",
        "protocol": "V4FinBench published protocol (5-fold, company-grouped within country)",
        "attribution": (
            "Protocol per github.com/genwro-ai/V4FinBench docs/benchmark_protocol.md (MIT), "
            "reimplemented not copied. Data: Kostrzewa et al., arXiv:2605.10896, CC BY 4.0."
        ),
        "config": {
            "model": model_path, "horizon": horizon, "folds": list(folds),
            "max_rows": max_rows, "n_features": int(X.shape[1]),
            "fold_seed": FOLD_SEED, "n_splits": N_SPLITS,
            "boosting": boosters,
            "classical": list(classical),
            "tuned": tune,
            "context_strategy": cfg.inference.context_strategy,
            "feature_transform": cfg.inference.feature_transform,
            "max_context": cfg.inference.max_context,
            "n_ensemble": cfg.inference.n_ensemble,
            "config_sources": list(cfg.sources),
        },
        "arms": [asdict(s) for s in summaries],
    }
    (out_dir / "v4_protocol.json").write_text(json.dumps(record, indent=2))
    return record


def summarise(record: dict) -> str:
    """Render fold-averaged ROC-AUC and F1, the way their table reports them."""
    c = record["config"]
    header = (
        f"V4FinBench published protocol — horizon {c['horizon']}, "
        f"{len(c['folds'])} folds, {c['n_features']} features"
    )
    lines = [
        header, "",
        f"{'arm':>22} {'ROC-AUC':>17} {'F1':>17} {'AP':>9} {'F1-oracle':>11}",
    ]
    for a in record["arms"]:
        lines.append(
            f"{a['arm']:>22}   {a['roc_auc_mean']:.4f} ± {a['roc_auc_std']:.4f}"
            f"   {a['f1_mean']:.4f} ± {a['f1_std']:.4f}"
            f"   {a.get('avg_precision_mean', float('nan')):.4f}"
            f"   {a.get('f1_oracle_mean', float('nan')):9.4f}"
        )
    lines += [
        "",
        (
            "ROC-AUC and F1 are their protocol's metrics. AP (average precision) and "
            "F1-oracle are OURS and are not comparable to their table: at this base rate "
            "ROC-AUC is dominated by the negative majority, and F1-oracle tunes the "
            "threshold on test, so it is an upper bound rather than a score. Read "
            "`F1-oracle - F1` as how much was lost in transferring the threshold "
            "(docs/results/FINDINGS.md §59)."
        ),
    ]
    tuned = c.get("tuned", False)
    lines += [
        "",
        (
            "Baselines were grid-searched on the validation fold, per their Table 5."
            if tuned
            else "*** BASELINES ARE UNTUNED (library defaults). Their protocol grid-searches "
                 "every baseline on the validation fold, so these are NOT their baselines "
                 "and understate the field -- re-run with --tune before quoting. ***"
        ),
        "",
        "Their published reference at this horizon is in docs/results/FINDINGS.md §36. Their TabPFN is",
        "fine-tuned on this data and ours never sees real data, so the like-for-like comparison",
        "is against their classical baselines, not their fine-tuned TabPFN.",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=str, default="runs/v4-protocol")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--horizon", type=int, default=0, choices=sorted(HORIZON_FILES))
    p.add_argument("--folds", type=str, default="0,1,2,3,4", help="comma-separated")
    p.add_argument("--max-rows", type=int, default=None, help="development cap")
    p.add_argument(
        "--device", type=str, default="cpu",
        help="torch device for the fintfm arm. Cell-attention checkpoints "
             "(n_cell_blocks>0) are far cheaper on mps/cuda than cpu",
    )
    p.add_argument(
        "--tune", action="store_true",
        help="grid-search every baseline on the validation fold, as their protocol and "
             "Table 5 specify. Expensive: ~76 fits per fold on 600k rows. Without it, the "
             "baselines run at library defaults and the output says so",
    )
    p.add_argument(
        "--classical", type=str, default="logistic_regression",
        help="comma-separated, from logistic_regression, mlp, random_forest",
    )
    p.add_argument(
        "--no-boosting", action="store_true",
        help="skip LightGBM/CatBoost/XGBoost; they are the baselines that compete, so this "
             "is for quick development runs only",
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    record = run(
        args.model, Path(args.out), horizon=args.horizon,
        folds=tuple(int(v) for v in args.folds.split(",")),
        max_rows=args.max_rows, with_boosting=not args.no_boosting, tune=args.tune,
        classical=tuple(v.strip() for v in args.classical.split(",") if v.strip()),
        cfg=cfg, device=args.device,
    )
    print("\n" + summarise(record))
    print(f"\nconfig: {cfg.provenance()}")
    print(f"\n{record['attribution']}")


if __name__ == "__main__":
    main()
