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
from typing import TYPE_CHECKING

import numpy as np

from fintfm.config import Config, load_config

if TYPE_CHECKING:  # sklearn is imported lazily below; this keeps the annotation free
    from sklearn.base import ClassifierMixin

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


def _parse_value(text: str) -> object:
    """Read one ``key=value`` value as JSON (int, float, bool, null), else keep the string."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_booster_params(specs: list[str]) -> list[tuple[str, dict]]:
    """Parse ``--booster-params`` specs into ``(booster, params)`` pairs.

    Args:
        specs: Each ``name:key=value,key=value``, e.g.
            ``lightgbm:n_estimators=1000,learning_rate=0.05``. ``name:`` with nothing after
            the colon means the library defaults, scored as an arm of its own.

    Returns:
        One ``(booster, params)`` per spec, in the order given.

    Raises:
        ValueError: On an unknown booster, a missing colon, or an item without ``=``.
    """
    from fintfm.evaluation.boosting import BOOSTING_MODELS

    out: list[tuple[str, dict]] = []
    for spec in specs:
        name, sep, rest = spec.partition(":")
        if not sep:
            raise ValueError(f"booster spec {spec!r} needs 'name:key=value,...'")
        if name not in BOOSTING_MODELS:
            raise ValueError(f"unknown booster {name!r}; expected one of {BOOSTING_MODELS}")
        params: dict = {}
        for item in filter(None, (t.strip() for t in rest.split(","))):
            key, eq, value = item.partition("=")
            if not eq or not key.strip():
                raise ValueError(f"booster parameter {item!r} in {spec!r} needs key=value")
            params[key.strip()] = _parse_value(value.strip())
        out.append((name, params))
    return out


def booster_arm_name(name: str, params: dict) -> str:
    """Arm name for a booster at fixed parameters: ``lightgbm[n_estimators=1000,...]``.

    The bare name is kept for empty parameters, so a defaults arm reads the same as the
    ordinary untuned booster arm it reproduces.
    """
    if not params:
        return name
    return f"{name}[" + ",".join(f"{k}={v}" for k, v in params.items()) + "]"


def _classical_estimator(name: str, params: dict) -> ClassifierMixin:
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


def company_codes(company: np.ndarray) -> np.ndarray:
    """Code each row's company as an integer, for storing beside per-row predictions.

    The code is the company's index among the panel's sorted unique ids, so it is fixed by the
    loaded panel alone and the same run's folds share one coding. Integers rather than the ids
    themselves: V4FinBench's ids are strings, which ``np.savez`` would store either as wide
    fixed-width unicode or as a pickled object array that ``np.load`` refuses by default.

    Args:
        company: ``(n,)`` company identifier per row.

    Returns:
        ``(n,)`` ``int32`` codes; equal codes mean the same company.
    """
    _, codes = np.unique(company, return_inverse=True)
    return codes.ravel().astype(np.int32)


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
    #: Average precision on the *validation* split, so a setting can be chosen without
    #: looking at test (§163). Our addition, like ``avg_precision``.
    val_avg_precision: float = float("nan")


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


def _load_horizon(
    root: Path, horizon: int, max_rows: int | None, drop_columns: tuple[str, ...] = ()
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Read one horizon file, returning features, target, and the fold-grouping columns.

    Args:
        root: Directory holding the parquet files.
        horizon: Paper horizon, a key of :data:`HORIZON_FILES`.
        max_rows: Development cap; companies are kept whole.
        drop_columns: Columns to drop **in addition to** :data:`DROP_COLUMNS`, for an
            ablation. Grouping columns are still read for the folds, so dropping ``country``
            removes it as a feature only.

    Returns:
        ``(X, y, country, company, feature_names)``.

    Raises:
        ValueError: If an extra column is absent from the file or is the target. A misspelled
            name must not complete a run that silently dropped nothing.
    """
    import pyarrow.parquet as pq

    path = root / HORIZON_FILES[horizon]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Fetch with:  uv run fintfm-fetch v4finbench"
        )
    table = pq.read_table(path)
    df = table.to_pandas()
    missing = [c for c in drop_columns if c not in df.columns]
    if missing:
        raise ValueError(f"--drop-columns names columns not in {path.name}: {missing}")
    if TARGET in drop_columns:
        raise ValueError(f"{TARGET!r} is the target and cannot be dropped as a feature")
    if max_rows is not None and len(df) > max_rows:
        # take whole companies, never a partial company, or the grouping guarantee breaks
        keep = df["company"].drop_duplicates()
        rng = np.random.RandomState(0)
        chosen = set(rng.permutation(keep.to_numpy())[: max(1, max_rows // 6)].tolist())
        df = df[df["company"].isin(chosen)].reset_index(drop=True)
    y = df[TARGET].to_numpy().astype(np.int64)
    country = df["country"].to_numpy()
    company = df["company"].to_numpy()
    features = df.drop(
        columns=[c for c in (*DROP_COLUMNS, *drop_columns, TARGET) if c in df.columns]
    )
    features = features.select_dtypes(include=[np.number])
    return features.to_numpy(dtype=np.float32), y, country, company, list(features.columns)


def selection_scores(
    y_val: np.ndarray, p_val: np.ndarray, metric: str
) -> tuple[float, float, float]:
    """Score one tuning candidate on the validation fold.

    Args:
        y_val: Validation labels.
        p_val: Candidate's validation scores.
        metric: ``"ap"`` or ``"auc"``, the one the first element reports.

    Returns:
        ``(selection score, average precision, ROC-AUC)``. All three are 0.0 for a single-class
        validation fold, where neither metric is defined, so such a fold selects nothing.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    if len(np.unique(y_val)) < 2:
        return 0.0, 0.0, 0.0
    ap = float(average_precision_score(y_val, p_val))
    auc = float(roc_auc_score(y_val, p_val))
    return (ap if metric == "ap" else auc), ap, auc


def run(
    model_path: str | None,
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
    select_metric: str = "ap",
    drop_columns: tuple[str, ...] = (),
    booster_params: tuple[tuple[str, dict], ...] = (),
) -> dict:
    """Score arms under V4FinBench's published protocol.

    Args:
        model_path: Checkpoint, or ``None`` to score the baselines alone (no fintfm arm) --
            for measuring them on folds whose fintfm scores already exist. Its
            classification head is used, so it must have been trained
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
        select_metric: What ``tune`` selects each baseline's configuration on, on the
            validation fold: ``"ap"`` (average precision, this project's headline metric) or
            ``"auc"`` (ROC-AUC, what runs before 2026-10-16 used; their protocol does not say).
            At a 0.36% base rate the two disagree, so the choice is recorded with the run.
        drop_columns: Feature columns to drop beyond :data:`DROP_COLUMNS`, for an ablation
            (audit AU2.5). Recorded with the run; ``n_features`` changes accordingly.
        booster_params: Extra booster arms at **fixed** parameters, as ``(booster, params)``
            pairs (see :func:`parse_booster_params`). Each is fitted once per fold, never
            tuned, and named by :func:`booster_arm_name` so it cannot collide with the
            ordinary booster arm; they run whether or not ``with_boosting`` is set.

    Returns:
        The recorded result dictionary.

    Raises:
        ValueError: On an unknown ``select_metric``, or two arms that would share a name.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    if select_metric not in ("ap", "auc"):
        raise ValueError(f"select_metric must be 'ap' or 'auc', got {select_metric!r}")
    fixed_arms = [(booster_arm_name(n, p), n, p) for n, p in booster_params]
    fixed_names = [a for a, _, _ in fixed_arms]
    if len(set(fixed_names)) != len(fixed_names):
        raise ValueError(f"--booster-params gives two arms the same name: {fixed_names}")
    out_dir.mkdir(parents=True, exist_ok=True)

    from fintfm.evaluation.boosting import available_boosting, fit_predict_boosting
    from fintfm.evaluation.datasets import CACHE_DIR
    from fintfm.inference.classifier import FinancialTFMClassifier
    from fintfm.modeling.model import FinancialTFM

    cfg = cfg or load_config()
    root = root or (CACHE_DIR / "v4finbench")
    X, y, country, company, _names = _load_horizon(root, horizon, max_rows, drop_columns)
    if drop_columns:
        print(f"dropping {len(drop_columns)} extra column(s) for an ablation: "
              f"{list(drop_columns)}")
    print(
        f"horizon {horizon} ({HORIZON_FILES[horizon]}): {len(y):,} rows, {X.shape[1]} features, "
        f"{y.sum():,} positive ({y.mean():.3%})"
    )
    assignments = build_fold_assignments(country, company)
    print(f"folds: {np.bincount(assignments).tolist()} rows per fold\n")
    codes = company_codes(company)

    model = FinancialTFM.load(model_path) if model_path is not None else None
    # §25: every gradient-boosting comparison in this project once used sklearn's weakest
    # implementation because the strong ones were silently absent. available_boosting()
    # announces what is missing rather than skipping it quietly.
    boosters = available_boosting() if with_boosting else []
    if with_boosting and not boosters:
        print("  no boosting baselines available; install with `uv sync --extra bench`")
    clash = set(fixed_names) & set(boosters)
    if clash:
        raise ValueError(f"--booster-params arms {sorted(clash)} collide with the default "
                         "booster arms; drop the empty spec or pass --no-boosting")
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
            best, best_score, best_val = None, -np.inf, (0.0, 0.0)
            for params in configs:
                est = _classical_estimator(name, params).fit(Xtr, y[tr])
                p_val = est.predict_proba(Xva)[:, 1]
                # selected on the validation fold, as their protocol specifies
                score, ap, auc = selection_scores(y[va], p_val, select_metric)
                if score > best_score:
                    best, best_score, best_val = (est, params), score, (ap, auc)
            est, params = best
            arms[name] = (est.predict_proba(Xva)[:, 1], est.predict_proba(Xte)[:, 1])
            if tune:
                print(f"    {name} best params: {params} (selected on val {select_metric}; "
                      f"val AP {best_val[0]:.4f}, val AUC {best_val[1]:.4f})")

        if model is None:
            pass  # baselines only
        elif model.cfg.max_features >= Xtr.shape[1]:
            clf = FinancialTFMClassifier(
                model,
                max_context=cfg.inference.max_context,
                context_strategy=cfg.inference.context_strategy,
                feature_transform=cfg.inference.feature_transform,
                n_ensemble=cfg.inference.n_ensemble,
                query_chunk=cfg.inference.query_chunk,
                feature_chunk=cfg.inference.feature_chunk,
                finetune_steps=cfg.inference.finetune_steps,
                finetune_lr=cfg.inference.finetune_lr,
                finetune_batch=cfg.inference.finetune_batch,
                finetune_query=cfg.inference.finetune_query,
                finetune_context=cfg.inference.finetune_context,
                finetune_query_group_frac=cfg.inference.finetune_query_group_frac,
                finetune_query_pos_frac=cfg.inference.finetune_query_pos_frac,
                finetune_seed=cfg.inference.finetune_seed,
                ensemble_transforms=cfg.inference.ensemble_transforms,
                ensemble_svd_components=cfg.inference.ensemble_svd_components,
                ensemble_class_shift=cfg.inference.ensemble_class_shift,
                max_features_strategy=cfg.inference.max_features_strategy,
                device=device,
                random_state=fold,
            ).fit(Xtr.astype(np.float32), y[tr], groups=company[tr])
            if cfg.inference.finetune_steps:
                fl = clf.finetune_losses_
                k = max(1, len(fl) // 10)
                print(f"  fine-tuned {len(fl)} steps: loss first-10% {np.mean(fl[:k]):.4f}, "
                      f"last-10% {np.mean(fl[-k:]):.4f}")
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
                best_pred, best_score, best_params, best_val = None, -np.inf, None, (0.0, 0.0)
                for params in configs:
                    p_both = fit_predict_boosting(name, Xtr, y[tr], both, params=params)
                    if p_both is None:
                        continue
                    score, ap, auc = selection_scores(y[va], p_both[: len(va)], select_metric)
                    if score > best_score:
                        best_pred, best_score, best_params, best_val = p_both, score, params, (ap, auc)
                if best_pred is None:
                    print(f"  {name} failed on fold {fold}; excluded from this fold")
                    continue
                arms[name] = (best_pred[: len(va)], best_pred[len(va) :])
                if tune:
                    print(f"    {name} best params: {best_params} (selected on val "
                          f"{select_metric}; val AP {best_val[0]:.4f}, val AUC {best_val[1]:.4f})")

        # fixed-parameter booster arms (audit AU4.1): one fit each, no search
        if fixed_arms:
            both = np.vstack([Xva, Xte])
            for arm, name, params in fixed_arms:
                p_both = fit_predict_boosting(name, Xtr, y[tr], both, params=params or None)
                if p_both is None:
                    print(f"  {arm} failed on fold {fold}; excluded from this fold")
                    continue
                arms[arm] = (p_both[: len(va)], p_both[len(va) :])

        # Per-row test predictions, kept so arms can be compared with a **paired** bootstrap.
        # Two models scored on the same rows have correlated errors, so an unpaired interval
        # overstates the uncertainty of their difference; without the raw vectors the only
        # available comparison is point estimates, which is how a 79-positive fold turns into
        # a league table nobody can check (docs/results/FINDINGS.md §60).
        np.savez_compressed(
            out_dir / f"predictions_fold{fold}.npz",
            y_true=y[te],
            **{f"pred_{a}": v[1] for a, v in arms.items()},
            # validation predictions too, so a setting can be chosen on validation (§163)
            # without anyone having to look at test to choose it
            y_val=y[va],
            **{f"val_{a}": v[0] for a, v in arms.items()},
            # company per row (codes from `company_codes`), so a comparison can resample whole
            # companies: a fold holds ~5 rows each, and they are not independent (audit AU2.1)
            company_test=codes[te],
            company_val=codes[va],
        )

        for arm, (p_val, p_test) in arms.items():
            thr, _ = best_f1_threshold(y[va], p_val)
            both = len(np.unique(y[te])) > 1
            auc = roc_auc_score(y[te], p_test) if both else float("nan")
            f1 = f1_score(y[te], (p_test >= thr).astype(int), zero_division=0)
            ap = average_precision_score(y[te], p_test) if both else float("nan")
            ap_val = (
                average_precision_score(y[va], p_val) if len(np.unique(y[va])) > 1 else float("nan")
            )
            # the same threshold search, but fitted on test: an upper bound no honest
            # procedure reaches, and useful only as the gap against `f1`
            _, f1_orc = best_f1_threshold(y[te], p_test) if both else (0.0, float("nan"))
            results.append(
                FoldResult(arm, fold, len(tr), len(te), int(y[te].sum()), float(auc),
                           float(f1), thr, float(ap), float(f1_orc), float(ap_val))
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
            "select_metric": select_metric if tune else None,
            "drop_columns": list(drop_columns),
            "booster_params": [
                {"arm": a, "booster": n, "params": p} for a, n, p in fixed_arms
            ],
            "context_strategy": cfg.inference.context_strategy,
            "feature_transform": cfg.inference.feature_transform,
            "max_context": cfg.inference.max_context,
            "n_ensemble": cfg.inference.n_ensemble,
            "finetune_steps": cfg.inference.finetune_steps,
            "finetune_lr": cfg.inference.finetune_lr,
            "finetune_batch": cfg.inference.finetune_batch,
            "finetune_query": cfg.inference.finetune_query,
            "finetune_context": cfg.inference.finetune_context,
            "finetune_query_group_frac": cfg.inference.finetune_query_group_frac,
            "finetune_query_pos_frac": cfg.inference.finetune_query_pos_frac,
            "finetune_seed": cfg.inference.finetune_seed,
            "ensemble_transforms": cfg.inference.ensemble_transforms,
            "ensemble_svd_components": cfg.inference.ensemble_svd_components,
            "ensemble_class_shift": cfg.inference.ensemble_class_shift,
            "max_features_strategy": cfg.inference.max_features_strategy,
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
            else "*** BASELINES ARE UNTUNED (library defaults: CatBoost 1,000 iterations, "
                 "LightGBM and XGBoost 100 trees). Their protocol grid-searches every baseline "
                 "on the validation fold, so these are NOT their baselines. Untuned is not "
                 "necessarily weaker: the --tune grid caps CatBoost at 200 iterations. Quote "
                 "a booster with its iteration count. ***"
        ),
        *(
            ["", "Fixed-parameter booster arms (one fit per fold, never tuned): "
             + ", ".join(b["arm"] for b in c["booster_params"]) + "."]
            if c.get("booster_params") else []
        ),
        *(
            ["", f"*** ABLATION: dropped beyond their drop list: {c['drop_columns']}. ***"]
            if c.get("drop_columns") else []
        ),
        "",
        "Their published reference at this horizon is in docs/results/FINDINGS.md §36.",
        (
            f"*** FINTFM WAS FINE-TUNED ON THIS FOLD'S TRAINING ROWS "
            f"({record['config']['finetune_steps']} steps, §163): it is not the zero-shot "
            f"model, and compares with their fine-tuned TabPFN as well as the baselines. ***"
        )
        if record["config"].get("finetune_steps")
        else (
            "Their TabPFN is fine-tuned on this data and ours is zero-shot here, so the "
            "like-for-like comparison is against their classical baselines, not their "
            "fine-tuned TabPFN."
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    """Run V4FinBench under its published five-fold protocol.

    Entry point for the ``v4-protocol`` console script; see
    ``--help`` for the flags. Writes its record as JSON under ``--out``.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None)
    p.add_argument(
        "--no-fintfm", action="store_true",
        help="score the baselines only; for folds whose fintfm scores already exist",
    )
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
    p.add_argument(
        "--select-metric", choices=("ap", "auc"), default="ap",
        help="what --tune selects each baseline on, on the validation fold (runs before "
             "2026-10-16 used auc)",
    )
    p.add_argument(
        "--drop-columns", type=str, default="",
        help="comma-separated feature columns to drop beyond their drop list, for an "
             "ablation (e.g. operational_status); recorded in the run config",
    )
    p.add_argument(
        "--booster-params", nargs="+", default=[], metavar="NAME:K=V,...",
        help="extra booster arms at fixed parameters, one fit per fold and never tuned, "
             "e.g. lightgbm:n_estimators=1000,learning_rate=0.05 catboost:iterations=100. "
             "'name:' alone is that booster at library defaults (pair with --no-boosting). "
             "Arms are named name[k=v,...]; recorded in the run config",
    )
    args = p.parse_args()
    try:
        booster_params = tuple(parse_booster_params(args.booster_params))
    except ValueError as e:
        p.error(str(e))
    if not args.no_fintfm and args.model is None:
        p.error("--model is required unless --no-fintfm")
    cfg = load_config(args.config)
    record = run(
        None if args.no_fintfm else args.model, Path(args.out), horizon=args.horizon,
        folds=tuple(int(v) for v in args.folds.split(",")),
        max_rows=args.max_rows, with_boosting=not args.no_boosting, tune=args.tune,
        select_metric=args.select_metric,
        drop_columns=tuple(v.strip() for v in args.drop_columns.split(",") if v.strip()),
        booster_params=booster_params,
        classical=tuple(v.strip() for v in args.classical.split(",") if v.strip()),
        cfg=cfg, device=args.device,
    )
    print("\n" + summarise(record))
    print(f"\nconfig: {cfg.provenance()}")
    print(f"\n{record['attribution']}")


if __name__ == "__main__":
    main()
