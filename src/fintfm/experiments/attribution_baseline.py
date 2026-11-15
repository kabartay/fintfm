"""SHAP-over-gradient-boosting attribution baseline (openspec task 12.2).

Why this exists
---------------
The incumbent in credit risk is gradient boosting paired with SHAP explainers
(``openspec/changes/zero-shot-attribution/proposal.md``). Before any in-context attribution is
compared with it (12.4), the baseline needs its own numbers: what GBM+SHAP ranks as important on
the UCI panels, how stable that ranking is, and what it costs.

Design
------
- LightGBM is fitted per stratified fold. Exact TreeSHAP comes from
  ``Booster.predict(..., pred_contrib=True)`` (LightGBM, MIT licence, already a dependency); the
  ``shap`` package is deliberately not added.
- Attributions are computed on the **held-out** fold, in the raw-margin (log-odds) space the
  TreeSHAP decomposition is exact in. The global importance of a feature is its mean absolute
  SHAP value over those rows, normalised to sum to one so panels and seeds are comparable.
- Stability is reported three ways, each on the normalised importance vectors: Spearman rank
  correlation between folds of one seed, between seeds (fold-averaged importance), and the overlap
  of the top-``k`` feature sets (Jaccard).
- Runtime is split into fit time and TreeSHAP time per fold.

**Run it as a script, not with ``-m``.** PyTorch and LightGBM each bring an OpenMP runtime and
the two segfault in one process on macOS (``fintfm/evaluation/boosting.py``). ``python -m
fintfm.experiments.attribution_baseline`` would import the ``fintfm`` package, and with it torch,
so this module imports neither: run ``python src/fintfm/experiments/attribution_baseline.py`` and
load the dataset module by file path (it needs only numpy).

Nothing here trains a FinTFM model. These are verification numbers about GBM+SHAP only; the
unit tests are SMOKE-TEST, and run numbers are MEASURED only when produced by the command
recorded beside them.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

#: LightGBM settings, fixed so the baseline is reproducible. Moderate size: this is a baseline
#: for explanations, not a tuned benchmark arm.
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

#: Size of the top-k feature set used for overlap.
TOP_K = 10


@dataclass
class FoldResult:
    """One fitted fold: importance vector, discrimination and cost."""

    seed: int
    fold: int
    importance: list[float]
    auc: float
    fit_seconds: float
    shap_seconds: float
    n_explained: int


@dataclass
class BaselineResult:
    """Summary of one dataset across seeds and folds."""

    dataset: str
    n_rows: int
    n_features: int
    default_rate: float
    n_seeds: int
    n_folds: int
    top_k: int
    mean_importance: list[float]
    top_features: list[dict]
    mean_auc: float
    fold_spearman_mean: float
    seed_spearman_mean: float
    fold_topk_jaccard_mean: float
    seed_topk_jaccard_mean: float
    fit_seconds_per_fold: float
    shap_seconds_per_fold: float
    shap_microseconds_per_row: float
    folds: list[FoldResult] = field(default_factory=list)


def mean_abs_shap(contrib: np.ndarray) -> np.ndarray:
    """Per-feature mean absolute SHAP value.

    Args:
        contrib: LightGBM ``pred_contrib`` output ``(n, d + 1)``; the last column is the
            expected value and carries no feature attribution, so it is dropped.

    Returns:
        ``(d,)`` array of mean ``|SHAP|``.
    """
    return np.abs(contrib[:, :-1]).mean(axis=0)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of two importance vectors; ``nan`` if either is constant."""
    from scipy.stats import rankdata

    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def topk_jaccard(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Jaccard overlap of the top-``k`` feature index sets of two importance vectors."""
    ta = set(np.argsort(-a, kind="stable")[:k].tolist())
    tb = set(np.argsort(-b, kind="stable")[:k].tolist())
    return len(ta & tb) / len(ta | tb)


def fit_and_explain(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_explain: np.ndarray,
    seed: int,
    params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Fit LightGBM and compute exact TreeSHAP on ``X_explain``.

    Returns:
        ``(contrib, margin, proba, fit_seconds, shap_seconds)``; ``contrib`` is
        ``(n, d + 1)`` with the expected value last, ``margin`` the raw log-odds.
    """
    import lightgbm as lgb

    model = lgb.LGBMClassifier(random_state=seed, **{**LGBM_PARAMS, **(params or {})})
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    booster = model.booster_
    t1 = time.perf_counter()
    contrib = booster.predict(X_explain, pred_contrib=True)
    shap_s = time.perf_counter() - t1
    margin = booster.predict(X_explain, raw_score=True)
    proba = model.predict_proba(X_explain)[:, 1]
    return np.asarray(contrib), np.asarray(margin), np.asarray(proba), fit_s, shap_s


def run_dataset(
    X: np.ndarray,
    y: np.ndarray,
    name: str,
    seeds: tuple[int, ...] = (0, 1, 2),
    n_folds: int = 5,
    top_k: int = TOP_K,
    feature_names: list[str] | None = None,
) -> BaselineResult:
    """Run the baseline on one panel.

    Each seed draws its own stratified fold split and its own LightGBM seed.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    d = X.shape[1]
    folds: list[FoldResult] = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for k, (tr, te) in enumerate(skf.split(X, y)):
            contrib, _margin, proba, fit_s, shap_s = fit_and_explain(X[tr], y[tr], X[te], seed)
            imp = mean_abs_shap(contrib)
            imp = imp / imp.sum() if imp.sum() > 0 else imp
            auc = float(roc_auc_score(y[te], proba)) if len(set(y[te])) > 1 else float("nan")
            folds.append(FoldResult(seed, k, imp.tolist(), auc, fit_s, shap_s, len(te)))

    imps = {(f.seed, f.fold): np.asarray(f.importance) for f in folds}
    fold_sp, fold_j = [], []
    for s in seeds:
        for i, j in itertools.combinations(range(n_folds), 2):
            fold_sp.append(spearman(imps[(s, i)], imps[(s, j)]))
            fold_j.append(topk_jaccard(imps[(s, i)], imps[(s, j)], top_k))
    seed_imp = {s: np.mean([imps[(s, k)] for k in range(n_folds)], axis=0) for s in seeds}
    seed_sp, seed_j = [], []
    for a, b in itertools.combinations(seeds, 2):
        seed_sp.append(spearman(seed_imp[a], seed_imp[b]))
        seed_j.append(topk_jaccard(seed_imp[a], seed_imp[b], top_k))

    mean_imp = np.mean(list(seed_imp.values()), axis=0)
    names = feature_names or [f"x{i}" for i in range(d)]
    order = np.argsort(-mean_imp, kind="stable")[:top_k]
    top = [{"index": int(i), "name": names[i], "importance": float(mean_imp[i])} for i in order]
    n_expl = sum(f.n_explained for f in folds)
    return BaselineResult(
        dataset=name,
        n_rows=len(y),
        n_features=d,
        default_rate=float(np.mean(y)),
        n_seeds=len(seeds),
        n_folds=n_folds,
        top_k=top_k,
        mean_importance=mean_imp.tolist(),
        top_features=top,
        mean_auc=float(np.nanmean([f.auc for f in folds])),
        fold_spearman_mean=float(np.nanmean(fold_sp)),
        seed_spearman_mean=float(np.nanmean(seed_sp)) if seed_sp else float("nan"),
        fold_topk_jaccard_mean=float(np.mean(fold_j)),
        seed_topk_jaccard_mean=float(np.mean(seed_j)) if seed_j else float("nan"),
        fit_seconds_per_fold=float(np.mean([f.fit_seconds for f in folds])),
        shap_seconds_per_fold=float(np.mean([f.shap_seconds for f in folds])),
        shap_microseconds_per_row=1e6 * sum(f.shap_seconds for f in folds) / n_expl,
        folds=folds,
    )


def main() -> None:
    """CLI: ``python src/fintfm/experiments/attribution_baseline.py`` (torch-free, see above)."""
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "evaluation" / "datasets.py"
    spec = importlib.util.spec_from_file_location("_fintfm_datasets", path)
    datasets = importlib.util.module_from_spec(spec)
    sys.modules["_fintfm_datasets"] = datasets  # dataclasses resolve their module by name
    spec.loader.exec_module(datasets)
    load_polish_bankruptcy = datasets.load_polish_bankruptcy
    load_taiwan_bankruptcy = datasets.load_taiwan_bankruptcy

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--polish-horizon", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("runs/attribution_baseline"))
    args = p.parse_args()

    panels = [load_polish_bankruptcy(args.polish_horizon), load_taiwan_bankruptcy()]
    args.out.mkdir(parents=True, exist_ok=True)
    for ds in panels:
        res = run_dataset(ds.X, ds.y, ds.name, tuple(args.seeds), args.folds)
        path = args.out / f"{ds.name}.json"
        path.write_text(json.dumps(asdict(res), indent=2))
        print(
            f"{ds.name}: n={res.n_rows} d={res.n_features} rate={res.default_rate:.4f} "
            f"AUC={res.mean_auc:.4f} fold-rho={res.fold_spearman_mean:.3f} "
            f"seed-rho={res.seed_spearman_mean:.3f} top{res.top_k}-J(fold)="
            f"{res.fold_topk_jaccard_mean:.3f} J(seed)={res.seed_topk_jaccard_mean:.3f} "
            f"fit={res.fit_seconds_per_fold:.2f}s shap={res.shap_seconds_per_fold:.2f}s "
            f"({res.shap_microseconds_per_row:.0f} us/row) -> {path}"
        )


if __name__ == "__main__":
    main()
