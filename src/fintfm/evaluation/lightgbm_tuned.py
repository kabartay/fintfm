"""A tuned LightGBM control for the credit suite (DR8.4, FINDINGS §189's open control).

**This file must not import the ``fintfm`` package.** ``fintfm.evaluation`` imports torch, and
LightGBM and torch in one process segfault on macOS. The tuning therefore runs as a script
(``python src/fintfm/evaluation/lightgbm_tuned.py data.npz out.json``), whose ``sys.path[0]`` is
this directory, and tests load it by file path in a subprocess. Only numpy, scikit-learn and
lightgbm are imported.

Hyperparameters are chosen on an inner validation split carved from the **training** rows
only; the test rows are scored once, after the refit, and never read during the search.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np

#: Cap on trees per config; early stopping usually stops well before it.
MAX_TREES = 1500
#: Early-stopping patience, in trees, on the inner validation average precision.
PATIENCE = 50


def inner_split(
    y: np.ndarray, time_: np.ndarray | None = None, groups: np.ndarray | None = None,
    embargo: float = 0.0, frac: float = 0.25, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Carve an inner (fit, validation) split from training rows.

    Dated rows: validation is the latest ``frac`` of rows by date and the fit side stops
    ``embargo`` before it (as the outer out-of-time split does). Undated rows: a stratified
    random ``frac``; with ``groups`` and no dates, a random ``frac`` of whole groups (a row
    split would put nearly every multi-year company on both sides). With dates and ``groups``,
    fit rows sharing a group with a validation row are dropped.

    Args:
        y: ``(n,)`` binary training labels.
        time_: ``(n,)`` dates, or ``None``.
        groups: ``(n,)`` group ids, or ``None``.
        embargo: Gap in ``time_`` units between the fit side and the validation start.
        frac: Validation share of the rows.
        seed: Seed of the random split.

    Returns:
        ``(fit_idx, val_idx)``.

    Raises:
        ValueError: If either side lacks a class, or ``frac`` is outside (0, 1).
    """
    if not 0 < frac < 1:
        raise ValueError(f"frac must be in (0, 1), got {frac}")
    n = len(y)
    if time_ is not None:
        cut = np.quantile(time_, 1 - frac)
        val = np.flatnonzero(time_ >= cut)
        fit_mask = time_ < cut - embargo
    elif groups is not None:
        rng = np.random.RandomState(seed)
        names = rng.permutation(np.unique(groups))
        size = {g: c for g, c in zip(*np.unique(groups, return_counts=True), strict=True)}
        cum = np.cumsum([size[g] for g in names])
        val_mask = np.isin(groups, names[: int(np.searchsorted(cum, frac * n)) + 1])
        val, fit_mask = np.flatnonzero(val_mask), ~val_mask
    else:
        rng = np.random.RandomState(seed)
        val_mask = np.zeros(n, dtype=bool)
        for c in np.unique(y):
            idx = np.flatnonzero(y == c)
            take = max(1, round(frac * len(idx)))
            val_mask[rng.choice(idx, size=take, replace=False)] = True
        val, fit_mask = np.flatnonzero(val_mask), ~val_mask
    if groups is not None:
        fit_mask &= ~np.isin(groups, groups[val])
    fit = np.flatnonzero(fit_mask)
    for side, idx in (("fit", fit), ("validation", val)):
        if len(np.unique(y[idx])) < 2:
            raise ValueError(f"inner {side} side has one class ({len(idx)} rows)")
    return fit, val


def sample_config(rng: np.random.RandomState) -> dict[str, Any]:
    """One random LightGBM config from the search space.

    Args:
        rng: Seeded generator.

    Returns:
        Estimator keyword arguments (without the tree count, set by early stopping).
    """
    return {
        "learning_rate": float(np.exp(rng.uniform(np.log(0.02), np.log(0.2)))),
        "num_leaves": int(np.exp(rng.uniform(np.log(4), np.log(256)))),
        "min_child_samples": int(np.exp(rng.uniform(np.log(5), np.log(200)))),
        "subsample": float(rng.uniform(0.5, 1.0)),
        "subsample_freq": 1,
        "colsample_bytree": float(rng.uniform(0.4, 1.0)),
        "reg_lambda": float(np.exp(rng.uniform(np.log(1e-3), np.log(30)))),
        "reg_alpha": float(np.exp(rng.uniform(np.log(1e-3), np.log(10)))),
        "scale_pos_weight": float(rng.choice([1.0, 1.0, 3.0, 10.0])),
    }


def tune_lightgbm(
    X: np.ndarray, y: np.ndarray, *, n_configs: int = 30, seed: int = 0,
    time_: np.ndarray | None = None, groups: np.ndarray | None = None, embargo: float = 0.0,
    n_jobs: int = 2, time_budget_s: float | None = None, max_trees: int = MAX_TREES,
    init_score: np.ndarray | None = None,
) -> dict[str, Any]:
    """Random-search LightGBM on an inner split of the training rows.

    Each config trains with early stopping on the inner validation average precision; the best
    config is the highest validation AP. The first config is the untuned control's (0.05
    learning rate, 31 leaves, up to 1,000 trees), so the search can only match or beat it on
    the validation side.

    Args:
        X: Training features (NaN allowed).
        y: Training labels.
        n_configs: Configs to try (fewer if ``time_budget_s`` runs out).
        seed: Seeds the search, the inner split and every fit.
        time_: Optional training dates (see :func:`inner_split`).
        groups: Optional training group ids.
        embargo: Embargo in ``time_`` units.
        n_jobs: LightGBM threads.
        time_budget_s: Stop starting new configs after this many seconds (the first always runs).
        max_trees: Tree cap per config after the first.
        init_score: Optional ``(n,)`` starting log-odds per training row (FinTFM's
            out-of-fold logit, DR8.4). The trees then fit what it leaves over, and the
            validation score is the starting score plus the trees. ``None`` leaves the
            search exactly as before.

    Returns:
        ``{"best_params", "best_iteration", "val_ap", "trials"}``; each trial has its params,
        validation AP, best iteration and seconds.
    """
    from lightgbm import LGBMClassifier, early_stopping
    from sklearn.metrics import average_precision_score

    fit, val = inner_split(y, time_, groups, embargo, seed=seed)
    rng = np.random.RandomState(seed)
    t_start = time.perf_counter()
    trials: list[dict[str, Any]] = []
    for i in range(n_configs):
        if time_budget_s is not None and i > 0 and time.perf_counter() - t_start > time_budget_s:
            break
        params = {"learning_rate": 0.05, "num_leaves": 31} if i == 0 else sample_config(rng)
        t0 = time.perf_counter()
        est = LGBMClassifier(
            n_estimators=min(1000, max_trees) if i == 0 else max_trees,
            metric="average_precision", n_jobs=n_jobs, random_state=seed, verbosity=-1, **params,
        )
        if init_score is None:
            est.fit(X[fit], y[fit], eval_set=[(X[val], y[val])],
                    callbacks=[early_stopping(PATIENCE, verbose=False)])
            p = est.predict_proba(X[val])[:, 1]
        else:
            est.fit(X[fit], y[fit], init_score=init_score[fit], eval_set=[(X[val], y[val])],
                    eval_init_score=[init_score[val]],
                    callbacks=[early_stopping(PATIENCE, verbose=False)])
            p = est.predict(X[val], raw_score=True) + init_score[val]
        trials.append({
            "params": params, "val_ap": float(average_precision_score(y[val], p)),
            "best_iteration": int(est.best_iteration_ or est.n_estimators),
            "seconds": round(time.perf_counter() - t0, 1),
        })
    best = max(trials, key=lambda t: t["val_ap"])
    return {"best_params": best["params"], "best_iteration": best["best_iteration"],
            "val_ap": best["val_ap"], "trials": trials}


def fit_predict_tuned(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, tuned: dict[str, Any], seed: int = 0,
    n_jobs: int = 2, init_train: np.ndarray | None = None, init_test: np.ndarray | None = None,
) -> np.ndarray:
    """Refit the chosen config on all training rows and score the test rows.

    The tree count is the early-stopped iteration times 1.1 (more rows than the inner fit).

    Args:
        Xtr: Training features.
        ytr: Training labels.
        Xte: Test features.
        tuned: Output of :func:`tune_lightgbm`.
        seed: Fit seed.
        n_jobs: LightGBM threads.
        init_train: Optional ``(n_train,)`` starting log-odds, as in :func:`tune_lightgbm`.
        init_test: Starting log-odds of the test rows; required with ``init_train``. The
            result is the sigmoid of the trees' raw score plus this.

    Returns:
        ``(n_test,)`` positive-class probabilities.

    Raises:
        ValueError: If only one of the two starting scores is given.
    """
    from lightgbm import LGBMClassifier

    if (init_train is None) != (init_test is None):
        raise ValueError("init_train and init_test go together")
    est = LGBMClassifier(
        n_estimators=max(10, round(tuned["best_iteration"] * 1.1)), n_jobs=n_jobs,
        random_state=seed, verbosity=-1, **tuned["best_params"],
    )
    if init_train is None:
        return est.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    raw = est.fit(Xtr, ytr, init_score=init_train).predict(Xte, raw_score=True) + init_test
    return 1.0 / (1.0 + np.exp(-raw))


def main(argv: list[str]) -> int:
    """Tune and score one split: ``data.npz out.json [n_configs] [time_budget_s] [seed] [preds.npy]``.

    With ``preds.npy`` the tuned model's probabilities for every ``Xte`` row are saved there.

    The npz holds ``Xtr, ytr, Xte, yte`` and optionally ``ttr, gtr, embargo``, and ``btr, bte``
    (starting log-odds of the training and test rows: the hybrid as init score; the feature
    hybrid is the same call with the logit as an extra column of ``Xtr`` and ``Xte``). The
    untuned control then starts from the same scores. The output JSON carries the tuned and the untuned (1,000 trees, 0.05) test average precision and ROC-AUC.

    Args:
        argv: Command-line arguments after the script name.

    Returns:
        Exit code.
    """
    from lightgbm import LGBMClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score

    d = np.load(argv[0])
    n_configs = int(argv[2]) if len(argv) > 2 else 30
    budget = float(argv[3]) if len(argv) > 3 else None
    seed = int(argv[4]) if len(argv) > 4 else 0
    btr = d["btr"] if "btr" in d.files else None
    bte = d["bte"] if btr is not None else None
    t0 = time.perf_counter()
    tuned = tune_lightgbm(
        d["Xtr"], d["ytr"], n_configs=n_configs, seed=seed, time_budget_s=budget,
        time_=d["ttr"] if "ttr" in d.files else None, init_score=btr,
        groups=d["gtr"] if "gtr" in d.files else None,
        embargo=float(d["embargo"]) if "embargo" in d.files else 0.0,
    )
    tune_s = time.perf_counter() - t0
    yte = d["yte"]
    p = fit_predict_tuned(d["Xtr"], d["ytr"], d["Xte"], tuned, seed=seed,
                          init_train=btr, init_test=bte)
    control = LGBMClassifier(n_estimators=1000, learning_rate=0.05, n_jobs=2, verbosity=-1,
                             random_state=seed)
    if btr is None:
        p0 = control.fit(d["Xtr"], d["ytr"]).predict_proba(d["Xte"])[:, 1]
    else:
        raw0 = control.fit(d["Xtr"], d["ytr"], init_score=btr).predict(
            d["Xte"], raw_score=True) + bte
        p0 = 1.0 / (1.0 + np.exp(-raw0))
    out = {
        "tuned_ap": float(average_precision_score(yte, p)),
        "tuned_auc": float(roc_auc_score(yte, p)),
        "untuned_ap": float(average_precision_score(yte, p0)),
        "untuned_auc": float(roc_auc_score(yte, p0)),
        "n_trials": len(tuned["trials"]), "tune_seconds": round(tune_s, 1), **tuned,
    }
    if len(argv) > 5:
        np.save(argv[5], p)
    with open(argv[1], "w") as f:
        json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
