"""Cheap held-out judge for whether a sampled task carries learnable signal.

Nori rejects unlearnable synthetic datasets with an ExtraTrees signal-quality filter.
``docs/results/FINDINGS.md`` §125 measured what that would do here: about a third of the
production prior mixture's tasks score at or below chance under this judge, including a fifth
that are single-class or otherwise unscorable outright, and on the rejected subset the trained
model itself extracts real signal (0.520 AUC where the judge scores 0.392). So the filter is
far from inert and the naive version of it is not obviously safe -- this module exists to let
``prior.mixture.PriorConfig.p_learnability_filter`` measure the effect rather than assume it.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

from fintfm.prior.base import Task

#: §125's judge, unchanged: 50 trees, a fixed seed so the filter's own randomness does not add
#: a second source of noise to whatever experiment is sweeping ``p_learnability_filter``.
_N_ESTIMATORS = 50
_JUDGE_SEED = 0
_TRAIN_FRACTION = 0.7
_MIN_SPLIT_ROWS = 2


def is_learnable(task: Task, *, auc_floor: float = 0.51) -> bool:
    """Return whether a held-out ``ExtraTreesClassifier`` clears ``auc_floor`` on ``task``.

    Splits the task's own rows 70/30 (no separate holdout is available at sampling time,
    before the batch's context/query split is chosen), fits on the first part and scores
    held-out AUC on the second. A task that is single-class overall, or single-class on either
    side of the split, cannot be scored and is treated as unlearnable -- §125 counted these
    separately as "degenerate" and folded them into the rejection rate, and this function does
    the same rather than crashing or silently passing them.

    Binary and multiclass tasks are both supported: multiclass uses macro one-vs-rest AUC, so
    every task family is judged on one scale. A task with fewer than two rows on either side of
    the split is also rejected, since no learner can be fit or scored on it.
    """
    X = np.nan_to_num(np.asarray(task.X, dtype=float))
    y = np.asarray(task.y).ravel()
    if len(np.unique(y)) < 2:
        return False

    n = len(y)
    cut = max(int(n * _TRAIN_FRACTION), _MIN_SPLIT_ROWS)
    if cut >= n:
        return False
    X_train, y_train, X_test, y_test = X[:cut], y[:cut], X[cut:], y[cut:]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return False

    judge = ExtraTreesClassifier(
        n_estimators=_N_ESTIMATORS, random_state=_JUDGE_SEED, n_jobs=2
    ).fit(X_train, y_train)
    proba = judge.predict_proba(X_test)

    # The single-class cases above are already excluded, so a warning here would be sklearn
    # objecting to something this function has already checked for, on every one of ~48,000
    # calls in a full pretraining run. Silence it rather than let it drown the training log.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if proba.shape[1] == 2:
            auc = roc_auc_score(y_test, proba[:, 1])
        else:
            present = np.unique(y_train)
            if not set(np.unique(y_test)).issubset(set(present)):
                return False
            auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=present)
    return bool(auc > auc_floor)
