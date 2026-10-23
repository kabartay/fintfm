"""Gradient boosting baselines, fitted in a torch-free subprocess.

**Why a subprocess.** PyTorch bundles its own OpenMP runtime and LightGBM loads the system
one. Two OpenMP runtimes in a single process segfault on macOS — confirmed by bisection, and
``KMP_DUPLICATE_LIB_OK=TRUE`` does not help (``docs/results/FINDINGS.md`` §25). Since this project
imports torch everywhere and the boosting libraries are the baselines that matter, the only
reliable separation is a process boundary.

This is heavy-handed for a benchmark, and it is the cost of comparing against the family that
actually competes rather than the weakest member of it. Every "gradient boosting beats us"
statement made before this existed was measured against sklearn's ``GradientBoostingClassifier``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

#: Baselines fitted out-of-process. sklearn's own GBM is deliberately absent: it runs in-process
#: fine and is kept only as a weak reference row.
BOOSTING_MODELS = ("lightgbm", "catboost", "xgboost")

_WORKER = '''
import json, sys
import numpy as np
d = np.load(sys.argv[1])
Xtr, ytr, Xte = d["Xtr"], d["ytr"], d["Xte"]
name = sys.argv[2]
params = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
imp = SimpleImputer(strategy="median")
if name == "lightgbm":
    from lightgbm import LGBMClassifier
    est = LGBMClassifier(verbosity=-1, **params)
elif name == "catboost":
    from catboost import CatBoostClassifier
    est = CatBoostClassifier(verbose=0, allow_writing_files=False, **params)
elif name == "xgboost":
    from xgboost import XGBClassifier
    est = XGBClassifier(verbosity=0, tree_method="hist", **params)
else:
    raise SystemExit(f"unknown model {name}")
if "btr" in d.files:
    # a per-row starting score (LightGBM only, checked by the caller): the trees fit what
    # the starting score leaves over, and the raw score never includes it, so it is added
    est.fit(imp.fit_transform(Xtr), ytr, init_score=d["btr"])
    raw = est.predict(imp.transform(Xte), raw_score=True) + d["bte"]
    np.save(sys.argv[3], 1.0 / (1.0 + np.exp(-raw)))
else:
    m = make_pipeline(imp, est).fit(Xtr, ytr)
    np.save(sys.argv[3], m.predict_proba(Xte)[:, 1])
print(json.dumps({"ok": True}))
'''

#: Baselines whose worker accepts a per-row starting score (``init_score``).
INIT_SCORE_MODELS = ("lightgbm",)


def fit_predict_boosting(
    name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    timeout: int = 1800, params: dict | None = None,
    init_score_train: np.ndarray | None = None, init_score_test: np.ndarray | None = None,
) -> np.ndarray | None:
    """Fit one boosting baseline out-of-process and return positive-class probabilities.

    Args:
        name: One of :data:`BOOSTING_MODELS`.
        X_train: Training features.
        y_train: Training labels.
        X_test: Rows to score.
        timeout: Seconds before the subprocess is abandoned.
        params: Hyperparameters for the estimator. **Defaults are not a neutral choice** —
            on V4FinBench's protocol, default LightGBM and XGBoost score *below* logistic
            regression on ROC-AUC while the published benchmark grid-searches every
            baseline, so reporting untuned boosters understates the field exactly as
            ``docs/results/FINDINGS.md`` §25 did in the other direction.
        init_score_train: Optional ``(n_train,)`` starting log-odds per training row, passed
            to LightGBM's ``fit(init_score=...)`` so the trees fit what it leaves over. Only
            for names in :data:`INIT_SCORE_MODELS`.
        init_score_test: Starting log-odds per test row, required with
            ``init_score_train``. The returned probability is the sigmoid of LightGBM's
            ``raw_score`` plus this, because the raw score never includes a starting score.

    Returns:
        ``(n_test,)`` probabilities, or ``None`` if the baseline could not be fitted — which
        is **printed, never silent**, because a quietly absent baseline flatters us.

    Raises:
        ValueError: For an unknown name, a starting score for a model that takes none, only
            one of the two starting scores, or a starting score whose length does not match
            its rows.
    """
    if name not in BOOSTING_MODELS:
        raise ValueError(f"unknown baseline {name!r}; expected one of {BOOSTING_MODELS}")
    if (init_score_train is None) != (init_score_test is None):
        raise ValueError("init_score_train and init_score_test go together")
    arrays: dict[str, np.ndarray] = {"Xtr": X_train, "ytr": y_train, "Xte": X_test}
    if init_score_train is not None and init_score_test is not None:
        if name not in INIT_SCORE_MODELS:
            raise ValueError(f"{name} takes no starting score here; only {INIT_SCORE_MODELS}")
        btr = np.asarray(init_score_train, dtype=np.float64)
        bte = np.asarray(init_score_test, dtype=np.float64)
        if btr.shape != (len(X_train),) or bte.shape != (len(X_test),):
            raise ValueError(f"starting scores of shape {btr.shape} and {bte.shape} do not "
                             f"match {len(X_train)} training and {len(X_test)} test rows")
        arrays |= {"btr": btr, "bte": bte}
    with tempfile.TemporaryDirectory() as tmp:
        data, out = Path(tmp) / "d.npz", Path(tmp) / "p.npy"
        np.savez(data, **arrays)
        argv = [sys.executable, "-c", _WORKER, str(data), name, str(out)]
        if params:
            argv.append(json.dumps(params))
        proc = subprocess.run(
            argv,
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        if proc.returncode != 0 or not out.exists():
            reason = (proc.stderr or "").strip().splitlines()
            detail = reason[-1] if reason else f"exit {proc.returncode}"
            print(f"  baseline skipped: {name} — {detail[:120]}")
            return None
        return np.load(out)


def available_boosting() -> list[str]:
    """Which boosting baselines can actually be imported, checked out-of-process."""
    ok = []
    for name in BOOSTING_MODELS:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {name}"],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode == 0:
            ok.append(name)
        else:
            print(f"  baseline unavailable: {name}")
    return ok
