"""Gradient boosting baselines, fitted in a torch-free subprocess.

**Why a subprocess.** PyTorch bundles its own OpenMP runtime and LightGBM loads the system
one. Two OpenMP runtimes in a single process segfault on macOS — confirmed by bisection, and
``KMP_DUPLICATE_LIB_OK=TRUE`` does not help (``docs/FINDINGS.md`` §25). Since this project
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
m = make_pipeline(imp, est).fit(Xtr, ytr)
np.save(sys.argv[3], m.predict_proba(Xte)[:, 1])
print(json.dumps({"ok": True}))
'''


def fit_predict_boosting(
    name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    timeout: int = 1800, params: dict | None = None,
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
            ``docs/FINDINGS.md`` §25 did in the other direction.

    Returns:
        ``(n_test,)`` probabilities, or ``None`` if the baseline could not be fitted — which
        is **printed, never silent**, because a quietly absent baseline flatters us.
    """
    if name not in BOOSTING_MODELS:
        raise ValueError(f"unknown baseline {name!r}; expected one of {BOOSTING_MODELS}")
    with tempfile.TemporaryDirectory() as tmp:
        data, out = Path(tmp) / "d.npz", Path(tmp) / "p.npy"
        np.savez(data, Xtr=X_train, ytr=y_train, Xte=X_test)
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
