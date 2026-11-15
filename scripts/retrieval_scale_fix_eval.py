"""Task 38.13: does fixing retrieval's zero-IQR distance scale make retrieval useful?

Scores arms on a company-grouped V4FinBench subsample, AP first: the default (uniform context),
retrieval as shipped, retrieval with the zero-IQR columns given their standard deviation as
scale, and plain ``n_ensemble`` + ``max_context`` without retrieval. The classifier is built
by ``vp.fintfm_classifier`` from a config, so every setting is the packaged one unless an arm
overrides it here.

Usage::

    python scripts/retrieval_scale_fix_eval.py --arms default,retrieval,retrieval_fixed \
        --max-rows 60000 --folds 0 --root <v4finbench dir> --checkpoint <v4-cellattn-labels.pt>
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import fintfm.inference.classifier as clfmod
from fintfm.config import load_config
from fintfm.evaluation.datasets import CACHE_DIR
from fintfm.experiments import v4_protocol as vp
from fintfm.inference import retrieval as rt
from fintfm.modeling.model import FinancialTFM

_ORIGINAL = rt.distance_stats


def std_fallback_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``distance_stats`` with the standard deviation replacing a zero interquartile range.

    Args:
        X: ``(n, F)`` training features, NaN for missing.

    Returns:
        ``(centre, scale)``; a column whose IQR is at most 1e-6 uses its standard deviation,
        or 1.0 when that is also at most 1e-6.
    """
    centre, scale = _ORIGINAL(X)
    with np.errstate(all="ignore"):
        sd = np.nan_to_num(np.nanstd(X, axis=0), nan=1.0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    return centre, np.where(scale > 1e-6, scale, sd).astype(np.float32)


#: arm -> (inference overrides, whether the scale fallback is on)
ARMS: dict[str, tuple[dict, bool]] = {
    "default": ({"context_strategy": "uniform"}, False),
    "retrieval": ({"context_strategy": "retrieval"}, False),
    "retrieval_fixed": ({"context_strategy": "retrieval"}, True),
    "retrieval_pc8": ({"context_strategy": "retrieval", "retrieval_per_class": 8}, False),
    "ens8_ctx4000": ({"context_strategy": "uniform", "n_ensemble": 8, "max_context": 4000}, False),
    "retrieval_fixed_ens8_ctx4000": (
        {"context_strategy": "retrieval", "n_ensemble": 8, "max_context": 4000},
        True,
    ),
}


def main() -> None:
    """Score the requested arms on the requested folds and print one line per arm and fold."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="default,retrieval,retrieval_fixed")
    ap.add_argument("--max-rows", type=int, default=60000)
    ap.add_argument("--folds", default="0")
    ap.add_argument("--root", type=Path, default=CACHE_DIR / "v4finbench")
    ap.add_argument("--checkpoint", default="runs/v4-cellattn-labels.pt")
    a = ap.parse_args()
    X, y, country, company, _ = vp._load_horizon(a.root, 0, a.max_rows)
    assign = vp.build_fold_assignments(country, company)
    model = FinancialTFM.load(a.checkpoint)
    base = load_config()
    print(f"rows {len(y):,}, positives {int(y.sum())} ({y.mean():.4%})")
    for fold in [int(f) for f in a.folds.split(",")]:
        tr, _va, te = vp.split_indices_for_fold(assign, fold)
        imp = SimpleImputer(strategy="median").fit(X[tr])
        sc = StandardScaler().fit(imp.transform(X[tr]))
        Xtr, Xte = (sc.transform(imp.transform(X[i])).astype(np.float32) for i in (tr, te))
        preds: dict[str, np.ndarray] = {}
        for name in a.arms.split(","):
            over, fixed = ARMS[name]
            cfg = dataclasses.replace(base, inference=dataclasses.replace(base.inference, **over))
            clfmod.distance_stats = std_fallback_stats if fixed else _ORIGINAL
            t0 = time.time()
            clf = vp.fintfm_classifier(model, cfg, device="cpu", random_state=fold)
            clf.fit(Xtr, y[tr], groups=company[tr])
            p = clf.predict_proba(Xte)[:, 1]
            yt = y[te]
            preds[name] = p
            ctx = getattr(clf, "pooled_context_rate_", None)
            print(
                f"fold {fold} {name}: AP {average_precision_score(yt, p):.4f} "
                f"AUC {roc_auc_score(yt, p):.4f} mean_pred {p.mean():.4%} "
                f"observed {yt.mean():.4%} test_n {yt.size} pos {int(yt.sum())} "
                f"ctx_rate {'' if ctx is None else f'{ctx:.3%}'} {time.time() - t0:.0f}s",
                flush=True,
            )
        if "default" in preds and "retrieval" in preds:
            for w in (0.5,):
                b = w * preds["default"] + (1 - w) * preds["retrieval"]
                print(
                    f"fold {fold} blend default/retrieval {w}: AP "
                    f"{average_precision_score(y[te], b):.4f} AUC {roc_auc_score(y[te], b):.4f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
