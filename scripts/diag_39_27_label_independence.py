"""39.27 diagnostic: is the unlabelled model's sub-chance AUC a fixed feature function?

Usage: ``uv run --frozen python scripts/diag_39_27_label_independence.py OUT.json``. CPU, about
40 s. Reads existing checkpoints under ``runs/``; trains nothing.
"""

import json
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from fintfm.experiments.capability import (
    _bayes_optimal_mu,
    _make_bayes_task,
    _probe_positive_proba,
)
from fintfm.modeling.model import FinancialTFM

R = "/Users/morganoko/proj/fintfm/runs"
arms = {
    "nolab_s0": f"{R}/dl/v4-cellattn-nolabels.pt",
    "nolab_s1": f"{R}/cellattn-s1-kaggle/v4-cellattn-nolabels-s1.pt",
    "labels_s0": f"{R}/dl/v4-cellattn-labels.pt",
}
out = {}
N, NE, SEEDS = 800, 4, (0, 1, 2, 3, 4)
for name, path in arms.items():
    m = FinancialTFM.load(path)
    rows = []
    for target in (0.9, 0.999):
        mu = _bayes_optimal_mu(target)
        for s in SEEDS:
            rng = np.random.default_rng(s)
            X, y = _make_bayes_task(rng, N, mu)
            nc = N // 2
            res = {"arm": name, "target": target, "seed": s}
            p_true = _probe_positive_proba(m, X, y, nc, NE)
            res["auc_true"] = roc_auc_score(y[nc:], p_true)
            res["rho_x0_true"] = spearmanr(p_true, X[nc:, 0])[0]
            yi = 1 - y
            p_inv = _probe_positive_proba(m, X, yi, nc, NE)
            res["rho_true_vs_inv_ctx"] = spearmanr(p_true, p_inv)[0]
            # mirrored task: class-1 shift reversed, same rows otherwise
            Xm = X.copy()
            Xm[:, 0] = X[:, 0] - 2 * mu * y
            p_m = _probe_positive_proba(m, Xm, y, nc, NE)
            res["auc_mirrored_task"] = roc_auc_score(y[nc:], p_m)
            res["rho_x0_mirrored"] = spearmanr(p_m, Xm[nc:, 0])[0]
            rows.append(res)
    out[name] = rows


def agg(rows: list[dict], key: str, target: float) -> tuple[float, float]:
    """Mean and standard deviation of ``key`` over the rows at one Bayes target."""
    v = [r[key] for r in rows if r["target"] == target]
    return float(np.mean(v)), float(np.std(v))


for name, rows in out.items():
    for t in (0.9, 0.999):
        print(
            name,
            t,
            {
                k: tuple(round(x, 3) for x in agg(rows, k, t))
                for k in (
                    "auc_true",
                    "rho_x0_true",
                    "rho_true_vs_inv_ctx",
                    "auc_mirrored_task",
                    "rho_x0_mirrored",
                )
            },
        )
with open(sys.argv[1], "w") as fh:
    json.dump(out, fh)
