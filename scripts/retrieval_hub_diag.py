"""Hub-effect diagnostic for task 38.13: why the same defaults recur across retrieval groups.

Builds the retrieval plan on a company-grouped V4FinBench subsample (no forward pass) and
measures, in the normalised distance space the plan uses: how far from the pool centre the
positives sit, how many groups each positive enters (its k-occurrence), what the rate is if
retrieval targets a real query instead of a group mean, and what it is for a target shrunk
toward the centre by averaging.

Usage::

    python scripts/retrieval_hub_diag.py --max-rows 60000 --fold 0 \
        --root /path/to/data/cache/v4finbench --checkpoint /path/to/v4-cellattn-labels.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import fintfm.inference.classifier as clfmod
from fintfm.config import load_config
from fintfm.evaluation.datasets import CACHE_DIR
from fintfm.experiments import v4_protocol as vp
from fintfm.inference.retrieval import normalise_for_distance, retrieve
from fintfm.modeling.model import FinancialTFM


def main() -> None:
    """Print the hub statistics for one fold of a company-grouped subsample."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=60000)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--root", type=Path, default=CACHE_DIR / "v4finbench")
    ap.add_argument("--checkpoint", default="runs/v4-cellattn-labels.pt")
    ap.add_argument("--n-real", type=int, default=200, help="targets for the exact-retrieval rate")
    ap.add_argument("--raw-queries", action="store_true", help="reproduce §194: unconditioned")
    ap.add_argument("--fix-scale", action="store_true", help="std replaces a zero IQR scale")
    a = ap.parse_args()
    if a.fix_scale:
        from retrieval_scale_fix_eval import std_fallback_stats

        clfmod.distance_stats = std_fallback_stats
    cfg = load_config("configs/arms/proxy/retrieval.yaml")
    X, y, country, company, _ = vp._load_horizon(a.root, 0, a.max_rows)
    assign = vp.build_fold_assignments(country, company)
    tr, _va, te = vp.split_indices_for_fold(assign, a.fold)
    imp = SimpleImputer(strategy="median").fit(X[tr])
    sc = StandardScaler().fit(imp.transform(X[tr]))
    Xtr, Xte = (sc.transform(imp.transform(X[i])).astype(np.float32) for i in (tr, te))
    clf = vp.fintfm_classifier(
        FinancialTFM.load(a.checkpoint), cfg, device="cpu", random_state=a.fold
    )
    clf.fit(Xtr, y[tr], groups=company[tr])
    # the pool is stored conditioned, so queries must be conditioned the way predict_proba does
    # before they are placed in the same distance space (scripts/retrieval_plan_diag.py did not)
    Xq = Xte if a.raw_queries else clf._condition(clf._select_query(Xte))
    plan = clf._retrieval_plan(Xq)
    Z, yp = clf._pool_Z, clf._pool_y
    pos = np.flatnonzero(yp == 1)
    neg = np.flatnonzero(yp == 0)
    k = clf.max_context
    print(
        f"rows {len(y):,}; fold {a.fold} pool {Z.shape[0]:,} rows, {pos.size} positives "
        f"({pos.size / Z.shape[0]:.4%}); test {Xte.shape[0]:,} rows, {int(y[te].sum())} "
        f"positives; groups {len(plan)}; max_context {k}; positives in a blind context "
        f"{k * pos.size / Z.shape[0]:.1f}"
    )
    print(f"pooled context rate {clf._pooled_context_rate(plan):.4%}")
    sq = Z.astype(np.float64) ** 2
    share = sq.sum(axis=0) / sq.sum()
    top = np.argsort(-share)[:5]
    print(
        f"features {clf._scale.size}; scale <= 1e-6: {int((clf._scale <= 1e-6).sum())}; "
        f"top-5 columns hold {share[top].sum():.1%} of the squared distance "
        f"(scales {[f'{clf._scale[j]:.3g}' for j in top]}); positives with |Z|>1 on the top "
        f"column {float((np.abs(Z[pos, top[0]]) > 1).mean()):.3f} vs negatives "
        f"{float((np.abs(Z[neg, top[0]]) > 1).mean()):.3f}"
    )

    norm = np.linalg.norm(Z, axis=1)
    print(
        f"median ||Z||: pool negatives {np.median(norm[neg]):.2f}, "
        f"pool positives {np.median(norm[pos]):.2f}"
    )
    occ = np.zeros(Z.shape[0], dtype=int)
    for _, c in plan:
        occ[c] += 1
    ever = occ[pos] > 0
    print(
        f"positives entering any context: {int(ever.sum())} of {pos.size}; median groups per "
        f"entering positive {np.median(occ[pos][ever]) if ever.any() else 0:.0f} of {len(plan)}; "
        f"negatives entering any context: {int((occ[neg] > 0).sum()):,} of {neg.size:,}"
    )
    print(
        f"Spearman(k-occurrence, ||Z||): positives {spearmanr(occ[pos], norm[pos])[0]:.3f}, "
        f"negatives {spearmanr(occ[neg], norm[neg])[0]:.3f}"
    )
    Zq = normalise_for_distance(Xq, clf._centre, clf._scale)
    tnorm = np.array([np.linalg.norm(Zq[q].mean(axis=0)) for q, _ in plan])
    print(
        f"||target||: group means median {np.median(tnorm):.2f}; "
        f"single test queries median {np.median(np.linalg.norm(Zq, axis=1)):.2f}"
    )
    rng = np.random.default_rng(0)
    pick = rng.choice(Xte.shape[0], size=min(a.n_real, Xte.shape[0]), replace=False)
    real = [float((yp[retrieve(Z, yp, Zq[i], k, 0)] == 1).mean()) for i in pick]
    print(
        f"positive rate in the {k} nearest rows, real-query targets, no floor ({pick.size}): "
        f"mean {np.mean(real):.4%}"
    )
    shr = [
        float(
            (
                yp[retrieve(Z, yp, Zq[rng.choice(Xte.shape[0], 100, replace=False)].mean(0), k, 0)]
                == 1
            ).mean()
        )
        for _ in range(a.n_real)
    ]
    origin = np.zeros(Z.shape[1], np.float32)
    print(
        f"mean of 100 random queries as target: mean rate {np.mean(shr):.4%}; centre target "
        f"{float((yp[retrieve(Z, yp, origin, k, 0)] == 1).mean()):.4%}"
    )
    obs = np.array([y[te][q].mean() for q, _ in plan])
    ctxr = np.array([(yp[c] == 1).mean() for _, c in plan])
    sizes = np.array([q.size for q, _ in plan])
    print(
        f"observed test rate per group (size-weighted) {np.average(obs, weights=sizes):.4%} vs "
        f"context rate {np.average(ctxr, weights=sizes):.4%}; "
        f"corr(obs, ctx rate) {np.corrcoef(obs, ctxr)[0, 1]:.3f}"
    )


if __name__ == "__main__":
    main()
