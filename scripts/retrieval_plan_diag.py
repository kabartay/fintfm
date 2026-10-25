"""Fold-0 diagnostic for §194: what each retrieval group's context holds, and where ranking breaks.

Rebuilds the classifier's retrieval plan exactly as the protocol run did (same split, same
preprocessing, random_state = fold), without any forward pass, and joins it to the saved
fold-0 predictions.
"""

from __future__ import annotations

import sys

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from fintfm.config import load_config
from fintfm.evaluation.datasets import CACHE_DIR
from fintfm.experiments import v4_protocol as vp
from fintfm.modeling.model import FinancialTFM

fold = 0
cfg = load_config("configs/arms/proxy/retrieval.yaml")
X, y, country, company, _ = vp._load_horizon(CACHE_DIR / "v4finbench", 0, None)
assign = vp.build_fold_assignments(country, company)
tr, va, te = vp.split_indices_for_fold(assign, fold)
imp = SimpleImputer(strategy="median").fit(X[tr])
sc = StandardScaler().fit(imp.transform(X[tr]))
Xtr, Xte = (sc.transform(imp.transform(X[i])).astype(np.float32) for i in (tr, te))

model = FinancialTFM.load("runs/v4-cellattn-labels.pt")
clf = vp.fintfm_classifier(model, cfg, device="cpu", random_state=fold)
clf.fit(Xtr, y[tr], groups=company[tr])
plan = clf._retrieval_plan(Xte)
rate = clf._pooled_context_rate(plan)

d = np.load("runs/v4-lever-retrieval/predictions_fold0.npz")
yt, p = d["y_true"], d["pred_fintfm"]
assert np.array_equal(yt, y[te]), "saved y_true does not match the rebuilt test split"

pos = int(clf._pool_y.max())
print(
    f"pool rows {clf._pool_y.size:,}, pool rate {np.mean(clf._pool_y == pos):.4%}, "
    f"pooled context rate {rate:.4%}, groups {len(plan)}, max_context {clf.max_context}"
)
rows = []
for q, c in plan:
    cpos = int((clf._pool_y[c] == pos).sum())
    gy = yt[q]
    auc = roc_auc_score(gy, p[q]) if 0 < gy.sum() < gy.size else np.nan
    rows.append((q.size, c.size, cpos, int(gy.sum()), float(p[q].mean()), auc))
r = np.array(rows, dtype=float)
print(
    f"context positives per group: min {r[:, 2].min():.0f}, median {np.median(r[:, 2]):.0f}, "
    f"max {r[:, 2].max():.0f}; groups at the floor of {clf.retrieval_min_positive}: "
    f"{int((r[:, 2] <= clf.retrieval_min_positive).sum())} of {len(r)}"
)
ok = ~np.isnan(r[:, 5])
w = r[ok, 3]
print(
    f"within-group ROC-AUC, weighted by test positives: {np.average(r[ok, 5], weights=w):.4f} "
    f"over {ok.sum()} groups holding {int(w.sum())} of {int(yt.sum())} positives"
)
gid = np.empty(yt.size, dtype=int)
for i, (q, _) in enumerate(plan):
    gid[q] = i
gmean = r[:, 4][gid]
print(f"pooled ROC-AUC {roc_auc_score(yt, p):.4f}, AP {average_precision_score(yt, p):.4f}")
print(
    f"group-mean score alone: ROC-AUC {roc_auc_score(yt, gmean):.4f}, "
    f"AP {average_precision_score(yt, gmean):.4f}"
)
ctx_rate = (r[:, 2] / r[:, 1])[gid]
obs = r[:, 3] / r[:, 0]
print(
    f"corr(context rate, group observed rate) over groups: "
    f"{np.corrcoef(r[:, 2] / r[:, 1], obs)[0, 1]:.3f}; corr(group mean score, observed rate): "
    f"{np.corrcoef(r[:, 4], obs)[0, 1]:.3f}"
)
print("largest-positive groups: size, ctx_pos, test_pos, mean_score, auc")
for row in sorted(rows, key=lambda t: -t[3])[:8]:
    print("  ", row[0], row[2], row[3], f"{row[4]:.4f}", f"{row[5]:.3f}")
sys.stdout.flush()

# Are the retrieved positives the same rows everywhere, and are they the data-poor ones?
ctx_pos = [c[clf._pool_y[c] == pos] for _, c in plan]
allpos = np.concatenate(ctx_pos)
uniq, cnt = np.unique(allpos, return_counts=True)
print(
    f"distinct positive rows across all contexts: {uniq.size} of {int((clf._pool_y == pos).sum())} "
    f"pool positives; median groups per retrieved positive {np.median(cnt):.0f} of {len(plan)}"
)
miss = np.isnan(X[tr]).mean(axis=1)
ypool = y[tr]
print(
    f"share of features missing (raw): pool negatives {miss[ypool == 0].mean():.3f}, "
    f"pool positives {miss[ypool == 1].mean():.3f}, retrieved positives {miss[uniq].mean():.3f}"
)
allneg = np.unique(np.concatenate([c[clf._pool_y[c] != pos] for _, c in plan]))
print(f"retrieved negatives: {allneg.size:,} distinct, share missing {miss[allneg].mean():.3f}")
