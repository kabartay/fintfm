"""Task 44.2 diagnostic: capability suite's own scoring on a reduced probe set and seed count."""

import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from fintfm.experiments.capability import bayes_ceiling_report, make_probe
from fintfm.inference.classifier import FinancialTFMClassifier
from fintfm.modeling.model import FinancialTFM

R = "/Users/morganoko/proj/fintfm/runs/"
OUT = sys.argv[1]
ARMS = {
    "pub_s0 (cells+labels, 48k, matched)": R + "kaggle-ts-pub-s0/pub-s0/pub-s0.pt",
    "nocell_s0 (n_cell_blocks=0, matched)": R + "kaggle-ts-nocell-s0/nocell-s0/nocell-s0.pt",
    "lab_s0 (cells+labels, s91 pair)": R + "dl/v4-cellattn-labels.pt",
    "nolab_s0 (cells, no labels, s91 pair)": R + "dl/v4-cellattn-nolabels.pt",
    "lab_s1 (s142 pair)": R + "cellattn-s1-kaggle/v4-cellattn-labels-s1.pt",
    "nolab_s1 (s142 pair)": R + "cellattn-s1-kaggle/v4-cellattn-nolabels-s1.pt",
    "dr62_induced_s0 (cells+labels, induced row stage)": R
    + "kaggle-ts-dr62-induced-s0/dr62-induced-s0/dr62-induced-s0.pt",
}
PROBES = ["antisymmetric", "symmetric_sum"]
SEEDS = (0, 1, 2)
res = {}
for name, path in ARMS.items():
    m = FinancialTFM.load(path)
    row = {}
    for probe in PROBES:
        v = []
        for s in SEEDS:
            Xtr, ytr, Xte, yte = make_probe(probe, n_features=20, seed=s)
            clf = FinancialTFMClassifier(
                m,
                max_context=1000,
                context_strategy="uniform",
                feature_transform="rank",
                random_state=s,
            ).fit(Xtr, ytr)
            v.append(float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])))
        row[probe] = v
    rep = bayes_ceiling_report(m, targets=(0.9, 0.999), seeds=3, n_ensemble=4)
    row["bayes"] = {
        str(t): {
            "auc_mean": float(np.mean([p.get("auc", np.nan) for p in r["per_task"]]))
            if "auc" in r["per_task"][0]
            else None,
            "agg": {k: v for k, v in r.items() if k != "per_task"},
        }
        for t, r in rep.items()
    }
    res[name] = row
    print(
        name,
        json.dumps(
            {
                k: (np.round(np.mean(v), 4).item() if isinstance(v, list) else v)
                for k, v in row.items()
            }
        ),
        flush=True,
    )
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
print("DONE", flush=True)
