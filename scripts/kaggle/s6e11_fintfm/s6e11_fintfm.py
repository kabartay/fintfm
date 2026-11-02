"""Kaggle kernel: FinTFM on Playground Series S6E11 (credit-card fraud), against LightGBM.

An out-of-domain check of the published checkpoint (FinTFM, synthetic-only prior) on a binary
fraud table: 699,960 training rows at a 1.67% positive rate, 25 features (5 categorical, 6
boolean), scored by ROC-AUC. Competition data is used for evaluation only and never leaves
Kaggle except as this kernel's own predictions.

Arms, each scored on a stratified 20% holdout of train (ROC-AUC and AP), then refitted on all of
train to predict test, writing its own submission as soon as it finishes so a timeout keeps the
arms already done:

- ``lightgbm``: default LightGBM with 500 rounds, the reference.
- ``fintfm``: the published checkpoint zero-shot, 2,000-row uniform context.
- ``fintfm_partition``: §169's partitioned contexts (depth-2 tree, 5% leaf floor).

CPU only, so it spends no GPU quota. Pinned to fintfm 4ad16a8; the checkpoint is the public
kabartay/fintfm-binary file, verified by SHA-256.
"""

import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CKPT_URL = "https://huggingface.co/kabartay/fintfm-binary/resolve/f116bfd43a2b15c65ed3551ea8c38e3364629ddc/v4-cellattn-labels.pt"
CKPT_SHA256 = "d231b062483c9261aae4c503ad5359db295fc7206fa0183f51763e4953c518d2"
OUT = Path("/kaggle/working")
TARGET = "is_fraud"


def load():
    """Train and test as numeric matrices: categoricals one-hot, booleans 0/1."""
    import numpy as np
    import pandas as pd

    root = min(Path("/kaggle/input").rglob("train.csv")).parent
    train, test = pd.read_csv(root / "train.csv"), pd.read_csv(root / "test.csv")
    y = train.pop(TARGET).to_numpy().astype(int)
    both = pd.concat([train, test], keys=["train", "test"])
    ids = both.pop("id")
    cats = [c for c in both.columns if both[c].dtype == object or str(both[c].dtype).startswith("str")]
    both = pd.get_dummies(both, columns=cats, dtype=np.float32).astype(np.float32)
    X, Xt = both.loc["train"].to_numpy(), both.loc["test"].to_numpy()
    print(f"train {X.shape}, test {Xt.shape}, positive rate {y.mean():.4f}, one-hot from {cats}", flush=True)
    return X, y, Xt, ids.loc["test"].to_numpy()


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    "fintfm @ git+https://github.com/kabartay/fintfm.git@4ad16a8"], check=True)
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    from fintfm.inference.classifier import FinancialTFMClassifier
    from fintfm.inference.large_n import PartitionedTFMClassifier

    ckpt = OUT / "v4-cellattn-labels.pt"
    urllib.request.urlretrieve(CKPT_URL, ckpt)
    if hashlib.sha256(ckpt.read_bytes()).hexdigest() != CKPT_SHA256:
        sys.exit("checkpoint hash mismatch")

    X, y, Xt, test_ids = load()
    Xa, Xh, ya, yh = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    def fintfm():
        return FinancialTFMClassifier(str(ckpt), device="cpu", random_state=0)

    def lightgbm():
        # Imported last: LightGBM's OpenMP loaded before torch segfaulted the local smoke run.
        from lightgbm import LGBMClassifier

        return LGBMClassifier(n_estimators=500, learning_rate=0.05, verbose=-1, random_state=0)

    arms = {
        "fintfm": fintfm,
        "fintfm_partition": lambda: PartitionedTFMClassifier(fintfm(), random_state=0),
        "lightgbm": lightgbm,
    }
    results = {}
    for name, make in arms.items():
        t0 = time.time()
        p = make().fit(Xa, ya).predict_proba(Xh)[:, 1]
        results[name] = {"holdout_roc_auc": float(roc_auc_score(yh, p)),
                         "holdout_ap": float(average_precision_score(yh, p)),
                         "holdout_mean_pred": float(p.mean()), "holdout_rate": float(yh.mean()),
                         "holdout_seconds": round(time.time() - t0)}
        print(name, results[name], flush=True)
        t0 = time.time()
        pt = make().fit(X, y).predict_proba(Xt)[:, 1]
        results[name]["test_seconds"] = round(time.time() - t0)
        pd.DataFrame({"id": test_ids, TARGET: pt}).to_csv(OUT / f"submission_{name}.csv", index=False)
        (OUT / "results.json").write_text(json.dumps(results, indent=1))
        print(f"{name}: wrote submission_{name}.csv ({results[name]['test_seconds']} s)", flush=True)
    ckpt.unlink()
    print("done", flush=True)


if __name__ == "__main__":
    main()
