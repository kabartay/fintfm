"""DR8.4 driver: tuned LightGBM on the credit suite's splits (FINDINGS §189's open control).

Stage A (this process, imports the suite and so torch) loads each panel, builds the suite's
splits and the same median-impute plus standardise preprocessing as §189, and writes one npz
per split. Stage B (a subprocess that never imports torch) runs
``src/fintfm/evaluation/lightgbm_tuned.py`` on it. Output: one JSON per split under ``--out``.

Usage: ``uv run --frozen --extra bench python scripts/credit_lightgbm_tuned.py --out DIR``
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WORKER = Path(__file__).resolve().parents[1] / "src/fintfm/evaluation/lightgbm_tuned.py"


def main() -> int:
    """Run every split of the chosen datasets within a wall-time budget."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", default=["sba", "ulb_fraud"])
    p.add_argument("--configs", type=int, default=30)
    p.add_argument("--per-split-budget", type=float, default=720.0, help="search seconds")
    p.add_argument("--total-budget", type=float, default=7200.0, help="wall seconds")
    p.add_argument(
        "--inner-embargo", type=float, default=None,
        help="embargo (time units) for the inner tuning split only; default: the panel's. "
        "SBA's 1,827-day embargo exceeds its training windows (split 0 spans 203 days), so "
        "its inner split needs 0; the test split keeps the panel's embargo either way",
    )
    a = p.parse_args()
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    from fintfm.evaluation.credit_suite import DATASETS, DEFAULT_ROOT, suite_splits

    a.out.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    for key in a.datasets:
        panel = DATASETS[key].loader(DEFAULT_ROOT)
        for k, (tr, te) in enumerate(suite_splits(panel)):
            dest = a.out / f"{key}_split{k}.json"
            if dest.exists():
                continue
            if time.perf_counter() - t_all > a.total_budget:
                print("total budget reached; stopping", flush=True)
                return 1
            imp = SimpleImputer(strategy="median").fit(panel.X[tr])
            sc = StandardScaler().fit(imp.transform(panel.X[tr]))
            arrays = {
                "Xtr": sc.transform(imp.transform(panel.X[tr])), "ytr": panel.y[tr],
                "Xte": sc.transform(imp.transform(panel.X[te])), "yte": panel.y[te],
                "embargo": np.array(panel.embargo if a.inner_embargo is None else a.inner_embargo),
            }
            if panel.time is not None:
                arrays["ttr"] = panel.time[tr]
            if panel.groups is not None:
                arrays["gtr"] = panel.groups[tr]
            npz = a.out / f"{key}_split{k}.npz"
            np.savez(npz, **arrays)
            r = subprocess.run(
                ["nice", "-n", "19", sys.executable, str(WORKER), str(npz), str(dest),
                 str(a.configs), str(a.per_split_budget)],
                env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
                check=False)
            npz.unlink()
            if r.returncode != 0:
                print(f"{key} split {k} failed", flush=True)
                continue
            res = json.loads(dest.read_text())
            res["inner_embargo"] = float(panel.embargo if a.inner_embargo is None else a.inner_embargo)
            res["panel_embargo"] = float(panel.embargo)
            dest.write_text(json.dumps(res, indent=2))
            print(f"{key} split {k}: tuned AP {res['tuned_ap']:.4f} AUC {res['tuned_auc']:.4f} | "
                  f"untuned AP {res['untuned_ap']:.4f} | {res['n_trials']} configs, "
                  f"{res['tune_seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
