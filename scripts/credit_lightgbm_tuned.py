"""DR8.4 driver: tuned LightGBM on the credit suite's splits (FINDINGS §189's open control).

Stage A (this process, imports the suite and so torch) loads each panel, builds the suite's
splits and the same median-impute plus standardise preprocessing as §189, and writes one npz
per split. Stage B (a subprocess that never imports torch) runs
``src/fintfm/evaluation/lightgbm_tuned.py`` on it. Output: one JSON per split under ``--out``.

With ``--oof-dir DIR`` (the directory ``credit_suite --hybrid --save-oof`` wrote) it tunes the
same search over FinTFM's logit too, once per ``--hybrid-modes`` entry: ``feature`` (the
out-of-fold logit as one more column; the test rows get the logit of the FinTFM fitted on all
training rows) and ``init`` (the logit as LightGBM's starting score). Files are then
``<dataset>_split<k>_<mode>.json``; without it, ``<dataset>_split<k>.json`` as before.

Usage: ``uv run --frozen --extra bench python scripts/credit_lightgbm_tuned.py --out DIR``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    """Run every split of the chosen datasets within a wall-time budget."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", default=["sba", "ulb_fraud"])
    p.add_argument("--configs", type=int, default=30)
    p.add_argument("--per-split-budget", type=float, default=720.0, help="search seconds")
    p.add_argument("--total-budget", type=float, default=7200.0, help="wall seconds")
    p.add_argument("--oof-dir", type=Path, default=None,
                   help="out-of-fold FinTFM logits from credit_suite --save-oof; tunes the "
                        "hybrids instead of the plain control")
    p.add_argument("--hybrid-modes", nargs="+", default=["feature", "init"],
                   choices=["feature", "init"])
    p.add_argument("--root", type=Path, default=None, help="raw data root (default: data/real)")
    p.add_argument("--max-rows", type=int, default=None,
                   help="the same development cap as credit_suite --max-rows (must match "
                        "the run that saved the logits)")
    p.add_argument(
        "--inner-embargo", type=float, default=None,
        help="embargo (time units) for the inner tuning split only; default: the panel's. "
        "SBA's 1,827-day embargo exceeds its training windows (split 0 spans 203 days), so "
        "its inner split needs 0; the test split keeps the panel's embargo either way",
    )
    a = p.parse_args()
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    from fintfm.evaluation.credit_suite import DATASETS, DEFAULT_ROOT, cap_panel, suite_splits
    from fintfm.evaluation.tuned_boosting import run_tuned

    a.out.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    for key in a.datasets:
        panel = DATASETS[key].loader(a.root or DEFAULT_ROOT)
        if a.max_rows is not None:
            panel = cap_panel(panel, a.max_rows)
        for k, (tr, te) in enumerate(suite_splits(panel)):
            modes = a.hybrid_modes if a.oof_dir is not None else [""]
            todo = [m for m in modes
                    if not (a.out / f"{key}_split{k}{'_' + m if m else ''}.json").exists()]
            if not todo:
                continue
            if time.perf_counter() - t_all > a.total_budget:
                print("total budget reached; stopping", flush=True)
                return 1
            imp = SimpleImputer(strategy="median").fit(panel.X[tr])
            sc = StandardScaler().fit(imp.transform(panel.X[tr]))
            base = {
                "Xtr": sc.transform(imp.transform(panel.X[tr])), "ytr": panel.y[tr],
                "Xte": sc.transform(imp.transform(panel.X[te])), "yte": panel.y[te],
                "embargo": np.array(panel.embargo if a.inner_embargo is None else a.inner_embargo),
            }
            if panel.time is not None:
                base["ttr"] = panel.time[tr]
            if panel.groups is not None:
                base["gtr"] = panel.groups[tr]
            saved = None
            if a.oof_dir is not None:
                saved = np.load(a.oof_dir / f"{key}_split{k}.npz")
                if not (np.array_equal(saved["train_rows"], tr)
                        and np.array_equal(saved["test_rows"], te)):
                    print(f"{key} split {k}: saved logits are for other rows (a different "
                          "--max-rows or suite?); skipped", flush=True)
                    continue
            for mode in todo:
                arrays = dict(base)
                if mode == "feature":
                    arrays["Xtr"] = np.column_stack([base["Xtr"], saved["oof_train"]])
                    arrays["Xte"] = np.column_stack([base["Xte"], saved["logit_test"]])
                elif mode == "init":
                    arrays["btr"], arrays["bte"] = saved["oof_train"], saved["logit_test"]
                tag = f"{key} split {k}{' ' + mode if mode else ''}"
                res = run_tuned(arrays, a.configs, a.per_split_budget)
                if res is None:
                    print(f"{tag} failed", flush=True)
                    continue
                res["inner_embargo"] = float(
                    panel.embargo if a.inner_embargo is None else a.inner_embargo)
                res["panel_embargo"] = float(panel.embargo)
                if mode:
                    res["hybrid_mode"] = mode
                (a.out / f"{key}_split{k}{'_' + mode if mode else ''}.json").write_text(
                    json.dumps(res, indent=2))
                print(f"{tag}: tuned AP {res['tuned_ap']:.4f} AUC {res['tuned_auc']:.4f} | "
                      f"untuned AP {res['untuned_ap']:.4f} | {res['n_trials']} configs, "
                      f"{res['tune_seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
