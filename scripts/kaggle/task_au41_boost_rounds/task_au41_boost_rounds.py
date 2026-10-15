"""Kaggle kernel: audit AU4.1 -- is the CatBoost/LightGBM gap boosting rounds? CPU only.

V4FinBench horizon 0, the published five-fold protocol, baselines only (no fintfm arm, no
logistic regression). Four booster arms, each fitted once per fold at fixed parameters:

* ``lightgbm[n_estimators=1000,learning_rate=0.05]`` and ``catboost[iterations=100]``, the
  two settings AU4.1 asks for;
* ``lightgbm`` and ``catboost`` at library defaults (LightGBM 100 trees, CatBoost 1,000
  iterations), re-measured on this machine so the pairing does not cross platforms. The
  existing Mac run of the same defaults is ``runs/boost/full``.

Per-fold predictions (validation and test) and ``v4_protocol.json`` land in
/kaggle/working/AU41/, ready for a paired comparison. Data is V4FinBench's own public Kaggle
dataset. The fintfm install is pinned to a commit; ``<COMMIT>`` below is a placeholder until
the commit carrying ``--booster-params`` is pushed, and the kernel refuses to run with it.
"""

import subprocess
import sys
from pathlib import Path

# Set to the pushed commit that carries `booster_params` in v4_protocol.run (audit AU4.1).
COMMIT = "<COMMIT>"
BOOSTER_PARAMS = (
    ("lightgbm", {}),
    ("catboost", {}),
    ("lightgbm", {"n_estimators": 1000, "learning_rate": 0.05}),
    ("catboost", {"iterations": 100}),
)


def main() -> None:
    """Find the parquet, install the pinned fintfm, and score the four booster arms."""
    if COMMIT.startswith("<"):
        sys.exit("COMMIT is still a placeholder; pin the pushed commit before running")
    hits = sorted(Path("/kaggle/input").rglob("company_years_h1.parquet"))
    if len(hits) != 1:
        sys.exit(f"expected one company_years_h1.parquet under /kaggle/input, found {hits}")
    root = hits[0].parent
    print(f"data: {root}", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    f"fintfm[bench] @ git+https://github.com/kabartay/fintfm.git@{COMMIT}"],
                   check=True)
    from fintfm.config import load_config
    from fintfm.experiments.v4_protocol import run, summarise
    record = run(None, Path("/kaggle/working/AU41"), horizon=0, folds=(0, 1, 2, 3, 4),
                 root=root, with_boosting=False, classical=(),
                 booster_params=BOOSTER_PARAMS, cfg=load_config(), device="cpu")
    print(summarise(record), flush=True)
    print("done")


if __name__ == "__main__":
    main()
