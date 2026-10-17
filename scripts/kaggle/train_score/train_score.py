"""Kaggle kernel template: pretrain the published recipe with one change, then score it.

Design review DR4.4 (noise yardstick from matched replicates) and audit AU2.8 (cell attention
at a matched prior). Each arm trains `configs/published/v4-cellattn-labels.yaml` (48,000 tasks,
fp32, masked attention, single GPU -- the published checkpoint's own protocol) at the
commit pinned below, with ARM_FLAGS appended, then scores the checkpoint zero-shot on
V4FinBench's five full-panel folds with the packaged default inference settings, as the
§160 matched control was scored. Outputs: the checkpoint, its training log, and per-fold
predictions with v4_protocol.json, in /kaggle/working/<ARM>/.

`make_train_score.py` writes one copy of this file per arm with ARM and ARM_FLAGS filled in.
"""

import subprocess
import sys
import urllib.request
from pathlib import Path

COMMIT = "__COMMIT__"
ARM = "__ARM__"
ARM_FLAGS: list[str] = __FLAGS__
RECIPE_URL = (f"https://raw.githubusercontent.com/kabartay/fintfm/{COMMIT}/"
              "configs/published/v4-cellattn-labels.yaml")


def main() -> None:
    hits = sorted(Path("/kaggle/input").rglob("company_years_h1.parquet"))
    if len(hits) != 1:
        sys.exit(f"expected one company_years_h1.parquet under /kaggle/input, found {hits}")
    root = hits[0].parent
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False)
    if smi.returncode != 0 or not smi.stdout.strip():
        sys.exit("no GPU in this session (quota exhausted?) -- refusing a CPU run")
    print(f"GPUs: {smi.stdout.strip()}; data: {root}", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    f"fintfm @ git+https://github.com/kabartay/fintfm.git@{COMMIT}"], check=True)
    out = Path("/kaggle/working") / ARM
    out.mkdir(parents=True, exist_ok=True)
    recipe = out / "recipe.yaml"
    urllib.request.urlretrieve(RECIPE_URL, recipe)
    ckpt = out / f"{ARM}.pt"
    train = [sys.executable, "-u", "-m", "fintfm.modeling.train", "--config", str(recipe),
             "--device", "cuda", "--checkpoint-every", "1000", "--out", str(ckpt), *ARM_FLAGS]
    print("$ " + " ".join(train), flush=True)
    with (out / "train.log").open("w") as log:
        rc = subprocess.run(train, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
    print((out / "train.log").read_text()[-3000:], flush=True)
    if rc != 0:
        sys.exit(f"training failed with exit {rc}")
    from fintfm.experiments.v4_protocol import run, summarise

    record = run(str(ckpt), out / "score", horizon=0, folds=(0, 1, 2, 3, 4), root=root,
                 with_boosting=False, classical=("logistic_regression",), device="cuda")
    print(summarise(record), flush=True)
    for p in out.glob(f"{ARM}.pt.step*"):
        p.unlink()  # keep the final checkpoint only
    print("done")


if __name__ == "__main__":
    main()
