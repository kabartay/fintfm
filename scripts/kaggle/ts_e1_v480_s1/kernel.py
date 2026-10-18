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

COMMIT = "d05ea19"
ARM = "e1-v480-s1"
ARM_FLAGS: list[str] = ["--amp", "fp16", "--attn-impl", "split", "--loader", "stream", "--ddp", "--batch-size", "4", "--seed", "1", "--steps", "60000"]
DDP = True
SCORE_PARTITION = False
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
    launcher = [sys.executable, "-u", "-m", "fintfm.modeling.train"]
    if DDP:
        launcher = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
                    "--nproc-per-node", "2", "-m", "fintfm.modeling.train"]
    train = [*launcher, "--config", str(recipe), "--device", "cuda",
             "--checkpoint-every", "1000", "--out", str(ckpt), *ARM_FLAGS]
    print("$ " + " ".join(train), flush=True)
    with (out / "train.log").open("w") as log:
        rc = subprocess.run(train, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
    print((out / "train.log").read_text()[-3000:], flush=True)
    if rc != 0:
        sys.exit(f"training failed with exit {rc}")
    from fintfm.config import load_config
    from fintfm.experiments.v4_protocol import run, summarise

    # the cell-attention arms ran out of memory scoring at the default chunks on a 14.56 GB T4
    # (2026-10-18); these chunks are identity-preserving, so no score changes
    score_cfg = out / "score.yaml"
    score_cfg.write_text("inference:\n  query_chunk: 512\n  feature_chunk: 8\n")
    cfg = load_config(score_cfg)
    record = run(str(ckpt), out / "score", horizon=0, folds=(0, 1, 2, 3, 4), root=root,
                 with_boosting=False, classical=("logistic_regression",), cfg=cfg, device="cuda")
    print(summarise(record), flush=True)
    if SCORE_PARTITION:
        part_cfg = out / "score_partition.yaml"
        part_cfg.write_text(score_cfg.read_text() + "large_n:\n  method: partition\n")
        record = run(str(ckpt), out / "score_partition", horizon=0, folds=(0, 1, 2, 3, 4),
                     root=root, with_boosting=False, classical=("logistic_regression",),
                     cfg=load_config(part_cfg), device="cuda")
        print(summarise(record), flush=True)
    for p in out.glob(f"{ARM}.pt.step*"):
        p.unlink()  # keep the final checkpoint only
    print("done")


if __name__ == "__main__":
    main()
