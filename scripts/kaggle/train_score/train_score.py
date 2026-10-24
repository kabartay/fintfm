"""Kaggle kernel template: pretrain the published recipe with one change, then score it.

Design review DR4.4 (noise yardstick from matched replicates) and audit AU2.8 (cell attention
at a matched prior). Each arm trains `configs/published/v4-cellattn-labels.yaml` (48,000 tasks,
fp32, masked attention, single GPU -- the published checkpoint's own protocol) at the
commit pinned below, with ARM_FLAGS appended, then scores the checkpoint zero-shot on
V4FinBench's five full-panel folds with the packaged default inference settings, as the
§160 matched control was scored. Outputs: the checkpoint, its training log, and per-fold
predictions with v4_protocol.json, in /kaggle/working/<ARM>/.

A curriculum arm (design review DR2.4, arm c) sets SWITCH_STEP: it trains ARM_FLAGS for
SWITCH_STEP steps of the same --steps schedule (`--run-steps`), keeps that checkpoint as
`<ARM>.phase1.pt`, then resumes the state with PHASE2_FLAGS and `--new-prior-phase` for the
rest. Optimiser, schedule and both RNG streams carry over, so only the prior changes.

A train-only arm (design review DR7.4, a multiclass checkpoint) sets SCORE_V4 False.
V4FinBench is binary, so such an arm is decided elsewhere (the proxy suite's multiclass tasks,
run off Kaggle), and the kernel stops after training without reading the V4FinBench data.

`make_train_score.py` writes one copy of this file per arm with ARM and ARM_FLAGS filled in.
"""

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

COMMIT = "__COMMIT__"
ARM = "__ARM__"
ARM_FLAGS: list[str] = []  # __FLAGS__
DDP = False  # __DDP__
SCORE_PARTITION = False  # __PART__
SWITCH_STEP = 0  # __SWITCH__
PHASE2_FLAGS: list[str] = []  # __PHASE2__
SCORE_V4 = True  # __SCORE__
RECIPE_URL = (f"https://raw.githubusercontent.com/kabartay/fintfm/{COMMIT}/"
              "configs/published/v4-cellattn-labels.yaml")


def main() -> None:
    root = None
    if SCORE_V4:
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
    base = [*launcher, "--config", str(recipe), "--device", "cuda",
            "--checkpoint-every", "1000", "--out", str(ckpt)]
    phases = [[*base, *ARM_FLAGS]]
    if SWITCH_STEP:
        phases = [[*base, *ARM_FLAGS, "--run-steps", str(SWITCH_STEP)],
                  [*base, *PHASE2_FLAGS, "--resume", f"{ckpt}.state", "--new-prior-phase"]]
    for i, train in enumerate(phases):
        print("$ " + " ".join(train), flush=True)
        with (out / "train.log").open("a" if i else "w") as log:
            rc = subprocess.run(train, stdout=log, stderr=subprocess.STDOUT,
                                check=False).returncode
        print((out / "train.log").read_text()[-3000:], flush=True)
        if rc != 0:
            sys.exit(f"training phase {i + 1} failed with exit {rc}")
        if i + 1 < len(phases):
            shutil.copy(ckpt, out / f"{ARM}.phase1.pt")  # the trunk-only weights at the switch
    if not SCORE_V4:
        _keep_final(out)
        print("done (train only: V4FinBench scoring skipped)")
        return
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
    _keep_final(out)
    print("done")


def _keep_final(out: Path) -> None:
    """Delete the intermediate step checkpoints, keeping the final one.

    Args:
        out: The arm's output directory.
    """
    for p in out.glob(f"{ARM}.pt.step*"):
        p.unlink()


if __name__ == "__main__":
    main()
