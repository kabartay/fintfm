"""Kaggle kernel script: task 48.3's matched binned-head control.

The existing `v4-regression.pt` (§106) is **not** a valid control for the quantile-head
checkpoint in `task_48_3_quantile_head/`: it was trained at `--p-financial 0.35 --p-regression
0.35`, a mixed prior, while the quantile head can only train on pure regression tasks
(`collate` refuses a classification task under `head_type="quantile"`). Comparing the two
would confound "quantile vs binned head" with "different prior composition", exactly the
mistake §141 warns about for a different pair of checkpoints. This kernel trains a fresh
binned-head checkpoint at the **identical** `--p-regression 1.0` composition, architecture and
step count as its quantile-head sibling, differing only in `--head-type`.

Push with:
    kaggle kernels push -p scripts/kaggle/task_48_3_binned_control

Then after it completes:
    kaggle kernels status  <username>/<slug>
    kaggle kernels output  <username>/<slug> -p runs/task48-3-kaggle/
"""

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    run([
        sys.executable, "-m", "pip", "install", "--quiet", "fintfm==0.5.6",
    ])

    import torch

    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run([
        sys.executable, "-u", "-m", "fintfm.modeling.train",
        "--steps", "6000",
        "--batch-size", "8",
        "--n-rows-choices", "256,512,1024",
        "--d-cell", "48",
        "--d-model", "128",
        "--n-layers", "4",
        "--n-col-layers", "2",
        "--n-cell-blocks", "1",
        "--cell-labels",
        "--column-id-dim", "16",
        "--feature-chunk", "8",
        "--max-features", "136",
        "--p-regression", "1.0",
        "--seed", "0",
        "--device", device,
        "--checkpoint-every", "500",
        "--max-classes", "10",
        "--out", "/kaggle/working/v4-binned48_3-control.pt",
    ])

    print("done")


if __name__ == "__main__":
    main()
