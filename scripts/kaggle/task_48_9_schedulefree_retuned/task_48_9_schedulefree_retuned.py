"""Kaggle kernel script: task 48.9 retested, schedule-free at its own defaults.

The first attempt (`task_48_9_schedulefree/`, Section 146) measured a clear negative --
but at `--lr 3e-4` (AdamW's tuned value, Section 127) and with **zero warmup**, because
`train.py`'s `AdamWScheduleFree` construction never passed `warmup_steps` despite a docstring
claiming the optimiser "carries its own internal warmup". Both are fixed as of commit
c8c3027: `warmup_steps` is now wired through, and this kernel passes `--lr 0.0025`, the
package's own `AdamWScheduleFree` default, instead of inheriting AdamW's tuning. Section 146's
original measurement is evidence about one mistuned configuration, not about schedule-free
optimisation on this architecture -- this is the actual comparison that was missing.

Everything else matches the exact §127 reference recipe (matching `v4-cellattn-labels.pt`,
seed 0) and the first attempt, so the only axis that differs from `task_48_9_schedulefree/`
is the learning rate and the now-functioning warmup.

Push with:
    kaggle kernels push -p scripts/kaggle/task_48_9_schedulefree_retuned

Then after it completes:
    kaggle kernels status  <username>/<slug>
    kaggle kernels output  <username>/<slug> -p runs/task48-9-retuned-kaggle/
"""

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    # Pinned to the commit that fixed warmup_steps never being wired to AdamWScheduleFree.
    run([
        sys.executable, "-m", "pip", "install", "--quiet",
        "fintfm[schedulefree] @ git+https://github.com/kabartay/fintfm.git@c8c3027",
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
        "--p-financial", "1.0",
        "--seed", "0",
        "--device", device,
        "--checkpoint-every", "500",
        "--optimizer", "schedulefree",
        "--lr", "0.0025",
        "--out", "/kaggle/working/v4-cellattn-schedulefree-retuned48_9.pt",
    ])

    print("done")


if __name__ == "__main__":
    main()
