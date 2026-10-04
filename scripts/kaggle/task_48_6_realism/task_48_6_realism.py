"""Kaggle kernel script: task 48.6, cheap realism augmentations.

Trains one new checkpoint at the exact §127 reference recipe (matching the existing
`v4-cellattn-labels.pt` control, seed 0), with Nori's four cheap realism augmentations
bundled on: discretized features, extra noise features, a correlated feature block, and
label noise. A first bundle-wide signal before any per-augmentation attribution, per the
"measurement before a rewrite" discipline this project otherwise follows.

Push with:
    kaggle kernels push -p scripts/kaggle/task_48_6_realism

Then after it completes:
    kaggle kernels status  <username>/<slug>
    kaggle kernels output  <username>/<slug> -p runs/task48-6-kaggle/
"""

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    # Pinned to the commit that added the --discretize-frac/--n-noise-features/
    # --n-correlated-block-features/--label-noise-rate flags, which postdate the 0.5.6
    # PyPI release used by every other kernel in this directory.
    run([
        sys.executable, "-m", "pip", "install", "--quiet",
        "git+https://github.com/kabartay/fintfm.git@5c7b359",
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
        "--discretize-frac", "0.2",
        "--n-noise-features", "5",
        "--n-correlated-block-features", "3",
        "--label-noise-rate", "0.02",
        "--out", "/kaggle/working/v4-cellattn-realism48_6.pt",
    ])

    print("done")


if __name__ == "__main__":
    main()
