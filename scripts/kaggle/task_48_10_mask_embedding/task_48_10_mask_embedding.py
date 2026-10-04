"""Kaggle kernel script: task 48.10, a dedicated missingness embedding.

Trains one new checkpoint at the exact §127 reference recipe (matching the existing
`v4-cellattn-labels.pt` control, seed 0), with `--mask-embedding` instead of the default
joint [value, missing] cell MLP. Tests the narrower question task 48.10's premise correction
left open: not naive imputation vs a mask-aware embedding (already settled in this project's
favour -- the existing architecture already learns a value/missingness interaction), but
whether a dedicated nn.Embedding(2, d_cell) table beats that joint small MLP.

Push with:
    kaggle kernels push -p scripts/kaggle/task_48_10_mask_embedding

Then after it completes:
    kaggle kernels status  <username>/<slug>
    kaggle kernels output  <username>/<slug> -p runs/task48-10-kaggle/
"""

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    # Pinned to the commit that added --mask-embedding, which postdates the 0.5.6 PyPI
    # release used by every other kernel that doesn't need this flag.
    run([
        sys.executable, "-m", "pip", "install", "--quiet",
        "git+https://github.com/kabartay/fintfm.git@e1b2c45",
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
        "--mask-embedding",
        "--out", "/kaggle/working/v4-cellattn-maskembed48_10.pt",
    ])

    print("done")


if __name__ == "__main__":
    main()
