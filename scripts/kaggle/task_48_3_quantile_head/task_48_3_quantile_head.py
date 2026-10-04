"""Kaggle kernel script: task 48.3, the pinball-loss quantile head.

Trains one new checkpoint at the §127 reference recipe's architecture (matching the existing
`v4-cellattn-labels.pt` control's size, seed 0), with `--head-type quantile --n-quantiles 99`
and `--p-regression 1.0` -- a quantile head has no class-index path, so every task must be a
regression task (`collate` refuses a classification task under this head_type). Its sibling
kernel in `task_48_3_binned_control/` trains the matched binned-head comparison at the
identical prior composition, so the only axis that differs between the two checkpoints is the
head itself.

Scored afterwards with `fintfm-capability --regression-sweep`, comparing both checkpoints
against the `binning_oracle` floor and `ridge_gaussian`, per task 48.3's verify clause.

Push with:
    kaggle kernels push -p scripts/kaggle/task_48_3_quantile_head

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
    # Pinned to the commit that added --head-type/--n-quantiles, which postdates the 0.5.6
    # PyPI release used by every other kernel that doesn't need this flag.
    run([
        sys.executable, "-m", "pip", "install", "--quiet",
        "git+https://github.com/kabartay/fintfm.git@d356358",
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
        "--head-type", "quantile",
        "--n-quantiles", "99",
        "--out", "/kaggle/working/v4-quantile48_3.pt",
    ])

    print("done")


if __name__ == "__main__":
    main()
