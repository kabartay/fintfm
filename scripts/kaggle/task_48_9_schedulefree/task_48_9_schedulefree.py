"""Kaggle kernel script: task 48.9, schedule-free optimisation.

Trains one new checkpoint at the exact §127 reference recipe (matching the existing
`v4-cellattn-labels.pt` control, seed 0), with `--optimizer schedulefree` instead of the
default cosine-annealed AdamW. The accuracy comparison against the control is the first
property task 48.9 asks for; the second -- that a schedule-free run can be *extended* past
its original step budget without restarting, which AdamW's cosine schedule structurally
cannot do -- is already demonstrated by a fast local test
(`test_schedulefree_run_extends_past_its_original_steps_without_restarting`) and is not
re-demonstrated here at GPU scale.

Push with:
    kaggle kernels push -p scripts/kaggle/task_48_9_schedulefree

Then after it completes:
    kaggle kernels status  <username>/<slug>
    kaggle kernels output  <username>/<slug> -p runs/task48-9-kaggle/
"""

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    # Pinned to the commit that added --optimizer/the schedulefree extra, which postdates
    # the 0.5.6 PyPI release used by every other kernel that doesn't need these flags.
    run([
        sys.executable, "-m", "pip", "install", "--quiet",
        "fintfm[schedulefree] @ git+https://github.com/kabartay/fintfm.git@e1b2c45",
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
        "--out", "/kaggle/working/v4-cellattn-schedulefree48_9.pt",
    ])

    print("done")


if __name__ == "__main__":
    main()
