"""Kaggle kernel script: task 48.5, the learnability filter's decisive comparison.

Section 125 already measured the filter's rejection rate (about a third of the production
mixture) and found it discards real signal (the model scores 0.520 AUC on the subset the
filter would reject, against the filter's own 0.392). What was not yet measured is the one
experiment task 48.5 actually asks for: a checkpoint trained *with* the filter on, against
the existing `v4-cellattn-labels.pt` control, at matched compute and seed.

Push with:
    kaggle kernels push -p scripts/kaggle/task_48_5_learnability

Then after it completes:
    kaggle kernels status  <username>/<slug>
    kaggle kernels output  <username>/<slug> -p runs/task48-5-kaggle/
"""

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    run([sys.executable, "-m", "pip", "install", "--quiet", "fintfm==0.5.6"])

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
        "--p-learnability-filter", "1.0",
        "--out", "/kaggle/working/v4-cellattn-learnfilter48_5.pt",
    ])

    print("done")


if __name__ == "__main__":
    main()
