"""Kaggle kernel script: task 39.28, a second-seed replication of section 91's ablation.

Reproduces the exact §91 reference recipe (`cell-attention-and-task-inference`, two
6,000-step runs identical but for `--cell-labels`) at `--seed 1`, on Kaggle's free GPU quota
instead of paid HF Jobs compute. Both checkpoints are written to `/kaggle/working/` and are
fetched afterwards with `kaggle kernels output <slug> -p <local_dir>` -- no HF upload or
token needed, since Kaggle's own kernel-output mechanism does the job.

Push with:
    kaggle kernels push -p scripts/kaggle

Then after it completes:
    kaggle kernels status  <username>/<slug>
    kaggle kernels output  <username>/<slug> -p runs/cellattn-s1-kaggle/
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

    common = [
        sys.executable, "-u", "-m", "fintfm.modeling.train",
        "--steps", "6000",
        "--batch-size", "8",
        "--n-rows-choices", "256,512,1024",
        "--d-cell", "48",
        "--d-model", "128",
        "--n-layers", "4",
        "--n-col-layers", "2",
        "--n-cell-blocks", "1",
        "--column-id-dim", "16",
        "--feature-chunk", "8",
        "--max-features", "136",
        "--p-financial", "1.0",
        "--seed", "1",
        "--device", device,
        "--checkpoint-every", "500",
    ]

    run([*common, "--cell-labels", "--out", "/kaggle/working/v4-cellattn-labels-s1.pt"])
    run([*common, "--out", "/kaggle/working/v4-cellattn-nolabels-s1.pt"])

    print("both arms done")


if __name__ == "__main__":
    main()
