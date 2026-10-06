"""Kaggle kernel: the matched control Section 158 needs to re-adjudicate Sections 144-147.

Those four ablations trained at the CLI default `--max-classes 10` but were scored against
`v4-cellattn-labels.pt`, trained at 2. Section 158 showed by hashing that head size (and what it
does to initialisation) is the ONLY difference -- seeded task streams and initialisations are
identical across every commit involved. This kernel removes it: the exact reference recipe the
four ablations used, seed 0, `--max-classes 10`, no change under test. Each ablation is then
re-scored against this checkpoint instead.

Pinned to e1b2c45, the commit 48.6 and 48.10 trained on (hash-identical to the others).

Push with:
    kaggle kernels push -p scripts/kaggle/task_158_matched_control
Then:
    kaggle kernels output muhakabartay/fintfm-task-158-matched-control -p runs/task158-kaggle/
"""

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    run([
        sys.executable, "-m", "pip", "install", "--quiet",
        "git+https://github.com/kabartay/fintfm.git@e1b2c45",
    ])
    import torch

    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run([
        sys.executable, "-u", "-m", "fintfm.modeling.train",
        "--steps", "6000", "--batch-size", "8", "--n-rows-choices", "256,512,1024",
        "--d-cell", "48", "--d-model", "128", "--n-layers", "4", "--n-col-layers", "2",
        "--n-cell-blocks", "1", "--cell-labels", "--column-id-dim", "16",
        "--feature-chunk", "8", "--max-features", "136", "--p-financial", "1.0",
        # The point of this run: the head size the four ablations actually had.
        "--max-classes", "10",
        "--seed", "0", "--device", device, "--checkpoint-every", "500",
        "--out", "/kaggle/working/v4-cellattn-labels-mc10.pt",
    ])
    print("done")


if __name__ == "__main__":
    main()
