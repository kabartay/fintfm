"""Kaggle kernel: measure a TPU session's host CPU for FinTFM's CPU inference (2026-11-10).

Does not use the TPU. FinTFM has no XLA path; the question is whether the TPU VM's host CPUs
can take the CPU-bound V4FinBench and credit-suite scoring off the shared Mac. Prints the host's
core count, memory and CPU model, then times zero-shot V4FinBench scoring of the published
checkpoint on a 30,000-row development slice (fold 0, logistic regression beside it) at two
torch thread counts. SMOKE-TEST only: the slice says nothing about model quality. Pinned to
8348c6c; the checkpoint is the public kabartay/fintfm-binary file, verified by SHA-256; the data
is V4FinBench's public Kaggle dataset (CC BY 4.0).
"""

import hashlib
import os
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

COMMIT = "8348c6c"
CKPT_URL = "https://huggingface.co/kabartay/fintfm-binary/resolve/f116bfd43a2b15c65ed3551ea8c38e3364629ddc/v4-cellattn-labels.pt"
CKPT_SHA256 = "d231b062483c9261aae4c503ad5359db295fc7206fa0183f51763e4953c518d2"
MAX_ROWS = 30_000


def host_facts() -> None:
    """Print the host's CPU count, memory and CPU model."""
    print(f"os.cpu_count: {os.cpu_count()}", flush=True)
    print(f"sched_getaffinity: {len(os.sched_getaffinity(0))}", flush=True)
    meminfo = Path("/proc/meminfo").read_text().splitlines()
    print(next(line for line in meminfo if line.startswith("MemTotal")), flush=True)
    print(next(line for line in meminfo if line.startswith("MemAvailable")), flush=True)
    model = [line for line in Path("/proc/cpuinfo").read_text().splitlines()
             if line.startswith("model name")]
    print(f"cpu model: {model[0] if model else platform.processor()}", flush=True)


def time_slice(root: Path, ckpt: Path, threads: int) -> None:
    """Score fold 0 of the development slice on CPU with ``threads`` torch threads."""
    import torch

    from fintfm.experiments.v4_protocol import run, summarise
    torch.set_num_threads(threads)
    out = Path(f"/kaggle/working/slice_t{threads}")
    t0 = time.perf_counter()
    record = run(str(ckpt), out, horizon=0, folds=(0,), max_rows=MAX_ROWS, root=root,
                 with_boosting=False, classical=("logistic_regression",), device="cpu")
    wall = time.perf_counter() - t0
    print(summarise(record), flush=True)
    print(f"threads {threads}: wall {wall:.1f} s for the {MAX_ROWS:,}-row slice", flush=True)


def main() -> None:
    """Report the host, install the pinned fintfm and time the slice at two thread counts."""
    host_facts()
    hits = sorted(Path("/kaggle/input").rglob("company_years_h1.parquet"))
    if len(hits) != 1:
        sys.exit(f"expected one company_years_h1.parquet under /kaggle/input, found {hits}")
    root = hits[0].parent
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    f"fintfm @ git+https://github.com/kabartay/fintfm.git@{COMMIT}"], check=True)
    import torch
    print(f"torch {torch.__version__}; default threads {torch.get_num_threads()}", flush=True)
    ckpt = Path("/kaggle/working/v4-cellattn-labels.pt")
    urllib.request.urlretrieve(CKPT_URL, ckpt)
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if digest != CKPT_SHA256:
        sys.exit(f"checkpoint hash {digest} != {CKPT_SHA256}")
    n = len(os.sched_getaffinity(0))
    for threads in sorted({min(16, n), n}):
        time_slice(root, ckpt, threads)


if __name__ == "__main__":
    main()
