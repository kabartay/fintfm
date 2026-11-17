"""Kaggle CPU kernel: task 32.7's binary context comparison, panel ``polish5``, seed 1 (FINDINGS §289).

§288's merged sweep could not reject anything: 2,000 resamples over a 360-comparison Holm family
put the adjusted-p floor at 0.18. This kernel runs one (panel, seed) slice of the pre-registered
smaller family: strategies uniform, prototype and retrieval, context sizes 500, 1000 and 2000,
``--n-ensemble 8``, ``--compare-against uniform --n-boot 20000 --family-size 72`` (Holm floor 0.0036
over the merged 72 comparisons per metric). The record keeps each cell's held-out probabilities, so
the test can be recomputed after the twelve slices are merged.

Inputs: the public ``kabartay/fintfm-binary`` checkpoint (SHA-256 checked before any work) and the
UCI Polish and Taiwanese bankruptcy tables (CC BY 4.0), downloaded inside the session. Nothing is
uploaded from a local machine. fintfm is installed from GitHub at a pinned commit. ``summary.json``
records the exit code and wall time.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

COMMIT = "d6a72de"
CKPT_URL = (
    "https://huggingface.co/kabartay/fintfm-binary/resolve/"
    "f116bfd43a2b15c65ed3551ea8c38e3364629ddc/v4-cellattn-labels.pt"
)
CKPT_SHA256 = "d231b062483c9261aae4c503ad5359db295fc7206fa0183f51763e4953c518d2"
WORK = Path("/kaggle/working")
OUT_NAME = "binary-context-sweep"
#: Session limit is 12 h; leave an hour for install, download and the final write.
TIMEOUT_S = 11 * 3600
#: The pre-registered flags of FINDINGS §289; only the paths differ.
SWEEP_ARGS = (
    "--model",
    "{ckpt}",
    "--out",
    "{out}",
    "--panels",
    "polish5",
    "--seeds",
    "1",
    "--strategies",
    "uniform,prototype,retrieval",
    "--compare-against",
    "uniform",
    "--n-boot",
    "20000",
    "--family-size",
    "72",
    "--n-ensemble",
    "8",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks.

    Args:
        path: File to hash.

    Returns:
        Lower-case hex digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sweep_command(ckpt: Path, out: Path) -> list[str]:
    """Build the sweep command, equal to the queued ``fintfm-binctx`` line.

    Args:
        ckpt: Verified checkpoint path.
        out: Output directory.

    Returns:
        The argument vector.
    """
    args = [a.format(ckpt=ckpt, out=out) for a in SWEEP_ARGS]
    return [sys.executable, "-m", "fintfm.experiments.binary_context_sweep", *args]


def main() -> None:
    """Install the pinned fintfm, verify the checkpoint, run the sweep, write ``summary.json``."""
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    cwd = WORK / "run"  # the dataset loaders cache under cwd/data/cache
    cwd.mkdir(exist_ok=True)
    ckpt = WORK / "v4-cellattn-labels.pt"
    urllib.request.urlretrieve(CKPT_URL, ckpt)
    digest = sha256_file(ckpt)
    if digest != CKPT_SHA256:
        sys.exit(f"checkpoint hash {digest} != {CKPT_SHA256}")
    print(f"checkpoint SHA-256 verified: {digest}", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            f"fintfm @ git+https://github.com/kabartay/fintfm.git@{COMMIT}",
        ],
        check=True,
    )
    out = WORK / OUT_NAME
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    summary = {"commit": COMMIT, "checkpoint_sha256": digest, "command": sweep_command(ckpt, out)}
    try:
        with (WORK / "run.log").open("w") as log:
            proc = subprocess.run(
                summary["command"],
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=TIMEOUT_S,
                check=False,
            )
        summary["exit_code"] = proc.returncode
    except subprocess.TimeoutExpired:
        summary["exit_code"] = "timeout"
    summary["seconds"] = round(time.time() - t0)
    summary["record_written"] = (out / "binary_context_sweep.json").exists()
    (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
    ckpt.unlink()  # the public file; no need to ship it back
    print(json.dumps(summary, indent=2), flush=True)
    print((WORK / "run.log").read_text()[-4000:], flush=True)


if __name__ == "__main__":
    main()
