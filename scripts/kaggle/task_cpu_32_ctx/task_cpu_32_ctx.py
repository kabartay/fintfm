"""Kaggle CPU kernel: the binary-path context-strategy sweep for openspec tasks 32.3, 32.4 and 32.6.

Runs the local queue job ``h-binctx`` (``runs/local_queue/queue.txt``) with the same flags:
``fintfm-binctx --model v4-cellattn-labels.pt --out binary-context-sweep`` (defaults: strategies
uniform, balanced, hybrid, prototype, retrieval; context sizes 500, 1000, 2000; seeds 0, 1, 2;
Polish 1y/3y/5y and Taiwan; ``--n-ensemble 8 --n-boot 2000``; 180 cells). The record is one Holm
family per metric, exactly as the local command would write it, so the run is one subprocess and
the output is ``/kaggle/working/binary-context-sweep/binary_context_sweep.json``.

The survival-path job ``h-ctxsweep175`` is NOT here: its checkpoint ``runs/v4-hazard-ldp.pt``
(SHA-256 db4696b8...) was trained locally and exists in no public place, so it cannot go to Kaggle.

Inputs: the public ``kabartay/fintfm-binary`` checkpoint (the file ``runs/v4-cellattn-labels.pt``,
SHA-256 checked before any work) and the UCI Polish and Taiwanese bankruptcy tables (CC BY 4.0),
which the loaders download inside the session. Nothing is uploaded from a local machine. fintfm is
installed from GitHub at a pinned commit. ``summary.json`` records the exit code and wall time.
Coverage here is measured against held-out labels of observed bankruptcies on these panels, so it
is validation of discrimination on those panels, not verification against a solver.
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

COMMIT = "e475151"
CKPT_URL = (
    "https://huggingface.co/kabartay/fintfm-binary/resolve/"
    "f116bfd43a2b15c65ed3551ea8c38e3364629ddc/v4-cellattn-labels.pt"
)
CKPT_SHA256 = "d231b062483c9261aae4c503ad5359db295fc7206fa0183f51763e4953c518d2"
WORK = Path("/kaggle/working")
OUT_NAME = "binary-context-sweep"
#: Session limit is 12 h; leave an hour for install, download and the final write.
TIMEOUT_S = 11 * 3600
#: Same flags as the queued line; only the paths differ.
SWEEP_ARGS = ("--model", "{ckpt}", "--out", "{out}")


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
        [sys.executable, "-m", "pip", "install", "--quiet",
         f"fintfm @ git+https://github.com/kabartay/fintfm.git@{COMMIT}"],
        check=True,
    )
    out = WORK / OUT_NAME
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    summary = {"commit": COMMIT, "checkpoint_sha256": digest, "command": sweep_command(ckpt, out)}
    try:
        with (WORK / "run.log").open("w") as log:
            proc = subprocess.run(
                summary["command"], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                timeout=TIMEOUT_S, check=False,
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
