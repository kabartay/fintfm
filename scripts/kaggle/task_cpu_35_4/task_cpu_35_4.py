"""Kaggle CPU kernel: openspec task 35.4, TabBench public academic set, checkpoint vs untrained control.

Runs exactly the measurement 35.4's Verify asks for (the suite's protocol: 5 stratified shuffle folds,
20% held out, ROC-AUC and accuracy per fold, native coverage per arm, checkpoint plus a seeded
untrained control) by calling the committed harness ``fintfm.experiments.tabbench_protocol``, one
subprocess per OpenML dataset, each with its own timeout. 2026-11-16. Prepared, not launched.

Pinned to fintfm e475151. The checkpoint is the public ``kabartay/fintfm-binary`` file (Apache-2.0)
at a fixed Hugging Face revision; its SHA-256 equals the local ``runs/v4-cellattn-labels.pt`` and is
checked before anything runs. The 90 ids are ``configs/tabbench/eligible_ids.json``, embedded here
and hash-checked, because the repository's ``configs/`` is not part of the pip install. Data comes
from OpenML's public API inside the kernel; nothing is uploaded to Kaggle. The benchmark suite's own
code is not installed or imported (the harness reimplements the folds with scikit-learn).

INTERNAL ONLY: the benchmark code's licence exempts private benchmarking only while no result is
communicated externally (docs/research/tabbench_v2.md, task 35.2, still open). A private Kaggle
kernel's output stays with the account; do not make the kernel or its output public.

Outputs under /kaggle/working: ``tabbench-35-4/datasets/openml-<id>.json`` (per-fold ROC-AUC and
accuracy per arm, or a skip reason), ``tabbench-35-4/tabbench_protocol.json`` (aggregate, native
coverage, paired difference), ``tabbench-35-4.tar.gz``, ``summary.json``, ``run.log``. If the session
budget ends first, ``summary.json`` lists ``unfinished_ids``; finish them in a second kernel by
editing ``IDS_JSON`` (and the hash check) and the slug, then merge the ``datasets/`` folders and
rerun the harness with ``--aggregate-only`` locally.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

FINTFM_COMMIT = "e475151"
CKPT_URL = (
    "https://huggingface.co/kabartay/fintfm-binary/resolve/"
    "f116bfd43a2b15c65ed3551ea8c38e3364629ddc/v4-cellattn-labels.pt"
)
CKPT_SHA256 = "d231b062483c9261aae4c503ad5359db295fc7206fa0183f51763e4953c518d2"
#: Exact text of configs/tabbench/eligible_ids.json at the pinned commit (no trailing newline).
IDS_JSON = "[3, 15, 29, 31, 37, 44, 137, 179, 251, 310, 334, 335, 741, 803, 930, 1037, 1043, 1046, 1049, 1050, 1063, 1067, 1068, 1120, 1220, 1444, 1462, 1464, 1471, 1480, 1487, 1489, 1494, 1496, 1510, 6332, 23381, 40646, 40649, 40701, 40900, 40981, 40983, 40994, 41156, 41721, 41865, 41875, 41882, 42178, 43892, 43895, 44089, 44122, 44123, 44124, 44130, 44160, 44162, 44186, 44489, 44528, 45547, 45553, 45558, 45578, 46905, 46910, 46911, 46916, 46919, 46920, 46922, 46924, 46925, 46927, 46929, 46930, 46932, 46935, 46937, 46938, 46940, 46947, 46950, 46951, 46956, 46962, 46969, 46979]"
IDS_SHA256 = "a9e37319ee9d9024fb8bb67d1f813426339bcb83c58a6088306461742671ce4a"
#: The suite has 189 datasets; the harness records the other 99 as not listed, never as scored.
SUITE_SIZE = 189
WORK = Path("/kaggle/working")
OUT = WORK / "tabbench-35-4"
#: Stop starting new datasets after this many seconds (Kaggle's session limit is 12 h).
BUDGET_S = 11 * 3600 + 1800
#: Kill one dataset's process after this many seconds, record it and go on.
DATASET_TIMEOUT_S = 3 * 3600
THREADS = "4"


def sha256(path: Path) -> str:
    """SHA-256 of a file, read in chunks.

    Args:
        path: The file to hash.

    Returns:
        The lower-case hex digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pinned_ids() -> list[int]:
    """The embedded id list, after checking its hash.

    Returns:
        The 90 OpenML dataset ids.

    Raises:
        SystemExit: When the embedded text does not hash to the pinned value.
    """
    got = hashlib.sha256(IDS_JSON.encode()).hexdigest()
    if got != IDS_SHA256:
        sys.exit(f"embedded ids SHA-256 {got} != pinned {IDS_SHA256}")
    return [int(i) for i in json.loads(IDS_JSON)]


def harness_command(checkpoint: Path, ids_file: Path, out: Path = OUT) -> list[str]:
    """The harness command for one id file (35.4's one-command protocol, Kaggle paths).

    Args:
        checkpoint: The verified checkpoint.
        ids_file: JSON list of OpenML ids to run.
        out: Output directory shared by every call, so finished datasets are skipped.

    Returns:
        The argument vector.
    """
    return [
        sys.executable,
        "-m",
        "fintfm.experiments.tabbench_protocol",
        "--model",
        str(checkpoint),
        "--out",
        str(out),
        "--ids",
        str(ids_file),
        "--suite-size",
        str(SUITE_SIZE),
    ]


def fetch_checkpoint() -> Path:
    """Download the public checkpoint and verify its SHA-256.

    Returns:
        The local path.

    Raises:
        SystemExit: When the hash differs.
    """
    ckpt = WORK / "v4-cellattn-labels.pt"
    urllib.request.urlretrieve(CKPT_URL, ckpt)
    got = sha256(ckpt)
    if got != CKPT_SHA256:
        sys.exit(f"checkpoint SHA-256 {got} != pinned {CKPT_SHA256}; refusing to run")
    print(f"checkpoint SHA-256 ok ({got})", flush=True)
    return ckpt


def main() -> None:
    """Verify, install, run every dataset in its own process, aggregate, tar, write summary."""
    started = time.time()
    ids = pinned_ids()
    ckpt = fetch_checkpoint()
    pkg = f"fintfm[overlap] @ git+https://github.com/kabartay/fintfm.git@{FINTFM_COMMIT}"
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", pkg], check=True)
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env.update({"OMP_NUM_THREADS": THREADS, "MKL_NUM_THREADS": THREADS, "PYTHONUNBUFFERED": "1"})
    deadline = started + BUDGET_S
    summary: dict = {
        "task": "35.4",
        "fintfm": FINTFM_COMMIT,
        "checkpoint_sha256": CKPT_SHA256,
        "ids_sha256": IDS_SHA256,
        "cpus": os.cpu_count(),
        "datasets": [],
        "unfinished_ids": [],
        "internal_only": True,
    }
    tmp = WORK / "ids_tmp"
    tmp.mkdir(exist_ok=True)
    (tmp / "all.json").write_text(json.dumps(ids))
    log_path = WORK / "run.log"
    for did in ids:
        if time.time() > deadline:
            summary["unfinished_ids"].append(did)
            print(f"openml-{did}: skipped, session budget spent", flush=True)
            continue
        one = tmp / f"{did}.json"
        one.write_text(json.dumps([did]))
        start = time.time()
        with log_path.open("a") as log:
            log.write(f"=== {time.strftime('%m-%d %H:%M:%S')} openml-{did}\n")
            log.flush()
            try:
                rc = subprocess.run(
                    harness_command(ckpt, one),
                    cwd=WORK,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=DATASET_TIMEOUT_S,
                ).returncode
            except subprocess.TimeoutExpired:
                rc = -1
        finished = (OUT / "datasets" / f"openml-{did}.json").exists()
        if not finished:
            summary["unfinished_ids"].append(did)
        seconds = round(time.time() - start)
        summary["datasets"].append({"id": did, "exit": rc, "seconds": seconds, "record": finished})
        print(f"openml-{did}: exit {rc}, {seconds} s, record {finished}", flush=True)
        (WORK / "summary.json").write_text(json.dumps(summary, indent=1))
    # The per-dataset calls each wrote a one-id aggregate; rebuild it over the whole list.
    with log_path.open("a") as log:
        subprocess.run(
            [*harness_command(ckpt, tmp / "all.json"), "--aggregate-only"],
            cwd=WORK,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    summary["total_seconds"] = round(time.time() - started)
    (WORK / "summary.json").write_text(json.dumps(summary, indent=1))
    with tarfile.open(WORK / "tabbench-35-4.tar.gz", "w:gz") as tar:
        tar.add(OUT, arcname="tabbench-35-4")
    ckpt.unlink()  # the public file; no need to ship it back
    print("done")


if __name__ == "__main__":
    main()
