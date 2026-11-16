"""Kaggle CPU kernel: sample-efficiency probe with held-out-slice calibrated boosting, per panel.

Openspec ``sample-efficiency-regime`` tasks 13.7 (``gboost_platt_holdout`` and
``gboost_isotonic_holdout`` arms) and 13.8 (``--panel``). Prepared 2026-11-16, NOT launched.

What this kernel runs, one subprocess per panel, each with its own timeout:

* ``taiwan``: the UCI Taiwanese bankruptcy panel (CC BY 4.0), downloaded inside the kernel.
* ``v4finbench``: V4FinBench (CC BY 4.0) from its public Kaggle dataset, company-grouped
  subsample of ``--v4-pool-rows 200000`` company-years.

``polish3`` is not run here: the local queue job ``h-sampleeff`` already measured it with
``--calibration-frac 0.3`` (``runs/sampleeff_1115``).

Nothing local is uploaded. The checkpoint is the published ``kabartay/fintfm-binary`` file at a
pinned Hugging Face revision, its SHA-256 checked before use (the same file and hash the CPU kernels
``make_cpu_kernels.py`` and ``make_v4_inference_kernels.py`` use). fintfm is installed from GitHub at
``COMMIT``. CPU only: the kernel exits if it finds an accelerator. Outputs go under
``/kaggle/working/<panel>/sample_efficiency.json`` plus ``summary.json`` and ``<panel>.log``.
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
CALIBRATION_FRAC = 0.3
V4_POOL_ROWS = 200_000
#: Stop starting new panels after this many seconds (Kaggle's session limit is 12 h).
BUDGET_S = 11 * 3600
#: Kill one panel's process after this many seconds, record it and go on.
PANEL_TIMEOUT_S = 4 * 3600
#: Panels in run order; the small public one first so a V4FinBench failure cannot cost it.
PANELS = ("taiwan", "v4finbench")
#: Keys every panel's record must carry for the Verify of 13.7 / 13.8 to be readable.
REQUIRED_KEYS = (
    "panel",
    "split",
    "panel_rows",
    "pool_rows",
    "test_rows",
    "calibration_frac",
    "sizes",
)
REQUIRED_ARMS = (
    "fintfm",
    "gboost",
    "gboost_sigmoid",
    "gboost_isotonic",
    "gboost_platt_holdout",
    "gboost_isotonic_holdout",
    "calibration_holdout",
)


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


def fetch_checkpoint(dest: Path) -> Path:
    """Download the published checkpoint and refuse to continue on a hash mismatch.

    Args:
        dest: Where to write the file.

    Returns:
        ``dest``.

    Raises:
        SystemExit: When the SHA-256 differs from :data:`CKPT_SHA256`.
    """
    urllib.request.urlretrieve(CKPT_URL, dest)
    got = sha256(dest)
    if got != CKPT_SHA256:
        sys.exit(f"checkpoint SHA-256 {got} != pinned {CKPT_SHA256}; refusing to run")
    print(f"checkpoint {dest} SHA-256 ok", flush=True)
    return dest


def panel_command(panel: str, checkpoint: Path, v4_root: Path | None) -> list[str]:
    """The ``fintfm-ablate`` command line for one panel.

    Args:
        panel: ``taiwan`` or ``v4finbench``.
        checkpoint: The verified checkpoint.
        v4_root: Directory of the V4FinBench parquet files (required for ``v4finbench``).

    Returns:
        The argv list, run with the kernel's interpreter.

    Raises:
        ValueError: For an unknown panel, or ``v4finbench`` without a root.
    """
    cmd = [
        sys.executable,
        "-P",
        "-m",
        "fintfm.experiments.prior_ablation",
        "--sample-efficiency",
        str(checkpoint),
        "--panel",
        panel,
        "--calibration-frac",
        str(CALIBRATION_FRAC),
        "--out",
        str(WORK / panel),
    ]
    if panel == "v4finbench":
        if v4_root is None:
            raise ValueError("v4finbench needs the parquet directory")
        cmd += ["--v4-pool-rows", str(V4_POOL_ROWS), "--v4-root", str(v4_root)]
    elif panel != "taiwan":
        raise ValueError(f"unknown panel {panel!r}")
    return cmd


def check_record(path: Path, panel: str) -> list[str]:
    """Check one panel's JSON carries the fields 13.7 and 13.8 need.

    Args:
        path: ``sample_efficiency.json`` of the panel.
        panel: The panel the record must name.

    Returns:
        A list of problems; empty when the record is complete.
    """
    if not path.exists():
        return ["no sample_efficiency.json"]
    rec = json.loads(path.read_text())
    problems = [f"missing key {k}" for k in REQUIRED_KEYS if k not in rec]
    if rec.get("panel") != panel:
        problems.append(f"panel {rec.get('panel')!r} != {panel!r}")
    sizes = rec.get("sizes", {})
    if not sizes:
        problems.append("no sizes")
    for size, row in sizes.items():
        problems += [f"n={size}: missing {a}" for a in REQUIRED_ARMS if a not in row]
    return problems


def main() -> None:
    """Check CPU-only, fetch the checkpoint, install fintfm, run each panel, write summary."""
    started = time.time()
    if (
        Path("/dev/accel0").exists()
        or os.environ.get("TPU_NAME")
        or os.environ.get("COLAB_TPU_ADDR")
    ):
        sys.exit("a TPU is attached -- this kernel is CPU only")
    hits = sorted(Path("/kaggle/input").rglob("company_years_h1.parquet"))
    if len(hits) != 1:
        sys.exit(f"expected one company_years_h1.parquet under /kaggle/input, found {hits}")
    v4_root = hits[0].parent
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            f"fintfm[bench] @ git+https://github.com/kabartay/fintfm.git@{COMMIT}",
        ],
        check=True,
    )
    import torch

    if torch.cuda.is_available():
        sys.exit("a GPU is attached -- this kernel is CPU only")
    torch.set_num_threads(len(os.sched_getaffinity(0)))
    print(f"cpus {len(os.sched_getaffinity(0))}, torch {torch.__version__}", flush=True)
    checkpoint = fetch_checkpoint(WORK / "v4-cellattn-labels.pt")
    summary: dict = {
        "fintfm": COMMIT,
        "checkpoint_sha256": CKPT_SHA256,
        "calibration_frac": CALIBRATION_FRAC,
        "v4_pool_rows": V4_POOL_ROWS,
        "device": "cpu",
        "cpus": os.cpu_count(),
        "panels": [],
    }
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"})
    for panel in PANELS:
        if time.time() - started > BUDGET_S:
            summary["panels"].append({"panel": panel, "exit": None, "seconds": 0})
            print(f"{panel}: skipped, session budget spent", flush=True)
            continue
        out = WORK / panel
        out.mkdir(parents=True, exist_ok=True)
        start = time.time()
        with (WORK / f"{panel}.log").open("w") as log:
            try:
                rc = subprocess.run(
                    panel_command(panel, checkpoint, v4_root),
                    cwd=out,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=PANEL_TIMEOUT_S,
                ).returncode
            except subprocess.TimeoutExpired:
                rc = -1
        problems = (
            check_record(out / "sample_efficiency.json", panel)
            if rc == 0
            else ["not run to completion"]
        )
        seconds = round(time.time() - start)
        summary["panels"].append(
            {"panel": panel, "exit": rc, "seconds": seconds, "problems": problems}
        )
        print(f"{panel}: exit {rc}, {seconds} s, problems {problems}", flush=True)
        (WORK / "summary.json").write_text(json.dumps(summary, indent=1))
    print("done")


if __name__ == "__main__":
    main()
