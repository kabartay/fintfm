"""Kaggle T4 kernel: FINDINGS §257's combined binary arm (task 49.22), the 10 large tables.

The arm is the readout ensemble ``xfit,noxfit`` with ``FINTFM_PROBE_BUDGET=50000`` on all 30 binary
TabArena-Lite datasets (repeat 0, fold 0). ``task_cpu_bin_ens2_budget`` fits the 20 tables whose
cross-fitted entry fit took under 700 s on the Mac on Kaggle CPU; this kernel fits the other 10
on a T4, the device the entry's binary run (``fintfm_lp_bin_both_kaggle``) used, airline last.
Both halves use the same fork and fintfm pins and settings and are merged into one entry locally.
§257 adopts the arm only if no dataset fails or hits AutoGluon's 3,600 s limit; with both levers
airline already took 3,008 s on a T4 before the second readout, so it is the one most at risk.

Launch only after the reproduction gate (``task_cpu_ens_gate``) passed at the same fork commit;
this kernel does not rerun the model tests. Binary settings equal ``runs/local_queue/lite_bin_one.sh``
and the gate. The published binary checkpoint is fetched by the fork from ``kabartay/fintfm-binary``
at a pinned commit and SHA-256. One process per dataset; ``summary.json`` records the pins and, per
dataset, the exit code, the wall time and whether ``TimeLimitExceeded`` appeared; results are tarred
into ``/kaggle/working/<run>.tar.gz``. Comparison runs locally (``FINTFM_SKIP_COMPARE=1`` here).
Built from ``scripts/kaggle/task_lite_air_levers``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

FINTFM_COMMIT = "34fb288"
FORK_COMMIT = "6ca0219"
FORK_BRANCH = "fintfm-readout-ensemble"
AUTOGLUON = "1.6.4b20260919"
WORK = Path("/kaggle/working")
#: Stop starting new datasets after this many seconds (Kaggle's limit is 12 h).
BUDGET_S = 11 * 3600
#: Kill one dataset's process after this many seconds, record it and go on.
DATASET_TIMEOUT_S = 2 * 3600

#: The 10 binary Lite tables over 700 s Mac fit time, by that time with airline moved last.
LARGE = [
    "NATICUSdroid",
    "Amazon_employee_access",
    "HR_Analytics_Job_Change_of_Data_Scientists",
    "Diabetes130US",
    "APSFailure",
    "credit_card_clients_default",
    "GiveMeSomeCredit",
    "bank-marketing",
    "kddcup09_appetency",
    "customer_satisfaction_in_airline",
]

BASE_ENV = {
    "FINTFM_MODEL": "FinTFM-LinProbe",
    "FINTFM_PROBLEM_TYPES": "binary",
    "FINTFM_PARTITION": "{}",
    "FINTFM_PARAMS": '{"feature_transform": "squash", "max_features_strategy": "select"}',
    "FINTFM_PROBE": '{"scale": true, "cv": true, "raw": true}',
}
ARM = {"FINTFM_PROBE_ENSEMBLE": "xfit,noxfit", "FINTFM_PROBE_BUDGET": "50000"}

#: Run name -> (datasets, extra environment), in run order.
RUNS = {"fintfm_lp_bin_ens2_budget_t4": (LARGE, ARM)}


def install(fork_root: Path) -> None:
    """Clone the fork at its pin and install it, then fintfm at its pin, into the session.

    Args:
        fork_root: Where the fork is cloned.
    """
    pip = [sys.executable, "-m", "pip", "install", "--quiet"]
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--branch",
            FORK_BRANCH,
            "https://github.com/kabartay/tabarena.git",
            str(fork_root),
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(fork_root), "checkout", "--quiet", FORK_COMMIT], check=True)
    subprocess.run(
        [*pip, f"autogluon.tabular=={AUTOGLUON}", f"autogluon.core[ray]=={AUTOGLUON}"], check=True
    )
    subprocess.run(
        [
            *pip,
            "-e",
            str(fork_root / "packages" / "bencheval"),
            "-e",
            str(fork_root / "packages" / "tabarena"),
        ],
        check=True,
    )
    # After the fork, so the pinned source wins over any fintfm the fork's metadata names.
    subprocess.run(
        [*pip, f"fintfm @ git+https://github.com/kabartay/fintfm.git@{FINTFM_COMMIT}"], check=True
    )


def check_fork(fork_root: Path) -> None:
    """Stop unless the fork has the runner, cross-fitting, the ensemble and the probe budget.

    Args:
        fork_root: The fork's root.

    Raises:
        SystemExit: When the runner, ``probe_crossfit_folds`` or an arm variable is missing.
    """
    runner = fork_root / "run_fintfm_lite.py"
    if not runner.exists():
        sys.exit(f"{runner} missing at fork commit {FORK_COMMIT}")
    sources = "\n".join(
        p.read_text() for p in (fork_root / "packages/tabarena/src/tabarena").rglob("*.py")
    )
    for needle in ("probe_crossfit_folds", "probe_ensemble", *ARM):
        if needle not in sources:
            sys.exit(f"{needle} not found in the fork at commit {FORK_COMMIT}")


def run_env(run_name: str, extra: dict[str, str], dataset: str) -> dict[str, str]:
    """The environment of one dataset's process.

    Args:
        run_name: ``FINTFM_RUN_NAME``.
        extra: The run's lever variables.
        dataset: ``FINTFM_DATASETS``.

    Returns:
        The inherited environment without ``VIRTUAL_ENV`` and any ``FINTFM_*`` variable, plus the
        binary entry's settings, the levers and the device.
    """
    env = {
        k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV" and not k.startswith("FINTFM_")
    }
    env.update(BASE_ENV)
    env.update(extra)
    env.update(
        {
            "FINTFM_RUN_NAME": run_name,
            "FINTFM_DATASETS": dataset,
            "FINTFM_DEVICE": "cuda",
            "FINTFM_SKIP_COMPARE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def run_one(fork_root: Path, run_name: str, extra: dict[str, str], datasets: list[str], deadline: float) -> list[dict]:
    """Fit one run, one process per dataset, and record what happened to each.

    Args:
        fork_root: The fork's root; results land in ``experiments/<run_name>/`` under it.
        run_name: The run's name.
        extra: The run's lever variables.
        datasets: Dataset names to fit.
        deadline: ``time.time()`` after which no new dataset is started.

    Returns:
        One record per dataset: name, exit code (``None`` when skipped for time, ``-1`` when it
        hit ``DATASET_TIMEOUT_S``), seconds, and whether ``TimeLimitExceeded`` was logged.
    """
    log_path = WORK / f"{run_name}.log"
    records = []
    for ds in datasets:
        if time.time() > deadline:
            records.append({"dataset": ds, "exit": None, "seconds": 0, "time_limit": None})
            print(f"[{run_name}] {ds}: skipped, session budget spent", flush=True)
            continue
        start = time.time()
        with log_path.open("a") as log:
            log.write(f"=== {time.strftime('%m-%d %H:%M:%S')} dataset {ds}\n")
            log.flush()
            before = log_path.stat().st_size
            try:
                rc = subprocess.run(
                    [sys.executable, "-P", "run_fintfm_lite.py", "--full"],
                    cwd=fork_root,
                    env=run_env(run_name, extra, ds),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=DATASET_TIMEOUT_S,
                ).returncode
            except subprocess.TimeoutExpired:
                rc = -1
        with log_path.open() as log:
            log.seek(before)
            hit = "TimeLimitExceeded" in log.read()
        seconds = round(time.time() - start)
        records.append({"dataset": ds, "exit": rc, "seconds": seconds, "time_limit": hit})
        print(f"[{run_name}] {ds}: exit {rc}, {seconds} s, time limit hit {hit}", flush=True)
    return records


def tar_results(fork_root: Path, run_name: str) -> None:
    """Tar one run's ``experiments/<run_name>`` directory into ``/kaggle/working/<run_name>.tar.gz``.

    Args:
        fork_root: The fork's root.
        run_name: The run's directory name.
    """
    results = fork_root / "experiments" / run_name
    if results.exists():
        with tarfile.open(WORK / f"{run_name}.tar.gz", "w:gz") as tar:
            tar.add(results, arcname=run_name)


def main() -> None:
    """Install, check the fork and the GPU, run phase 1 then phase 2, tar results, write summary."""
    started = time.time()
    fork_root = Path("/kaggle/tmp/tabarena")
    fork_root.parent.mkdir(parents=True, exist_ok=True)
    install(fork_root)
    check_fork(fork_root)
    head = subprocess.run(
        ["git", "-C", str(fork_root), "rev-parse", "--short=7", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != FORK_COMMIT:
        sys.exit(f"fork HEAD {head} != pinned {FORK_COMMIT}")
    import torch

    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        sys.exit("no GPU in this session (quota exhausted?) -- refusing a CPU run")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    deadline = started + BUDGET_S
    summary = {
        "fintfm": FINTFM_COMMIT,
        "fork": FORK_COMMIT,
        "fork_branch": FORK_BRANCH,
        "autogluon": AUTOGLUON,
        "gpu": torch.cuda.get_device_name(0),
        "base_env": BASE_ENV,
        "runs": {},
    }
    for run_name, (datasets, extra) in RUNS.items():
        records = run_one(fork_root, run_name, extra, list(datasets), deadline)
        summary["runs"][run_name] = {"env": extra, "datasets": records}
        tar_results(fork_root, run_name)
        (WORK / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: [r["exit"] for r in v["datasets"]] for k, v in summary["runs"].items()}))
    print("done")


if __name__ == "__main__":
    main()
