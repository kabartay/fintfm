"""Kaggle kernel: TabArena-Lite binary large-table levers for FinTFM-LinProbe on a T4.

Why: on the Mac CPU the three local airline arms (``air-depth``, ``air-budget``, ``air-both`` in
``runs/local_queue/queue.txt``) all stopped with AutoGluon's ``TimeLimitExceeded`` (2026-11-16):
each lever makes one bagged fold slower than its share of the 3,600 s limit. The entry is GPU work
(the fork's ``fintfm_linprobe/hpo.py``), so the levers are measured on the device the entry
declares.

Phase 1, ``customer_satisfaction_in_airline`` only, four arms on the same GPU:

1. ``fintfm_lp_air_ctrl_kaggle``: the binary entry as measured in §254 (cross-fitting on by default).
2. ``fintfm_lp_air_depth_kaggle``: plus ``FINTFM_PARTITION_SCALE_N=on``.
3. ``fintfm_lp_air_budget_kaggle``: plus ``FINTFM_PROBE_BUDGET=50000``.
4. ``fintfm_lp_air_both_kaggle``: plus both.

Phase 2, ``fintfm_lp_bin_both_kaggle``: all 30 binary Lite datasets with both levers, airline last
so it is the one skipped when time runs out. Both levers only act on large tables (the partition
depth grows with n; the probe budget caps readout rows at 20,000 by default), so on small datasets
this arm should equal the control. Phase 2 runs whatever phase 1 shows; the decision whether to
use it is taken locally, on the paired comparison.

Binary settings equal ``runs/local_queue/lite_bin_one.sh``: ``FINTFM_PARTITION={}``,
``FINTFM_PARAMS={"feature_transform": "squash", "max_features_strategy": "select"}``,
``FINTFM_PROBE={"scale": true, "cv": true, "raw": true}``. The published binary checkpoint is
fetched by the fork itself from ``kabartay/fintfm-binary`` at a pinned commit and SHA-256
(``fintfm/checkpoints.py`` in the fork). One process per dataset, each with its own time limit;
each run's raw results are tarred into ``/kaggle/working/<run>.tar.gz``; ``summary.json`` records
the pins and, per dataset, the exit code, the wall time and whether ``TimeLimitExceeded`` appeared.
Comparison runs locally (``FINTFM_SKIP_COMPARE=1`` here).

Pinned: fintfm at ``FINTFM_COMMIT``, the fork (``kabartay/tabarena``, branch ``fintfm-large-levers``)
at ``FORK_COMMIT``, AutoGluon at the nightly the local runs used. Built from
``scripts/kaggle/task_lite_xfit_mcreg``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

FINTFM_COMMIT = "e09fd6f"
FORK_COMMIT = "6981265"
FORK_BRANCH = "fintfm-large-levers"
AUTOGLUON = "1.6.4b20260919"
WORK = Path("/kaggle/working")
#: Stop starting new datasets after this many seconds (Kaggle's limit is 12 h).
BUDGET_S = 11 * 3600
#: Kill one dataset's process after this many seconds, record it and go on.
DATASET_TIMEOUT_S = 2 * 3600

AIRLINE = "customer_satisfaction_in_airline"
#: The 30 binary Lite datasets, in ``lite_bin_probe.sh``'s order with airline moved last.
BINARY_DATASETS = [
    "Amazon_employee_access",
    "Bank_Customer_Churn",
    "Diabetes130US",
    "E-CommereShippingData",
    "Fitness_Club",
    "GiveMeSomeCredit",
    "HR_Analytics_Job_Change_of_Data_Scientists",
    "Is-this-a-good-customer",
    "Marketing_Campaign",
    "NATICUSdroid",
    "bank-marketing",
    "blood-transfusion-service-center",
    "churn",
    "coil2000_insurance_policies",
    "credit-g",
    "credit_card_clients_default",
    "diabetes",
    "hazelnut-spread-contaminant-detection",
    "heloc",
    "in_vehicle_coupon_recommendation",
    "jm1",
    "online_shoppers_intention",
    "polish_companies_bankruptcy",
    "qsar-biodeg",
    "seismic-bumps",
    "taiwanese_bankruptcy_prediction",
    "APSFailure",
    "Bioresponse",
    "kddcup09_appetency",
    AIRLINE,
]

BASE_ENV = {
    "FINTFM_MODEL": "FinTFM-LinProbe",
    "FINTFM_PROBLEM_TYPES": "binary",
    "FINTFM_PARTITION": "{}",
    "FINTFM_PARAMS": '{"feature_transform": "squash", "max_features_strategy": "select"}',
    "FINTFM_PROBE": '{"scale": true, "cv": true, "raw": true}',
}
DEPTH = {"FINTFM_PARTITION_SCALE_N": "on"}
BUDGET = {"FINTFM_PROBE_BUDGET": "50000"}

#: Run name -> (datasets, extra environment), in run order.
RUNS = {
    "fintfm_lp_air_ctrl_kaggle": ([AIRLINE], {}),
    "fintfm_lp_air_depth_kaggle": ([AIRLINE], DEPTH),
    "fintfm_lp_air_budget_kaggle": ([AIRLINE], BUDGET),
    "fintfm_lp_air_both_kaggle": ([AIRLINE], {**DEPTH, **BUDGET}),
    "fintfm_lp_bin_both_kaggle": (BINARY_DATASETS, {**DEPTH, **BUDGET}),
}


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
    """Stop unless the fork has the runner, cross-fitting and both large-table levers.

    Args:
        fork_root: The fork's root.

    Raises:
        SystemExit: When the runner, ``probe_crossfit_folds`` or a lever variable is missing.
    """
    runner = fork_root / "run_fintfm_lite.py"
    if not runner.exists():
        sys.exit(f"{runner} missing at fork commit {FORK_COMMIT}")
    sources = "\n".join(
        p.read_text() for p in (fork_root / "packages/tabarena/src/tabarena").rglob("*.py")
    )
    for needle in ("probe_crossfit_folds", *DEPTH, *BUDGET):
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
