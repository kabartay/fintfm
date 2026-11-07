"""Kaggle kernel: TabArena-Lite arm with tree-partitioned contexts against the default, same GPU.

Fix 4 of the TabArena loss analysis (``docs/results/drafts/tabarena_loss_analysis_2026-11.md``,
FINDINGS §204): ``PartitionedTFMClassifier`` (``src/fintfm/inference/large_n.py``), the lever that
gave +0.0375 mean AP on V4FinBench with 5 of 5 folds positive (§169), run on TabArena-Lite for
breadth only. On the Mac (2026-11-06) the arm finished 3 of 27 datasets on MPS before
HR_Analytics_Job_Change_of_Data_Scientists failed with an MPS out-of-memory error at 85 GB of
Metal allocations on a 64 GB machine, with swap near full; this kernel runs it on CUDA.

Two arms, the 27 binary Lite datasets FinTFM covers, split 0, the published checkpoint through
the TabArena fork's adapter (default config, weights fetched by the adapter at its pinned
revision):

1. ``fintfm_lite_partition_kaggle``, ``FINTFM_PARTITION={}`` (the classifier's defaults: depth 2,
   ``min_samples_leaf`` 0.05, one bag), adapter defaults otherwise (``n_ensemble`` 8,
   ``max_context`` 1000), so it is not §169's configuration;
2. ``fintfm_lite_default_kaggle``, no override, the pairing reference on the same device.

The partition arm runs first; if the session budget runs out, the default arm is the one cut
short, and the MPS default (``fintfm_lite_nan_1019``) is the fallback reference, with the device
difference stated. One process per dataset, so a failure costs one dataset rather than the arm.
Each arm's raw results are tarred into ``/kaggle/working/<run>.tar.gz``; the leaderboard
comparison runs locally from the cache (``FINTFM_SKIP_COMPARE=1`` here). ``summary.json`` records
per dataset the exit code, the wall time and whether the adapter logged partitioning on.

Built from ``task_lite_retrieval`` (the Fix 5 template, never launched). Pinned: fintfm at
``FINTFM_COMMIT``, the fork (``kabartay/tabarena``, branch ``fintfm-audit-fixes``) at
``FORK_COMMIT``, AutoGluon at the nightly the local runs used.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

FINTFM_COMMIT = "eb3fb7f"
FORK_COMMIT = "86ab7af"
AUTOGLUON = "1.6.4b20260919"
WORK = Path("/kaggle/working")
#: Stop starting new datasets after this many seconds (Kaggle's limit is 12 h).
BUDGET_S = 11 * 3600
#: The 27 binary datasets of ``fintfm_lite_nan_1019`` (``docs/results/TABARENA.md``).
DATASETS = [
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
    "customer_satisfaction_in_airline",
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
]
ARMS = (
    ("fintfm_lite_partition_kaggle", "{}"),
    ("fintfm_lite_default_kaggle", ""),
)

#: One dataset, one config, split 0 of Lite; written into the fork's root and run from there,
#: because Ray and the result cache resolve paths against the working directory.
RUNNER = '''"""Fit FinTFM's default config on one TabArena-Lite dataset (written by the Kaggle kernel)."""
from __future__ import annotations

import os
from pathlib import Path

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext

experiments = TabArenaV0pt1ExperimentBundle(models=[("FinTFM", 0)]).build_experiments()
TabArenaContext().build_and_run_jobs(
    experiments,
    expname=str(Path.cwd() / "experiments" / os.environ["FINTFM_RUN_NAME"]),
    subset="lite",
    build_kwargs={"dataset_names": [os.environ["FINTFM_DATASETS"]]},
    new_result_prefix="[New] ",
    debug_mode=True,
)
'''


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
            "fintfm-audit-fixes",
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


def run_arm(
    fork_root: Path,
    run_name: str,
    partition: str,
    datasets: list[str],
    device: str,
    deadline: float,
    log_dir: Path,
) -> list[dict]:
    """Fit one arm, one process per dataset, and record what happened to each.

    Args:
        fork_root: The fork's root; results land in ``experiments/<run_name>/`` under it.
        run_name: Results directory name; distinct per arm, because results are cached per
            (config, dataset, split) and the override does not change the config name.
        partition: ``FINTFM_PARTITION`` JSON, or ``""`` for the default (unpartitioned).
        datasets: Dataset names to fit.
        device: ``FINTFM_DEVICE``: ``cuda`` on Kaggle, ``cpu`` for the local smoke test.
        deadline: ``time.time()`` after which no new dataset is started.
        log_dir: Where ``<run_name>.log`` is appended.

    Returns:
        One record per dataset: name, exit code (``None`` when skipped for time), seconds, and
        whether the adapter logged partitioning on (``None`` for the default arm).
    """
    runner = fork_root / "run_one_lite.py"
    runner.write_text(RUNNER)
    log_path = log_dir / f"{run_name}.log"
    records = []
    for ds in datasets:
        if time.time() > deadline:
            records.append({"dataset": ds, "exit": None, "seconds": 0, "override_seen": None})
            print(f"[{run_name}] {ds}: skipped, session budget spent", flush=True)
            continue
        env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "FINTFM_PARAMS")}
        env.update(
            FINTFM_RUN_NAME=run_name,
            FINTFM_DATASETS=ds,
            FINTFM_PARTITION=partition,
            FINTFM_DEVICE=device,
            FINTFM_SKIP_COMPARE="1",
            PYTHONUNBUFFERED="1",
        )
        start = time.time()
        with log_path.open("a") as log:
            log.write(f"=== {time.strftime('%m-%d %H:%M:%S')} dataset {ds}\n")
            log.flush()
            before = log_path.stat().st_size
            rc = subprocess.run(
                [sys.executable, "-P", str(runner)],
                cwd=fork_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        with log_path.open() as log:
            log.seek(before)
            seen = "partitioned contexts on" in log.read() if partition else None
        seconds = round(time.time() - start)
        records.append({"dataset": ds, "exit": rc, "seconds": seconds, "override_seen": seen})
        print(f"[{run_name}] {ds}: exit {rc}, {seconds} s, override seen {seen}", flush=True)
    return records


def main() -> None:
    """Install, check the GPU, run both arms, tar their results and write ``summary.json``."""
    started = time.time()
    fork_root = Path("/kaggle/tmp/tabarena")
    fork_root.parent.mkdir(parents=True, exist_ok=True)
    install(fork_root)
    import torch

    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        sys.exit("no GPU in this session (quota exhausted?) -- refusing a CPU run")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    deadline = started + BUDGET_S
    summary = {
        "fintfm": FINTFM_COMMIT,
        "fork": FORK_COMMIT,
        "autogluon": AUTOGLUON,
        "gpu": torch.cuda.get_device_name(0),
        "arms": {},
    }
    for run_name, partition in ARMS:
        records = run_arm(fork_root, run_name, partition, list(DATASETS), "cuda", deadline, WORK)
        summary["arms"][run_name] = {"partition": partition, "datasets": records}
        results = fork_root / "experiments" / run_name
        if results.exists():
            with tarfile.open(WORK / f"{run_name}.tar.gz", "w:gz") as tar:
                tar.add(results, arcname=run_name)
        (WORK / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: [r["exit"] for r in v["datasets"]] for k, v in summary["arms"].items()}))
    print("done")


if __name__ == "__main__":
    main()
