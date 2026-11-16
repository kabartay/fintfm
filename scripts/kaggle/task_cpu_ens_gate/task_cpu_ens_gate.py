"""Kaggle CPU kernel: FINDINGS §256's reproduction gate for the binary readout ensemble (task 49.22).

The fork branch ``fintfm-readout-ensemble`` (``FORK_COMMIT``) adds ``probe_ensemble`` /
``FINTFM_PROBE_ENSEMBLE``: several readout members fitted inside one FinTFM-LinProbe model and
averaged. Its tests could not run on the development machine (load 180-410), so this kernel:

1. Runs the fork's ``fintfm_linprobe`` model tests (they include an equality test: the in-model
   ensemble equals the mean of separately fitted members to 1e-9) and fits nothing if any fails.
2. On ``blood-transfusion-service-center`` and ``credit-g`` (TabArena-Lite repeat 0 fold 0,
   binary entry settings), fits ``fintfm_lp_bin_gate_xfit_kaggle`` (the current readout, no
   ensemble), then ``..._ens3_kaggle`` (``xfit,noxfit,rawonly``) and ``..._ens2_kaggle``
   (``xfit,noxfit``). The local comparison is against the stored Mac runs
   (``fintfm_lp_bin_scr_xfit_1115``, ``fintfm_lp_bin_scalecvraw_1114``,
   ``fintfm_lp_bin_rawonly_1115``); the Kaggle xfit run measures the machine-to-machine drift, so
   a difference in the ensemble that matches it is explained rather than a defect.
3. Times ``xfit,noxfit`` on ``customer_satisfaction_in_airline`` (``..._airens2_kaggle``) against
   AutoGluon's 3,600 s limit (cross-fitting alone took 1,915 s on the Mac CPU).

The published binary checkpoint is fetched by the fork from Hugging Face at a pinned commit and
SHA-256. One process per dataset; ``summary.json`` records exit code, seconds and whether
``TimeLimitExceeded`` appeared; results are tarred per run into ``/kaggle/working``.
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
FORK_COMMIT = "169402a"
FORK_BRANCH = "fintfm-readout-ensemble"
AUTOGLUON = "1.6.4b20260919"
WORK = Path("/kaggle/working")
BUDGET_S = 11 * 3600 + 1800
DATASET_TIMEOUT_S = 3 * 3600
SMALL = ["blood-transfusion-service-center", "credit-g"]
AIRLINE = "customer_satisfaction_in_airline"
BASE_ENV = {
    "FINTFM_MODEL": "FinTFM-LinProbe",
    "FINTFM_PROBLEM_TYPES": "binary",
    "FINTFM_PARTITION": "{}",
    "FINTFM_PARAMS": '{"feature_transform": "squash", "max_features_strategy": "select"}',
    "FINTFM_PROBE": '{"scale": true, "cv": true, "raw": true}',
}
#: Run name -> (datasets, extra environment), in run order.
RUNS: dict[str, tuple[list[str], dict[str, str]]] = {
    "fintfm_lp_bin_gate_xfit_kaggle": (SMALL, {}),
    "fintfm_lp_bin_gate_ens3_kaggle": (SMALL, {"FINTFM_PROBE_ENSEMBLE": "xfit,noxfit,rawonly"}),
    "fintfm_lp_bin_gate_ens2_kaggle": (SMALL, {"FINTFM_PROBE_ENSEMBLE": "xfit,noxfit"}),
    "fintfm_lp_bin_gate_airens2_kaggle": ([AIRLINE], {"FINTFM_PROBE_ENSEMBLE": "xfit,noxfit"}),
}


def install(fork_root: Path) -> None:
    """Clone the fork at its pin and install it, then fintfm at its pin, then pytest.

    Args:
        fork_root: Where the fork is cloned.
    """
    pip = [sys.executable, "-m", "pip", "install", "--quiet"]
    url = "https://github.com/kabartay/tabarena.git"
    subprocess.run(
        ["git", "clone", "--quiet", "--branch", FORK_BRANCH, url, str(fork_root)], check=True
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
    subprocess.run(
        [*pip, f"fintfm @ git+https://github.com/kabartay/fintfm.git@{FINTFM_COMMIT}"], check=True
    )
    subprocess.run([*pip, "pytest"], check=True)


def run_env(run_name: str, extra: dict[str, str], dataset: str) -> dict[str, str]:
    """The environment of one dataset's process.

    Args:
        run_name: ``FINTFM_RUN_NAME``.
        extra: The run's extra variables.
        dataset: ``FINTFM_DATASETS``.

    Returns:
        The inherited environment without ``VIRTUAL_ENV`` and ``FINTFM_*``, plus the settings.
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
            "FINTFM_DEVICE": "cpu",
            "FINTFM_SKIP_COMPARE": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "4",
        }
    )
    return env


def main() -> None:
    """Install, run the model tests, then every run, writing ``summary.json`` as it goes."""
    started = time.time()
    fork_root = Path("/kaggle/tmp/tabarena")
    fork_root.parent.mkdir(parents=True, exist_ok=True)
    install(fork_root)
    head = subprocess.run(
        ["git", "-C", str(fork_root), "rev-parse", "--short=7", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != FORK_COMMIT:
        sys.exit(f"fork HEAD {head} != pinned {FORK_COMMIT}")
    summary: dict = {
        "fintfm": FINTFM_COMMIT,
        "fork": FORK_COMMIT,
        "fork_branch": FORK_BRANCH,
        "autogluon": AUTOGLUON,
        "cpus": os.cpu_count(),
        "runs": {},
    }
    tests = fork_root / "packages/tabarena/src/tabarena/models/fintfm_linprobe/tests"
    t0 = time.time()
    with (WORK / "model_tests.log").open("w") as log:
        rc = subprocess.run(
            [sys.executable, "-P", "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tests)],
            cwd=fork_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    summary["model_tests"] = {"exit": rc, "seconds": round(time.time() - t0)}
    (WORK / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"model tests exit {rc}", flush=True)
    if rc != 0:
        sys.exit("model tests failed; see model_tests.log -- fitting nothing")
    deadline = started + BUDGET_S
    for run_name, (datasets, extra) in RUNS.items():
        records = []
        log_path = WORK / f"{run_name}.log"
        for ds in datasets:
            if time.time() > deadline:
                records.append({"dataset": ds, "exit": None, "seconds": 0})
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
            print(f"[{run_name}] {ds}: exit {rc}, {seconds} s, time limit {hit}", flush=True)
        summary["runs"][run_name] = {"env": extra, "datasets": records}
        results = fork_root / "experiments" / run_name
        if results.exists():
            with tarfile.open(WORK / f"{run_name}.tar.gz", "w:gz") as tar:
                tar.add(results, arcname=run_name)
        (WORK / "summary.json").write_text(json.dumps(summary, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
