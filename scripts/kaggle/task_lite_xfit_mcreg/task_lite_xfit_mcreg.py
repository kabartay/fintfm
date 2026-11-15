"""Kaggle kernel: TabArena-Lite (repeat 0, fold 0) for FinTFM-LinProbe with cross-fitting, multiclass and regression.

The two local queue entries ``mc-xfit`` and ``reg-xfit`` (``runs/local_queue/queue.txt``, run through
``lite_mc_probe.sh`` and ``lite_reg_probe.sh``) are slow on the Mac CPU, so this kernel runs the
same two arms on a Kaggle T4 in parallel with the local binary run. Settings equal the local
scripts', except the device:

1. ``fintfm_lp_mc_scr_xfit_kaggle``: the 8 multiclass Lite datasets, the ``dr74-mc-s0`` checkpoint
   (``scripts/kaggle/ts_dr74_mc_s0``), ``FINTFM_PARTITION={"n_bags": 8}``,
   ``FINTFM_PROBE={"scale": true, "cv": true, "raw": true}``. Local pair: ``fintfm_lp_mc_scr_xfit_1115``;
   its baseline is ``fintfm_lp_mc_scr_bag8_1113``.
2. ``fintfm_lp_reg_xfit_kaggle``: the 13 regression Lite datasets, the ``v4-binned48_3-control``
   checkpoint (``scripts/kaggle/task_48_3_binned_control``), ``FINTFM_PARTITION={"n_bags": 4}``,
   ``FINTFM_PROBE={"raw": true}``. Local pair: ``fintfm_lp_reg_xfit_1115``; its baseline is
   ``fintfm_lp_reg_rawpart4_pure_1114``.

Both use ``FINTFM_PARAMS={"feature_transform": "squash", "max_features_strategy": "select"}``. The
cross-fitting readout is the fork's default (``probe_crossfit_folds=5`` at ``FORK_COMMIT``); the
kernel refuses to run if the cloned fork does not carry it.

The checkpoints come from the two training kernels' outputs (``kernel_sources`` in
``kernel-metadata.json``), and each loaded file's SHA-256 is checked against the constants below
before anything is installed or fitted: a mismatch stops the kernel. One process per dataset
(``run_fintfm_lite.py`` with ``FINTFM_DATASETS``), each with its own time limit, so a failure or a
hang costs one dataset rather than the arm. Each arm's raw results are tarred into
``/kaggle/working/<run>.tar.gz``; the comparison runs locally (``FINTFM_SKIP_COMPARE=1`` here).
``summary.json`` records the pins, the checkpoint hashes, and per dataset the exit code, the wall
time and whether the adapter logged partitioning on.

Local smoke test of the logic on one dataset (no install, CPU, results under a ``_smoke`` run name)::

    .venv/bin/python -P scripts/kaggle/task_lite_xfit_mcreg/task_lite_xfit_mcreg.py --local \
        --arm mc --datasets website_phishing --ckpt runs/kaggle-ts-dr74-mc-s0/dr74-mc-s0/dr74-mc-s0.pt

Built from ``scripts/kaggle/task_lite_partition``. Pinned: fintfm at ``FINTFM_COMMIT``, the fork
(``kabartay/tabarena``, branch ``fintfm-reg-levers``) at ``FORK_COMMIT``, AutoGluon at the nightly
the local runs used.

Four more regression arms follow the two above (order: multiclass, regression, then these). Each
is the regression cross-fit arm plus exactly one variable the fork reads (the levers are off by
default at ``FORK_COMMIT``, so the first two arms equal ``b48cb22``'s): ``..._log1p_kaggle``
(``FINTFM_REG_TARGET_TRANSFORM=log1p``), ``..._quantile_kaggle`` (``=quantile``),
``..._blendcv_kaggle`` (``FINTFM_REG_BLEND=cv``), ``..._wide_kaggle`` (``FINTFM_REG_ALPHAS=wide``).
All arms share one deadline; an arm is not started when the time left is below
``ARM_ESTIMATE_S``, and ``summary.json`` lists those under ``arms_not_started``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

FINTFM_COMMIT = "03d0270"
FORK_COMMIT = "551259a"
#: The fork branch holding ``FORK_COMMIT`` (b48cb22 plus regression readout levers, off by default).
FORK_BRANCH = "fintfm-reg-levers"
AUTOGLUON = "1.6.4b20260919"
WORK = Path("/kaggle/working")
#: Stop starting new datasets after this many seconds (Kaggle's limit is 12 h).
BUDGET_S = 11 * 3600
#: Kill one dataset's process after this many seconds, record it and go on.
DATASET_TIMEOUT_S = 2 * 3600

PARAMS = '{"feature_transform": "squash", "max_features_strategy": "select"}'

#: The 8 multiclass Lite datasets (all multiclass tasks of the suite, at most 10 classes).
MC_DATASETS = [
    "anneal",
    "hiva_agnostic",
    "maternal_health_risk",
    "MIC",
    "SDSS17",
    "splice",
    "students_dropout_and_academic_success",
    "website_phishing",
]
#: The 13 regression Lite datasets, in ``lite_reg_probe.sh``'s order.
REG_DATASETS = [
    "QSAR_fish_toxicity",
    "concrete_compressive_strength",
    "healthcare_insurance_expenses",
    "airfoil_self_noise",
    "Another-Dataset-on-used-Fiat-500",
    "QSAR-TID-11",
    "wine_quality",
    "miami_housing",
    "houses",
    "superconductivity",
    "Food_Delivery_Time",
    "physiochemical_protein",
    "diamonds",
]

#: Per-arm time estimate (s) the scheduler needs before it starts an arm; ESTIMATED, not measured.
#: Local CPU train+infer of the matching non-cross-fit runs: multiclass 3,926 s, regression
#: 16,060 s; the T4 is assumed no slower than that CPU, plus a margin for the cross-fit readout.
ARM_ESTIMATE_S = {"mc": 5400, "reg": 10800}

#: Arm key -> everything that differs between the two arms. ``env`` is exactly what the local
#: script exports besides the run name, the device and ``OMP_NUM_THREADS``.
ARMS = {
    "mc": {
        "run_name": "fintfm_lp_mc_scr_xfit_kaggle",
        "datasets": MC_DATASETS,
        "checkpoint_file": "dr74-mc-s0.pt",
        "checkpoint_sha256": "14d5b4d561fbea6e35c7a30dc2644cb8f64f437c25d62da087ee960717cc8f4d",
        "checkpoint_env": "FINTFM_CHECKPOINT_MULTICLASS",
        "source_kernel": "muhakabartay/fintfm-ts-dr74-mc-s0",
        "env": {
            "FINTFM_MODEL": "FinTFM-LinProbe",
            "FINTFM_PROBLEM_TYPES": "multiclass",
            "FINTFM_ALLOW_MULTICLASS": "1",
            "FINTFM_PARTITION": '{"n_bags": 8}',
            "FINTFM_PARAMS": PARAMS,
            "FINTFM_PROBE": '{"scale": true, "cv": true, "raw": true}',
        },
    },
    "reg": {
        "run_name": "fintfm_lp_reg_xfit_kaggle",
        "datasets": REG_DATASETS,
        "checkpoint_file": "v4-binned48_3-control.pt",
        "checkpoint_sha256": "31d5b3ef49e89bbe405fc85ab1d746a9fd6c3a1c5282e1770111902cc4af526c",
        "checkpoint_env": "FINTFM_CHECKPOINT_REGRESSION",
        "source_kernel": "muhakabartay/fintfm-task-48-3-matched-binned-head-control",
        "env": {
            "FINTFM_MODEL": "FinTFM-LinProbe",
            "FINTFM_PROBLEM_TYPES": "regression",
            "FINTFM_ALLOW_REGRESSION": "1",
            "FINTFM_PARTITION": '{"n_bags": 4}',
            "FINTFM_PARAMS": PARAMS,
            "FINTFM_PROBE": '{"raw": true}',
        },
    },
}


#: The regression cross-fit arm plus exactly one lever variable each (coordinator, 2026-11-15).
LEVERS = {
    "reg_log1p": ("fintfm_lp_reg_xfit_log1p_kaggle", "FINTFM_REG_TARGET_TRANSFORM", "log1p"),
    "reg_quantile": ("fintfm_lp_reg_xfit_quantile_kaggle", "FINTFM_REG_TARGET_TRANSFORM", "quantile"),
    "reg_blendcv": ("fintfm_lp_reg_xfit_blendcv_kaggle", "FINTFM_REG_BLEND", "cv"),
    "reg_wide": ("fintfm_lp_reg_xfit_wide_kaggle", "FINTFM_REG_ALPHAS", "wide"),
}
for _key, (_run, _var, _val) in LEVERS.items():
    ARMS[_key] = {
        **ARMS["reg"],
        "run_name": _run,
        "env": {**ARMS["reg"]["env"], _var: _val},
        "lever": _var,
    }
    ARM_ESTIMATE_S[_key] = ARM_ESTIMATE_S["reg"]


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


def find_checkpoint(arm: str, inputs: Path = Path("/kaggle/input")) -> Path:
    """Locate one arm's checkpoint among the attached kernel outputs and verify its hash.

    Args:
        arm: ``"mc"`` or ``"reg"``.
        inputs: Where Kaggle mounts ``kernel_sources``.

    Returns:
        The one matching file, whose SHA-256 equals the arm's pinned digest.

    Raises:
        SystemExit: When the file is missing, ambiguous, or its hash differs.
    """
    spec = ARMS[arm]
    hits = sorted(inputs.rglob(spec["checkpoint_file"]))
    if len(hits) != 1:
        sys.exit(f"[{arm}] expected one {spec['checkpoint_file']} under {inputs}, found {hits}")
    return verify_checkpoint(arm, hits[0])


def verify_checkpoint(arm: str, path: Path) -> Path:
    """Refuse a checkpoint whose SHA-256 differs from the arm's pinned digest.

    Args:
        arm: ``"mc"`` or ``"reg"``.
        path: The checkpoint file.

    Returns:
        ``path`` unchanged when the digest matches.

    Raises:
        SystemExit: On a mismatch.
    """
    got = sha256(path)
    want = ARMS[arm]["checkpoint_sha256"]
    if got != want:
        sys.exit(f"[{arm}] {path}: SHA-256 {got} != pinned {want}; refusing to run")
    print(f"[{arm}] checkpoint {path} SHA-256 ok", flush=True)
    return path


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


def check_fork(fork_root: Path, arms: list[str] | None = None) -> None:
    """Stop unless the fork has the runner, the cross-fitting readout and each arm's lever.

    Args:
        fork_root: The fork's root.
        arms: Arms about to run (default: all); each lever arm's environment variable must
            appear somewhere in the fork's ``tabarena`` sources.

    Raises:
        SystemExit: When ``run_fintfm_lite.py``, ``probe_crossfit_folds`` or a lever is missing.
    """
    runner = fork_root / "run_fintfm_lite.py"
    model = fork_root / "packages/tabarena/src/tabarena/models/fintfm_linprobe/model.py"
    if not runner.exists():
        sys.exit(f"{runner} missing at fork commit {FORK_COMMIT}")
    if "probe_crossfit_folds" not in model.read_text():
        sys.exit(f"{model} has no probe_crossfit_folds at fork commit {FORK_COMMIT}")
    sources = "\n".join(
        p.read_text() for p in (fork_root / "packages/tabarena/src/tabarena").rglob("*.py")
    )
    for arm in arms or list(ARMS):
        lever = ARMS[arm].get("lever")
        if lever and lever not in sources:
            sys.exit(f"[{arm}] {lever} not read anywhere in the fork at commit {FORK_COMMIT}")


def arm_env(arm: str, checkpoint: Path, device: str, run_name: str | None = None) -> dict[str, str]:
    """The environment of one dataset's process.

    Args:
        arm: ``"mc"`` or ``"reg"``.
        checkpoint: The verified checkpoint.
        device: ``FINTFM_DEVICE``: ``cuda`` on Kaggle, ``cpu`` for the local smoke test.
        run_name: Overrides the arm's run name (the smoke test).

    Returns:
        The inherited environment without ``VIRTUAL_ENV`` and any ``FINTFM_*`` variable, plus the
        arm's settings.
    """
    spec = ARMS[arm]
    env = {
        k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV" and not k.startswith("FINTFM_")
    }
    env.update(spec["env"])
    env.update(
        {
            "FINTFM_RUN_NAME": run_name or spec["run_name"],
            spec["checkpoint_env"]: str(checkpoint),
            "FINTFM_DEVICE": device,
            "FINTFM_SKIP_COMPARE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def run_arm(
    fork_root: Path,
    arm: str,
    checkpoint: Path,
    datasets: list[str],
    device: str,
    deadline: float,
    log_dir: Path,
    run_name: str | None = None,
) -> list[dict]:
    """Fit one arm, one process per dataset, and record what happened to each.

    Args:
        fork_root: The fork's root; results land in ``experiments/<run_name>/`` under it.
        arm: ``"mc"`` or ``"reg"``.
        checkpoint: The verified checkpoint.
        datasets: Dataset names to fit.
        device: ``FINTFM_DEVICE``.
        deadline: ``time.time()`` after which no new dataset is started.
        log_dir: Where ``<run_name>.log`` is appended.
        run_name: Overrides the arm's run name (the smoke test).

    Returns:
        One record per dataset: name, exit code (``None`` when skipped for time, ``-1`` when it
        hit ``DATASET_TIMEOUT_S``), seconds, and whether the adapter logged partitioning on.
    """
    run_name = run_name or ARMS[arm]["run_name"]
    log_path = log_dir / f"{run_name}.log"
    records = []
    for ds in datasets:
        if time.time() > deadline:
            records.append({"dataset": ds, "exit": None, "seconds": 0, "override_seen": None})
            print(f"[{run_name}] {ds}: skipped, session budget spent", flush=True)
            continue
        env = arm_env(arm, checkpoint, device, run_name)
        env["FINTFM_DATASETS"] = ds
        start = time.time()
        with log_path.open("a") as log:
            log.write(f"=== {time.strftime('%m-%d %H:%M:%S')} dataset {ds}\n")
            log.flush()
            before = log_path.stat().st_size
            try:
                rc = subprocess.run(
                    [sys.executable, "-P", "run_fintfm_lite.py", "--full"],
                    cwd=fork_root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=DATASET_TIMEOUT_S,
                ).returncode
            except subprocess.TimeoutExpired:
                rc = -1
        with log_path.open() as log:
            log.seek(before)
            seen = "partitioned contexts on" in log.read()
        seconds = round(time.time() - start)
        records.append({"dataset": ds, "exit": rc, "seconds": seconds, "override_seen": seen})
        print(f"[{run_name}] {ds}: exit {rc}, {seconds} s, override seen {seen}", flush=True)
    return records


def tar_results(fork_root: Path, run_name: str, dest: Path) -> None:
    """Tar one run's ``experiments/<run_name>`` directory into ``dest/<run_name>.tar.gz``.

    Args:
        fork_root: The fork's root.
        run_name: The run's directory name.
        dest: Output directory.
    """
    results = fork_root / "experiments" / run_name
    if results.exists():
        with tarfile.open(dest / f"{run_name}.tar.gz", "w:gz") as tar:
            tar.add(results, arcname=run_name)


def main() -> None:
    """Verify the checkpoints, install, check the GPU, run both arms, tar results, write summary."""
    started = time.time()
    # Hashes first: a wrong checkpoint should cost seconds, not an install.
    checkpoints = {arm: find_checkpoint(arm) for arm in ARMS}
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
        "autogluon": AUTOGLUON,
        "gpu": torch.cuda.get_device_name(0),
        "fork_branch": FORK_BRANCH,
        "arm_estimate_s": ARM_ESTIMATE_S,
        "arms": {},
        "arms_not_started": [],
    }
    for arm, spec in ARMS.items():
        remaining = deadline - time.time()
        if remaining < ARM_ESTIMATE_S[arm]:
            summary["arms_not_started"].append(
                {"run": spec["run_name"], "remaining_s": round(remaining)}
            )
            print(f"[{spec['run_name']}] not started: {remaining:.0f} s left", flush=True)
            (WORK / "summary.json").write_text(json.dumps(summary, indent=1))
            continue
        records = run_arm(
            fork_root, arm, checkpoints[arm], list(spec["datasets"]), "cuda", deadline, WORK
        )
        summary["arms"][spec["run_name"]] = {
            "checkpoint": spec["checkpoint_file"],
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "env": spec["env"],
            "datasets": records,
        }
        tar_results(fork_root, spec["run_name"], WORK)
        (WORK / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: [r["exit"] for r in v["datasets"]] for k, v in summary["arms"].items()}))
    print("done")


def local_smoke(args: argparse.Namespace) -> None:
    """Run one arm's logic on chosen datasets on CPU against a local fork checkout.

    No install and no GPU check. The checkpoint hash is still verified. Results go under
    ``<run name>_smoke``, so they cannot be mistaken for the arm's.

    Args:
        args: Parsed ``--arm``, ``--datasets``, ``--ckpt``, ``--fork`` and ``--out``.
    """
    fork_root = Path(args.fork)
    check_fork(fork_root, [args.arm])
    checkpoint = verify_checkpoint(args.arm, Path(args.ckpt))
    run_name = f"{ARMS[args.arm]['run_name']}_smoke"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = run_arm(
        fork_root,
        args.arm,
        checkpoint,
        args.datasets.split(","),
        "cpu",
        time.time() + 3600,
        out,
        run_name,
    )
    (out / "summary_smoke.json").write_text(json.dumps(records, indent=1))
    tar_results(fork_root, run_name, out)
    print(json.dumps(records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--local", action="store_true", help="CPU smoke test, no install")
    parser.add_argument("--arm", choices=sorted(ARMS), default="mc")
    parser.add_argument("--datasets", default="website_phishing")
    parser.add_argument("--ckpt", help="local checkpoint path (--local)")
    parser.add_argument("--fork", default=str(Path.home() / "proj" / "tabarena"))
    parser.add_argument("--out", default="/tmp/xfit_smoke")
    cli = parser.parse_args()
    if cli.local:
        local_smoke(cli)
    else:
        main()
