"""Kaggle kernel: §295's arm on QSAR-TID-11 alone, on Kaggle CPU where §293's entry fitted it (task 49.25).

§295's run put QSAR-TID-11 in the Mac half, but the entry's QSAR-TID-11 comes from Kaggle CPU
(``fintfm_lp_reg_spline_small_cpu``). The Mac fits it 0.12% worse with and without the clip
(bit-identical either way), which is machine drift, not the arm. This kernel measures the arm on the
entry's machine. A copy of ``task_lite_clip_reg`` with only the kernel name, the run name
``fintfm_lp_reg_clip_cpu_qsar`` and the dataset list changed; see fintfm
``docs/results/FINDINGS.md`` §295 and §300.
"""


from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

# ---- PLACEHOLDERS: the owner fills these in before pushing the kernel ----------------------------
#: Full SHA of the pull-request branch head in the fork: 2e5019e, the binary entry with four bags up
#: to 5,000 fit rows and splines up to 10,000 (FINDINGS §290), and one GPU requested when CUDA is
#: available (2026-11-18). The CPU kernels first pushed at e5cb05b run identically: the GPU request
#: changes nothing where CUDA is absent.
FORK_COMMIT = "fe2b0724f9cf190335e6947bb9898f0fca76f5d7"
#: The fintfm release the fork's extra pins: 0.12.0 publishes the multiclass and regression
#: checkpoints. The preflight refuses to run until it is on PyPI.
FINTFM_VERSION = "0.12.0"
# --------------------------------------------------------------------------------------------------

FORK_URL = "https://github.com/kabartay/tabarena.git"
FORK_BRANCH = "fintfm-pr-clip"
#: The registry name of the entry, default configuration (config index 0).
MODEL = "FinTFM-LinProbe"
#: The floor the pull-request branch's ``packages/tabarena/pyproject.toml`` requires
#: (``autogluon.tabular>=1.6.4b20261109``; upstream raised it for autogluon/autogluon#6057). The stored
#: entry runs used 1.6.4b20260919, which that branch no longer installs (pip: ResolutionImpossible,
#: 2026-11-17), so this reproduction also changes the AutoGluon nightly.
AUTOGLUON = "1.6.4b20261109"
KERNEL = "task_lite_clip_qsar"
#: ``gpu``: refuse to run without CUDA (the binary entry ran on a T4); ``cpu``: run on the CPU.
DEVICE = "cpu"
WORK = Path("/kaggle/working")
#: Stop starting new datasets after this many seconds (Kaggle's limit is 12 h).
BUDGET_S = 11 * 3600
#: Kill one dataset's process after this many seconds, record it and go on.
DATASET_TIMEOUT_S = 2 * 3600

#: Run name -> (problem type, datasets). The run name is the directory under ``experiments/``.
RUNS: dict[str, tuple[str, list[str]]] = {
    'fintfm_lp_reg_clip_cpu_qsar': ('regression', ['QSAR-TID-11'])
}

#: The experiment script each dataset's process runs, in the fork root. Upstream-style API only.
WORKER = '''\
"""Fit one TabArena-Lite dataset (repeat 0, fold 0) with the registry entry's default config."""
import sys

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext

model, dataset, results_dir = sys.argv[1:4]
experiments = TabArenaV0pt1ExperimentBundle(models=[(model, 0)]).build_experiments()
TabArenaContext().build_and_run_jobs(
    experiments,
    expname=results_dir,
    subset="lite",
    build_kwargs={"dataset_names": [dataset]},
    new_result_prefix="[New] ",
    debug_mode=True,
)
'''

_COMMIT_RE = re.compile(r"[0-9a-f]{7,40}")
_VERSION_RE = re.compile(r"\d+\.\d+\.\d+")


def commit_problem(commit: str) -> str | None:
    """Why ``commit`` cannot be a pinned fork commit, or ``None`` when it can.

    Args:
        commit: The value of ``FORK_COMMIT``.

    Returns:
        A message, or ``None`` when it is 7 to 40 lower-case hex digits.
    """
    if _COMMIT_RE.fullmatch(commit):
        return None
    return f"FORK_COMMIT is {commit!r}, a placeholder: set it to the pull-request branch head SHA"


def version_problem(version: str, is_available: Callable[[str], bool]) -> str | None:
    """Why ``fintfm==version`` cannot be installed, or ``None`` when it can.

    Args:
        version: The value of ``FINTFM_VERSION``.
        is_available: Tells whether the package index serves that version (injected so tests
            need no network).

    Returns:
        A message, or ``None`` when the version is well formed and available.
    """
    if not _VERSION_RE.fullmatch(version):
        return f"FINTFM_VERSION is {version!r}, not a release number such as 0.11.0"
    if not is_available(version):
        return f"fintfm=={version} is not installable from the package index (not released yet?)"
    return None


def unpublished_checkpoints(source: str, problem_types: tuple[str, ...]) -> list[str]:
    """The problem types whose checkpoint in the fork's ``checkpoints.py`` cannot be fetched.

    A problem type counts as unpublished when a top-level assignment whose name contains the
    upper-cased type holds a dict literal with an ``"unpublished"`` key, or when no such
    assignment exists (the module could not be verified, so the kernel does not guess).

    Args:
        source: The text of ``fintfm/checkpoints.py``.
        problem_types: ``binary``, ``multiclass`` and/or ``regression``.

    Returns:
        One message per offending problem type, empty when all are published.
    """
    tree = ast.parse(source)
    found: dict[str, list[bool]] = {t: [] for t in problem_types}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
        for name in names:
            for ptype in problem_types:
                if ptype.upper() in name.upper():
                    found[ptype].append("unpublished" in keys)
    problems = []
    for ptype in problem_types:
        if not found[ptype]:
            problems.append(f"{ptype}: no checkpoint reference found in checkpoints.py")
        elif any(found[ptype]):
            problems.append(f"{ptype}: checkpoint is still an unpublished placeholder")
    return problems


def pip_has_version(version: str) -> bool:
    """Whether ``pip download fintfm==version --no-deps`` succeeds (needs network).

    Args:
        version: The release to look for.

    Returns:
        ``True`` when pip can fetch it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, "-m", "pip", "download", "--quiet", "--no-deps", "-d", tmp]
        return (
            subprocess.run(
                [*cmd, f"fintfm=={version}"], capture_output=True, check=False
            ).returncode
            == 0
        )


def refuse(problems: list[str]) -> None:
    """Print each problem and exit non-zero when there are any.

    Args:
        problems: Messages from the preflight checks.

    Raises:
        SystemExit: With a message naming every problem when ``problems`` is not empty.
    """
    if problems:
        sys.exit("PREFLIGHT REFUSED:\n  - " + "\n  - ".join(problems))


def problem_types() -> tuple[str, ...]:
    """The distinct problem types this kernel runs, in run order."""
    return tuple(dict.fromkeys(ptype for ptype, _ in RUNS.values()))


def preflight_pins() -> None:
    """Stage 1: refuse a placeholder commit or an uninstallable fintfm version."""
    problems = [commit_problem(FORK_COMMIT), version_problem(FINTFM_VERSION, pip_has_version)]
    refuse([p for p in problems if p])


def clone_fork(fork_root: Path) -> None:
    """Clone the fork's branch and check out ``FORK_COMMIT`` (a detached head).

    Args:
        fork_root: Where to clone.
    """
    subprocess.run(
        ["git", "clone", "--quiet", "--branch", FORK_BRANCH, FORK_URL, str(fork_root)], check=True
    )
    subprocess.run(["git", "-C", str(fork_root), "checkout", "--quiet", FORK_COMMIT], check=True)


def preflight_fork(fork_root: Path) -> None:
    """Stage 2: refuse when a problem type's checkpoint in the fork is unpublished.

    Args:
        fork_root: The cloned fork.
    """
    path = fork_root / "packages/tabarena/src/tabarena/models/fintfm/checkpoints.py"
    if not path.exists():
        refuse([f"{path} missing at fork commit {FORK_COMMIT}"])
    refuse(unpublished_checkpoints(path.read_text(), problem_types()))


def install(fork_root: Path) -> None:
    """Install AutoGluon, the fork, and the released fintfm into the session.

    Args:
        fork_root: The cloned fork.
    """
    pip = [sys.executable, "-m", "pip", "install", "--quiet"]
    subprocess.run(
        [*pip, f"autogluon.tabular=={AUTOGLUON}", f"autogluon.core[ray]=={AUTOGLUON}"], check=True
    )
    subprocess.run(
        [
            *pip,
            "-e",
            str(fork_root / "packages/bencheval"),
            "-e",
            str(fork_root / "packages/tabarena"),
        ],
        check=True,
    )
    # After the fork, so the released package wins over any other fintfm in the metadata.
    subprocess.run([*pip, f"fintfm=={FINTFM_VERSION}"], check=True)


def child_env() -> tuple[dict[str, str], list[str]]:
    """The environment for the fit processes: the session's, without ``FINTFM_*``.

    Returns:
        The environment, and the names of the removed variables.
    """
    removed = sorted(k for k in os.environ if k.startswith("FINTFM_"))
    env = {k: v for k, v in os.environ.items() if k not in removed and k != "VIRTUAL_ENV"}
    env["PYTHONUNBUFFERED"] = "1"
    return env, removed


def run_dataset(
    fork_root: Path, run: str, dataset: str, env: dict[str, str], log_path: Path
) -> dict:
    """Fit one dataset in its own process.

    Args:
        fork_root: The cloned fork; results land in ``experiments/<run>/``.
        run: The run name.
        dataset: The dataset name.
        env: The child environment.
        log_path: The log file to append to.

    Returns:
        ``dataset``, ``exit`` (``-1`` on timeout) and ``seconds``.
    """
    start = time.time()
    results_dir = str(fork_root / "experiments" / run)
    with log_path.open("a") as log:
        log.write(f"=== {time.strftime('%m-%d %H:%M:%S')} dataset {dataset}\n")
        log.flush()
        try:
            rc = subprocess.run(
                [sys.executable, "-P", str(WORK / "repro_worker.py"), MODEL, dataset, results_dir],
                cwd=fork_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=DATASET_TIMEOUT_S,
            ).returncode
        except subprocess.TimeoutExpired:
            rc = -1
    return {"dataset": dataset, "exit": rc, "seconds": round(time.time() - start)}


def tar_results(fork_root: Path, run: str, dest: Path) -> None:
    """Tar ``experiments/<run>`` into ``dest/<run>.tar.gz``.

    Args:
        fork_root: The cloned fork.
        run: The run name.
        dest: Output directory.
    """
    results = fork_root / "experiments" / run
    if results.exists():
        with tarfile.open(dest / f"{run}.tar.gz", "w:gz") as tar:
            tar.add(results, arcname=run)


def main(preflight_only: bool) -> None:
    """Preflight, install, check the device, run every dataset of every run, tar, summarise.

    Args:
        preflight_only: Stop after the preflight (clone included, no install, no fits).
    """
    started = time.time()
    preflight_pins()
    fork_root = (
        Path("/kaggle/tmp/tabarena")
        if Path("/kaggle").exists()
        else Path(tempfile.mkdtemp()) / "tabarena"
    )
    fork_root.parent.mkdir(parents=True, exist_ok=True)
    clone_fork(fork_root)
    preflight_fork(fork_root)
    if preflight_only:
        print("preflight ok")
        return
    install(fork_root)
    head = subprocess.run(
        ["git", "-C", str(fork_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not head.startswith(FORK_COMMIT):
        sys.exit(f"fork HEAD {head} != pinned {FORK_COMMIT}")
    import torch

    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}", flush=True)
    if DEVICE == "gpu" and not torch.cuda.is_available():
        sys.exit("no GPU in this session (quota exhausted?) -- refusing a CPU run")
    (WORK / "repro_worker.py").write_text(WORKER)
    env, removed = child_env()
    summary = {
        "kernel": KERNEL,
        "fork": head,
        "fork_branch": FORK_BRANCH,
        "fintfm": FINTFM_VERSION,
        "model": MODEL,
        "autogluon": AUTOGLUON,
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "removed_fintfm_env": removed,
        "runs": {},
    }
    deadline = started + BUDGET_S
    for run, (ptype, datasets) in RUNS.items():
        records = []
        for ds in datasets:
            if time.time() > deadline:
                records.append({"dataset": ds, "exit": None, "seconds": 0})
                continue
            records.append(run_dataset(fork_root, run, ds, env, WORK / f"{run}.log"))
            print(
                f"[{run}] {ds}: exit {records[-1]['exit']}, {records[-1]['seconds']} s", flush=True
            )
        summary["runs"][run] = {"problem_type": ptype, "datasets": records}
        tar_results(fork_root, run, WORK)
        (WORK / "summary.json").write_text(json.dumps(summary, indent=1))
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--preflight-only", action="store_true", help="preflight (clones the fork), fit nothing"
    )
    main(parser.parse_args().preflight_only)
