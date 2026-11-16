"""Kaggle CPU kernel: openspec ``pd-term-structure`` task 11.8, the multi-seed term-structure run.

The Verify of task 11.8 is one command, ``fintfm-termstruct --steps 1200 --seeds 0 1 2 --n-boot
2000``. It trains every arm from scratch on the synthetic prior (joint, per-horizon matched total,
per-horizon matched per-model), so it needs **no checkpoint, no dataset and no ``kernel_sources``**:
nothing local is uploaded and nothing is downloaded except the pinned fintfm and its dependencies.

The model is small (``d_model`` 48, 3 layers, 5 horizons) and the run is 3 seeds x 7 x 1,200 =
25,200 gradient steps at batch 16, so it is a CPU job. Time ESTIMATED, not measured on Kaggle: the
task's own note puts it at about 0.5 s/step or more on the loaded Mac, i.e. roughly 3.5 to 5 hours.

The command runs as one subprocess under a timeout so that a hang costs the session and not the log.
Each seed writes ``seed<S>/term_structure.json`` as it finishes, so a timeout still leaves the
completed seeds; the pooled ``term_structure_seeds.json`` exists only if all three finish. Outputs
go under ``/kaggle/working/term-structure-11-8`` plus ``term_structure.log`` and ``summary.json``.

Local logic check without installing anything: ``--dry-run`` prints the command and writes nothing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

FINTFM_COMMIT = "e475151"
WORK = Path("/kaggle/working")
OUT = WORK / "term-structure-11-8"
#: The 11.7 recipe from the task text.
STEPS = 1200
SEEDS = [0, 1, 2]
N_BOOT = 2000
#: Kill the training command after this many seconds (Kaggle's session limit is 12 h).
TIMEOUT_S = 11 * 3600
THREADS = "4"


def command(out: Path = OUT) -> list[str]:
    """The Verify command of task 11.8, launched through the module the console script wraps.

    Args:
        out: The ``--out`` directory.

    Returns:
        The argument vector, equal to ``fintfm-termstruct --steps 1200 --seeds 0 1 2 --n-boot 2000
        --out <out>``.
    """
    return [
        sys.executable,
        "-m",
        "fintfm.experiments.term_structure",
        "--steps",
        str(STEPS),
        "--seeds",
        *[str(s) for s in SEEDS],
        "--n-boot",
        str(N_BOOT),
        "--out",
        str(out),
    ]


def installed_commit() -> str | None:
    """The git commit pip recorded for the installed fintfm, or ``None`` if not a VCS install."""
    direct = metadata.distribution("fintfm").read_text("direct_url.json")
    if not direct:
        return None
    return json.loads(direct).get("vcs_info", {}).get("commit_id")


def main() -> None:
    """Install the pinned fintfm, run the Verify command, write ``summary.json``."""
    if "--dry-run" in sys.argv:
        print(" ".join(command()))
        return
    started = time.time()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            f"fintfm @ git+https://github.com/kabartay/fintfm.git@{FINTFM_COMMIT}",
        ],
        check=True,
    )
    commit = installed_commit()
    if not commit or not commit.startswith(FINTFM_COMMIT):
        sys.exit(f"installed fintfm commit {commit} != pinned {FINTFM_COMMIT}")
    env = {
        **os.environ,
        "OMP_NUM_THREADS": THREADS,
        "MKL_NUM_THREADS": THREADS,
        "PYTHONUNBUFFERED": "1",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = WORK / "term_structure.log"
    try:
        with log_path.open("w") as log:
            rc = subprocess.run(
                command(),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=TIMEOUT_S,
            ).returncode
    except subprocess.TimeoutExpired:
        rc = -1
    seeds_done = [s for s in SEEDS if (OUT / f"seed{s}" / "term_structure.json").exists()]
    summary = {
        "task": "pd-term-structure 11.8",
        "fintfm": FINTFM_COMMIT,
        "command": command(),
        "exit": rc,
        "timed_out": rc == -1,
        "seconds": round(time.time() - started),
        "seeds_done": seeds_done,
        "pooled_written": (OUT / "term_structure_seeds.json").exists(),
        "device": "cpu",
        "cpus": os.cpu_count(),
        "threads": THREADS,
    }
    (WORK / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
