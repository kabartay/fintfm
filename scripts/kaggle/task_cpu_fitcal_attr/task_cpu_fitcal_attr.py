"""Kaggle CPU kernel: openspec 4.3 (fitted calibration) and 12.5 (attribution stability), 2026-11-16.

Two measurements that were blocked only by a saturated Mac, each in its own subprocess with its
own timeout, so one failing or timing out does not cost the other:

* **4.3**: ``fintfm-fitcal --model runs/v4-cellattn-labels.pt --out runs/fitted-calibration``, the same
  command and arguments as the local queue's ``h-fitcal`` entry (``fintfm-fitcal`` is called through
  its entry point function, so no console script is needed). Output: ``fitted-calibration/`` with
  ``fitted_calibration.json``.
* **12.5**: ``FINTFM_SLOW_ATTRIBUTION=1 pytest tests/test_attribution_stability.py -k thresholds -rx``.
  The pre-registered thresholds are constants in ``attribution_stability.py`` and the test is a strict
  xfail, so a small pytest plugin (``-p attr_dump``) wraps ``stability`` and writes the measured
  statistics to ``attribution_stability.json`` beside the test outcome. The plugin changes nothing
  the test sees.

The checkpoint is the public Hugging Face file (``kabartay/fintfm-binary``, Apache-2.0) at the revision
every published binary number used, verified by SHA-256 against ``runs/v4-cellattn-labels.pt``
before anything runs. It is not a Kaggle kernel output, so there is no ``kernel_sources``, and
nothing local is uploaded. fintfm is cloned at ``FINTFM_COMMIT`` and installed from the clone, because
the 12.5 test lives in the repository's ``tests/``. Polish bankruptcy data is fetched from UCI by
fintfm itself. Outputs land in ``/kaggle/working/`` and ``summary.json`` records exit codes, wall
times and timeouts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

FINTFM_COMMIT = "e475151"
FINTFM_URL = "https://github.com/kabartay/fintfm.git"
CKPT_URL = (
    "https://huggingface.co/kabartay/fintfm-binary/resolve/"
    "f116bfd43a2b15c65ed3551ea8c38e3364629ddc/v4-cellattn-labels.pt"
)
CKPT_SHA256 = "d231b062483c9261aae4c503ad5359db295fc7206fa0183f51763e4953c518d2"
WORK = Path("/kaggle/working")
CLONE = Path("/kaggle/tmp/fintfm")
#: Per-measurement timeouts in seconds (Kaggle's session limit is 12 h).
FITCAL_TIMEOUT_S = 5 * 3600
ATTR_TIMEOUT_S = 2 * 3600
THREADS = "4"

#: ``fintfm-fitcal`` with the arguments of the queued ``h-fitcal`` command, via its entry point.
FITCAL_CMD = [
    sys.executable,
    "-P",
    "-c",
    "from fintfm.experiments.fitted_calibration import main; main()",
    "--model",
    "runs/v4-cellattn-labels.pt",
    "--out",
    "runs/fitted-calibration",
]
ATTR_CMD = [
    sys.executable,
    "-P",
    "-m",
    "pytest",
    "tests/test_attribution_stability.py",
    "-k",
    "thresholds",
    "-rx",
    "-p",
    "attr_dump",
    "-p",
    "no:cacheprovider",
]

#: Pytest plugin: wrap ``stability`` so the measured statistics reach a JSON file.
PLUGIN = '''\
"""Write the statistics of every ``stability`` call to $ATTR_JSON (no effect on the test)."""
import dataclasses
import json
import os

from fintfm.experiments import attribution_stability as ast_

_orig = ast_.stability
_calls = []


def _wrapped(*args, **kwargs):
    res = _orig(*args, **kwargs)
    _calls.append(dataclasses.asdict(res))
    record = {
        "thresholds": {
            "MIN_RESAMPLE_SPEARMAN": ast_.MIN_RESAMPLE_SPEARMAN,
            "MIN_SEED_SPEARMAN": ast_.MIN_SEED_SPEARMAN,
            "MIN_SEED_TOPK_JACCARD": ast_.MIN_SEED_TOPK_JACCARD,
        },
        "results": _calls,
    }
    with open(os.environ["ATTR_JSON"], "w") as f:
        json.dump(record, f, indent=1)
    return res


ast_.stability = _wrapped
'''


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


def fetch_checkpoint(dest: Path) -> None:
    """Download the public checkpoint and stop unless its SHA-256 matches the pin.

    Args:
        dest: Where to write it.

    Raises:
        SystemExit: When the hash differs.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(CKPT_URL, dest)
    got = sha256(dest)
    if got != CKPT_SHA256:
        sys.exit(f"{dest}: SHA-256 {got} != pinned {CKPT_SHA256}; refusing to run")
    print(f"checkpoint {dest} SHA-256 ok", flush=True)


def install() -> None:
    """Clone fintfm at its pin, verify HEAD, and install it with pytest."""
    CLONE.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--quiet", FINTFM_URL, str(CLONE)], check=True)
    subprocess.run(["git", "-C", str(CLONE), "checkout", "--quiet", FINTFM_COMMIT], check=True)
    head = subprocess.run(
        ["git", "-C", str(CLONE), "rev-parse", "--short=7", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != FINTFM_COMMIT:
        sys.exit(f"fintfm HEAD {head} != pinned {FINTFM_COMMIT}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "-e", str(CLONE), "pytest"], check=True
    )


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The environment of one measurement's process.

    Args:
        extra: Variables to add.

    Returns:
        The inherited environment without ``VIRTUAL_ENV`` and ``FINTFM_*``, with CPU threads set.
    """
    env = {
        k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV" and not k.startswith("FINTFM_")
    }
    env.update(
        {
            "OMP_NUM_THREADS": THREADS,
            "MKL_NUM_THREADS": THREADS,
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    env.update(extra or {})
    return env


def run_step(
    name: str, cmd: list[str], env: dict[str, str], timeout: int, log_path: Path
) -> dict[str, object]:
    """Run one measurement in its own subprocess with a timeout.

    Args:
        name: Step name for the summary.
        cmd: The command, run with the fintfm clone as working directory.
        env: Its environment.
        timeout: Seconds before it is killed.
        log_path: Where stdout and stderr go.

    Returns:
        ``{"step", "exit", "seconds", "timed_out"}``; ``exit`` is -1 on timeout.
    """
    start = time.time()
    timed_out = False
    with log_path.open("w") as log:
        try:
            rc = subprocess.run(
                cmd,
                cwd=CLONE,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout,
            ).returncode
        except subprocess.TimeoutExpired:
            rc, timed_out = -1, True
    seconds = round(time.time() - start)
    print(f"{name}: exit {rc}, {seconds} s, timed out {timed_out}", flush=True)
    return {"step": name, "exit": rc, "seconds": seconds, "timed_out": timed_out}


def pytest_counts(log_path: Path) -> dict[str, int]:
    """Outcome counts from pytest's final summary line.

    Args:
        log_path: The pytest log.

    Returns:
        E.g. ``{"xfailed": 1, "deselected": 2}``; empty when no summary line is found.
    """
    lines = [
        ln for ln in log_path.read_text().splitlines() if re.search(r"\d+ \w+.* in [\d.]+s", ln)
    ]
    if not lines:
        return {}
    return {w: int(n) for n, w in re.findall(r"(\d+) (\w+)", lines[-1].split(" in ")[0])}


def main() -> None:
    """Verify the checkpoint, install, run 12.5 and 4.3 in separate subprocesses, write summary."""
    started = time.time()
    install()
    ckpt = CLONE / "runs" / "v4-cellattn-labels.pt"
    fetch_checkpoint(ckpt)
    summary: dict[str, object] = {
        "fintfm": FINTFM_COMMIT,
        "checkpoint_sha256": CKPT_SHA256,
        "device": "cpu",
        "cpus": os.cpu_count(),
        "threads": THREADS,
        "fitcal_cmd": "fintfm-fitcal --model runs/v4-cellattn-labels.pt --out runs/fitted-calibration",
        "attr_cmd": "FINTFM_SLOW_ATTRIBUTION=1 pytest tests/test_attribution_stability.py "
        "-k thresholds -rx",
        "steps": [],
    }
    steps: list[dict[str, object]] = summary["steps"]  # type: ignore[assignment]

    # 12.5 first: it is the shorter one, so a fault in the harness shows early.
    plugin_dir = Path("/kaggle/tmp/plugin")
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "attr_dump.py").write_text(PLUGIN)
    attr_json = WORK / "attribution_stability.json"
    attr_log = WORK / "attribution_stability.log"
    steps.append(
        run_step(
            "12.5",
            ATTR_CMD,
            child_env(
                {
                    "FINTFM_SLOW_ATTRIBUTION": "1",
                    "ATTR_JSON": str(attr_json),
                    "PYTHONPATH": str(plugin_dir),
                }
            ),
            ATTR_TIMEOUT_S,
            attr_log,
        )
    )
    counts = pytest_counts(attr_log)
    steps[-1]["pytest"] = counts
    steps[-1]["json_written"] = attr_json.exists()
    # A strict xfail that fails as expected reports "1 xfailed" and exits 0. A skip means the
    # gate did not open; an xpass under strict is a failure and exits non-zero.
    steps[-1]["as_registered_xfail"] = counts.get("xfailed") == 1 and steps[-1]["exit"] == 0
    (WORK / "summary.json").write_text(json.dumps(summary, indent=1))

    steps.append(
        run_step(
            "4.3",
            FITCAL_CMD,
            child_env(),
            FITCAL_TIMEOUT_S,
            WORK / "fitted_calibration.log",
        )
    )
    out = CLONE / "runs" / "fitted-calibration"
    steps[-1]["json_written"] = (out / "fitted_calibration.json").exists()
    if out.exists():
        shutil.copytree(out, WORK / "fitted-calibration", dirs_exist_ok=True)
    summary["total_seconds"] = round(time.time() - started)
    (WORK / "summary.json").write_text(json.dumps(summary, indent=1))
    ckpt.unlink()  # the public file; no need to ship it back
    print(json.dumps(steps))
    print("done")


if __name__ == "__main__":
    main()
