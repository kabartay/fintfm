"""Kaggle CPU kernel for openspec ``time-based-evaluation`` task 3.3: the credit panels with a time split.

Task 3.3's Verify is ``fintfm-bench --credit`` printing random and time-based numbers side by side,
with the gap recorded in FINDINGS from the run output. The local queue entry ``h-timesplit50k``
(``fintfm-bench --time-split --max-rows 50000``) measures only the V4FinBench window pair at 50,000
rows; it does not run the UCI credit panels and does not scale past 50,000 rows. This kernel adds
what that run does not:

1. ``credit_ts100k``: ``fintfm-bench --credit --time-split --max-rows 100000``, the Verify's own
   command line. It prints the Polish (horizons 1, 3, 5) and Taiwanese panels as random-split only,
   states per panel that no time split exists, then the V4FinBench random vs time block at 100,000
   rows. ``--out`` writes the time block as JSON. If this unit fails before writing its JSON (for
   example the UCI download is refused), ``ts100k_fallback`` runs the same time block without
   ``--credit``.
2. ``ts200k``: ``fintfm-bench --time-split --max-rows 200000``, only when enough of the deadline is
   left. It shows how the gap moves with panel size (the local run is 50,000, unit 1 is 100,000).

Inputs, all legitimate to fetch inside the kernel: V4FinBench (``dataset_sources``, Kostrzewa et
al., CC BY 4.0, Kaggle dataset ``sebastiantomczak10/v4-group-corporate-bankruptcy``), the UCI
Polish and Taiwanese bankruptcy sets (downloaded by ``fintfm.evaluation.datasets``, CC BY 4.0), and
the published binary checkpoint ``kabartay/fintfm-binary`` on Hugging Face (Apache-2.0), checked
against its SHA-256 (equal to the local ``runs/v4-cellattn-labels.pt``). No local data or
checkpoint is uploaded. fintfm is installed from GitHub at ``COMMIT``.

Time costs are ESTIMATED, not measured (see ``tasks.md`` 3.3: about 40 ms per row of train plus
test for the fintfm arm at 6,000 rows, scaling past that unmeasured). One subprocess per unit, each
with a timeout; ``summary.json`` records pins, hashes, exit codes, wall times and what was not
started. Outputs land in ``/kaggle/working``.
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
INPUT = Path("/kaggle/input")
#: Kaggle's session limit is 12 h; stop starting work after this many seconds.
BUDGET_S = 11 * 3600
#: Per-unit timeouts (s).
CREDIT_TIMEOUT_S = 5 * 3600
FALLBACK_TIMEOUT_S = 4 * 3600
#: ``ts200k`` starts only with at least this much of the budget left (ESTIMATED need ~4.4 h linear).
TS200K_MIN_LEFT_S = 3 * 3600
#: Margin kept free at the end of the budget for writing ``summary.json``.
MARGIN_S = 10 * 60
V4_FILES = tuple(f"company_years_h{i}.parquet" for i in range(1, 7))


def sha256(path: Path) -> str:
    """SHA-256 of a file, read in chunks.

    Args:
        path: File to hash.

    Returns:
        Lower-case hex digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoint(path: Path, expected: str = CKPT_SHA256) -> Path:
    """Stop the kernel when the checkpoint's SHA-256 differs from the pinned digest.

    Args:
        path: Downloaded checkpoint.
        expected: Pinned digest.

    Returns:
        ``path`` when the hash matches.

    Raises:
        SystemExit: On a mismatch.
    """
    digest = sha256(path)
    if digest != expected:
        sys.exit(f"checkpoint hash {digest} != {expected}")
    return path


def find_v4_root(inputs: Path = INPUT) -> Path:
    """Locate the directory holding all six V4FinBench horizon files.

    Args:
        inputs: Where Kaggle mounts ``dataset_sources``.

    Returns:
        The directory with ``company_years_h1.parquet`` ... ``h6``.

    Raises:
        SystemExit: When the files are missing or ambiguous.
    """
    hits = sorted(inputs.rglob(V4_FILES[0]))
    if len(hits) != 1:
        sys.exit(f"expected one {V4_FILES[0]} under {inputs}, found {hits}")
    root = hits[0].parent
    missing = [f for f in V4_FILES if not (root / f).exists()]
    if missing:
        sys.exit(f"V4FinBench under {root} lacks {missing}")
    return root


def bench_command(ckpt: Path, v4_root: Path, out: Path, max_rows: int, credit: bool) -> list[str]:
    """Build one ``fintfm-bench`` command line.

    Args:
        ckpt: Checkpoint path.
        v4_root: V4FinBench directory.
        out: JSON path for the time block.
        max_rows: Company-grouped subsample size.
        credit: Whether to add ``--credit`` (the UCI panels and the sample-efficiency probe).

    Returns:
        The argument vector.
    """
    cmd = [sys.executable, "-m", "fintfm.evaluation.bench"]
    if credit:
        cmd.append("--credit")
    cmd += [
        "--time-split",
        "--max-rows",
        str(max_rows),
        "--v4-root",
        str(v4_root),
        "--model",
        str(ckpt),
        "--out",
        str(out),
    ]
    return cmd


def run_unit(name: str, cmd: list[str], timeout_s: float) -> dict:
    """Run one unit as a subprocess with a timeout, logging to ``<name>.log``.

    Args:
        name: Unit name; also the log stem.
        cmd: Argument vector.
        timeout_s: Kill the process after this many seconds.

    Returns:
        ``{"unit", "exit_code" (None on timeout), "timed_out", "wall_s"}``.
    """
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    t0 = time.time()
    timed_out = False
    with (WORK / f"{name}.log").open("w") as log:
        try:
            rc: int | None = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout_s, check=False
            ).returncode
        except subprocess.TimeoutExpired:
            rc, timed_out = None, True
    return {"unit": name, "exit_code": rc, "timed_out": timed_out, "wall_s": round(time.time() - t0)}


def should_run_fallback(elapsed_s: float, credit_json_written: bool) -> bool:
    """Whether ``ts100k_fallback`` should start (pure, for tests).

    Args:
        elapsed_s: Seconds since the kernel started.
        credit_json_written: Whether ``credit_ts100k`` wrote its JSON.

    Returns:
        True when the credit unit left no JSON and over an hour plus the margin remains.
    """
    return (not credit_json_written) and BUDGET_S - elapsed_s > MARGIN_S + 3600


def should_run_ts200k(elapsed_s: float) -> bool:
    """Whether ``ts200k`` should start (pure, for tests).

    Args:
        elapsed_s: Seconds since the kernel started.

    Returns:
        True when at least ``TS200K_MIN_LEFT_S`` of the budget remains.
    """
    return BUDGET_S - elapsed_s >= TS200K_MIN_LEFT_S


def main() -> None:
    """Install the pinned fintfm, fetch and verify the checkpoint, run the units, write the summary."""
    t_start = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    v4_root = find_v4_root()
    print(f"data: {v4_root}", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet",
         f"fintfm[bench] @ git+https://github.com/kabartay/fintfm.git@{COMMIT}"],
        check=True,
    )
    ckpt = WORK / "v4-cellattn-labels.pt"
    urllib.request.urlretrieve(CKPT_URL, ckpt)
    verify_checkpoint(ckpt)
    results: list[dict] = []
    not_started: list[str] = []

    out1 = WORK / "credit_ts100k.json"
    results.append(
        run_unit("credit_ts100k", bench_command(ckpt, v4_root, out1, 100_000, True),
                 CREDIT_TIMEOUT_S)
    )
    if not out1.exists():
        if should_run_fallback(time.time() - t_start, False):
            left = BUDGET_S - (time.time() - t_start) - MARGIN_S
            results.append(
                run_unit("ts100k_fallback",
                         bench_command(ckpt, v4_root, WORK / "ts100k_fallback.json", 100_000, False),
                         min(FALLBACK_TIMEOUT_S, left))
            )
        else:
            not_started.append("ts100k_fallback")
    elapsed = time.time() - t_start
    if should_run_ts200k(elapsed):
        results.append(
            run_unit("ts200k",
                     bench_command(ckpt, v4_root, WORK / "ts200k.json", 200_000, False),
                     BUDGET_S - elapsed - MARGIN_S)
        )
    else:
        not_started.append("ts200k")
    ckpt.unlink()  # the public file; no need to ship it back
    summary = {
        "commit": COMMIT,
        "checkpoint_sha256": CKPT_SHA256,
        "checkpoint_url": CKPT_URL,
        "v4_root": str(v4_root),
        "units": results,
        "not_started": not_started,
        "total_wall_s": round(time.time() - t_start),
    }
    (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
