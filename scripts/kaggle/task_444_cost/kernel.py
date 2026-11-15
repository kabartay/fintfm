"""Kaggle kernel: task 44.4, training cost of the factorized cell encoder against attention.

Nothing is trained to keep and no real data is read. In one session on one T4 it runs
`fintfm-train --benchmark-steps N` (the project's own step: optimiser, schedule, fp16 and split
attention as the 48,000-task arms train them) once per cell of a grid, each in its own
subprocess so an out-of-memory in one cell is recorded and the next still runs:

    cell mixing {attention (the default), factorized} x rows per task {512, 1024, 2048, 4096}
    at the published model (batch 4, the per-rank batch of ts_445_*), and
    cell mixing x rows {512, 1024} at the 5M-parameter "medium" shape of FINDINGS §94
    (d_cell 64, d_model 256, n_layers 6, n_col_layers 3, d_ff 1024) at batch 2 and 8.

Every cell reports peak training memory (`torch.cuda.max_memory_allocated` over the timed
steps, from the benchmark record), seconds per step, the sample/step split and the parameter
count, into /kaggle/working/cost444/summary.json. `--max-query-rows 512` caps queries as AU5.4
did, so "rows" is the context size plus at most 512 queries; the records carry the flag.

A T4 has 14.56 GB, not the 24 GB card 44.4 names: the 5M-parameter cell at `n_rows=1024` and
batch 8 may be refused here and is then reported as such, with the batch-2 peak beside it, from
which the 24 GB statement is a linear extrapolation (ESTIMATED), not a measurement. The
statement itself has to be made on a 24 GB card.

ESTIMATED wall time 1.5 to 2.5 h (arithmetic in openspec/changes/factorized-attention/tasks.md
44.4), each cell capped at CELL_TIMEOUT_S.

Local smoke test (CPU, tiny model, no GPU, no pip install): set FINTFM_444_SMOKE=1 and run this
file from the repository root with `uv run`; the output goes to ./cost444_smoke/.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Pinned to a pushed commit that carries cell_mixing="factorized" (44.3).
COMMIT = "d58141e"
RECIPE_URL = (f"https://raw.githubusercontent.com/kabartay/fintfm/{COMMIT}/"
              "configs/published/v4-cellattn-labels.yaml")
SMOKE = os.environ.get("FINTFM_444_SMOKE") == "1"
OUT = Path("./cost444_smoke") if SMOKE else Path("/kaggle/working/cost444")

MIXINGS = ("attention", "factorized")
#: (shape name, model flags beyond the recipe, rows grid, batch sizes)
SHAPES = [
    ("published", [], (512, 1024, 2048, 4096), (4,)),
    ("medium5m", ["--d-cell", "64", "--d-model", "256", "--n-layers", "6", "--n-col-layers", "3",
                  "--d-ff", "1024"], (512, 1024), (2, 8)),
]
#: timed steps per cell; the largest cells take fewer so the session stays bounded
STEPS_SMALL, STEPS_LARGE = 200, 100
CELL_TIMEOUT_S = 1800
EXEC_FLAGS = ["--amp", "fp16", "--attn-impl", "split", "--seed", "0", "--eval-every", "1000000",
              "--max-query-rows", "512"]
DEVICE = "cuda"

if SMOKE:
    SHAPES = [("published", ["--d-cell", "16", "--d-model", "32", "--n-heads", "2", "--d-ff", "64",
                             "--n-layers", "1", "--n-col-layers", "1", "--max-features", "16",
                             "--feature-chunk", "8"], (64, 128), (2,))]
    STEPS_SMALL = STEPS_LARGE = 3
    CELL_TIMEOUT_S = 600
    EXEC_FLAGS = ["--seed", "0", "--eval-every", "1000000", "--threads", "2",
                  "--warmup-steps", "2", "--max-query-rows", "32"]
    DEVICE = "cpu"


def cells() -> list[dict]:
    """The grid, ordered so the cheap cells run first and a timeout loses the expensive tail.

    Returns:
        One dict per cell: shape name, model flags, mixing, rows, batch and timed steps.
    """
    out = []
    for name, flags, rows_grid, batches in SHAPES:
        for rows in rows_grid:
            for batch in batches:
                for mixing in MIXINGS:
                    out.append({"shape": name, "flags": flags, "mixing": mixing, "rows": rows,
                                "batch": batch,
                                "steps": STEPS_LARGE if rows >= 2048 else STEPS_SMALL})
    return sorted(out, key=lambda c: (c["rows"] * c["batch"], c["shape"], c["mixing"]))


def command(recipe: Path, cell: dict) -> list[str]:
    """The benchmark command of one cell.

    Args:
        recipe: The recipe YAML.
        cell: One element of :func:`cells`.

    Returns:
        The argv; the mixing flag is the only thing differing between a cell's two arms.
    """
    return [sys.executable, "-u", "-m", "fintfm.modeling.train", "--config", str(recipe),
            "--device", DEVICE, *EXEC_FLAGS, *cell["flags"],
            "--batch-size", str(cell["batch"]), "--n-rows-choices", str(cell["rows"]),
            "--cell-mixing", cell["mixing"], "--benchmark-steps", str(cell["steps"]),
            "--out", str(OUT / "unused.pt")]


def run_cell(recipe: Path, cell: dict) -> dict:
    """Run one cell in a subprocess and parse its JSON line.

    Args:
        recipe: The recipe YAML.
        cell: One element of :func:`cells`.

    Returns:
        The cell with ``result`` (the benchmark record or None), ``exit``, ``wall_seconds`` and,
        on a failure, ``oom`` (whether the output names CUDA out of memory) and ``tail``.
    """
    cmd = command(recipe, cell)
    print("$ " + " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              check=False, timeout=CELL_TIMEOUT_S)
        rc, text = proc.returncode, proc.stdout
    except subprocess.TimeoutExpired as exc:
        rc, text = -9, (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
    wall = time.perf_counter() - t0
    lines = [ln for ln in text.splitlines() if ln.startswith("{")]
    rec = {k: v for k, v in cell.items() if k != "flags"}
    rec.update({"command": cmd, "exit": rc, "wall_seconds": wall,
                "result": json.loads(lines[-1]) if rc == 0 and lines else None})
    if rec["result"] is None:
        rec["oom"] = "out of memory" in text.lower()
        rec["tail"] = text[-1500:]
    res = rec["result"]
    print(f"  -> exit {rc}, {wall:.0f} s" + (
        f", {res['step_seconds'] / res['steps']:.3f} s/step, "
        f"peak {(res['peak_memory_bytes'] or 0) / 1e9:.2f} GB, {res['parameters']:,} params"
        if res else f", failed (oom={rec['oom']})"), flush=True)
    return rec


def main() -> None:
    """Measure every cell and write ``summary.json`` after each, so a timeout keeps the rest."""
    if not SMOKE:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             check=False)
        if smi.returncode != 0 or not smi.stdout.strip():
            sys.exit("no GPU in this session (quota exhausted?) -- refusing a CPU run")
        print(f"GPUs: {smi.stdout.strip()}", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        f"fintfm @ git+https://github.com/kabartay/fintfm.git@{COMMIT}"],
                       check=True)
    OUT.mkdir(parents=True, exist_ok=True)
    recipe = OUT / "recipe.yaml"
    if SMOKE:
        recipe = Path("configs/published/v4-cellattn-labels.yaml")
    else:
        urllib.request.urlretrieve(RECIPE_URL, recipe)
    summary: dict = {"commit": COMMIT, "smoke": SMOKE, "records": []}
    for cell in cells():
        summary["records"].append(run_cell(recipe, cell))
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
