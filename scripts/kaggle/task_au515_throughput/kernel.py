"""Kaggle kernel: audit AU5.15, the one-session throughput measurement before a 2M-task run.

Nothing is trained to keep and no real data is read. In one session on a T4x2 it measures,
at the published recipe (`configs/published/v4-cellattn-labels.yaml`, fp16, split attention):

1. Prior generation throughput: `write_task_dump` (what `scripts/dump_prior_tasks.py` calls),
   `--stream per_shard`, K = 1, 2 and 4 workers on this session's CPUs, with the dump size in
   MB per batch (the 2M-task dump must fit /kaggle/working's disk).
2. Training throughput, `fintfm-train --benchmark-steps 50` on one GPU: live generation on
   the training process (sync), `--task-dump` from the K=4 dump, and the streaming loader.
3. A short real training run per source, 300 steps at batch 8 with `--checkpoint-every 100`
   (and one live run without checkpoints, so the checkpoint cost shows): the single-GPU live
   run, the single-GPU `--task-dump` run, and the two-GPU DDP streaming run at batch 4 per
   rank. `--task-dump` with `--ddp` is refused by fintfm-train, so DDP has no dump arm.
   Rates come from the train log's cumulative `Ns` stamps, which exclude start-up.

It writes /kaggle/working/au515/summary.json with every figure and the derived tasks per
GPU-hour (a two-GPU run is divided by two GPUs, for GPU-hours as the quota counts them). The
2M-task cost is computed from that file as docs/results/drafts/au5.15_2026-11.md describes.

Estimated wall time 25 to 30 minutes (ESTIMATED from FINDINGS §170's 0.66 s per step for fp16
split at batch 8 on a T4, and about 4.9 batches/s generation on one core); GPUs are in use
only during the benchmark and training arms.

Local smoke test (CPU, tiny model, no GPU, no pip install): set FINTFM_AU515_SMOKE=1 and run
this file from the repository root with `uv run`; the output goes to ./au515_smoke/.
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Pin to a pushed commit that contains the task dump and --checkpoint-every (3705a71 and
# b1084ce, then everything after). The placeholder refuses to run rather than install a wrong ref.
COMMIT = "<pin to a pushed commit>"
RECIPE_URL = (f"https://raw.githubusercontent.com/kabartay/fintfm/{COMMIT}/"
              "configs/published/v4-cellattn-labels.yaml")
SMOKE = os.environ.get("FINTFM_AU515_SMOKE") == "1"

OUT = Path("./au515_smoke") if SMOKE else Path("/kaggle/working/au515")
BATCH = 8                 # tasks per global step, the reference recipe's
N_GEN_BATCHES = 320       # batches per generation arm (2,560 tasks); the training arms read <= 303
SHARD = 32                # batches per shard, so K = 4 has 10 shards to share
BENCH_STEPS = 50
TRAIN_STEPS = 300
CKPT_EVERY = 100
WORKERS = (1, 2, 4)
# Flags every training and benchmark command shares: fp16 and split attention are E1's recipe
# (FINDINGS §170 addendum, §171).
RECIPE_FLAGS = ["--amp", "fp16", "--attn-impl", "split", "--seed", "0",
                "--eval-every", "1000000", "--log-every", "20"]
DEVICE = "cuda"
TINY: list[str] = []

if SMOKE:
    # a few steps of a tiny model on CPU with two threads; checks the script path only
    N_GEN_BATCHES, SHARD, BENCH_STEPS, TRAIN_STEPS, CKPT_EVERY, WORKERS = 12, 4, 3, 6, 3, (1, 2)
    BATCH = 2
    DEVICE = "cpu"
    RECIPE_FLAGS = ["--seed", "0", "--eval-every", "1000000", "--log-every", "2",
                    "--threads", "2", "--warmup-steps", "2"]
    # tiny model and prior; d_ff is pinned in the recipe so it is passed too
    TINY = ["--d-cell", "16", "--d-model", "32", "--n-heads", "2", "--d-ff", "64",
            "--n-layers", "1", "--n-col-layers", "1", "--max-features", "16",
            "--n-rows-choices", "32,64", "--feature-chunk", "8"]


def sh(cmd: list[str], log: Path) -> tuple[int, float]:
    """Run a command, write its output to ``log`` and the kernel log, return (exit, seconds).

    Args:
        cmd: The command.
        log: File receiving the combined output.

    Returns:
        The exit code and the wall-clock seconds.
    """
    print("$ " + " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          check=False)
    wall = time.perf_counter() - t0
    log.write_text(proc.stdout)
    print(proc.stdout[-2500:], flush=True)
    return proc.returncode, wall


def dir_bytes(path: Path) -> int:
    """Total size in bytes of the files directly in ``path``."""
    return sum(f.stat().st_size for f in path.iterdir() if f.is_file())


def generation_arms(recipe: Path) -> dict:
    """Generate ``N_GEN_BATCHES`` batches per-shard at each worker count and time it.

    Args:
        recipe: The recipe YAML.

    Returns:
        One record per K: seconds, batches/s, tasks/s, MB per batch and the load average.
    """
    from fintfm.config import load_config
    from fintfm.modeling.train import build_configs, build_parser
    from fintfm.prior.dump import write_task_dump

    args = build_parser().parse_args(["--config", str(recipe), "--out", "/dev/null",
                                      "--batch-size", str(BATCH), *TINY])
    _, prior_cfg, _ = build_configs(args, load_config(recipe))
    records = {}
    for k in WORKERS:
        out = OUT / f"dump_k{k}"
        t0 = time.perf_counter()
        write_task_dump(out, prior_cfg, seed=0, batch_size=BATCH, n_batches=N_GEN_BATCHES,
                        shard_size=SHARD, stream="per_shard", workers=k)
        secs = time.perf_counter() - t0
        size = dir_bytes(out)
        records[f"k{k}"] = {
            "workers": k, "batches": N_GEN_BATCHES, "tasks": N_GEN_BATCHES * BATCH,
            "seconds": secs, "batches_per_s": N_GEN_BATCHES / secs,
            "tasks_per_s": N_GEN_BATCHES * BATCH / secs,
            "mb_per_batch": size / N_GEN_BATCHES / 1e6, "load_average": os.getloadavg(),
        }
        print(json.dumps(records[f"k{k}"]), flush=True)
        if k != max(WORKERS):
            for f in out.iterdir():  # keep only the largest-K dump for the training arms
                f.unlink()
            out.rmdir()
    return records


def log_rate(log_text: str, batch: int) -> dict | None:
    """Training rate from the cumulative ``step N/M ... Ns`` stamps in a train log.

    Args:
        log_text: The training log.
        batch: Global tasks per step.

    Returns:
        Seconds per step and tasks per second between the first and last stamp (start-up and
        the first interval excluded, checkpoint saves included), or ``None`` with fewer than
        two stamps.
    """
    stamps = [(int(m[1]), float(m[2])) for m in
              re.finditer(r"step (\d+)/\d+\s+loss \S+\s+lr \S+\s+(\d+)s", log_text)]
    if len(stamps) < 2:
        return None
    (s0, t0), (s1, t1) = stamps[0], stamps[-1]
    per_step = (t1 - t0) / (s1 - s0)
    return {"stamps": len(stamps), "seconds_per_step": per_step,
            "tasks_per_s": batch / per_step if per_step > 0 else None}


def main() -> None:
    """Run the three measurement stages and write summary.json."""
    OUT.mkdir(parents=True, exist_ok=True)
    gpus = 0
    names: list[str] = []
    if not SMOKE:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, check=False)
        names = [n.strip() for n in smi.stdout.splitlines() if n.strip()]
        if smi.returncode != 0 or not names:
            sys.exit("no GPU in this session (quota exhausted?) -- refusing a CPU run")
        gpus = len(names)
        if COMMIT.startswith("<"):
            sys.exit("COMMIT is still a placeholder: pin it to a pushed commit first")
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        f"fintfm @ git+https://github.com/kabartay/fintfm.git@{COMMIT}"],
                       check=True)
        recipe = OUT / "v4-cellattn-labels.yaml"
        urllib.request.urlretrieve(RECIPE_URL, recipe)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    else:
        recipe = Path("configs/published/v4-cellattn-labels.yaml")
    summary: dict = {"commit": COMMIT, "smoke": SMOKE, "gpus": names, "cpus": os.cpu_count(),
                     "batch": BATCH, "train_steps": TRAIN_STEPS, "bench_steps": BENCH_STEPS}

    # 1. generation (no GPU used); keeps the largest-K dump for the --task-dump arms
    summary["generation"] = generation_arms(recipe)
    dump = OUT / f"dump_k{max(WORKERS)}"
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    base = [sys.executable, "-u", "-m", "fintfm.modeling.train", "--config", str(recipe),
            "--device", DEVICE, "--batch-size", str(BATCH), *RECIPE_FLAGS, *TINY]

    # 2. benchmark arms, single process
    bench = {}
    for name, extra in [("live", []), ("task_dump", ["--task-dump", str(dump)]),
                        ("stream_w2", ["--loader", "stream", "--loader-workers", "2"])]:
        cmd = [*base, "--benchmark-steps", str(BENCH_STEPS), "--out", str(OUT / "unused.pt"),
               *extra]
        rc, wall = sh(cmd, OUT / f"bench_{name}.log")
        lines = [ln for ln in (OUT / f"bench_{name}.log").read_text().splitlines()
                 if ln.startswith("{")]
        bench[name] = {"exit": rc, "wall_s": wall,
                       "result": json.loads(lines[-1]) if lines else None}
    summary["benchmark"] = bench

    # 3. short real runs with checkpoints
    runs = {}
    single = [("live_nockpt", [], 0), ("live", [], CKPT_EVERY),
              ("task_dump", ["--task-dump", str(dump)], CKPT_EVERY)]
    for name, extra, every in single:
        ckpt = OUT / f"run_{name}.pt"
        cmd = [*base, "--steps", str(TRAIN_STEPS), "--checkpoint-every", str(every),
               "--out", str(ckpt), *extra]
        rc, wall = sh(cmd, OUT / f"run_{name}.log")
        state = Path(f"{ckpt}.state")
        runs[name] = {"exit": rc, "wall_s": wall, "checkpoint_every": every, "gpus": 1,
                      "rate": log_rate((OUT / f"run_{name}.log").read_text(), BATCH),
                      "checkpoint_mb": ckpt.stat().st_size / 1e6 if ckpt.exists() else None,
                      "state_mb": state.stat().st_size / 1e6 if state.exists() else None}
        for f in OUT.glob(f"run_{name}.pt*"):
            f.unlink()
    if gpus >= 2:
        ckpt = OUT / "run_ddp_stream.pt"
        cmd = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
               "--nproc-per-node", "2", "-m", "fintfm.modeling.train", "--config", str(recipe),
               "--device", "cuda", *RECIPE_FLAGS, "--batch-size", str(BATCH // 2), "--ddp",
               "--loader", "stream", "--loader-workers", "1", "--steps", str(TRAIN_STEPS),
               "--checkpoint-every", str(CKPT_EVERY), "--out", str(ckpt)]
        rc, wall = sh(cmd, OUT / "run_ddp_stream.log")
        runs["ddp_stream"] = {"exit": rc, "wall_s": wall, "checkpoint_every": CKPT_EVERY,
                              "gpus": 2,
                              "rate": log_rate((OUT / "run_ddp_stream.log").read_text(), BATCH)}
        for f in OUT.glob("run_ddp_stream.pt*"):
            f.unlink()
    summary["runs"] = runs

    # derived: tasks per GPU-hour (quota counts GPU-hours, so a two-GPU run is halved)
    derived = {}
    for name, rec in runs.items():
        if rec["rate"] and rec["rate"]["tasks_per_s"]:
            derived[name] = 3600.0 * rec["rate"]["tasks_per_s"] / rec["gpus"]
    for name, rec in bench.items():
        if rec["result"]:
            derived[f"bench_{name}"] = rec["result"]["tasks_per_hour"]
    summary["tasks_per_gpu_hour"] = derived
    gen = summary["generation"]
    summary["generation_tasks_per_cpu_hour"] = {
        k: 3600.0 * v["tasks_per_s"] / v["workers"] for k, v in gen.items()}
    summary["dump_mb_per_2m_tasks"] = gen[f"k{max(WORKERS)}"]["mb_per_batch"] * 2e6 / BATCH
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    for f in list(OUT.glob("dump_k*/*")):
        f.unlink()
    for d in OUT.glob("dump_k*"):
        d.rmdir()
    print("done")


if __name__ == "__main__":
    main()
