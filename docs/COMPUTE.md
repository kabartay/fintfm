# Compute

Measured throughput, what a real run costs, and where to run it. Every number here was
measured on this machine on the date given — none is estimated. Re-measure rather than
trusting these after an architecture change, because they will move.

## The machine

Apple M3 Max, 16 CPU cores (12 performance, 4 efficiency), 40 GPU cores, 64 GB unified
memory, macOS. **Shared** with a genomics pipeline (`bwa`, `samtools`) that runs
continuously — see `CLAUDE.md`. Observed CPU load averages on 2026-09-08 ranged from 21 to
**145** across the day, the upper end being roughly ninefold oversubscription of 16 cores.

## Metal beats CPU by 3.5x, and that is not the main reason to use it

Measured 2026-09-08 at the Phase 1 configuration (2,175,234 parameters; `max_features=64`,
`d_cell=64`, `d_model=192`, `n_col_layers=2`, `n_layers=6`, batch 32, 256 rows per task).
Batches were pre-generated so this timed compute alone:

| device | s/step | 5,000 steps × 3 variants | 20,000 steps × 3 variants |
| --- | --- | --- | --- |
| CPU (6 threads) | 2.476 | 10.3 h | 41.3 h |
| **Metal (`--device mps`)** | **0.702** | **2.9 h** | **11.7 h** |

Both produced an identical loss of 0.5949 at the same seed, so Metal is numerically sound
for this model rather than merely fast.

**The decisive advantage is not the speedup.** Metal runs on the GPU, which the genomics
pipeline does not touch, so a training run stops competing for the cores that pipeline needs.
On a shared machine that turns "wait until the pipeline finishes" into "run now". Prefer
`--device mps` here even when CPU would be fast enough.

Synthetic task generation is NumPy on the CPU and runs every step, but it is cheap enough
not to matter: 0.036 s per batch of 32 for the financial prior, 0.070 s for the generic one.
Total per step is therefore about 0.74-0.77 s on Metal, and the CPU/GPU split means it
partly overlaps anyway.

## What a real run costs

| run | scale | Metal | notes |
| --- | --- | --- | --- |
| smoke test | 200-400 steps, ~130k params | < 1 min | pipeline check only, says nothing about quality |
| **Phase 1 first pass** | 5,000 steps × 3 variants, 2.2M params | **~3 h** | 80k tasks/variant; directional signal |
| Phase 1 full | 20,000 steps × 3 variants | ~12 h | 640k tasks/variant; overnight |
| ~12M params | `d_model=384`, `n_layers=8`, 20k steps × 3 | ~1.5-2 days | ESTIMATED by scaling, not measured |

Note the gap between the current configuration and `docs/STRATEGY.md`'s stated Phase 1
target of 10-50M parameters: `d_model=192` with 6 layers yields only **2.2M**. Reaching 10M+
needs roughly `d_model=384` and 8 layers, which is a weekend on Metal. The first pass is
deliberately below target — its job is to find out whether the effect exists at all before
spending a weekend measuring it precisely.

## Do we need NVIDIA?

**Not for Phase 1.** A single A100 or H100 would run this perhaps 5-15x faster than Metal,
turning 12 hours into one or two, but Phase 1's answer does not depend on getting there
sooner and the local GPU is free.

**Rent it for Phase 2.** The scaling curve requires training several model sizes at several
data scales, which multiplies the runs; that is where rented compute stops being a
convenience and starts being the difference between a week and a month. A few hundred dollars
of A100 time is the right order of magnitude, and it is not needed until Phase 1 returns a
positive result.

## Rules for running here

- **Check `uptime` first.** At load 145 a native training run is what `CLAUDE.md` records as
  having frozen this machine. Metal reduces but does not eliminate this, since the process
  still needs CPU for the prior.
- **Use `--device mps`**, and `--threads` below the free core count when on CPU.
- **Set `PYTHONUNBUFFERED=1`** for a long background run. Without it, Python's stdout
  buffering plus `tee` hides all progress until the process exits, which is how the first
  Phase 1 run became unmonitorable.
- **Verify a long run is alive by accumulated CPU time, not instantaneous `%CPU`** — read the
  `TIME` column of `ps -o pid,etime,time -p <pid>` twice and check it grew. An instantaneous
  reading on a wrapper shell says nothing about the worker.

## The OpenMP conflict, and why boosting baselines run out-of-process

Discovered 2026-09-09, and any contributor on macOS will hit it.

**Symptom:** a process that has imported `torch` segfaults (exit 139) when it fits LightGBM,
CatBoost or XGBoost. No Python traceback — the crash is below the interpreter.

**Cause,** read directly from the macOS crash report, which lists two OpenMP runtimes mapped
into one process:

```
.../site-packages/torch/lib/libomp.dylib      bundled with PyTorch
/opt/homebrew/opt/libomp/lib/libomp.dylib     Homebrew, loaded by LightGBM
```

**Confirmed by bisection:** LightGBM alone works; `import torch` then LightGBM crashes.
**`KMP_DUPLICATE_LIB_OK=TRUE` does not fix it** — still 139. That workaround is widely
recommended and it is not sufficient here.

**Fix:** `evaluation/boosting.py` fits these models in a subprocess that never imports torch,
exchanging arrays through a temporary `.npz`. Heavy for a benchmark, and the only reliable
separation for a native-library conflict we do not control. Guarded by
`tests/test_metrics.py::test_boosting_baselines_fit_with_torch_loaded`, which imports torch
deliberately.

**Before this existed, LightGBM had never once run** and CatBoost and XGBoost were absent, so
every gradient-boosting comparison used sklearn's weakest implementation and understated the
field by 0.018-0.040 AUC (`docs/FINDINGS.md` §25).

**The wider lesson:** an exception guard converted a crash into a silent skip, and a silently
absent baseline flatters us. Skips are now announced.
