# AU5.15: one-session throughput measurement before a 2M-task pretraining run (2026-11)

**Status: kernel prepared and smoke-tested; nothing launched.** The owner launches it at the next
Kaggle quota reset. No number below is a throughput result for this task; the only numbers from
the new kernel are SMOKE-TEST. Every other figure is quoted from the finding named beside it.

## What is already known

- Generation, local (MEASURED, AU5.15 task text, load 514-521 on 16 cores, 128 batches of the
  reference recipe): K=1 4.94 batches/s, K=4 3.43 batches/s (compressed 2.36). No speed-up on a
  saturated machine; an idle machine has not been measured.
- Training on a Kaggle T4 (MEASURED, FINDINGS §170 addendum, fintfm 49f415c, published recipe,
  50 timed steps, batch 8): fp32 masked 14,256 tasks/h; fp16 split 37,367 tasks/h (0.66 s/step),
  fp16 masked 33,959 tasks/h. CPU task generation was about 5% of a step (4.9 of 101.6 s,
  §170 v3, fp32), so a dump should buy little on one GPU.
- DDP on two T4s (MEASURED, §170, fp32 split, 20 steps): about 27,000 tasks/h for the pair, about
  1.9x one GPU. fp16 with DDP was not measured together; §170 estimates 65,000 to 70,000 tasks/h
  per session (ESTIMATED).
- **A gap that decides the plan:** `fintfm-train` refuses `--task-dump` together with `--ddp`
  (`_check_train_config`). A dump run is therefore single-GPU, about 37,367 tasks/h (MEASURED
  by §170 for live generation, not yet for a dump); the two-GPU path needs `--loader stream`.
  At 2M tasks that is the difference between about 53.5 and about 29 to 31 session-hours
  (ESTIMATED, below). Whether a dump is worth it is a question the session answers; the
  dump's only likely gain is when generation (about 5% of a step on a T4) is the bottleneck.

## What the session measures (`scripts/kaggle/task_au515_throughput/`)

Kernel `muhakabartay/fintfm-task-au515-throughput`, T4x2 with internet, fp16 and split attention
(E1's recipe, §171), batch 8, seed 0. Output `/kaggle/working/au515/summary.json`.

1. **Generation**, `write_task_dump(..., stream="per_shard", workers=K)` (the call behind
   `scripts/dump_prior_tasks.py`), K = 1, 2, 4 on the session's CPUs, 320 batches (2,560 tasks)
   in 32-batch shards: batches/s, tasks/s, MB per batch, load average.
2. **Benchmarks**, `--benchmark-steps 50`, one GPU: live sampling, `--task-dump` (the K=4 dump),
   `--loader stream --loader-workers 2`. Tasks per hour and the sample/step split.
3. **Short real runs**, 300 steps at batch 8, `--checkpoint-every 100`: live without checkpoints,
   live with, `--task-dump` with, and two-GPU DDP stream (batch 4 per rank) with. Rates from
   the train log's cumulative seconds, so start-up is excluded and checkpoint saves included.
   Checkpoint and state sizes recorded.

Derived in the JSON: tasks per GPU-hour (a two-GPU run is halved, since the quota counts
GPU-hours), generation tasks per CPU-hour, and the dump size for 2M tasks in MB.

Estimated wall time **25 to 30 minutes** (ESTIMATED: install about 2 min; generation 3 arms about
3 min at 4.9 batches/s per core; 3 benchmarks about 4 min including start-up; 4 runs of 300
steps at 0.66 s/step plus start-up and saves about 15 min). GPU quota used: about 20 GPU-minutes
single plus about 8 minutes on two GPUs (ESTIMATED).

## How the 2M-task cost is computed from it

With R = tasks per GPU-hour of the arm chosen (take the `runs` rate, which includes checkpoints,
not the benchmark rate):

- GPU-hours for 2M tasks = 2,000,000 / R. At §170's 37,367 tasks/h that is **53.5 GPU-hours**
  (ESTIMATED); at 33,959 (fp16 masked) 58.9 GPU-hours (ESTIMATED). Against the 30 GPU-hour
  Kaggle week (preregs_2026-11.md) that is about 1.8 weeks, in sessions of about 9 hours.
- Sessions = GPU-hours / (session length x GPUs used); steps = 2,000,000 / 8 = 250,000.
- Generation side: CPU-hours = 2,000,000 / (generation tasks per CPU-hour from stage 1). A dump
  needs `dump_mb_per_2m_tasks` of disk; if that exceeds /kaggle/working's limit the dump is
  written and consumed in chunks, or the stream loader is used instead.
- Dump versus stream: the dump is worth it only if the `task_dump` run's tasks/GPU-hour beats
  `live` by more than the generation CPU-hours cost, and DDP stream's per-GPU rate below shows
  what a dump forfeits by being single-GPU. Choose the arm with the highest tasks per GPU-hour
  that fits the disk.
- Cost in dollars: Kaggle quota is free, so the cost is GPU-hours of quota; on paid GPUs use
  §127's 1.02 s/step on an L4 (about $28 per million tasks, ESTIMATED, CLAUDE.md) as the
  comparison, rechecked against the measured rate.

Then, as AU5.15 requires, after the full run: score on V4FinBench five folds first, average
precision first, against a 48k-task control at the same commit, under §159's rule (positive at
both seeds and a mean difference above 0.0073 mean AP, from §158's no-change gap), then
TabArena-Lite. A volume arm needs seed replicates; the design is §171's `V480` against `V48`.

## SMOKE-TEST of the script path (not a throughput result)

Run locally, CPU, 2 threads, tiny model (d_model 32, 1 layer, 16 features), batch 2, 12 batches
generation, 3 benchmark steps, 6 training steps, `--checkpoint-every 3`, on a machine at load
average about 157 (SMOKE-TEST). Exit 0; every stage wrote its record (generation K=1, 2;
three benchmarks; three single-process runs; the DDP arm is skipped without two GPUs).
`summary.json` copy: `runs/agents1104/AU5.15/smoke_summary.json`. Its rates, including K=2 at
0.51 batches/s against K=1 at 82.6 batches/s, say nothing: with 12 tiny batches process
start-up dominates, and the machine was saturated. Not quoted as evidence for any decision.
Not exercised by the smoke test: the pip install, the recipe download, CUDA, and the DDP arm.

## Launch (owner)

1. Push a commit containing the dump and `--checkpoint-every` code (at least b1084ce) and set
   `COMMIT` in `scripts/kaggle/task_au515_throughput/kernel.py` to it; the kernel refuses a
   placeholder.
2. After the quota reset:

```bash
cd /Users/morganoko/proj/fintfm
kaggle kernels push -p scripts/kaggle/task_au515_throughput
```

3. Monitor the step rate within 15 minutes (CLAUDE.md): the fp16 split single-GPU arm should
   show about 0.66 s per step. Fetch `summary.json` into `runs/kaggle-au515/`, then fill the
   figures above and close the task with measured numbers.

## Open

- Whether `--task-dump` could be allowed with `--ddp` (each rank reading its own batch indices)
  is a code change, not made here; the session shows whether it would matter.
- The measurement is at 8 tasks per step and 320 generated batches; per_shard generation at
  K=4 on 4 vCPUs shares cores with the training process only in the real run, not here.
