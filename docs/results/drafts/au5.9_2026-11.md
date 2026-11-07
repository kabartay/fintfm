# AU5.9: hazard context carries the default period (pre-registration draft)

**DRAFT, 2026-11-04, before any arm runs.** Not yet in FINDINGS.md. What exists: the Verify task
and scorer (built here), a SMOKE-TEST of the pipeline, two arm configs and two unpinned Kaggle
kernels. What does not exist: a trained flag-on checkpoint, so no quality number. Every cost is
ESTIMATED from measured step times, quoted with their source.

## What is open, stated first

Hazard context carried only the binary label (`model.py`, audit AU5.9). `--hazard-period-context`
(6ec5534, 0a3b8c8) and `fit(X, y, periods=...)` (f69f5a4) exist, off by default. No flag-on
checkpoint exists, and the published one is classification-only
(`inference_tasks_2026-11.md`, AU5.9 section). Until now there was no task on which the periods
could be shown to matter. The Verify below is that task. Whether a pretrained flag-on model reads
periods on it is **unmeasured**.

## The Verify (built)

`src/fintfm/experiments/hazard_period_probe.py`, tests in `tests/test_hazard_period_probe.py`
(9 tests) and `tests/test_au59_arms.py` (3 tests).

- **Task.** K = 6 periods. Two regimes, **early** (default-period pmf proportional to 0.5^k, so
  F(0) = 0.508) and **late** (the reverse, F(0) = 0.016). A task's features and binary labels
  come from its seed alone, so an early and a late task of the same seed have **identical X and
  y** (asserted); only the defaulters' periods differ. The binary default rate is about 0.25 by
  design (logit intercept -1.3, signal 1.5 on feature 0). Truth for a query row: cumulative PD at
  horizon k = p(x) * F_regime(k). A model that sees only (X, y) cannot tell the regimes apart.
- **Two arms per checkpoint.** *With periods*: the context's true default periods. *Blind*: the
  same defaulters with periods redrawn uniformly, which carries no regime information. A
  flag-off checkpoint ignores periods, so its two arms coincide: the control that cannot pass.
- **Errors.** Primary, *shape RMSE*: the normalised mean predicted curve m_k / m_(K-1) against
  F_regime(k), per task, level-free. Secondary, *cumulative-PD RMSE* against p(x) * F_regime(k).
  The level is printed beside the shape (`shape[0]` for early and late tasks, truth 0.508 and
  0.016), per the repository's rule on printing the level.
- **Floor of a periods-blind model.** It sees the mixture F_mix = (F_early + F_late) / 2 and can
  do no better than predict it: shape floor = rms(F_early - F_mix) = **0.2965** at K = 6, decay
  0.5 (computed by the scorer; unit-tested as the error of the mixture predictor). The
  cumulative-PD floor is per battery (smoke battery: 0.1133). It is the error of a blind model
  that knows p(x) exactly, so a real model sits above it unless it reads the periods.
- **Pass criterion (pre-registered).** On the shape error, (1) the with-periods mean is below
  the blind floor, and (2) the paired per-task difference with-minus-blind has a 95% bootstrap
  interval (2,000 resamples over tasks) wholly below zero, and (3) the checkpoint is flag-on.
  **Strong pass**: with-periods shape RMSE at or below half the floor (0.148). 200 tasks, 400
  context rows, 200 query rows per task, seed 0.
- **Caveat.** The task is synthetic and outside the prior's own period structure (the prior draws
  a per-task hazard shape, `_sample_survival`, with periods tied to latent distress). A fail
  means the model does not use periods *on this task*; it does not by itself show the flag is
  useless on the prior's tasks or on V4FinBench.

## SMOKE-TEST: the pipeline runs (no quality claim)

Tiny training on CPU, `nice -n 19`, 2 threads, 2026-11-04. Outputs in
`runs/agents1104/AU5.9/` (`smoke-on.pt`, `smoke-off.pt`, `probe-smoke/*.json`).

```bash
uv run --frozen --extra bench --extra hf fintfm-train --device cpu --steps 40 --batch-size 2 \
  --n-rows-choices 64,128 --max-features 8 --d-model 32 --d-cell 16 --n-layers 2 --n-col-layers 1 \
  --n-horizons 6 --hazard-period-context --p-financial 1.0 --checkpoint-every 20 --seed 0 \
  --out runs/agents1104/AU5.9/smoke-on.pt        # smoke-off.pt: the same without the flag
uv run --frozen --extra bench --extra hf python -m fintfm.experiments.hazard_period_probe \
  --model runs/agents1104/AU5.9/smoke-on.pt runs/agents1104/AU5.9/smoke-off.pt \
  --n-tasks 20 --n-ctx 200 --n-query 100 --out runs/agents1104/AU5.9/probe-smoke
```

- **SMOKE-TEST.** Both trainings finished (40 steps, batch 2, checkpoints at steps 20 and 40);
  30,240 parameters flag-on against 30,016 flag-off (the 224 extra are the period embedding, 7
  tokens x 32).
- **SMOKE-TEST.** The flag-on checkpoint loads, `fit(..., periods=...)` accepts periods,
  `predict_term_structure` runs, and the scorer reports both arms. Shape RMSE 0.3388 with and
  without periods (floor 0.2965), identical to four decimals: after 40 steps at the warm-up
  rate the period embedding has not moved enough to change the curve. Verdict: not passed, as it
  must be for a model this untrained. Flag-off: 0.3391 in both arms, not passed.
- This says the plumbing works. It says nothing about the architecture or the flag.

## Update 2026-11-07: hazard loss fixed for half precision

The two kernels were held because `_autocast` refused `--amp` for any model with a hazard
head: the survival NLL takes `log1p(-h)` of hazards clamped at 0.999, which half precision
rounds to one (bf16 spacing near one is 2^-8, so 0.999 becomes 1.0 and the log is `-inf`). The
owner chose to fix the loss rather than train in float32 (which would cost about 2.4x the GPU
time, ESTIMATED from section 86).

- **Fix.** `HazardHead.hazards` (`src/fintfm/modeling/hazard.py`) casts its input to float32
  and runs the projection, sigmoid and clamp with autocast disabled. `cumulative_pd` and
  `loss` both go through it, so the whole NLL is float32. Float32 input gives bit-identical
  output (tested), so `--amp none` is unchanged.
- **Refusal.** `_autocast` (`train.py`) now refuses only a non-binned (quantile) head. The
  quantile pinball loss is not made safe here: it would need a float32 cast of predictions and
  targets in `model.py`, which is not trivially safe, so the refusal stays.
- **Tests (SMOKE-TEST, CPU).** bf16 autocast NLL equals the float32 NLL within 1e-6 on hazards
  at the clamp, where the bf16-rounded hazards are exactly one; `_autocast` accepts hazard
  under bf16 and refuses quantile; a 3-step `--amp bf16 --n-horizons 6` run through `train`
  and `_training_step` has finite float32 loss. CUDA fp16 with the grad scaler on a hazard
  model is **not run**: no GPU here. The first T4 step is the real check.
- **Kernels.** Both reset to `COMMIT = "UNPINNED"` (refuse-unpinned guard intact) by running
  `make_au59_kernels.py` without `--commit`. `ARM_FLAGS` match `configs/arms/au59-hazard-*.yaml`
  (n_horizons 6; the ctx arm adds `--hazard-period-context`), checked by
  `tests/test_au59_arms.py` (3 pass). Pin after pushing.

## The arms

| arm | `--n-horizons` | `--hazard-period-context` | config | kernel |
| --- | --- | --- | --- | --- |
| `au59-hazard-ctx-s0` | 6 | on | `configs/arms/au59-hazard-period-context.yaml` | `scripts/kaggle/ts_au59_hazard_ctx_s0/` |
| `au59-hazard-ctrl-s0` | 6 | off | `configs/arms/au59-hazard-ctrl.yaml` | `scripts/kaggle/ts_au59_hazard_ctrl_s0/` |

Both are the published recipe (`configs/published/v4-cellattn-labels.yaml`: 6,000 steps, batch 8,
48,000 tasks, `p_financial` 1.0) with `n_horizons` 6, K matching V4FinBench's six horizons and the
earlier hazard checkpoints (`v4-hazard-136f.pt`, `v4-hazard-ldp.pt`, K = 6). The control is a
**second run**, not the published checkpoint: the survival objective itself changes training
(the classification head is untrained, `trained_objectives` `['survival']`), so only a
flag-off survival run isolates the flag. `tests/test_au59_arms.py` pins that the two configs
differ in `hazard_period_context` alone and that the arm differs from the recipe in two keys.
Kaggle execution flags are the AU5.3 kernel's (`--amp fp16 --attn-impl split --loader stream
--ddp --batch-size 4`, seed 0); every `fintfm-train` carries `--checkpoint-every 1000`. Each
kernel trains, then runs the Verify on the checkpoint on the same GPU and writes
`probe/<arm>.json`. **Kernels are UNPINNED** (`COMMIT = "UNPINNED"`, refuse to run) until a
commit containing the configs and `hazard_period_probe.py` is pushed.

## Decision rule

1. **Mechanism (the Verify).** `au59-hazard-ctx-s0` passes as defined above, **and**
   `au59-hazard-ctrl-s0` does not (it cannot read periods: its two arms coincide). If the arm
   does not pass: the flag is not learned at 48,000 tasks on this prior; record it, do not
   rerun with more steps unprompted.
2. **Benefit on real data (before anything is believed, per CLAUDE.md).** Only if 1 passes: score
   both checkpoints on V4FinBench's out-of-time term structure
   (`fintfm-v4oot --model <ckpt> ... `, the §30 split, which passes `ds.period` through
   `fit(..., periods=...)`) and compare mean AUC per horizon, calibration level beside the
   observed rate, and coherence violations, arm against control. Adopt the flag as default for
   hazard pretraining only if it is no worse than the control on those and the Verify passes.
   The yardstick for a V4FinBench difference is the project's noise yardstick (§173).
3. A single seed is a screening result. A pass triggers a second seed before any claim.

## Cost (ESTIMATED)

- Basis, MEASURED: `e1-v48-s0` trained 6,000 steps in 3,150 s on a Kaggle T4x2 (0.525 s per
  step) with the same execution flags; whole sessions took 1.50 to 1.91 h of which 0.6 to 1.1 h
  was installation and five-fold scoring (`preregs_2026-11.md`, "What every cost here rests on",
  from §171 and the kernel logs).
- ESTIMATED: training about 3,150 s (0.88 h) per arm. Not measured: the survival prior
  (`_sample_survival`, a 60-step bisection per task) adds CPU time in the stream loader, and the
  hazard head and period embedding add negligible GPU time. Install, the Verify on 200 tasks and
  upload: about 0.2 to 0.4 h (the Verify's full-size time is not measured). Per arm about 1.1 to
  1.3 h; **both arms about 2.2 to 2.6 h of T4x2**, two sessions, one weekly-quota slot each. Not
  checked against Kaggle's billing.
- **The check:** the step-1,000 line at about 525 s. Over about 700 s means the prior is the
  bottleneck; stop and report. Arm a monitor within 15 minutes (CLAUDE.md).
- V4FinBench out-of-time scoring (decision rule 2) is off Kaggle and costs CPU time on this
  machine; not estimated here.

## Launch (owner; GPU)

After the push, pin and regenerate, then push each kernel directly. `launch_queue.sh` is not
suitable: it scores the fetched checkpoint with `fintfm-v4protocol`, the classification path,
which `assert_trained_for` refuses on a survival checkpoint.

```bash
uv run python scripts/kaggle/make_au59_kernels.py --commit <pushed sha>
kaggle kernels push -p scripts/kaggle/ts_au59_hazard_ctx_s0
kaggle kernels push -p scripts/kaggle/ts_au59_hazard_ctrl_s0   # second quota slot
# fetch both outputs to runs/kaggle-ts-au59-hazard-{ctx,ctrl}-s0/, then, off Kaggle:
uv run --frozen --extra bench --extra hf python -m fintfm.experiments.hazard_period_probe \
  --model <ctx>.pt <ctrl>.pt --n-tasks 200 --out runs/au59/probe   # re-score, CPU, optional
```

## Not done

No flag-on pretraining, no V4FinBench inference, no Kaggle push, no second seed. The Verify's
pass criterion has never been seen to pass: a positive control (a checkpoint that does read
periods) exists only once the arm trains. A quicker positive control would be a small flag-on
model trained on this task family itself; not built.
