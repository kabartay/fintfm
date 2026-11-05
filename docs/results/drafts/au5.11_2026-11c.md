# AU5.11 part 3: the prior-diversity arms, prepared and not launched (2026-11-05)

Draft; not a FINDINGS entry. **PRE-REGISTRATION, written before any arm trains.** Nothing here
has been trained or scored on a GPU; costs are ESTIMATED unless marked MEASURED. The flags
(`heavy_tail_column_prob`, `tree_split_by: uniform`, `missing_label`) are in 7dfd663, all off by
default (`au5.11_2026-11b.md`). This file adds the arm configs, the kernels and the rule.

## What was prepared

| item | path |
| --- | --- |
| primary arm: mixture + the three new flags | `configs/arms/au511-diverse.yaml`, kernel `scripts/kaggle/ts_au511_diverse_s0/` |
| attribution arm: the same mixture, new flags off | `configs/arms/au511-mixture.yaml`, kernel `scripts/kaggle/ts_au511_mixture_s0/` |
| generator | `scripts/kaggle/train_score/make_train_score.py` (arms `au511-diverse-s0`, `au511-mixture-s0`; `AU511_MIXTURE`, `AU511_NEW`) |
| drift guard | `tests/test_au511_arm.py` |

Kernels are the usual train-and-score template (T4x2, fp16, split attention, streamed loader,
DDP at 4 tasks per rank, `--checkpoint-every 1000`, then five-fold V4FinBench scoring), pinned at
**62a73c3**, which contains 7dfd663 (`git merge-base --is-ancestor 7dfd663 62a73c3`, exit 0,
checked 2026-11-05). The pin is the latest the other 2026-11 kernels use. Neither kernel has been
pushed to Kaggle.

## The arms

Both train `configs/published/v4-cellattn-labels.yaml` (model, optimiser, 6,000 steps x 8 tasks =
48,000 tasks, `max_classes` 2, rows {256, 512, 1024}, seed 0) and change only the prior:

- **Mixture (both arms).** `--p-financial 0.3 --p-tree 0.3 --p-task-family 0.3`, the arm named in
  AU5.11. The chain is sequential, so these are conditional weights; the realised sources are
  about scm 0.35, tree 0.30, family 0.22, financial 0.13 (MEASURED, 300 tasks,
  `au5.11_2026-11b.md`). Plus the existing `categorical_mode unordered`, `missing_mode mixed` and
  DR2.4's three feature-ratio bands (`0.5:0:0.05,0.35:0.05:0.2,0.15:0.2:0.5`).
- **`au511-diverse-s0` only.** `heavy_tail_column_prob 0.15`, `tree_split_by uniform`,
  `missing_label` on.
- Not changed: `max_classes` (2, the control's; V4FinBench is binary), gating, tie, irrelevant,
  duplicate and flip knobs. The financial source reads only the bands among the knobs
  (`make_train_score.py` comment on `FIN_V2`), so the new flags act on the ~87% that is not
  financial.

**Why two arms.** `au511-diverse-s0` against the control moves the mixture and the three flags
together, and §180 already says the mixture alone is large: the general trunk (arm b, financial
share 0 at 48,000 tasks) scored 0.0713 V4FinBench mean AP against 0.1975, -0.1262, 0 of 5 folds
positive. Without `au511-mixture-s0` a harmful verdict could not be attributed to the flags or to
the mixture. The attribution arm is the cheaper to drop if the quota is short, but then the
result reads as a bundle only.

## Controls and yardstick

- **Control: `e1-v48-s0`** (d05ea19; published recipe, 48,000 tasks, seed 0, p_financial 1.0), five-fold
  mean AP **0.1975** (0.2071, 0.2031, 0.1821, 0.1734, 0.2215), scores in
  `runs/kaggle-ts-e1-v48-s0/e1-v48-s0/score/`. Reported beside it: `e1-v48-s1`, 0.1878. A matched
  control exists; none is trained. The kernels use 62a73c3, not d05ea19; with the new flags off
  the generated tasks are byte-identical to 63500ad (SHA-256 pinned in
  `tests/test_prior_diversity.py`), and the model and training code are the published recipe's, but
  no flags-off run at 62a73c3 has been compared with d05ea19's training stream here. The
  `au511-mixture-s0` arm carries no such risk for the flags comparison, since both new arms run at
  62a73c3.
- **Task count:** 48,000 per arm (batch 4 per rank x 2 ranks x 6,000 steps, asserted in
  `tests/test_au511_arm.py`); seed 0; same as the control, matched.
- **Primary measure:** five-fold zero-shot V4FinBench mean AP, full panel, packaged inference
  defaults, scored by the kernel; paired company-level bootstrap per fold, Holm across folds
  (`scripts/compare_v4_scores.py --cluster --control runs/kaggle-ts-e1-v48-s0/e1-v48-s0/score
  au511-diverse-s0=...`), as `preregs_2026-11.md`.
- **Rule (§175's, unchanged).** **Promising** if the arm's mean AP exceeds the control's by more than
  **0.0117** (§173) with at least 4 of 5 folds positive; **harmful** under the mirror (below by
  more than 0.0117, at least 4 of 5 folds negative); otherwise **null**. One seed screens; it
  does not adopt. Promising: a seed-1 replicate, then the matched comparison. Null or harmful: not
  carried into any bundle on this evidence.
- **Second comparison, decided the same way:** `au511-diverse-s0` against `au511-mixture-s0`. This
  is the test of the three flags themselves.
- **Reported, not decided on:** against `e1-v48-s1`; mean ROC-AUC; seconds per step against the
  control; the proxy suite's 16 binary tasks (`proxy_suite`, run off Kaggle, one job at a time); and
  TabArena-Lite, only after the V4FinBench result (CLAUDE.md: V4FinBench before TabArena), since the
  arm's stated purpose is a general-tabular checkpoint.

## Prediction (stated before the result)

**Harmful on V4FinBench for the mixture, with the three flags unlikely to rescue it.** §180's trunk
arms were all harmful at 48,000 tasks (-0.0116 to -0.1166 against `e1-v48-s1`), the proxy gained,
and this mixture keeps only ~13% financial tasks. If so, the arm is a candidate for §180's separate
general family (TabArena, proxy), not for the credit checkpoint, and the result to read is the
diverse-minus-mixture difference on both V4FinBench and the proxy.

## What falsifies it

- The prediction fails if `au511-diverse-s0` is **null or promising** against `e1-v48-s0` on V4FinBench
  (mean AP within 0.0117 of 0.1975 or above it), despite the ~87% non-financial prior.
- The flags are shown **useless** if `au511-diverse-s0` minus `au511-mixture-s0` is within +-0.0117
  on V4FinBench mean AP and on fewer than 4 of 5 folds in one direction; **useful** if above
  +0.0117 with 4 of 5 folds positive; **harmful** under the mirror.
- The prior-diversity hypothesis behind AU5.11 (heavy tails, uniform splits and label-driven
  missingness missing from the prior are what limits the model) is not supported if the flags are
  useless on both V4FinBench and the proxy.

## Step time and cost (ESTIMATED from MEASURED rates)

MEASURED inputs: `e1-v48-s0` 0.525 s/step on T4x2 (6,000 steps, 3,150 s; §171); whole sessions 1.50
to 1.91 h, of which 0.6 to 1.1 h is install and scoring (`preregs_2026-11.md`); AU5.15 live
single-GPU 0.80 s/step at batch 8 (`au5.15_2026-11.md`: 0.800 with checkpoints, 0.804 without),
DDP stream 0.500 s/step.

**The flags are not free in generation.** SMOKE-TEST, this Mac, load average about 95 with the
process at `nice -n 19` and one thread, `sample_batch` of 8 at the published recipe's rows and 136
columns, 24 batches after one warm-up, `/private/tmp/.../gen_time.py`: control 0.076 s/batch,
`au511-mixture` 0.474 (6.2x), `au511-diverse` 0.858 (11x). Absolute times are inflated by the load
and not a Kaggle rate; only the ratio is informative, and it says AU5.15's "generation is not the
bottleneck" (about 5% of a step, measured for the financial prior) does not carry over to
this mixture. The tree and family sources cost more per task than the financial source, and the new
flags add to the tree source and to the per-column maps.

Per-step bounds for the DDP stream kernel (ESTIMATED):

| case | s/step | 6,000 steps | session (+0.6 to 1.1 h) |
| --- | --- | --- | --- |
| generation hidden by the stream workers (control's rate) | 0.50 to 0.53 | 3,000 to 3,150 s | 1.5 to 2.0 h |
| generation-bound, 0.86 s per 8 tasks as measured above, which is a loaded-Mac upper estimate | up to about 0.9 | up to 5,400 s | up to 2.6 h |

So **`au511-diverse-s0` costs about 1.5 to 2.6 session-hours, `au511-mixture-s0` about 1.5 to
2.3 h; both arms 3.0 to 4.9 h** against the 30 h Kaggle week (ESTIMATED; counted in session hours as
`preregs_2026-11.md` does; whether Kaggle's quota counts a T4x2 session once or twice is not settled here). The
first-hour check replaces the estimate: read the step-1,000 line of `train.log` against 525 s
(control) and, if it exceeds about 700 s, the run is generation-bound and a second look at
`--loader-workers` is warranted before the remaining 5,000 steps are spent. Memory is not
expected to change (the model and batch shape are the control's; width is padded to 136).

## SMOKE-TEST (does not measure quality)

- `uv run --frozen pytest -q tests/test_au511_arm.py tests/test_config.py`: the arm files differ from the
  published recipe in prior keys only; the kernel flags on the published recipe and the arm file build
  the same `ModelConfig` (apart from the kernel's execution flag `attn_impl`) and `PriorConfig`; the
  flags reach generated tasks (batches differ from the flags-off mixture); both kernels are pinned and
  carry `--checkpoint-every`.
- `fintfm-train --config configs/arms/au511-diverse.yaml --device cpu` with a 14,706-parameter model
  (d_model 32, 1 layer, 16 columns, rows 64/128, batch 2), 4 steps, `--checkpoint-every 2`, one thread,
  `nice -n 19`: exit 0, losses finite (0.616, 0.720, 0.620, 0.536), checkpoints at steps 2 and 4. This
  proves the config loads and the flagged prior trains a step; it says nothing about the prior's
  statistics (`au5.11_2026-11b.md`) or about quality.

## Launch (owner)

```
cd scripts/kaggle && kaggle kernels push -p ts_au511_diverse_s0     # primary
cd scripts/kaggle && kaggle kernels push -p ts_au511_mixture_s0     # attribution
```

Push 62a73c3 (or a descendant carrying `configs/published/v4-cellattn-labels.yaml`) to
`kabartay/fintfm` first, since the kernel installs from that commit. Arm a monitor within 15
minutes (CLAUDE.md): step-1,000 time against 525 s.
