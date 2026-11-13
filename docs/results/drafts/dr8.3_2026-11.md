# DR8.3: the `max_features` 512 checkpoint, prepared (draft for FINDINGS, 2026-11-04)

Draft, not merged into `docs/results/FINDINGS.md`. **Preparation only: no pretraining run, no
GPU run, no V4FinBench or TabArena scoring was done.** Everything below that ran is a CPU smoke
test on a shared Mac (load average 140 to 280 during the measurements, so times are noisy and
only ratios are used). Code: the worktree branch of this commit; recipe
`configs/arms/dr83-width512.yaml`; kernel `scripts/kaggle/ts_dr83_width512_s0/` (unpinned).

## Result first

1. **The model had no width limit beyond `max_features`; the prior did.** The financial prior
   cannot emit more than about 370 columns however wide it is asked to go: its 20 accounts give
   380 ordered ratios. Asked for 500 columns it returned 363 to 375 (MEASURED, 8 draws). A
   `max_features` 512 run on the old prior would never have shown the model a table wider than
   about 375 columns. Fixed (`_extend_ratio_family`, `src/fintfm/prior/financial.py`): a task
   asked for 300, 450 or 500 columns now carries at least that many, within 512. Tasks within the
   old reach are bit-identical (golden outputs and the prior, mixture and v2 test files pass,
   111 tests; a new test fails if the extension runs for a 100-column task).
2. **The default width draw would have wrecked the recipe.** Uniform on [4, 512] gives a median of
   258 columns and 74% of tasks above 136 (MEASURED, 6,000 draws per row count); the reference
   draws a median of 70 and none above 136. The arm uses three `feature_ratio_bands` instead:
   median 69, 29% of tasks above 136 (MEASURED, same draws).
3. **Training memory: a per-GPU batch of 4 (the reference's global 8 on two T4s) probably does not
   fit a T4 at the widest batch.** ESTIMATED 14 to 18 GB against 14.56 GB (method below). A
   per-GPU batch of 2 is ESTIMATED at 7 to 9 GB and fits. The kernel uses that, a stated deviation
   (global batch 4, 12,000 steps, the same 48,000 tasks). A matched global batch of 8 needs a
   40 GB card.
4. **Inference pads every table to `max_features`.** A 130-column V4FinBench table scored by a
   512-column checkpoint is padded to 512 (`predict_context`, `pad_features=True`, what every
   estimator does). On CPU, fp32, one forward pass of the reference-size model at 512 rows cost
   17 times the time and 6.8 times the memory at 512 columns as at 136 (SMOKE-TEST, below). A
   GPU would show less (fused fp16 attention does not form the F x F score matrix) but nothing
   here measures it. So **V4FinBench scoring of this checkpoint is not in the kernel** and its
   cost is the first thing to measure (plan below).

## 1. What limits width

Read from `src/fintfm/modeling/model.py`:

| stage | shape | cost in F (columns) |
| --- | --- | --- |
| cell embedding, column identities | `(B, N, F, d_cell)`, ids `(B, F, 16)` | linear |
| row-within-feature attention (cell block) | `B*F` sequences of `N` rows | linear in F; `feature_chunk` 8 bounds transient memory, not what backward keeps |
| column attention (2 layers, plus 1 in the block) | `B*N` sequences of `F` columns | **quadratic**: `B*N*heads*F^2` scores unless a fused kernel is used |
| pooling, row stage | `(B, N, d_model)` | independent of F |

- Nothing in the weights depends on F. Only `encode_rows` and `predict_context` refuse a table
  wider than `cfg.max_features` (DR1.5). A tiny model built at 512 encodes a 512-wide batch,
  back-propagates, and refuses 513 (`tests/test_wide_width.py`).
- The column stage is not chunked over rows. In fp32 or with the stock module it holds
  `B*N*heads*F^2` scores: B 4, N 1,024, 4 heads, F 512, fp32 is 17 GB for one layer. The
  published recipe's `--amp fp16 --attn-impl split` is therefore required, not optional, at 512
  (§170 addendum: fp16 split 9.5 GB at F 136, batch 8, via fused attention with 12-wide heads
  padded to a fused-kernel width, DR5.5).
- **Training pad.** The reference pads every batch to 136. At 512 that would make every step cost
  the widest case, so the arm sets `prior.pad_to_batch_width: true` (DR6.1: a batch is padded to
  its widest task, and column identities are drawn at 512 and cut, so what a narrow batch sees is
  unchanged).

## 2. SMOKE-TEST measurements (CPU, fp32, `attn_impl` split, `feature_chunk` 8)

Reference model size (d_cell 48, d_model 128, 4 heads, 2 column layers, 4 row layers, 1 cell
block, cell labels, 885,650 parameters), random inputs, one process per row, delta of peak RSS
over the RSS before the pass.

| pass | rows N | batch B | F | peak RSS delta (MB) | seconds |
| --- | --- | --- | --- | --- | --- |
| train step (fwd + bwd) | 256 | 2 | 136 | 668 | 7.5 |
| train step | 256 | 2 | 272 | 1,008 | 12.5 |
| train step | 256 | 2 | 512 | 1,936 | 27.1 |
| forward, no grad | 512 | 1 | 136 | 374 | 2.1 |
| forward, no grad | 512 | 1 | 512 | 2,533 | 36.3 |

End to end at SMOKE scale: `fintfm-train --config configs/arms/dr83-width512.yaml --device cpu
--steps 4 --batch-size 2 --d-cell 8 --d-model 16 --n-layers 1 --n-col-layers 1 --n-rows-choices
256 --checkpoint-every 2 --eval-every 2` (19,898 parameters, 4 steps, about 5 s): trained,
evaluated twice, wrote `.step2`, `.step4`, `.state` and the final file
(`runs/agents1104/DR8.3/smoke512.*`). The checkpoint stores
`max_features` 512, and `FinancialTFMClassifier` scored a 500-column table with the default
`max_features_strategy: raise` (11.7 s). **This says nothing about quality.**

**Memory fit to F (train step):** `m(F) = a + bF + cF^2` through the three points gives
463 + 1.01 F + 0.0036 F^2 MB, so 512 columns cost 2.9 times 136 (about half of it the quadratic
term, which is the fp32 score matrix a fused fp16 kernel avoids).

## 3. Projection to a Kaggle T4 (ESTIMATED)

**Method.** The anchor is MEASURED on a T4: fp16 split, batch 8, F 136, 9.5 GB, 0.66 s per step
(§170 addendum); batch 4 is 38,150 tasks per hour. Batch 4 per GPU is taken as half the memory
(4.75 GB), as fp32 split went 13.3 to 6.7 GB from batch 8 to 4 in the same table. The F factor is
bracketed by the CPU fit (2.9) and pure linear growth (3.76, the width ratio). The widest batch
(F 512, 1,024 rows) is the one that decides an out-of-memory.

| per-GPU batch | peak at F 512 and 1,024 rows | fits 14.56 GB | share of batches padded to 512 | mean step time vs the reference's pad to 136 |
| --- | --- | --- | --- | --- |
| 2 | 7 to 9 GB | yes | 11% | 1.61x |
| 4 | 14 to 18 GB | no, not with margin | 20% | 2.01x |
| 8 | 28 to 36 GB | no | 33% | 2.47x |

Step-time ratios are the CPU step times (7.5, 12.5, 27.1 s) interpolated at each batch's padded
width over 6,000 simulated batches; noisy under load, so read them to one significant figure.

**Time.** §170 estimates 65,000 to 70,000 tasks per hour on a T4x2 with DDP and fp16. At the
arm's 1.6x that is about 41,000 to 44,000 tasks per hour, so 48,000 tasks in about 1.1 to 1.2 h
(ESTIMATED; small per-GPU batches under-use a GPU, so allow up to 2 h). V4FinBench and proxy
scoring are separate. Volume 480,000 tasks (`e1-v480`) would be about 11 to 12 h and needs the
two-session path in the kernel template.

**Which GPU.** Cheapest: the Kaggle T4x2 kernel as written (free quota, per-GPU batch 2, 12,000
steps). It is not a matched batch. Cheapest matched (global batch 8 on one card): a 40 to 48 GB
card on HF Jobs, ESTIMATED 1 to 3 h; I did not check current HF prices, so the owner should read
them before launching. A 24 GB card (L4, A10G) holds batch 4 (14 to 18 GB, ESTIMATED), not 8.

## 4. The arm

`configs/arms/dr83-width512.yaml`: a full copy of `configs/published/v4-cellattn-labels.yaml`
(`--config` does not layer over it) with exactly three keys changed, enforced by
`tests/test_dr83_arm.py`:

| key | reference | arm |
| --- | --- | --- |
| `model.max_features` | 136 | 512 |
| `prior.feature_ratio_bands` | none (U[4, 136]) | `[0.6, 0.004, 0.15]`, `[0.3, 0.15, 0.5]`, `[0.1, 0.5, 2.0]` (features per row) |
| `prior.pad_to_batch_width` | false | true |

Everything else is the reference: financial prior only (`p_financial` 1.0), rows 256, 512, 1,024,
`max_classes` 2, 48,000 tasks, seed 0. Bands are features per row, so a 256-row batch is rarely
wide and a 1,024-row batch often is (MEASURED, 6,000 draws: share above 136 columns 10%, 30%, 47%
at 256, 512, 1,024 rows), as real wide tables tend to be (a judgement, not measured). The three
proxy wide tasks have 500 columns (madelon), 216 (mfeat-factors) and 1,558
(Internet-Advertisements); I did not re-check their row counts. Internet-Advertisements is
wider than any task the arm sees, so it stays a test of subsampling on top of the checkpoint
(round-robin column subsets of at most 512 per ensemble member, `dr75-subsample.yaml`'s
mechanism).

**Kernel** `scripts/kaggle/ts_dr83_width512_s0/` follows `ts_au516_rbf_s0`: DDP on two T4s,
`--amp fp16 --attn-impl split --loader stream --batch-size 2 --seed 0 --steps 12000`,
`--checkpoint-every 1000`, recipe fetched from the arm file at `COMMIT`. `COMMIT` is
`"UNPINNED"` and the kernel exits until a pushed sha is set (the `make_au51_kernels.py`
pattern). `SCORE_V4` is False: train only, like DR7.4.

Deviations a reader must keep beside any result:

- global batch 4 and 12,000 steps instead of 8 and 6,000 (same 48,000 tasks); the control's
  batch is 8, so a null or a loss here has a batch confound. If it is ambiguous, rerun on a 40 GB
  card at batch 8.
- the width mix changes the whole prior's shape, not only its cap.
- a binary checkpoint (`max_classes` 2). **DR7.5 also excluded 4 multiclass datasets, and DR7.6
  wants binary and multiclass declared together**, so the multiclass half needs the same change
  on the DR7.4 recipe (`max_classes` 10, `--permute-classes`), another run.

## 5. Evaluation plan (pre-registered here, before any run)

Controls: the 48,000-task checkpoints at the same recipe, `e1-v48-s0` and `e1-v48-s1`
(V4FinBench mean AP 0.1975 and 0.1878, §180; proxy Elo 563 and 546, §182); and for the wide tasks
the §190 table.

**Step 0, cost (before anything else).** One V4FinBench fold with the arm on a T4, timed beside
`e1-v48-s0`'s fold, with `query_chunk` 512, `feature_chunk` 8. If one fold takes more than 4
times the control's (130 columns padded to 512), stop and write the unpadded-inference task first
(a runtime option on `predict_context`, with `column_ids_full_width` so the 130 columns see the
identities they would padded; its test is allclose of padded against unpadded). Not built here.

**Step 1, proxy wide tasks, per task** (`proxy_suite`, splits 0 to 2, `runs/dr83/` style, no
baselines needed; sources: §190 and §182), the arm with `max_features_strategy: subsample`
(Internet-Advertisements needs it; madelon and mfeat-factors fit and run unmodified):

| task | metric | select (§190) | subsample (§182/§190) | best baseline | arm |
| --- | --- | --- | --- | --- | --- |
| madelon (500) | 1 - ROC-AUC | 0.385 | 0.377 | 0.094 LightGBM | to measure |
| Internet-Advertisements (1,558) | 1 - ROC-AUC | 0.447 | 0.269 | 0.013 RF | to measure |
| mfeat-factors (216, multiclass) | log-loss | 1.96 | 1.99 | 0.082 Linear | **binary checkpoint cannot score it** |

mfeat-factors is multiclass, so the binary arm cannot be scored on it: it waits for the
multiclass 512 checkpoint (the uniform guess is 2.30).

**Decision rule (wide tasks, binary).** The arm is *useful* if on madelon and
Internet-Advertisements it beats the better of select and subsample on both (madelon below
0.377, Internet-Advertisements below 0.269), 3-split means. It *closes the gap* if it reaches
half the distance to the best baseline: madelon at or below 0.236, Internet-Advertisements at or
below 0.141. Two tasks cannot carry a significance test; the Lite rerun supplies the count.

**Step 2, V4FinBench sanity** (130 features, five folds, full panel, the packaged default
inference settings, `compare_v4_scores.py --cluster`). Pass if mean AP is at least 0.1810, which
is the mean of the two controls (0.1927, computed here from 0.1975 and 0.1878) minus §173's
yardstick 0.0117, and no fold is a Holm-significant loss. The arm's batch deviation and the
changed width mix are both live causes of a loss; a loss does not say which. Per CLAUDE.md the
V4FinBench verdict is read first, and a proxy gain with a V4FinBench regression is not adopted.

**Step 3, versus the other routes.** Subsample and select numbers are the §190 table, already
measured at 4 members; no new run.

**Step 4, DR7.5: TabArena-Lite with no width exclusion.** Needs both checkpoints (binary here,
multiclass still to train), the declared types together as DR7.6 requires, one Lite run under a
new `FINTFM_RUN_NAME` with `max_features_strategy` `subsample` for tables above 512 and no
capability exclusion. Verify (from DR7.5): every binary and multiclass dataset is run. It
reports and never selects (DR4.1). Run it only if Steps 1 and 2 pass; otherwise the width cap
stays and the 7 excluded datasets stay a documented gap.

## 6. Cost summary

| item | cost | label |
| --- | --- | --- |
| binary arm, T4x2 kernel, 48,000 tasks | about 1.1 to 2 h of Kaggle quota | ESTIMATED |
| the same, matched batch 8, one 40 to 48 GB card | 1 to 3 h; price unchecked | ESTIMATED |
| Step 0, one V4FinBench fold | not estimated (the unknown it measures) | |
| Step 1, proxy, two wide tasks | about the 16-task suite's per-task cost, CPU hours | not measured here |
| multiclass 512 arm | the same again | ESTIMATED |
| the same arms at 480,000 tasks | about 11 to 12 h each | ESTIMATED |

## Open

- Nothing was trained or scored: no claim about quality of a 512-column checkpoint exists yet.
- The kernel is unpinned: push the branch, set `COMMIT`, then launch (owner).
- Padded-inference cost on a GPU is unmeasured (Step 0).
- The multiclass 512 checkpoint is not prepared; mfeat-factors and 4 TabArena multiclass
  datasets depend on it.
- The extension family (differences over a third account) is new prior content that only the
  tail of a wide task above about 380 columns uses; whether it teaches anything is part of what
  the run tests.

## 7. The multiclass 512-column arm, prepared (2026-11-07, DR7.5 / DR7.6)

**Preparation only: nothing was launched, trained at scale or scored.** The arm is
`configs/arms/dr75-width512-mc.yaml`; the kernel is `scripts/kaggle/ts_dr75_width512_mc_s0/`,
unpinned (`COMMIT = "UNPINNED"`, refuses to run until a pushed sha that holds the yaml is set).

**Recipe.** DR7.4's multiclass recipe (§185, kernel `ts_dr74_mc_s0`: the published recipe plus
`--max-classes 10 --p-financial 0.5 --scm-legacy --permute-classes`, 48,000 tasks) with DR8.3's
three width levers. The file equals `dr83-width512.yaml` except four keys, enforced by
`tests/test_dr83_arm.py`:

| key | DR8.3 binary arm | multiclass arm |
| --- | --- | --- |
| `model.max_classes` | 2 | 10 |
| `prior.p_financial` | 1.0 | 0.5 (rest: generic SCM, classes uniform on 2 to 10) |
| `prior.scm_legacy` | false | true (the control's sampler, §185) |
| `prior.permute_classes` | false | true |

The settings sit in the file, so the kernel's flags are DR8.3's (`--batch-size 2 --steps 12000`,
DDP, fp16, split attention), not DR7.4's `--batch-size 4 --steps 6000`: the same 48,000 tasks at
global batch 4 against DR7.4's global 8, the same batch confound as DR8.3's binary arm.

**Prior check (SMOKE-TEST, CPU, 300 single-task draws, seed 1).** The multiclass prior emits wide
tasks too: maximum width 512, median 65, 27% of tasks above 136 columns and 11% above 370 (the
old financial cap). Among the 123 draws with more than 2 classes, 25% are above 136 and 11% above
370 (maximum 512). Class counts run 2 to 10. The wide multiclass tasks come from the generic SCM,
which is not width-capped like the financial prior, so the DR8.3 top-up matters for the financial
half only (binary tasks: 28% above 136, maximum 512, so the top-up reaches the cap here too).
Training smoke at `max_features` 512 with these settings (`fintfm-train --config
configs/arms/dr75-width512-mc.yaml --device cpu --steps 6 --batch-size 2 --d-cell 8 --d-model 16
--n-layers 1 --n-col-layers 1 --n-rows-choices 256 --checkpoint-every 3 --eval-every 3`, 20,226
parameters): trained, evaluated twice, wrote `.step3`, `.step6` and the final file. This says
nothing about quality.

**GPU-hours (ESTIMATED, not measured).** Same shape as the binary arm (§3 above): about 41,000
to 44,000 tasks per hour on a T4x2, so 48,000 tasks in about 1.1 to 1.2 h, allow up to 2 h:
**about 2.2 to 4 T4 GPU-hours (1.1 to 2 h of the T4x2 quota)**. The SCM half costs more than the
financial half at equal width, which the binary arm's ratio does not capture, so read the upper
end as the likelier. For reference DR7.4's narrow arm took 3,233 s at global batch 8 (MEASURED,
§185/§197). At 480,000 tasks, about 11 to 12 h (ESTIMATED), two sessions. Peak RAM on the Mac:
not needed, nothing runs here.

### Launch gate (decided before the binary result exists)

The multiclass arm launches when the binary 512 arm passes, in this order, the draft's own
sanity steps from section 5:

1. **Step 0, cost.** One V4FinBench fold with the binary 512 checkpoint on a T4 takes at most 4
   times `e1-v48-s0`'s fold (130 columns padded to 512). If not, the unpadded-inference task is
   written first and the launch waits.
2. **Step 2, V4FinBench sanity.** Mean AP over the five full-panel folds is at least 0.1810 (the
   mean of the controls 0.1975 and 0.1878, which is 0.1927, minus §173's yardstick 0.0117), and
   no fold is a Holm-significant loss, read with AP first.
3. **Training health.** The binary arm's training log shows no divergence and 12,000 steps
   completed (`tasks_seen` 48,000), so the same recipe shape is known to train at 512 columns.

Step 1 (the wide proxy tasks) does not gate the launch, because the multiclass checkpoint is what
lets mfeat-factors and the 4 excluded multiclass datasets be scored at all, and 1.1 to 2 h of
quota is cheap beside that. It gates the TabArena-Lite rerun (DR7.5), as before. If Step 0 or 2
fails, do not launch; record that DR7.5's 4 multiclass datasets and DR7.6's joint declaration stay
blocked.

**Multiclass sanity after training (pre-registered here).** On the proxy suite's multiclass tasks
(6, 11, 12, 14, 16, 18, 22, 23, splits 0 to 2, DR7.4's command) the arm is compared with
`dr74-mc-s0` (994 proxy Elo, 6 of 8 tasks scorable at 136 columns, §185) by the same per-task and
per-split sign test; `letter` stays out (26 classes against the 10-class cap) and mfeat-factors
(216 columns) is newly scorable and reported on its own against the 2.30 uniform guess. A checkpoint that regresses against `dr74-mc-s0` on 4 or more of the 6 shared tasks is reported as a loss of
the width change, and the TabArena-Lite rerun does not use this checkpoint for multiclass.
This rule was proposed with the arm and confirmed by the owner on 2026-11-07, before any training.
