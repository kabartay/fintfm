# Pre-registration drafts, November 2026: the training-measurement tasks

**DRAFT, 2026-11-02, before any arm trains.** Not yet in FINDINGS.md; each arm's section is meant
to be merged there as its own pre-registration before its kernel is pushed. Covers the open tasks
whose measurement needs a training run: AU5.2, AU5.3, AU5.4, AU5.5, AU5.6, AU5.10, AU5.16, AU5.17
(`openspec/changes/audit-2026-10-15/tasks.md`) and DR5.4, DR6.1, DR6.4
(`openspec/changes/design-review-2026-10-16/tasks.md`). Nothing here has been trained or scored;
every cost below is ESTIMATED unless marked MEASURED.

## Launch plan for the next 30 GPU-hour week

**Five 48,000-task screening arms, about 11.5 GPU-hours (ESTIMATED), in two lanes of about 6 h
each.** The other 18 h of the week stay unspent until the five results are in, held for the
seed-1 replicate that any promising arm earns under §175's rule.

| order | lane A | GPU-h (ESTIMATED) | lane B | GPU-h (ESTIMATED) |
| --- | --- | --- | --- | --- |
| 1 | `au517-lenscale-s0` | 1.6 to 2.0 | `au54-ctx2k-s0` | 2.2 to 3.2 |
| 2 | `au517-muon-s0` | 1.6 to 2.1 | `au53-cell2-s0` | 1.9 to 2.6 |
| 3 | `au516-rbf-s0` | 1.6 to 2.0 | | |
| lane total | | 4.8 to 6.1 | | 4.1 to 5.8 |

- **Why this order.** Lane A carries the three arms that cost the same as the control, cheapest
  first by rank. Lane B starts with `au54-ctx2k-s0` because it is the one arm that could run out
  of memory on a T4 (its section below), and an out-of-memory failure shows within minutes of
  the first 2,223-row batch; starting it first leaves the whole week to react.
- **Reserve, 18 h.** Spend it only on what the screening results earn: a seed-1 replicate of each
  promising arm (each costs what its screening arm cost), then, if two or more levers are
  promising at both seeds, one combined 48,000-task arm of those levers at seeds 0 and 1. DR6.4's
  240,000-task matched comparison does not fit this week beside the replicates (about 22 h for
  four arms, ESTIMATED below) and runs the week after, if at all.
- **Checks during each run** (CLAUDE.md: a launch is not fire and forget). The step-1,000 line of
  `train.log` against `e1-v48-s0`'s 525 s (MEASURED, §171 progress: 6,000 steps in 3,150 s), with
  each arm's own expected ratio below; for `au54-ctx2k-s0` and `au53-cell2-s0`, the first logged
  step against an out-of-memory failure.
- **After each session.** `uv run python scripts/compare_v4_scores.py --cluster --control
  runs/kaggle-ts-e1-v48-s0/e1-v48-s0/score <arm>=runs/kaggle-ts-<arm>/<arm>/score`, then the same
  against `e1-v48-s1`. That reads stored predictions and needs no scoring on this Mac.

### What every cost here rests on

- **MEASURED.** `e1-v48-s0` trained 6,000 steps in 3,150 s on a Kaggle T4x2 (0.525 s per step,
  §171 progress). Whole 48,000-task train-and-score sessions, from the kernel logs in `runs/`:
  `e1-v48-s0` 5,403 s, `dr62-induced-s0` 6,891 s, `dr24-a-s0` 6,686 s, so 1.50 to 1.91 h per
  session, of which 0.6 to 1.1 h is installation and five-fold scoring.
- **MEASURED.** At fp32 with masked attention on a single T4, the published recipe trained
  6,000 steps in 13,093 s with its cell-attention block and 4,644 s without it (`pub-s0` and
  `nocell-s0` logs, §172), so the cell stage is about 65% of a step.
- **MEASURED.** E0 (§170): rows pinned at 1,024 cost 4.11 s per step against 1.93 s for the mixed
  {256, 512, 1,024} draw (fp32, masked), which fits cost growing as about N^1.7 in rows.
- **ESTIMATED.** Kaggle charges a T4x2 session by the hour, as §202's 25.3 h for five sessions
  implies. Not checked against Kaggle's documentation.

## Which tasks still deserve a run

§202 closed scale at this model size: ten times the tasks is null on V4FinBench at both seeds
(§188), and neither depth (§195) nor width (§202) at about twice the parameters moves it at
480,000 tasks. §178 found both DR6 architecture levers harmful at screening. So a run is funded
here only where it tests something none of those measured, and only at 48,000 tasks, the screening
scale (§175), where its control already exists.

| task | decision | why |
| --- | --- | --- |
| AU5.17 (attention half) | **run**, `au517-lenscale-s0` | Training contexts hold 76 to 921 rows and V4FinBench is scored at `max_context` 2,000 (§162). §162's confirmation ruled out shrinking the context at inference; nothing has tested making the model length-aware in training. Costs the same as the control. |
| AU5.4 | **run**, `au54-ctx2k-s0` | The direct fix for the same gap: train on the context lengths inference uses. Verify asks for exactly this. §83's flat full-panel context curve lowers the expected gain but was measured on the pre-cell-attention architecture, never trained past 921 rows. |
| AU5.17 (optimiser half) | **run**, `au517-muon-s0` | The optimiser axis has been tested once: schedule-free at AdamW's rate, -0.0098 mean AP against its matched control, no different at the 0.0117 yardstick (§146, §160, §173). Muon at `rms_match` reuses the tuned rate (§127) by construction. Costs about the same as the control. |
| AU5.3 | **narrow to one arm**, `au53-cell2-s0` | Heads, width and summary tokens are superseded: width and heads by §202 (DR8.2 width null, with `--n-heads 6`), summary tokens by DR6.3's readout tokens, harmful at screening (§178), which DR6.3 itself re-specified from AU5.3. A second cell-attention block is the one AU5.3 arm left, and the cell stage is the only architecture component measured to move V4FinBench (+0.0520 mean AP, §172); DR8.2 left it fixed. |
| AU5.16 | **run**, `au516-rbf-s0` | Cheap, and no input-representation change has been trained at the current recipe. Ranked last: §88 found the inference rank transform already gives the cell-attention model near-invariance to monotone warps, which lowers what a value basis can add. The ECDF channel is not run (below). |
| AU5.6 | **close as superseded (§87, §88)** | §87 found the train/inference marginal shift; §88 measured that the rank transform does the invariance work, so "§87's mismatch needs no retraining", and marginal augmentation was worth about 0.001. A per-task transform draw in pretraining targets the same shift. |
| AU5.5 | **close without a GPU run** | Its Verify needs a test (present: `tests/test_prior_levers.py::test_sample_batch_permutes_classes_per_task`) and a label-symmetry probe before and after. Before is §165 (published checkpoint). After needs a checkpoint trained with `--permute-classes`, and two exist (`dr74-mc-s0`, `dr74-mc-s1`, §186, §197) beside a matched one without (`v4-multiclass.pt`). That probe is a small CPU job on synthetic tasks. A binary credit arm is not worth GPU time: in credit the default is always class 1, so a class-index prior is information, not a leak. |
| AU5.10 | **close without a run** | Its Verify is a test, and it exists (`tests/test_train.py::test_decay_norm_bias_false_exempts_norms_biases_and_embeddings`). The lever removes AdamW decay of 0.01 from norms, biases and embeddings; a single 48,000-task arm resolves only 0.0117 mean AP (§173), far above any effect this could plausibly have. The final-LayerNorm nit stays unimplemented and unfunded. |
| AU5.2 | **close, measured in §170** | Its Verify (tasks per GPU-hour before and after on one kernel, outputs equal within tolerance) is §170 and its addendum: split attention 1.21x fp32 and 1.10x under fp16 (37,367 against 33,959 tasks per hour), logits within 1.2e-4 and 9.4e-4. Split and fp16 are E1's recipe. |
| DR6.1 | **close, measured in §170** | Its Verify (identical outputs at a widest-136 batch, 2d22ba9's test; step time measured) is met: 1.29x against 1.21x for split alone in §170, fp32. Not added to these arms, so each stays one lever away from its control; worth turning on in the next recipe for about 7% more tasks per hour (18,256 against 17,101, MEASURED at fp32 only). |
| DR5.4 | **close, superseded by §188** | Its Verify, a test that a stable-phase resume reproduces the uninterrupted weights, passes. What stays open is a 2,000,000-task run branched to a longer horizon, and §188 funded no 2M or 10M run at this model size and prior. Reopen if a later recipe earns a volume arm. |
| DR6.4 | **defer; no bundle exists** | Its bundle was DR6.2 and DR6.3, and §178 excluded both. Re-specify it as the bundle of whichever levers here are promising at both seeds, at 240,000 tasks and two seeds. Cost if it runs (ESTIMATED): 30,000 steps at 0.53 s is about 4.4 h of training per arm, about 5.5 h with scoring; the bundle and v1 at two seeds is about 22 h, a week of its own. |

### Ranked by expected value per GPU-hour

1. `au517-lenscale-s0` (AU5.17): 1.6 to 2.0 GPU-h. The cheapest test of the one measured gap
   between training and scoring (contexts to 921 rows trained, 2,000 scored).
2. `au54-ctx2k-s0` (AU5.4): 2.2 to 3.2 GPU-h. Same gap, the direct fix, at about 1.6 times the
   cost and one memory risk.
3. `au517-muon-s0` (AU5.17): 1.6 to 2.1 GPU-h. A fresh optimiser axis at control cost, with a
   failed precedent (§146) and no project evidence for it.
4. `au53-cell2-s0` (AU5.3): 1.9 to 2.6 GPU-h. The component that has worked, a second pass of it
   (weight-tied, see its section), against two null capacity results (§195, §202).
5. `au516-rbf-s0` (AU5.16): 1.6 to 2.0 GPU-h. Control cost; §88 lowers its expected gain.

The ranking is judgement on the evidence cited, not a computed probability.

## Common to every arm

- **Recipe.** `e1-v48-s0`'s exactly (§171): `configs/published/v4-cellattn-labels.yaml` with
  `--amp fp16 --attn-impl split --loader stream --ddp --batch-size 4` on a Kaggle T4x2 (global
  batch 8), `--seed 0 --steps 6000` (48,000 tasks), cosine schedule, `--checkpoint-every 1000`,
  plus one lever. Pinned at **d05ea19**, the control's own commit, which is on `origin/main`
  (`git branch -r --contains d05ea19`) and carries every flag used here, so no code differs
  between arm and control.
- **Kernels.** `scripts/kaggle/ts_<arm>/`, written by `uv run python
  scripts/kaggle/train_score/make_train_score.py <arm>`; one session, train then score.
- **Checked before launch (SMOKE-TEST, 2026-11-02).** Each arm's flags parsed by d05ea19's own
  `build_parser` and `build_configs` against d05ea19's recipe file, the model built and its
  optimiser constructed on CPU (`--ddp` left out, since it needs torchrun). Parameter counts:
  control 885,650; `au517-lenscale-s0`, `au517-muon-s0`, `au53-cell2-s0`, `au54-ctx2k-s0`
  885,650; `au516-rbf-s0` 886,418. No arm has trained a step; fp16 and DDP run only on the GPU.
- **Control: `e1-v48-s0`** (d05ea19), five-fold mean AP **0.1975** (0.2071, 0.2031, 0.1821,
  0.1734, 0.2215; §171 progress), scores in `runs/kaggle-ts-e1-v48-s0/e1-v48-s0/score/`.
  Reported beside it: `e1-v48-s1`, 0.1878, the other 48,000-task seed.
- **Primary measure.** Five-fold zero-shot V4FinBench mean AP, full panel, packaged inference
  defaults (uniform context, `max_context` 2,000, one member), scored by the kernel template;
  paired company-level bootstrap per fold, Holm across folds
  (`scripts/compare_v4_scores.py --cluster`).
- **Rule (§175's, unchanged).** **Promising** if the arm's mean AP exceeds the control's by more
  than the pretraining yardstick, **0.0117** (§173), with at least 4 of 5 folds positive;
  **harmful** under the mirror; otherwise **null**. One seed screens; it does not adopt.
- **Pre-declared actions (§175's).** Promising: a seed-1 replicate from this week's reserve, then
  DR6.4's matched comparison. Null or harmful: not carried into any bundle on this evidence, and
  the task closes with the result.
- **Reported, not decided on.** The difference against `e1-v48-s1` (§178's caveat: the control is
  the higher 48,000-task seed, so a promising verdict against it is the conservative one); seconds
  per step against the control's 0.525; mean ROC-AUC; and the proxy suite's 16 binary tasks
  against `runs/dr73/proxy_e1_v48-s0.csv` (`proxy_suite run --no-baselines --out ...`, then
  `compare --unit task --anchor e1-v48-s0`), run on this Mac one job at a time. The proxy is
  not validated against TabArena (DR4.2) and decides nothing.

## Arm 1: `au517-lenscale-s0` (AU5.17, attention half)

- **Lever.** `--attn-length-scale`: row-attention logits multiplied by log(n_ctx)/log(512)
  (`attn_length_ref` stays 512), in the row encoder only; the row-within-feature cell stage is
  not scaled. Identity at a 512-row context, about 1.09 at 921 rows (the longest trained) and
  1.22 at 2,000 (the scoring default). No new weight.
- **Question.** Does a model trained to sharpen its attention with context length score better at
  the 2,000-row default, which is past every context it trained on (§162)?
- **Cost (ESTIMATED).** One scalar per call, so the control's step time: 6,000 steps in about
  3,150 s, a 1.6 to 2.0 h session.
- **Prediction: null.** §83 found AP nearly flat in context length on the full panel, and §162's
  confirmation found an in-range 500-row context no better than 2,000, so the length gap does not
  look like what holds the model back.

## Arm 2: `au54-ctx2k-s0` (AU5.4)

- **Lever.** `--n-rows-choices 256,512,1024,2223 --max-query-rows 512`. A 2,223-row task holds
  contexts of 667 to 2,000 rows (`int(0.9 * 2223) == 2000`), so the trained range reaches the
  scoring default, and every task's query rows are capped at 512, inference's `query_chunk`. The
  cap also trims the query of a 1,024-row task whose context is under 512 rows: stated as part of
  the lever, not separated from it.
- **Question.** Does training on the context lengths inference uses raise V4FinBench AP? This is
  AU5.4's own Verify: the trained range covers `max_context`, and five-fold AP against the
  current recipe.
- **Cost (ESTIMATED).** From E0's N^1.7 fit (above), a 2,223-row batch at about 1,810 rows
  (mean context plus the capped query) costs about 2.6 times a 1,024-row one, so a step averages
  about 1.8 to 2.4 times the control's: 0.95 to 1.27 s, 1.6 to 2.1 h of training, a 2.2 to 3.2 h
  session. **Memory risk:** a 2,223-row batch carries up to 2.45 times the control's largest
  cell-token count per rank. The control used 9.5 GB at batch 8 on one T4 (§170 addendum), so
  about 4.8 GB at 4 per rank; linear memory in N (§95) puts this arm near 12 GB of 14.56 GB. An
  out-of-memory failure stops the run at the first such batch; if it happens, the arm is
  re-registered at `--n-rows-choices 256,512,1024,2000` (contexts to 1,800) rather than at a
  smaller batch, which would change the recipe.
- **The check.** The first logged steps (a quarter of batches draw 2,223 rows), then the
  step-1,000 line against about 950 to 1,270 s.
- **Prediction: null.** As Arm 1, and §83's flat curve. If both arms are null the train/inference
  length gap is closed as a cause.

## Arm 3: `au517-muon-s0` (AU5.17, optimiser half)

- **Lever.** `--optimizer muon`: Muon on the attention and feed-forward matrices at `--lr` 3e-4
  under the default `muon_scale: rms_match` (update RMS matched to AdamW's, so AdamW's tuned rate
  carries over), AdamW on the rest, the same warmup and cosine schedule, weight decay 0.01 on the
  matrices as AdamW gave them.
- **Question.** Does an orthogonalised update extract more from 48,000 tasks than AdamW?
- **Cost (ESTIMATED).** A few Newton-Schulz iterations per matrix per step, on matrices no larger
  than 512 by 128; a few percent of a step, so a 1.6 to 2.1 h session.
- **Untested combination.** Muon has never run under fp16 loss scaling or DDP here
  (`tests/test_train.py` covers it on CPU in fp32, resume included). A failure would show in the
  first steps; a loss that does not fall below the control's at step 1,000 is reported, not
  decided on.
- **Prediction: null.** §127 found the default AdamW rate wins at this recipe, and the one other
  optimiser tried did not beat it (§146).

## Arm 4: `au53-cell2-s0` (AU5.3, narrowed)

- **Lever.** `--n-cell-blocks 2`. **At d05ea19 the blocks share weights:** the model builds one
  row-within-feature encoder and one column encoder and runs the pair `n_cell_blocks` times
  (`model.py`, the `for _ in range(self.cfg.n_cell_blocks)` loop), so this arm is a second,
  weight-tied pass through the cell stage, not more parameters (885,650, as the control). It
  tests more cell-stage computation, not cell-stage capacity.
- **Question.** Does a second round of row-then-column cell attention, the stage worth +0.0520
  mean AP (§172), add to V4FinBench AP?
- **Cost (ESTIMATED).** The cell stage is about 65% of a step (fp32 masked, above), so a second
  pass costs about 1.65 times the control: about 0.87 s per step, 1.3 to 1.5 h of training, a 1.9
  to 2.6 h session. The share under fp16 split attention is not measured. Activation memory for
  the cell stage doubles; about 8 GB per rank, inside a T4.
- **The check.** The step-1,000 line against about 870 s.
- **Prediction: null.** Depth in the row stages was null even at ten times the tasks (§195), and
  a weight-tied pass adds less than an untied layer.

## Arm 5: `au516-rbf-s0` (AU5.16)

- **Lever.** `--value-embedding rbf`: each normalised cell value expanded on 16 fixed Gaussian
  centres (`value_basis_size` 16, not learnable) before the cell MLP, in place of the MLP on
  (z, missing) alone. 886,418 parameters. The ECDF channel (`--value-ecdf`) is not part of the
  arm: inference already rank-transforms every column, fitted on the training rows, before the
  model sees it (§87), so a context-only rank channel would repeat it.
- **Question.** Does a basis expansion of heavy-tailed financial ratios help the model read them?
- **Cost (ESTIMATED).** One small expansion per cell; the control's step time, a 1.6 to 2.0 h
  session.
- **Prediction: null.** §88: under the rank transform the cell-attention model is already close
  to invariant to monotone warps of a feature.

## What this does not decide

- **TabArena.** No arm here is a TabArena claim; the proxy is reported only (CLAUDE.md: V4FinBench
  first).
- **Adoption.** A promising arm at one seed adopts nothing; a seed-1 replicate and DR6.4's
  matched comparison come first.
- **Volume.** Every arm is at 48,000 tasks; a lever that helps here is not tested at 480,000.
