# DR6.4: the re-specified v2 bundle at 240,000 tasks (pre-registration draft)

**DRAFT, 2026-11-05, before any bundle arm exists.** Not yet in FINDINGS.md. Nothing here has been
trained or scored; every cost is ESTIMATED from MEASURED step times quoted with their section.
Numbers below were re-read from `docs/results/FINDINGS.md` and `preregs_2026-11.md` on 2026-11-05,
not recalled.

## Verdict, stated first

**No lever currently qualifies, so no bundle can be built and DR6.4 stays deferred.** Every
pretraining lever with a V4FinBench verdict is null or harmful at its one measured seed, and no
lever has a seed-1 arm at all. The 2026-11 screening arms (five from §207, AU5.9, DR8.3 width 512)
have no result yet. The bundle is built only under the composition rule below, once those arms and
their seed-1 replicates report.

## Evidence table

Measure for all rows unless marked: five-fold zero-shot V4FinBench mean AP, full panel (1,000,087
rows, five folds), 48,000 tasks (6,000 steps at global batch 8) unless marked. Controls:
`e1-v48-s0` 0.1975 and `e1-v48-s1` 0.1878 (§171 progress). Screening rule (§175, §173): promising
is a gain above 0.0117 with at least 4 of 5 folds positive; harmful is the mirror; otherwise null.
"Both seeds" needs a seed-1 arm judged by the same rule against `e1-v48-s1`.

### Screened and reported

| lever (section) | seed 0 | seed 1 | qualifies |
| --- | --- | --- | --- |
| DR6.2 induced cell rows (§178) | 0.1789, -0.0186, 5 of 5 folds negative; vs `e1-v48-s1` -0.0089 | not run | no: harmful (§178: better read as "no gain" against seed 1) |
| DR6.3 readout tokens + MLP head (§178) | 0.1713, -0.0261, 5 of 5 negative; vs s1 -0.0165 | not run | no: harmful |
| DR2.4 (a) financial prior v2 alone (§191) | -0.0212, 0 of 5 positive; vs s1 -0.0116 | not run | no: harmful at s0, "no gain" vs s1 |
| DR2.4 (b) general trunk (§191) | -0.1262, 0 of 5 positive | not run | no: harmful |
| DR2.4 (c) trunk then financial (§191) | -0.0546, 0 of 5 | not run | no: harmful (proxy Elo 764; the proxy is unvalidated and decides nothing) |
| DR2.4 (d) interleaved (§191) | -0.0358, 0 of 5 | not run | no: harmful |
| E1 volume, 480k vs 48k (§176, §188) | -0.0109, 1 of 5 positive | -0.0103, 0 of 5 positive | no: null at both seeds, same sign negative |
| DR8.2 depth 8 at 480k vs `e1-v480-s0` (§193, §195) | -0.0047, 3 of 5 positive; proxy 8 wins, 8 losses | not run | no: null |
| DR8.2 width 192 at 480k (§202) | +0.0003, 4 of 5 positive (fold 3 -0.0214, p 0.002); proxy 10 wins, 6 losses, p 0.45 | not run | no: null |
| DR7.4 `--permute-classes`, multiclass (§186, §197, §198) | candidate on the proxy | candidate, weaker | not eligible: a multiclass proxy result (6 scorable tasks), not V4FinBench binary credit AP; its 480k volume step was null (§198) |

The one lever measured to move V4FinBench is cell attention, +0.0520 mean AP (§172), which is
already in the v1 recipe. The one inference-side lead, `ens8` (+0.0058, 3 folds up and 2 down,
§196), is an inference lever on one checkpoint, not a pretraining arm.

### Pending (no result exists; the bundle cannot be fixed until these report)

All at `e1-v48-s0`'s recipe, one lever each, seed 0, pre-registered in `preregs_2026-11.md` and
summarised in §207 (stated prediction in every case: null).

| arm | lever | state |
| --- | --- | --- |
| `au517-lenscale-s0` (AU5.17) | row-attention length scale | queued, no result |
| `au517-muon-s0` (AU5.17) | Muon optimiser | queued, no result |
| `au54-ctx2k-s0` (AU5.4) | train contexts to 2,000 rows (`--n-rows-choices 256,512,1024,2223 --max-query-rows 512`) | queued, no result; out-of-memory risk |
| `au53-cell2-s0` (AU5.3) | second weight-tied cell-attention pass | queued, no result |
| `au516-rbf-s0` (AU5.16) | RBF value embedding | queued, no result |
| `dr83-width512` (DR8.3) | `max_features` 512 with three feature-ratio bands; a prior and width change, at per-GPU batch 2 and 12,000 steps, a batch confound (`dr8.3_2026-11.md`) | prepared, not run; judged under that draft's own rule, and eligible here only if it also passes the §175 rule |
| `au59-hazard-*` (AU5.9) | `--hazard-period-context` | arms configured; judged by its period probe (`au5.9_2026-11.md`), not V4FinBench AP, so it needs a §175 arm to be eligible |

Not levers for this bundle: AU5.1 (a fine-tuning recipe at the published checkpoint,
`au5.1_2026-11.md`, not pretraining) and AU5.11 (prior-diversity flags, checkpoint not trained,
`au5.11_2026-11.md`; eligible only if its arm runs as a §175 screening arm).

## Composition rule (fixed before any pending result)

A lever enters the bundle if and only if all of these hold:

1. **Promising at seed 0:** a §175 arm at `e1-v48-s0`'s recipe and seed 0 beats `e1-v48-s0` by
   more than 0.0117 mean AP with at least 4 of 5 folds positive.
2. **Promising at seed 1:** the same flags at `--seed 1` beat `e1-v48-s1` by more than 0.0117 with
   at least 4 of 5 folds positive. A seed-1 replicate is run only for a seed-0 promising arm
   (§175's pre-declared action). "Promising at both seeds" is this and nothing looser.
3. **Not harmful at either seed**, restated so a seed-1 harm cannot be averaged away.
4. **Composable:** the flags can be set together on one commit, and the combined step time is
   measured by a short smoke run before the bundle is launched (see cost).
5. **Credit, not proxy:** the verdict is V4FinBench AP. A proxy-only gain does not count (§188,
   §191).

Then: **two or more qualifying levers: the bundle is all of them.** **Exactly one: the bundle is
that lever alone**, a single-lever 240,000-task test, and leave-one-out is moot. **None: no bundle
is built**, DR6.4 stays deferred and the owner may close it as "nothing earned a bundle" with the
evidence table updated. The membership is fixed by this rule, not by judgement after seeing any
240,000-task outcome.

## Arms, seeds, volume

Four arms, **240,000 tasks each (30,000 steps at global batch 8)**, `--checkpoint-every` set,
Kaggle T4x2, fp16, split attention, streaming loader, DDP, per `e1-v48-s0`'s kernel template, v1
and bundle at the same commit:

| arm | seed |
| --- | --- |
| `dr64-v1-s0` | 0 |
| `dr64-v1-s1` | 1 |
| `dr64-bundle-s0` | 0 |
| `dr64-bundle-s1` | 1 |

v1 is `configs/published/v4-cellattn-labels.yaml`. **No 240,000-task v1 exists:** E1's long arms are
480,000 tasks (§176, §188), so a matched control must be trained; `e1-v480-s*` is not matched and
is only reported beside. Each bundle arm is judged against the v1 arm of the same seed.

## Adoption rule (the task's Verify, copied)

Adopt the bundle if five-fold mean AP is **at least +0.010 with 4 of 5 folds positive at both
seeds**, each against the same-seed v1 arm; then leave-one-out on the bundle's own levers (the
task's "DR6.2 and DR6.3 only" is re-read this way, since §178 excluded those two). Verify: the
verdict in FINDINGS.

Order of scoring (CLAUDE.md): **V4FinBench first** (`scripts/compare_v4_scores.py --cluster`,
paired company-level bootstrap per fold, Holm), **TabArena after**, for breadth, and only for a
bundle that passes V4FinBench. A TabArena gain without the V4FinBench gain is not a result (§115,
§116). Reported, not decided on: ROC-AUC, the proxy suite, seconds per step.

Note: +0.010 is below the 0.0117 single-run yardstick (§173). The same-sign-at-two-seeds and
4-of-5-fold requirements are what guard it (§173's caveat says 0.0117 may understate noise). The
task's number is kept as written and is not moved after seeing results.

## Cost (ESTIMATED from MEASURED step times)

- MEASURED: `e1-v48-s0` 6,000 steps in 3,150 s, 0.525 s per 8-task step (§171 progress); whole
  48,000-task sessions 1.50 to 1.91 h, of which 0.6 to 1.1 h is install and scoring
  (`preregs_2026-11.md`, "What every cost here rests on"). `dr82-width192-s0` ran 0.535 s per step
  (§202), a sanity bound only.
- ESTIMATED, one control-cost arm: 30,000 x 0.525 s = 15,750 s = **4.4 h training, 5.0 to 5.5 h
  session**. If the bundle's step costs the control's, four arms are 20 to 22 h, the "about 22
  GPU-h" the task states.
- **That holds only for levers at control step cost** (lenscale, muon, rbf). ESTIMATED in
  `preregs_2026-11.md`: `au53-cell2` about 1.65x (0.87 s per step, 7.3 h of training per 240k
  arm), `au54-ctx2k` 1.8 to 2.4x (0.95 to 1.27 s per step, 7.9 to 10.6 h). A bundle containing
  either costs those per bundle arm, and the weekly quota is about 30 GPU-h, so such a bundle
  takes more than a week. Recompute from the smoke-measured step time before launching.
- Leave-one-out adds one bundle-sized arm per lever per seed, not budgeted here; sized once the
  bundle is known.

## What would falsify it

- Mean AP gain over v1 below +0.010, or fewer than 4 of 5 folds positive, at either seed: not
  adopted. Opposite signs at the two seeds: not adopted, whatever the mean.
- If the 240,000-task v1 arm is itself below `e1-v48` (volume cost about 0.011 in §176 and §188),
  a bundle that merely reverses that loss is not a v2; report the bundle against `e1-v48` too.
- The stated prediction for every pending arm is null (§207). Each confirmation shrinks the bundle
  toward the empty set, which is an acceptable outcome of this task, not a failure of it.

## Condition for building

Build when every pending arm above has a seed-0 verdict in FINDINGS and every seed-0 promising arm
has a seed-1 verdict. Count qualifiers by the rule. Write the bundle's flags and commit into an
update of this draft before any kernel is pushed, and merge it into FINDINGS as the DR6.4
pre-registration at that point. Until then nothing is launched.

## Progress, 2026-11-07: the scoring and verdict script

`scripts/dr64_arm_verdict.py` (tests: `tests/test_dr64_arm_verdict.py`, 10 pass on synthetic
outputs; no real scoring was run). No earlier script did this: `scripts/compare_v4_scores.py`
compares two scored directories and `scripts/c5_verdict.py` applies §159/§163 at the 0.0073
yardstick, so the new script reuses `compare` and states §175's 0.0117 rule itself.

- `prepare <fetched dir>`: finds the one checkpoint (ignores `.step<N>`, `.phase1`, `.session1`),
  reads the arm's flags from its `scripts/kaggle/ts_*/kernel.py`, checks the kernel log for a GPU
  and the command echo, and train.log for the DDP line, the last step and `model parameters:`,
  then writes `dr64_score_command.sh`: the command of `launch_queue.sh`, with horizon, folds,
  classical arms and boosting read from the control's `v4_protocol.json` and the control's
  `score.yaml` as `--config`.
- `verdict <fetched dir>`: `compare(..., cluster=True)` against `e1-v48-s<seed>`, §175's rule,
  writes `dr64_verdict.json` (per-fold table, verdict, flags, whether batch size and steps match
  the control).
- `compose <runs root> [--expect lever,...] [--not-composable lever,...]`: the composition rule
  above; outcome `bundle`, `single-lever`, `none`, or `wait` while any lever is pending.

Findings while building it (each checked, none fixed, since the kernels are queued):

1. **The `ts_*` kernels print `GPUs: Tesla T4`, not `CUDA available: True`** (that line comes
   only from `make_real_arm_kernel.py`). The check accepts either and fails on `False`.
2. **Most arms are already scored on the Kaggle T4** in `<ARM>/score/`; the verdict uses that.
   `dr83-width512-s0` is train-only (`SCORE_V4 = False`) and the two AU5.9 kernels run a period
   probe instead, so those three need the CPU command. Whether a six-horizon survival checkpoint
   scores under `fintfm-v4protocol --horizon 0` is untested.
3. **The two AU5.9 kernels pass `--amp fp16` with `--n-horizons 6`.** `_autocast` refuses fp16
   for a model with a hazard head (MEASURED on a tiny model, 2026-11-07, and the check is present
   at the pinned commit 62a73c3), so they are likely to fail at training start. The arm yamls set
   `amp: "none"`; the CLI flag overrides that.
4. `dr83-width512` runs at batch 2 and 12,000 steps against the control's batch 4 and 6,000; the
   verdict file records that confound.

### Cost of one checkpoint's five-fold CPU scoring (ESTIMATED from existing logs)

From `runs/kaggle_queue/*.log` start and end stamps of `fintfm-v4protocol --device cpu`, full
panel, `--no-boosting`, one classical arm: `fintfm-r-edgar-s0` 3.05 h and `-s1` 3.4 h (2026-10-10
to 11, serial chain, bwa running), `task-48-9-schedulefree-retuned` 2.65 h. These overlapped a
second scoring chain, so about 2.7 to 3.4 h per checkpoint is a fair figure, not a clean one.
Peak RAM: 12 to 14 GB per scoring, recorded in the comment of `scripts/kaggle/launch_queue.sh`
(observed with `top`, 2026-10-10), not as a per-run peak; only one scoring at a time. For
comparison the Kaggle T4 scored the control in about 35 minutes (MEASURED, kernel log).
