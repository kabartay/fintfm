# Scaling-curve grid: PROPOSED (task 9.1) and priced (task 9.2), 2026-11-15

**Status: PROPOSED. The owner has not decided.** Task 9.1 stays open until the owner accepts,
trims or rejects this grid. This is a draft, not a FINDINGS entry. `openspec/changes/scaling-curve/tasks.md`
carries a SUPERSEDED banner (2026-09-25) saying this grid should not be run on the evidence of
§93, §114 and §110; that banner stands, and nothing here launches anything. The proposal exists
so that, if the owner ever wants the volume question answered at a peer-like scale, the grid is
fixed before any result is seen.

Every price below is **ESTIMATED**, derived from **MEASURED** step times cited by section.

## What the existing evidence already says

| point | model | tasks | result | source |
| --- | --- | --- | --- | --- |
| 5x volume | 885K | 240,000 vs 48,000 | -0.0012 AP, V4FinBench, five folds, full 1,000,087-row panel | §93 |
| 10x volume, two seeds | 885K | 480,000 vs 48,000 | -0.0106 AP, 1/5 and 0/5 folds better | §176, §188 |
| 5.7x parameters, matched tasks | 5.0M (4,979,778) vs 885K | 48,000 | -0.0049 ROC-AUC, 12/27 TabArena binary, p 0.701 | §114 |
| 10x volume, regression-only | see §233 | 480,000 | Elo 642 against 582 unpartitioned, 7 of 13 better (p 1.0) | §233 |

Nothing has been trained above 480,000 tasks. The field norm is around 10^7 (CLAUDE.md, §43).
The only untested region is **above 480,000 tasks**, so that is where the grid's new
information lives. Its other cells replicate or bracket what is already measured.

## The proposed grid (9.1)

**Axes.**

- **Tasks:** 48,000, 480,000 and 4,800,000. Steps are tasks / 8 at batch 8: 6,000, 60,000 and
  600,000. These are decades apart, matching the proposal's "100k, 1M, 10M" in spacing but
  anchored on the two volumes already run (48,000 and 480,000), so two columns are directly
  comparable to §93, §176 and §188. 4.8M rather than 10M because 10M costs 187.5 T4x2 wall hours
  for the smallest model alone (priced below as a row not proposed).
- **Model sizes:** 885K parameters (`d_model` 128, `n_layers` 4; `configs/published/v4-cellattn-labels.yaml`) and
  5.0M parameters (4,979,778, §108/§114). Two sizes, as the proposal asks for "more than one".
  Not 100M: the peer curve (Nori +0.0049 R² for 6M to 100M, §110) and §114 say parameters buy
  least, and §114's deep-narrow arm could not finish the benchmark in time.
- **Seeds:** 3 at 48,000 tasks, 2 at 480,000, 1 at 4,800,000. Reasoning: seed spread is the
  noise floor, and re-trains of one recipe have scored 0.1947, 0.1858 and 0.1845 AP
  (`docs/results/drafts/path-to-bar_2026-11-08.md`, from the 48.16 draft), a spread of about 0.010. Cheap cells
  get the seeds; the 4.8M cell gets one because a second would cost another 90 to 138 wall hours.
  A single 4.8M seed can show only a difference larger than that spread.
- **Metric:** real-data performance, scored on V4FinBench (average precision first, five folds,
  full panel) before TabArena, per CLAUDE.md. Synthetic held-out accuracy is not used.
- **Verdict rule, written before running:** monotone if each step up in tasks improves the
  five-fold mean AP by more than the seed spread at both sizes; saturating if the 480,000 to
  4,800,000 step gains less than half the 48,000 to 480,000 step; flat otherwise. Three points do
  not fit an exponent (proposal non-goal).

**Cells (12 training runs).**

| cell | model | tasks | seeds | status |
| --- | --- | --- | --- | --- |
| S-48k | 885K | 48,000 | 0, 1, 2 | seed 0 exists (§93's published checkpoint); seeds 1, 2 new |
| S-480k | 885K | 480,000 | 0, 1 | both exist (§176, §188); reuse only if the recipe is identical |
| S-4.8M | 885K | 4,800,000 | 0 | new |
| M-48k | 5.0M | 48,000 | 0, 1, 2 | new (§114's arm was trained at batch 4 and its recipe predates this grid) |
| M-480k | 5.0M | 480,000 | 0, 1 | new |
| M-4.8M | 5.0M | 4,800,000 | 0 | new |

## Pricing (9.2)

### Step times used, and where each comes from

| use | s/step | hardware, batch 8 | source |
| --- | --- | --- | --- |
| 885K, all rows | **0.54** | Kaggle T4x2, DDP, split attention, published recipe | `docs/infra/COMPUTE.md` published-recipe table, from §197 and §198 (6,000 steps in 3,233 s and 3,285 s) |
| 5.0M multiplier | **1.531** (= 0.98 / 0.64) | Tesla T4, 4.9M against 847K params, batch 8 | `docs/infra/COMPUTE.md` "Rented NVIDIA, measured" (pre-cell-attention architecture) |
| 5.0M, derived | 0.827 (= 0.54 x 1.531) | T4x2 | derived, ESTIMATED |
| cross-check, 885K | 0.558 and 0.588 | T4x2, 60,000 steps | 33,505 s (§202, `dr74-mc-v480-s0`); 35,300 s (§233) |
| cross-check, single GPU | 1.90 / 1.02 | T4 (§86) / L4 (§127) | `docs/infra/COMPUTE.md` |

### Assumptions

1. **Quota hours are session wall hours.** §233 records 35,300 s as "about 9.8 h of the weekly
   GPU quota", so a T4x2 session is billed as wall time, not as two GPU-hours. GPU-hours below
   are 2 x wall hours.
2. The Kaggle weekly quota is **30 h** per week.
3. The 5.0M step time is **not measured on the published recipe**: it applies the pre-cell-attention
   T4 ratio (4.9M over 847K, 0.98 / 0.64) to the measured 885K T4x2 time. COMPUTE.md warns
   these numbers move after an architecture change. This is the weakest number in the table.
4. The 480,000-task runs measured 0.558 and 0.588 s/step, 3 to 9% above 0.54 (longer runs carry
   checkpointing and loader overhead). Totals are not inflated for this; add up to 9%.
5. Excluded: installation, scoring (V4FinBench scoring is a separate CPU-bound step, §202), failed
   sessions, and the 12 h session limit. Any run above about 11 h needs `--resume` across
   sessions, and a cross-session resume is **not yet measured** (AU5.15 note in
   `openspec/changes/audit-2026-10-15/tasks.md`): M-480k (13.8 h), S-4.8M (90 h) and M-4.8M (137.8 h)
   all depend on it.
6. Single L4 is shown for 885K only as a hardware cross-check, at 1.02 s/step (§127).

### Per run and per cell

| cell | steps | wall h per run (T4x2) | GPU-h per run | runs | cell wall h (quota h) | cell GPU-h | L4-h per run (885K scale; 5.0M x 1.531) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S-48k | 6,000 | 0.90 | 1.80 | 3 | 2.70 | 5.4 | 1.7 |
| S-480k | 60,000 | 9.00 | 18.00 | 2 | 18.00 | 36.0 | 17.0 |
| S-4.8M | 600,000 | 90.00 | 180.00 | 1 | 90.00 | 180.0 | 170.0 |
| M-48k | 6,000 | 1.38 | 2.76 | 3 | 4.13 | 8.3 | 2.6 |
| M-480k | 60,000 | 13.78 | 27.56 | 2 | 27.56 | 55.1 | 26.0 |
| M-4.8M | 600,000 | 137.81 | 275.62 | 1 | 137.81 | 275.6 | 260.3 |

### Totals

| scope | wall (quota) h | GPU-h | Kaggle weeks at 30 h |
| --- | --- | --- | --- |
| Full grid, 12 runs | **280.2** | **560.4** | **9.3** |
| Excluding the three existing runs (S-48k seed 0, S-480k seeds 0 and 1) | 261.3 | 522.6 | 8.7 |
| Stage A only (48,000 and 480,000 columns, both sizes, 10 runs) | 52.4 | 104.8 | 1.7 |
| Stage B only (the two 4.8M runs) | 227.8 | 455.6 | 7.6 |
| Not proposed: a 10,000,000-task run, one seed (S / M) | 187.5 / 287.1 | 375.0 / 574.2 | 6.3 / 9.6 |

Staging is the recommendation if the owner accepts any of it: Stage A first (about 1.7 quota
weeks), and Stage B only if Stage A shows a signal above the seed spread. Both columns have prior
nulls at 480,000 tasks (§93, §176, §188, §202), so the 4.8M cells are the expensive bet that a
trend appears only past 480,000 tasks.

Consistency check against existing figures: AU5.15's 2M-task estimate is 49 to 56 single-GPU
hours (`openspec/changes/audit-2026-10-15/tasks.md`, from MEASURED 40,934 and 36,000 tasks per
GPU-hour with fp16 split, 0.70 and 0.80 s/step on one T4). This grid uses COMPUTE.md's fp32
T4x2 figure of 0.54 s/step, which is 1.08 GPU-seconds per step, so it costs about 1.4 to 1.5
times more per GPU-hour than that fp16 figure (derived). If fp16 is adopted for these cells the
prices fall by about that factor, but fp16 at 5.0M is untested and neither fp16 time appears in
COMPUTE.md, so it is not used here.

## Gaps stated before capabilities

- The 5.0M step time is a ratio from a different architecture, not a measurement. A 50-step
  benchmark of a 5.0M published-recipe model on a T4x2 would turn it into a measured number
  (about 1 session-hour, ESTIMATED).
- Cross-session resume is unmeasured.
- Task 9.2's Verify says "numbers from `docs/infra/COMPUTE.md`, not estimated afresh". The 885K
  step times are COMPUTE.md's; the 5.0M step time is a new extrapolation, so 9.2 is not closed.
- Only 4.8M differs from what has been tested; at 480,000 tasks the prior evidence is null.
