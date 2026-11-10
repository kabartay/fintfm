# Path to the TabArena-Lite bar: decision memo for DR7.6 (2026-11-08)

Draft, not a FINDINGS entry, and `tasks.md` is not edited. Every number is labelled MEASURED,
ESTIMATED or SMOKE-TEST, with its source. V4FinBench decides before TabArena (CLAUDE.md), so
V4FinBench results are read first and every Lite figure is reporting, not selecting (DR4.1).

## Result first

- **The bar is not reachable by tuning the current binary checkpoint, on the evidence held.**
  The bar is about 1,010 Elo (`docs/roadmap/TABARENA_BAR.md`, DR7.1); the binary model is 756
  (+126/-180), 94th of 95 (`docs/results/TABARENA.md`, DR3.5). The gap is +254 Elo.
- The largest measured Lite lever is the linear-probe readout, +154 Elo (910 against 756).
  It lands at `Linear (default)` level, and the bar sits above `Linear (tuned + ensembled)`
  (984) and `RF (default)` (1000). Stacking every measured Lite lever gives about 960
  (ESTIMATED, additivity unmeasured), still about 50 short and inside the +/-150 interval.
- What the bar needs is nonlinear capability, which §204 and §137 say the checkpoint lacks.
  The only measured sign of movement there is on the proxy suite, which DR4.2 has not yet
  validated against TabArena.
- DR7.6 also says "binary and multiclass declared together". Multiclass is at Elo 306
  (+76/-2619), 94th of 95, over 7 datasets (`docs/results/TABARENA.md`, 2026-10-26, MEASURED). That is a
  second gap, not counted in the +254.

## 1. Inventory of measured levers

### 1a. TabArena-Lite (27 binary datasets, split 0, ROC-AUC, Elo against the 95-method pool)

Baseline: published `v4-cellattn-labels.pt`, NaN passed through, 756 (`docs/results/TABARENA.md`, DR3.5,
MEASURED). Intervals are +/-125 to 215 wide; every arm below overlaps the baseline.
Reference rows (`docs/roadmap/TABARENA_BAR.md`): KNN (tuned) 859, Linear (default) 936, Linear (tuned + ens.)
984, RF (default) 1000, KNN (tuned + ens.) 1008.

| lever | Lite Elo (change) | paired result | source |
| --- | --- | --- | --- |
| Linear probe on the representation (§132, zero-fill era) | 910 (+154) | n/a | `docs/results/TABARENA.md` §132 line; §204 |
| Probe, stratified probe rows (fix 1) | 914 (+4 over probe) | 5/2/20, p 0.45 | §210, MEASURED |
| Member transforms `rank, squash` | 807 (+51) | 20/6/1, p 0.009, +0.01023 AUC | `docs/results/TABARENA.md` AU5.7/AU5.12, §210 |
| `feature_transform="squash"` (fix 6) | 777 (+21) | 18/8/1, p 0.076, +0.01004 | §210 |
| SVD 16 components | 782 (+26) | 17/10, p 0.25 | `docs/results/TABARENA.md` |
| Class-shift ensemble | 716 (-40) | 10/17, p 0.25 | `docs/results/TABARENA.md` |
| One member instead of 8 | 696 (-60) | 7/20, p 0.019 | `docs/results/TABARENA.md` |
| Context 500 / 750 rows against 1,000 | 702 (-54) / 745 (-11) | p 0.85 / 0.29 | `docs/results/TABARENA.md` AU5.14, DR3.5 pairing |
| Minority coded as 1; positive cap 0.3 | 756 (0) | 4/1/22; 9/2/16 | §210 |
| E1, 480,000 tasks (`e1-v480-s0`) | 842 (+86) | not paired | `docs/results/TABARENA.md`, 2026-10-24 |
| Other checkpoints: priorctl-s1 / treeclean-s1 / widerprior / mixedprior | 737 / 826 / 761 / 813 | n/a | `drafts/dr4.2_2026-11-07.md` section 2, 5 (stored pool; match by name and timestamp) |
| NaN passed through against filled with 0 | 756 against 763 | 2/4 of 6 differing, p 0.69 | `docs/results/TABARENA.md` DR3.5 |
| Multiclass checkpoint, 7 datasets | 306 | n/a | `docs/results/TABARENA.md` 2026-10-26 |
| `retrieval` | not run | 0 of 4 completed (MPS OOM, then time limit) | `docs/results/TABARENA.md`, §210 |
| Fine-tuning, partition | **not measured on Lite** | | see section 2 |

Why the gap exists (§204, MEASURED from cached results): FinTFM is below `LR (default)` on 23 of
27 datasets by a roughly constant mean 0.0297 ROC-AUC (0.7824 against 0.8120), below CatBoost
(0.8449) and the best default TFM (0.8572) on 27 of 27. The gap to CatBoost tracks how much a
dataset rewards a nonlinear model (rho -0.69, p < 0.001). Size, width, categorical share and
missingness explain nothing measurable. §204 concludes that adapter and inference settings can
at best close the gap to LR. Local slope for converting AUC to Elo, ESTIMATED from the arms above
(probe +0.0272 for +154; rank+squash +0.0102 for +51): about 5 Elo per 0.001 mean ROC-AUC, so
+254 Elo is about +0.045 AUC.

### 1b. V4FinBench (five-fold mean AP, full panel, zero-shot published checkpoint 0.1986)

Context for every row: untuned CatBoost 0.4005, LightGBM 0.2666, XGBoost 0.2229, logistic
regression 0.1614 (§163, MEASURED). Bar for a lever: +0.0073 with 4 of 5 folds (§159); noise
yardstick 0.0117 (§173).

| lever | change in mean AP | folds up | verdict | source |
| --- | --- | --- | --- | --- |
| Partitioned contexts, depth 2, zero-shot | +0.0375 | 5/5 | helps | §169, MEASURED |
| Partition depth 1 / 2 / 3 | 0.2347 / 0.2361 / 0.2357 | n/a | flat | §169 addendum, quoted in `drafts/dr3.11_2026-11-07.md` |
| Fine-tuning 200 steps (§163) | +0.0076 | 5/5 | helps, barely | §163 |
| Partition against §163's fine-tuned arm (AU5.8) | +0.0299 | 5/5 (4 Holm) | partition wins | §205 |
| Fine-tuning per leaf of the partition (DR3.11) | +0.0049 over partition | 4/5 | null | §208 |
| Fine-tuning: second seed / enriched queries / 600 steps | +0.0071 / +0.0009 / +0.0024 over zero-shot or §163 arm (as stated in §164) | 5/5, 3/5, 3/5 | not replicated / no different | §164 |
| 8 members (`ens8`) | +0.0058 | 3/5 | below bar | §196 |
| Positive floor exact / floor16 | +0.0024 / +0.0055 | split | null | §200 |
| Prototype context (AU4.2), full panel | +0.0002 | 2/5 | null | task brief, MEASURED 2026-11-08; not yet in FINDINGS (predictions in `runs/au42/prototype`) |
| `retrieval` context | AP 0.0126 (collapse) | 0/5 | harmful | §194 |
| Drop `operational_status`, zero-shot / fine-tuned | -0.0022 / -0.0023 | 4/5, 5/5 down | uses the column little | §200, §202 |
| FinTFM inside CatBoost, baseline / feature (a system) | +0.0039 / +0.0018 | 4/5, 2/5 | null | §201 |
| Probe, 2,000 context rows, against head | 0.1976 against 0.1986 | tie | label count, not representation | §209 |
| Probe, 20,000 stratified rows | 0.2213 (+0.0227 on the head) | n/a | more labels help | §209 |
| Cell attention against none (architecture) | +0.052 | 5/5 | the one large architecture effect | §172 |
| Ten times the tasks (480,000), two seeds | -0.0106 | 1/5, 0/5 | null | §176, §188 |
| Depth 8 / width 192 at 480,000 tasks | -0.0047 / +0.0003 | 3/5 | null | §193, §195, §202 |
| Multiclass arm at 480,000 tasks | proxy only | | null | §202 |
| Prior changes: mixing task families, realism augmentations, learnability filter, schedule-free, mask embedding, real-data pretraining, DR6.2/6.3 bundle | negative or null | | each a loss or null | §141, §144-§147, §161, §178 (numbers not re-quoted here) |
| Prior general-trunk arms (c)/(b)/(d)/(a) | V4 AP 0.1428 / 0.0713 / 0.1616 / 0.1762 against 0.1975 | | all below control | §180, §191 |

### 1c. Proxy suite (16 binary tasks; not validated against TabArena, DR4.2 rho 0.8 unmet)

| lever | result | source |
| --- | --- | --- |
| Fine-tuning 200 steps | 16 wins, 0 losses, p 3e-5; split-0 arena 726 [527, 865] against 419 [209, 555] (+307) | §192, MEASURED, one split |
| `ens8` / `retrieval` | 13/2 (p 0.0074) / 8/1 (p 0.039), candidates; V4 then +0.0058 / collapse | §187, §194, §196 |
| General-trunk prior, then financial v2 | proxy Elo 764 against 548 for the control (+216), 13/3 per task, p 0.021; V4 AP 0.1428 against 0.1975 | §191 |
| 480,000 tasks | positive on proxy, null on V4 | §181, §188 |
| Depth 8 / width 192 at 480,000 | null (8/8; 10/6) | §195, §202 |

The two benchmarks reward different levers. V4's two real wins (partition, 8 members) are
large-n, low-base-rate credit levers with no route to Lite, where the median training split is
a few thousand rows and §204 finds no size effect. Lite's wins (probe, transforms) are
general-tabular and have no V4 result that is recorded (see section 2, `tfsq8`).

## 2. Pending, and what it could be worth on Lite

Queue read from `scripts/kaggle/launch_queue.sh` (a generic runner) and the live lane command
lines on 2026-11-08, plus the progress notes in the three `tasks.md` files. "Lite effect" is
stated only where something is measured; otherwise "none measured".

| item | state on 2026-11-08 | plausible Lite Elo effect |
| --- | --- | --- |
| `task_lite_partition` (Fix 4, §204; DR3.11 gate) | pushed 11-07 22:54, running or finished, not scored | none measured. Prediction in `dr3.11_2026-11-07.md`: loses below about 2,000 training rows, ties above. Nothing to gain; it is a "must not lose" gate |
| AU5.12 `tfsq8` on V4 (8 members, rank+squash) | all three kernels complete; folds fetched to `runs/au512-tfsq8`; `cmp_vs_ens8.txt` is empty, so the verdict is not recorded | the Lite half is MEASURED (+51). V4 verdict is the missing credit number |
| AU4.3 `ctx900e3`, AU5.7 etc. | `ctx900e3` and `au517-muon`, `au54-ctx2k` kernels complete and fetched; not scored or recorded | none measured; Lite context 500/750/1,000 is flat |
| Lane 1 still to run: AU5.16 rbf, AU5.17 lenscale, AU5.3 cell2, AU5.9 hazard, DR8.3 width512, 48.25 head-only, AU5.1 lr 1e-5 / 1e-4 | queued | none measured; each is a 48,000-task arm with stated prediction null (`preregs_2026-11.md`, §207), and one arm resolves only 0.0117 AP (§173) |
| Lane 2 still to run: AU5.1 lr 3e-5, AU5.11 diverse and mixture, 48.15 maskcol, 48.27 latent, `dr311_bag4_f23` | queued | AU5.11 predicted harmful on V4 and a candidate only for a separate general family (`au5.11_2026-11c.md`); the rest null predicted; none measured on Lite |
| DR3.11 `ens8`, `bag8` | not queued, held on bag4's futility rule | none measured; Lite already uses 8 members |
| AU5.8 `boost`, `pc16` | prepared, not queued; boost ESTIMATED 18 GPU-h (`v4_inference_kernels_2026-11.md`) | none measured; large-n method |
| 48.16 `column_id_dim` 18 | drafted, not run | none measured |
| AU5.15 2M-task run | throughput measured (AU5.15 draft), run not started | none measured. Only point: E1 at 480,000 tasks is +86 on Lite, inside the interval, one seed |
| DR4.2 n = 9 proxy runs | three proxy runs not run; 0.6 to 6 h local (ESTIMATED, `dr4.2_2026-11-07.md`) | decides whether the proxy can be trusted; no Elo of its own |
| DR3.2 class-shift on V4, DR4.3 credit runs, DR8.4, DR7.5, DR6.4 | prepared, not run | none measured; not Lite levers |
| Lite fine-tuning, Lite partition + probe combination, Lite run of §191's (c) checkpoint | **not proposed anywhere in the three `tasks.md` files** | none measured |

## 3. Ranked recommendation

Costs are ESTIMATED from MEASURED step times, quoted with their basis. Kaggle quota is
30 GPU-h per week (`preregs_2026-11.md`).

**1. Stack the two Lite levers that are measured, in one local run (about 1 h, no GPU).**
Linear probe with stratified rows (914) plus `rank, squash` member transforms (807). Expected
about 960 if additive (ESTIMATED: 914 plus part of +51; additivity is unmeasured), so this run
either confirms a ceiling near LR level or finds more. It also tells us where the stack stops,
which decides whether anything else is worth buying. Cost: a single-fold Lite run is about an hour
(CLAUDE.md), one process per dataset on MPS (`docs/results/TABARENA.md`), nothing while `bwa` and a solver
campaign contend. Label the result "FinTFM with a trained linear readout", not zero-shot. The probe's
V4 advantage is a label-count effect (§209), so on Lite it is a use of the training rows beyond
the context, which is legitimate but not the zero-shot claim.

**2. Measure fine-tuning on Lite, six datasets first.** It is the only lever with a large measured
effect on a non-V4 benchmark (+307 proxy Elo, 16 of 16, §192) and it acts in gradients on the very
nonlinearity §204 says is missing. On V4 it is small (+0.0076). Pilot on the five datasets where RF
beats LR by more than 0.03 (`churn`, `Bank_Customer_Churn`, `in_vehicle_coupon_recommendation`,
`polish_companies_bankruptcy`, `E-CommereShippingData`) plus one control. Cost, ESTIMATED: no Lite
step time is measured; the proxy fit was about 2,000 s of CPU per split (§187) and V4's 200-step fit
was 2,400 s per fold on a T4 with a 200,000-row score (`dr3.11_2026-11-07.md`), so plan 2 to 6
GPU-h for the pilot. Two risks: TabArena's protocol bags 8 folds per dataset, so the full 27 would
be about 8 times a single fit (16 to 48 GPU-h, ESTIMATED), and TabArena's time limit already
defeated `retrieval` (§210). Go to the full 27 only if the pilot gives at least +100 Elo-equivalent
on those five.

**3. Run Lite on §191's general-trunk checkpoints, after DR4.2's n = 9 check.** The proxy says
trunk-then-financial is +216 proxy Elo over the control, and V4 says it loses 0.0547 AP
(0.1428 against 0.1975). If the proxy holds for TabArena, the right home is a separate general
family, never the credit checkpoint (`au5.11_2026-11c.md` states the same). Cost: DR4.2's three
proxy runs 0.6 to 6 h local, then one Lite run per checkpoint at about 1 h each, no GPU, if the
weights are still on disk (not checked). This is the cheapest test of whether a prior change can
buy nonlinear capability at all.

**Combination that could reach the bar.** (1) plus (2) plus a general-trunk checkpoint (3).
Nothing measured shows these add, and the sum is not an estimate I can defend; I am not giving one.

### Drop, with reasons

| item | verdict | reason |
| --- | --- | --- |
| AU5.8 boost (not queued) | do not queue | ESTIMATED 18 GPU-h; V4-only large-n method; the system hybrid AU4.4 gave +0.0039 (§201); Lite size effect is nil (§204). Capped pool means a null would not close it anyway |
| DR3.11 `ens8` and `bag8` | drop both, keep `bag4_f23` | stated prediction null, +0.002; ESTIMATED 21 GPU-h each; AU4.2's null (+0.0002) and §208 (+0.0049) point the same way; bag4 f23 is 4.2 GPU-h and is already queued, so let its futility rule decide whether anything follows |
| 48.16 `column_id_dim` 18 | drop | the two surrogates differ by 0.0095, below the 0.0117 bar, so one run cannot separate them, and 16's re-trains scored 0.1947, 0.1858 and 0.1845, so the peak is partly seed. Neither surrogate beat the mean predictor on this curve (48.16 draft). About 1.6 T4x2 session-hours plus a CPU rescore |
| AU5.15 2M-task run | drop | ESTIMATED 49 to 56 GPU-h single GPU (2,000,000 tasks at 40,934 and 36,000 per GPU-hour, MEASURED rates), about 1.6 to 1.9 quota weeks. 480,000 tasks was null on V4 at two seeds (-0.0106), depth and width at 480,000 null, and on Lite +86 within an interval of +/-175. A log-linear read of that one point gives about +50 for 2M (ESTIMATED, one seed, shape unknown), a tenth of what is needed. §207 already recommends no 2M run (DR5.4) |
| More single-lever 48,000-task arms (AU5.16/17, AU5.3, AU5.4, AU5.1, 48.15, 48.27) | let queued ones finish; add no seed-1 replicates unless an arm is promising | all stated predictions null; each resolves 0.0117 AP |

### Is the bar reachable with this architecture?

Honestly: not by anything measured. Evidence against: inference and head levers are bounded by LR
(§204); scale in volume, depth and width is null on V4 and the proxy (§188, §195, §202); the
checkpoint is at chance on any interaction of order two (§137); and the best stack I can build from
measured parts sits near 960 against a bar of 1,010, with a multiclass arm that drags a joint
declaration down. Against that: the intervals are wide (+/-150), Lite is one split, the bar is itself
an Elo in a pool that moves (§150), and the one large non-V4 effect (fine-tuning, +307 on the proxy)
has not been tried on Lite.

**Evidence that would change this reading:**

- Probe plus transforms on Lite at or above about 980 (then the bar is a few inside-interval steps
  away, and a paired per-dataset pass, as `docs/roadmap/TABARENA_BAR.md` requires, is the next test).
- Fine-tuning pilot at least +100 on the five nonlinear datasets.
- DR4.2 n = 9 with rho at or above 0.8 and a general-trunk checkpoint at or above about 900 on Lite
  (then a separate general family is the route).
- A checkpoint with order-two interaction above chance in the §137 probes and a `churn` AUC that
  moves toward RF's 0.9252 (FinTFM 0.7316).

**Evidence that would close it:** the stack landing at or below 930 and the fine-tuning pilot
below +50 on the five nonlinear datasets. Then DR7.2's "reopen only with a number at the bar"
should stand with no resubmission planned until a prior or architecture change shows
nonlinear capability on the proxy and on Lite.

## Not checked

- `runs/au512-tfsq8`, `ctx900e3`, `au517-muon` and `au54-ctx2k` results exist locally and are
  unrecorded; I did not score them (heavy, and out of scope).
- AU4.2's prototype result comes from the task brief; it is not yet in FINDINGS.
- Whether §191's checkpoints are still on disk, and whether the probe path runs inside TabArena's
  time limit for the 27 datasets.

## Owner decisions, 2026-11-08

Recommendations accepted: the Lite stack run, the drops (AU5.8 boost, DR3.11 8-member and 8-bag
arms, the 48.16 colid 18 check) and the deferral of AU5.15's 2M-task run, each recorded in its task.

**Correction to recommendation 1.** The probe cannot be stacked with the `rank, squash` *member*
transforms: `FinancialTFMClassifier.transform_representation` reads one forward pass on the base
context, so ensemble members never reach the probe, and that run would reproduce the probe alone.
The run stacks the probe with the single-pass `feature_transform="squash"` instead (§210 fix 6,
777, +0.01004 mean ROC-AUC over the default), which the probe does read through the conditioner.
Queued locally as `fintfm_linprobe_squash_1108` (`runs/local_queue/lite_stack.sh`: CPU, one
process per dataset, 27 binary datasets), paired against `fintfm_linprobe_d1fix_1104` (914) and
the default `fintfm_lite_nan_1019` (756). SMOKE-TEST on blood-transfusion-service-center: exit 0,
70 s, 2.4 GB peak, override seen. Label any result "FinTFM with a trained linear readout", and if
it wins, a Lite-only declared configuration: squash is null on V4FinBench (§211), so it never
becomes the credit default.

## Results, 2026-11-08 (FINDINGS §212)

- Probe plus squash: **969 Elo**, MEASURED, against the ESTIMATED 960 above; 41 short of 1,010.
  Against the probe alone 18 / 9, sign p 0.12; Elo intervals about ±150.
- Partition on Lite (Kaggle T4): **888 Elo**, 23 / 4 against the same-GPU default (751).
- The next candidate stack is probe plus partition, but the adapter cannot run it: the
  partition wrapper has no `transform_representation`, so it needs a probe per leaf (new fork
  code, then a CPU smoke test, then a Kaggle run). Not started; owner to decide.
  **Update 2026-11-09 (FINDINGS §215):** built (probe per leaf, fork working tree, uncommitted)
  and run on local CPU instead of Kaggle: **1,000.2 Elo**, MEASURED, level with `RF (default)`
  (1000.0, 13 / 14 on paired datasets), not above it. The 1,010 figure above came from an older
  leaderboard; the bar's definition, parity with RF (default), is met as a tie and not exceeded.
  **Update 2026-11-10 (FINDINGS §218):** on V4FinBench the probe per leaf is harmful, −0.0570
  mean AP, 0 of 5 folds, all Holm-significant, while the partition head (+0.0375, 5 of 5) and the
  global probe (+0.0227, 4 of 5) each help alone. The Lite configuration stays Lite-only and must
  not be described as a credit configuration.
- AU5.13's reduced T+E: about 14.7 to 21 GPU-h for 8 draws, ESTIMATED from the 1.84 T4-h
  default; recommended deferral, owner to decide.
