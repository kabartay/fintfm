Merged into FINDINGS §205 and §206 (2026-11-04).

# Inference-only open tasks, 2026-11-02/03 (draft for FINDINGS)

Draft, not yet merged into `docs/results/FINDINGS.md`. Covers the ten inference-only open tasks
AU4.2, AU4.3, AU5.7, AU5.8, AU5.9, AU5.12 (`openspec/changes/audit-2026-10-15/tasks.md`) and DR3.5,
DR7.5, DR8.1, DR8.3 (`openspec/changes/design-review-2026-10-16/tasks.md`). For each: what
existing results already answer, what the proxy suite was run for here, and what still needs a
full-panel V4FinBench run (command and estimate given, not run).

**Constraints on this session.** Shared, memory-constrained Mac (a full-panel job froze it on
2026-10-27). No full-panel V4FinBench scoring and no `--device mps`. Proxy-suite runs: CPU, one
arm at a time, `nice -n 10`, `--n-jobs 2`, each started only when `vm_stat` free plus inactive was
at least 12 GB (18-19 GB at each start). Per CLAUDE.md ("Score on the benchmark this project
claims"), a proxy result advances a task and never decides a credit claim; V4FinBench does.

**Checkpoint.** Every number is the published `runs/v4-cellattn-labels.pt` (48,000 tasks),
zero-shot unless stated. Commit 82238f6 plus this branch's arm configs.

## Summary

| task | action | result |
| --- | --- | --- |
| AU4.2 | floor half answered by §200; prototype half needs V4FinBench | floor16 +0.0055 mean AP, null; prototype not run (command below) |
| AU4.3 | proxy run; V4FinBench needed | proxy 11 wins, 5 losses, p 0.21, not a candidate |
| AU5.7 | existing results collected; one re-read of existing CSVs | V4 `ens8` +0.0058 (§196); TabArena-Lite arm still open |
| AU5.8 | partition arm compared against §163's arm from saved predictions | +0.0299 mean AP, 5 of 5 folds, 4 Holm-significant; boost and per-class retrieval not run |
| AU5.9 | not inference-only on any existing checkpoint | needs a pretraining run; not run |
| AU5.12 | two member axes run on the proxy, two re-read | no axis a candidate against 8 plain members; SVD a lead (12 wins, 4 losses, p 0.077) |
| DR3.5 | context 500/750 paired per dataset from existing Lite results; class shift and transforms need the fork | 500: 12 better, 14 worse; 750: 14 better, 8 worse; neither significant |
| DR7.5 | no inference lever left | blocked on a wider checkpoint (DR8.3) |
| DR8.1 | Verify already met | closed, citing §187, §192, §194, §196 |
| DR8.3 | no inference lever left | blocked on the `max_features` 512 checkpoint |

## Proxy arms run in this session (pre-registered here before any ran)

Proxy suite, the 16 binary tasks within the 136-feature cap (of the 18 requested;
`madelon` and `Internet-Advertisements` skipped by the cap as in §181), OpenML splits 0-2. Rule:
§183's, unchanged: an arm is a **candidate** if its per-task two-sided sign test against its
control gives p < 0.05 with more wins than losses. Controls are existing CSVs from 2026-10-23/24
(`runs/dr73/`), run from an earlier commit; inference code for these settings has not changed
since, but the comparison is across days, as §194 and §196 were.

| arm (config `configs/arms/proxy/`) | change | control | prediction (written before running) |
| --- | --- | --- | --- |
| `au43-ctx900-ens3` | `max_context` 900, `n_ensemble` 3 | `FinTFM-binary`, `proxy_binary.csv` | not a candidate (`ctx8k` was 6/2/8, so shorter context should not help) |
| `au512-ens8-cshift` | `n_ensemble` 8, `ensemble_class_shift` | `lever-ens8`, `proxy_lever_ens8.csv` | not a candidate (`ens8-swap`, the same axis at twice the cost, was not) |
| `au512-ens8-svd16` | `n_ensemble` 8, `ensemble_svd_components` 16 | `lever-ens8` | not a candidate |

Command (one arm shown; outputs `runs/inf2611/`):

```bash
PYTHONUNBUFFERED=1 nice -n 10 uv run --frozen --extra bench python -m fintfm.evaluation.proxy_suite \
  --cache-dir data/cache/proxy_suite run --model runs/v4-cellattn-labels.pt \
  --config configs/arms/proxy/au43-ctx900-ens3.yaml --name au43-ctx900-ens3 \
  --tasks 3,15,43,49,3918,7592,9946,9976,9977,9978,10093,14952,14954,125920,146819,146820,167120,167125 \
  --no-baselines --n-jobs 2 --device cpu --out runs/inf2611/proxy_au43-ctx900-ens3.csv
uv run --frozen --extra bench python -m fintfm.evaluation.proxy_suite compare \
  runs/inf2611/proxy_au43-ctx900-ens3.csv runs/dr73/proxy_binary.csv --unit task --anchor RF
```

Wall time on the loaded machine (load average about 33 on 16 cores): `au43-ctx900-ens3` 48 min,
`au512-ens8-cshift` 2 h 58 min, `au512-ens8-svd16` 2 h 21 min.

## AU4.2: context composition

**Floor half: already answered, null (§200).** `configs/arms/dr33-floor16.yaml` (a uniform context
with at least 16 positives, D5 correction on), full panel, five folds: 0.2042 mean AP against the
same-day control's 0.1986, +0.0055, 3 folds up and 2 down with significant moves both ways; exact
positive count +0.0024. Both below §173's 0.0117 yardstick. (The task names §159's rule; §200 used
§173's yardstick, which replaced it for this kind of comparison; the verdict is null under either,
since both need a consistent sign across folds.)

**Prototype half: not measured on the full panel.** The only V4FinBench number is §162's fold 0 at
the 200,000-row development cap, +0.0483 AP, Holm p 0.054, and §162's own confirmation step showed
the cap changes the shape of context results. §38's prototype win was a survival result on an
older checkpoint.

**Proxy not run, on purpose.** The question is how a context carries about 7 positives from a
0.35% pool. The proxy's binary tasks have a median of 2,310 training rows and are mostly not
low-base-rate, so a prototype result there would not bear on it.

**For later (not run):**

```bash
PYTHONUNBUFFERED=1 nice -n 10 uv run --frozen fintfm-v4protocol --model runs/v4-cellattn-labels.pt \
  --config configs/arms/au42-prototype.yaml --no-boosting --device cpu --out runs/au42/prototype \
  > runs/au42/prototype.log 2>&1; echo "EXIT=$?"
uv run --frozen python scripts/compare_v4_scores.py --cluster --control runs/lrsweep/eval-published \
  prototype=runs/au42/prototype
```

ESTIMATED: about 1.1 h per fold for the forward passes (one member, from §194's one-member
retrieval run), plus k-means over a ~600,000-row pool per fold (not measured); 6 to 8 h in all.
Memory ESTIMATED 12-14 GB resident (this machine's record for one full-panel scoring); run alone.

## AU4.3: context length beyond training

**The checkpoint does not record its `n_rows` choices.** `torch.load(...)["config"]` holds only
the model's architecture (`max_features` 136, `d_cell` 48, `d_model` 128, `n_heads` 4,
`n_col_layers` 2, `n_layers` 4, `d_ff` 512, `pooling` meanmax, `column_id_dim` 16,
`n_cell_blocks` 1, `cell_labels` true) and `trained_objectives` `['classification']`. The training
command recorded in FINDINGS (the §127 reference recipe, quoted in the 48.6 entry) is
`--n-rows-choices 256,512,1024`; with `min_ctx_frac` 0.3 and `max_ctx_frac` 0.9
(`prior/mixture.py:208-209,1042`) every training context held 76 to 921 rows. A gap worth fixing:
the prior configuration should travel with the checkpoint.

**Existing evidence.** §162's five-fold full-panel confirmation: `max_context` 500 with 8 members,
−0.0029 mean AP, 2 of 5 folds positive, so the in-range context did not help on credit.
TabArena-Lite (AU5.14): 702, 745 and 756 Elo at 500, 750 and 1,000.

**Proxy, MEASURED** (`runs/inf2611/proxy_au43-ctx900-ens3.csv` against `FinTFM-binary` in
`runs/dr73/proxy_binary.csv`): **11 wins, 5 losses, 0 ties, p = 0.21 per task; 25 wins, 21 losses,
2 ties per split (p = 0.66). Not a candidate; prediction held.** Elo 589 [334, 754] against the
default's 566 [304, 740], RF at 1,000. Mean 1 − ROC-AUC difference +0.0022 in the arm's favour; on
the 8 tasks with more than 2,000 training rows (where the default's context is capped) 5 wins, 3
losses, +0.0017. Against 8 members at 2,000 rows (`lever-ens8`): 6 wins, 9 losses, 1 tie. Three
in-range members are no better than one out-of-range one here, and below eight.

**For later (not run):**

```bash
PYTHONUNBUFFERED=1 nice -n 10 uv run --frozen fintfm-v4protocol --model runs/v4-cellattn-labels.pt \
  --config configs/arms/proxy/au43-ctx900-ens3.yaml --no-boosting --device cpu \
  --out runs/au43/ctx900-ens3 > runs/au43/ctx900-ens3.log 2>&1; echo "EXIT=$?"
uv run --frozen python scripts/compare_v4_scores.py --cluster --control runs/lrsweep/eval-published \
  ctx900ens3=runs/au43/ctx900-ens3
```

ESTIMATED 1.5 h per fold (three members at 900 rows, about 1.35 times one member at 2,000 if cost
is linear in context rows), about 7.5 h; memory as AU4.2. Given §162 and the proxy, a low-priority
run.

## AU5.7: stronger default ensembling

Already measured: `ens8` on the proxy 13 wins, 2 losses, 1 tie (§187), on V4FinBench +0.0058 mean
AP, 3 folds up and 2 down, not adopted (§196); `max_features_strategy: subsample` covers the wide
tasks and scores them near the floor (§182). New here, from existing CSVs (no inference): `ens8-tf`
(transforms cycled) against `ens8`, **12 wins, 4 losses, p = 0.077** per task, 28 wins, 19 losses,
1 tie per split (p = 0.24); not a candidate.

**Open: the TabArena-Lite half of Verify.** The fork's adapter defaults to `n_ensemble` 8 and
`max_context` 1,000 and exposes only `FINTFM_MAX_CONTEXT` as an override
(the fork's `packages/tabarena/src/tabarena/models/fintfm/model.py`, `_default_max_context` and
`_set_default_params`), so the `n_ensemble` 1 arm needs a hyperparameter
override or an env var added to the fork first. **Added 2026-11-04 (fork, uncommitted):
`FINTFM_PARAMS`, a JSON object of `FinancialTFMClassifier` arguments that overrides the
adapter's defaults; an unknown or reserved key raises, and an explicit hyperparameter still
wins.** Arms: `FINTFM_PARAMS='{"n_ensemble": 1}'` (AU5.7),
`'{"n_ensemble": 8, "ensemble_class_shift": true}'` and
`'{"n_ensemble": 8, "ensemble_svd_components": 16}'` (AU5.12),
`'{"n_ensemble": 8, "ensemble_transforms": ["rank", "squash"]}'` (DR3.5). Then, as AU5.14 ran: `FINTFM_PROBLEM_TYPES=binary
FINTFM_RUN_NAME=fintfm_lite_ens1 nice -n 10 .venv/bin/python -P run_fintfm_lite.py --full`, about
an hour per arm (CLAUDE.md's figure for a 27-dataset single-fold run), and pair per dataset as in
DR3.5 below.

## AU5.8: large-n methods

**Partition arm against §163's arm, MEASURED from saved predictions (no new inference).**
`scripts/compare_v4_scores.py --cluster --control runs/ft/B_lr1e5 partition=runs/dr31/partition`,
output `runs/inf2611/cmp_partition_vs_ft200.txt`. Same full-panel folds; the control's fold 0 AP
(0.2181) matches §163's table.

| fold | §163 fine-tuned (200 steps) | partition (§169, zero-shot) | diff | 95% CI, companies | Holm p, companies |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.2181 | 0.2241 | +0.0059 | [−0.0129, +0.0248] | 0.487 |
| 1 | 0.2068 | 0.2262 | +0.0194 | [+0.0049, +0.0320] | 0.020 |
| 2 | 0.1901 | 0.2411 | +0.0510 | [+0.0333, +0.0681] | 0.002 |
| 3 | 0.1955 | 0.2370 | +0.0415 | [+0.0249, +0.0584] | 0.002 |
| 4 | 0.2205 | 0.2524 | +0.0319 | [+0.0139, +0.0494] | 0.003 |
| mean | 0.2062 | 0.2362 | **+0.0299** | | |

Under §159's rule (five-fold mean beyond the 0.0073 bar with at least 4 of 5 folds the same sign;
it also clears §173's later 0.0117 yardstick) the zero-shot partition arm
**beats** §163's fine-tuned arm: 5 of 5 folds positive, 4 significant after Holm. This settles the
partition third of AU5.8's Verify.

**Not measured: `--large-n boost` and `retrieval_per_class`.** The proxy suite builds FinTFM from
the `inference` section only, so it cannot run the `large_n` wrappers, and its tasks are too small
for a large-n method to matter. Per-class retrieval on the proxy was not run either: §194 showed
proxy retrieval (a candidate) collapsing on V4FinBench, so a proxy number would not change the
decision.

**For later (not run):**

```bash
# BoostPFN-style, 8 rounds of 512 context rows; cap the residual pool, or every round scores ~600k rows
PYTHONUNBUFFERED=1 nice -n 10 uv run --frozen fintfm-v4protocol --model runs/v4-cellattn-labels.pt \
  --large-n boost --no-boosting --device cpu --out runs/au58/boost > runs/au58/boost.log 2>&1
# LoCalPFN-style per-class retrieval, 16 positives + 1,984 negatives per query group
PYTHONUNBUFFERED=1 nice -n 10 uv run --frozen fintfm-v4protocol --model runs/v4-cellattn-labels.pt \
  --config configs/arms/au58-retrieval-pc16.yaml --no-boosting --device cpu \
  --out runs/au58/retrieval-pc16 > runs/au58/retrieval-pc16.log 2>&1
```

Each compared with `compare_v4_scores.py --cluster --control runs/ft/B_lr1e5`. ESTIMATED: per-class
retrieval about 1.1 h per fold (§194's one-member retrieval); boost unknown, between 8 and 25 h per
fold with an uncapped pool (8 rounds, each scoring the training pool), so time fold 0 first with
`large_n.boost_pool_rows` set (for example 100,000). Memory as AU4.2.

## AU5.9: hazard context periods

**Not run; not inference-only.** The published checkpoint is classification-only
(`trained_objectives` `['classification']`), and `hazard_period_context` is ignored on a flag-off
checkpoint. No flag-on checkpoint exists (`runs/` holds `v4-hazard-136f.pt` and
`v4-hazard-ldp.pt`, both earlier than the flag). The Verify asks for a hazard test where the term
structure is learnable only from context periods, which needs a pretraining run with
`fintfm-train --n-horizons 3 --hazard-period-context` (GPU, `--checkpoint-every`) and a synthetic
evaluation task built for it. Out of scope for an inference session.

## AU5.12: per-member inference diversity

The label-symmetry probe is recorded (§165). Member axes on the proxy, each against `ens8` so only
the axis differs:

| axis | arm | per task (wins / losses / ties) | p | per split | source |
| --- | --- | --- | --- | --- | --- |
| class shift (label swap on odd members) | `au512-ens8-cshift` | 11 / 5 / 0 | 0.21 | 30 / 17 / 1, p 0.079 | MEASURED here |
| label-swap twin per member | `ens8-swap` | 9 / 7 / 0 | 0.80 | 29 / 18 / 1, p 0.14 | re-read of §187's CSV |
| transforms `rank, power, squash` | `ens8-tf` | 12 / 4 / 0 | 0.077 | 28 / 19 / 1, p 0.24 | re-read of §187's CSV |
| SVD, 16 components on odd members | `au512-ens8-svd16` | 12 / 4 / 0 | 0.077 | 31 / 14 / 3, p 0.016 | MEASURED here |

**No axis is a candidate against 8 plain members under §183's per-task rule; the prediction held.**
All four lean positive. **SVD is the closest and is a lead:** 12 wins, 4 losses per task (p 0.077),
31 wins, 14 losses per split (p 0.016, not the registered unit), mean 1 − ROC-AUC +0.0170, Elo 1126
[1021, 1259] with `ens8` at 1,000. The mean is carried by two low-width tasks: wilt (5 features)
0.2018 to 0.0139, below §192's fine-tuned 0.049, and banknote-authentication (4 features) 0.0693 to
0.0056. Against the single-member default, `au512-ens8-svd16` is 13 wins, 3 losses (p 0.021), but
that bundles 8 members with SVD. **Why it will not carry to V4FinBench as is:** components are
capped at `max_features` minus the table width, so a 130-column V4FinBench fold gets at most 6;
the gain here comes from narrow tables. A V4FinBench run would test a 6-component arm, not this one. The
largest per-task moves for class shift are on the weakest tasks (climate-model-simulation-crashes
1 − ROC-AUC 0.2167 to 0.1600, wilt 0.2018 to 0.1538), mean +0.0080 over the 16. Against the
single-member default, `au512-ens8-cshift` is 11 wins, 5 losses (p 0.21), fewer than `ens8`'s 13
wins, 2 losses.

**Open: the TabArena-Lite half of Verify** ("each member axis scored on TabArena-Lite"). The fork
adapter exposes none of `ensemble_class_shift`, `ensemble_transforms` or
`ensemble_svd_components`; one env var or hyperparameter pass-through is needed, then one Lite run
per axis, about an hour each.

## DR3.5: TabArena-Lite levers

**Context 500 and 750 against 1,000, paired per dataset, MEASURED from existing results (no new
fit).** The fork's `experiments/fintfm_lite_{ctx500,ctx750}_1026/` against
`experiments/fintfm_lite_nan_1019/` (the 1,000-row run the AU5.14 table uses), `results.pkl` per
dataset, split 0, ROC-AUC error:

| `max_context` | better / worse / tied (of 27) | sign test p | mean ROC-AUC change |
| --- | --- | --- | --- |
| 500 | 12 / 14 / 1 | 0.85 | −0.00048 |
| 750 | 14 / 8 / 5 | 0.29 | +0.00096 |

Largest moves at 500: Marketing_Campaign +0.0180, churn +0.0084; Is-this-a-good-customer −0.0118,
Diabetes130US −0.0092. At 750: in_vehicle_coupon_recommendation +0.0155, churn +0.0096;
seismic-bumps −0.0061. Null both ways, consistent with AU5.14's Elo; default 1,000 stays. The
1,000-row arm is a 2026-10-19 run, not same-day. This belongs in `docs/results/TABARENA.md` beside
the AU5.14 table when merged.

**Open: `ensemble_class_shift` and `ensemble_transforms=("rank", "squash")` on Lite**, for the same
reason as AU5.12 (no adapter override). The proxy rows above are the nearest evidence: neither is a
candidate.

## DR7.5: wide data on TabArena

No inference lever is left. Subsampling (§182) and column selection (§190) both leave the three
wide proxy tasks near the floor, and §190 shows the selected columns reach FinTFM and it still
cannot use them. Blocked on DR8.3's `max_features` 512 checkpoint, a pretraining run; nothing run.

## DR8.1: inference-lever sweep

**Verify is met; close.** Every §183 arm has its per-task sign test against the default in FINDINGS:
`ens8` 13/2/1 p 0.0074, `retrieval` 8/1/7 p 0.039, `ens8-swap` and `ens8-tf` 11/5/0 p 0.21, `ctx8k`
6/2/8 p 0.29 (§187), `ft200` 16/0 p 3e-5 on split 0 only (§192, a stated deviation). Both proxy
candidates were then scored on V4FinBench: `retrieval` 0.0126 mean AP, harmful (§194); `ens8`
+0.0058, not adopted (§196). The cause of the retrieval collapse stays an open diagnostic; the
per-class retrieval arm above (AU5.8) is the cheapest test of whether the class share is the
mechanism.

## DR8.3: wide data handled

No inference lever left (§190: selection no better than subsampling, worse on
Internet-Advertisements). The remaining arm is a checkpoint trained at `max_features` 512, a
pretraining run that goes through V4FinBench first; nothing run here.

## Housekeeping

- A second copy of `au512-ens8-svd16` was launched from the main checkout by another session at
  01:17 on 2026-11-03 (`runs/inf2611/svd16.sh`, same config, same output paths) and was gone by
  01:22, along with its script; its start line is the second `START` in `runs/inf2611/queue.log`.
  It truncated this session's log once, so `proxy_au512-ens8-svd16.log` may hold a gap; the CSV
  and JSON were written by this session's copy at 03:37 (the JSON's `config_sources` name this
  worktree).
- `vm_stat` free plus inactive was 18-19 GB at every start.
