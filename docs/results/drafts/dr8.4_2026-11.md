# DR8.4: a tuned LightGBM control on the credit suite (draft, 2026-11-04)

**Result.** On SBA 7(a), tuning LightGBM alone gains more than §189's FinTFM-logit feature did:
the tuned control beats the untuned-booster hybrid on 3 of 3 splits (+0.0049 mean AP), so §189's
+0.011 is not yet evidence that FinTFM adds anything a tuned booster lacks. On ULB fraud the
init-score hybrid still beats the tuned control on 3 of 3 splits (+0.0235 mean AP), most of it on
split 0. The matched comparison, a tuned booster with and without the FinTFM logit, is not run.

All numbers MEASURED, 2026-11-04, `runs/agents1104/DR8.4/*.json` (tuned and the untuned rerun) and
§189 (`runs/credit_suite/`) for the hybrid, FinTFM and logistic-regression columns. Same splits as
§189 (`credit_suite.suite_splits`): SBA out-of-time, 23,763 / 83,052 / 138,004 training rows; ULB
stratified, about 190,000 training rows, 164 test positives in split 0.

## Method

`src/fintfm/evaluation/lightgbm_tuned.py` (runs in a process that does not import torch: LightGBM
and torch in one process segfault on macOS), driven by `scripts/credit_lightgbm_tuned.py`.

- Hyperparameters chosen on an inner split of each training split only; the test split is never
  read. SBA: the latest 25% of training rows by approval date are the inner validation side; ULB:
  a stratified random 25%.
- **SBA's inner split has no embargo** (`--inner-embargo 0`). SBA's embargo is 1,827 days (the
  five-year outcome horizon) and split 0's training window spans 203 days, so an embargoed inner
  split has no fit rows. The test split keeps the full embargo. Consequence: inner fit rows'
  outcome windows overlap the inner validation period, which can bias the selection, not the
  test score. Recorded per split as `inner_embargo` in each JSON.
- Seeded random search of up to 30 configs, config 0 being the untuned control's settings; early
  stopping (patience 50 trees) on inner-validation AP; 600 s search budget per split; refit on all
  training rows at 1.1 times the best iteration count; `n_jobs=2`. Same median-impute and
  standardise as §189.
- Search coverage: SBA 30 of 30 configs on every split; ULB 22, 15 and 11 (budget-bound).

## Average precision per split

| dataset | split | tuned LightGBM | LightGBM (§189, untuned) | + logit as feature (§189) | + logit as init score (§189) | logistic regression (§189) |
| --- | --- | --- | --- | --- | --- | --- |
| SBA 7(a) | 0 | 0.4503 | 0.4392 | 0.4497 | 0.4366 | 0.1342 |
| | 1 | 0.4571 | 0.4320 | 0.4473 | 0.4282 | 0.1455 |
| | 2 | 0.2988 | 0.2872 | 0.2944 | 0.2967 | 0.1087 |
| | **mean** | **0.4021** | 0.3861 | 0.3971 | 0.3872 | 0.1295 |
| ULB fraud | 0 | 0.7988 | 0.8539 | 0.4135 | 0.8510 | 0.7603 |
| | 1 | 0.8419 | 0.4845 | 0.6620 | 0.8517 | 0.7437 |
| | 2 | 0.8490 | 0.6871 | 0.5957 | 0.8576 | 0.7427 |
| | **mean** | 0.8299 | 0.6751 | 0.5571 | **0.8534** | 0.7489 |

Tuned ROC-AUC: SBA 0.9530 / 0.9539 / 0.9288; ULB 0.9760 / 0.9804 / 0.9855.

Paired per split, AP:

- SBA, tuned minus untuned: +0.0111, +0.0251, +0.0116 (mean +0.0159, 3 of 3).
- SBA, tuned minus the feature hybrid: +0.0006, +0.0098, +0.0044 (mean +0.0049, 3 of 3).
- ULB, init-score hybrid minus tuned: +0.0522, +0.0098, +0.0086 (mean +0.0235, 3 of 3).
- ULB, tuned minus logistic regression: +0.0385, +0.0982, +0.1063 (mean +0.0810, 3 of 3).

## Reproduction check

The worker re-runs the untuned control (1,000 trees, lr 0.05). SBA reproduces §189 exactly on all
three splits (0.4392, 0.4320, 0.2872). ULB reproduces splits 0 and 2 (0.8539, 0.6871) but not split
1: 0.5369 here against §189's 0.4845. A likely cause is LightGBM's thread-count nondeterminism
(`n_jobs=2` here; §189's setting not checked); not confirmed. It shows how unstable untuned
LightGBM is on ULB, which is §189's own reading.

## What this settles, and what it does not

- **SBA: §189's feature gain is smaller than a tuning gain.** A reader of §189 would credit FinTFM
  with +0.011; tuning alone gives +0.0159 and beats the hybrid on every split. Whether the logit
  still adds anything on top of a tuned booster is the open question.
- **ULB: the init-score hybrid survives a tuned control**, by +0.0235 mean AP on 3 of 3 splits,
  though splits 1 and 2 are within 0.01. ULB's inner validation holds about 80 positives, so its
  search is noisy, and it was budget-bound at 11 to 22 configs.
- **Not run:** the tuned booster plus the FinTFM logit (feature and init score). It needs per-split
  out-of-fold FinTFM logits, which §189 did not save; regenerating them took up to 1,791 s per split
  on SBA (§189). The V4FinBench half of DR8.4 stays open.
- Three splits per dataset; no interval. A system result, as §189 says, not a model result.

## 2026-11-07: tooling for the open halves, smoke-tested, not run at scale

No new result. Both open halves now have code, tests and a SMOKE-TEST; the full runs are queued for
the coordinator (commands below). Every number in this section is SMOKE-TEST or ESTIMATED.

**(a) A tuned booster plus the FinTFM logit, credit suite.**

- `credit_suite --save-oof DIR` (needs `--hybrid`) writes `<dataset>_split<k>.npz` per split: the
  out-of-fold logits of the training rows, the test logits of the one FinTFM fitted on all training
  rows, the row indices, the inner-fold index and the labels. Off by default; with it off the
  code path is untouched, and a test checks that the CSV rows (every column but `seconds`) equal
  those of a run with it on. The folds are the ones §189's hybrid already used: stratified, and
  grouped by borrower where a dataset has an id (neither SBA nor ULB has one), and every inner
  context is checked against the test rows before any fit. SBA's embargo is therefore met by
  construction: contexts hold training rows only, as in §189.
- `credit_suite --max-rows N` is a seeded development cap (smoke tests only).
- `scripts/credit_lightgbm_tuned.py --oof-dir DIR` runs the same seeded search of up to 30 configs
  as the tuned control (same inner split, same early stopping, same budget) twice per split:
  `feature` (logit as one more column) and `init` (logit as the starting score). Inner validation
  scores include the starting score. The worker's untuned control starts from the same logits, so it
  reproduces §189's hybrid rows: SMOKE-TEST, SBA capped at 6,000 rows, split 0, untuned feature AP
  0.1082 and init AP 0.1173 in the tuned JSONs, equal to the hybrid CSV rows' 0.1082 and 0.1173.
- SMOKE-TEST numbers (6,000 rows, 4 configs, 60 s budget, 2 inner folds) are plumbing only and
  say nothing about quality: tuned feature AP 0.1457 / 0.1477 / 0.2562 and tuned init AP 0.2258 /
  0.1897 / 0.2245 on splits 0 to 2, against 36 test positives in split 0. Peak RAM
  (`/usr/bin/time -l`, includes children): 8.1 GB for the capped credit run, which loads the full
  SBA file before capping, and 1.8 GB for the tuned stage.

**(b) V4FinBench.** `hybrid_catboost --booster lightgbm_tuned` adds three arms to the AU4.4 layout
(`lightgbm_tuned`, `lightgbm_tuned_fintfm_baseline`, `lightgbm_tuned_fintfm_feature`), each a
separate seeded search in the torch-free worker, on an inner split of the training rows by whole
company (new in `inner_split`: undated rows with groups split by group; a row split would put
nearly every multi-year company on both sides). Validation and test rows are never read by the
search. Output files pair with `scripts/compare_v4_scores.py`. Default `--booster catboost`
leaves AU4.4's outputs as before. SMOKE-TEST (fold 0, panel capped at 20,000 rows, 4,000 training
rows, 3 configs, 2 inner folds): all four arms scored, 210 s wall on a loaded machine, peak RAM
14.2 GB (full 1,000,087-row panel is loaded before the cap). Capped-run APs
(0.2288 / 0.2184 / 0.2239 tuned alone / init / feature; FinTFM 0.3131) are SMOKE-TEST.

**Commands for the coordinator, serial, run from the main checkout after merging**
(`cd .`; prefix each with `PYTHONUNBUFFERED=1 nice -n 19 uv run --frozen
--extra bench`, redirect to a log, check `$?`, one at a time, check `uptime` first). Times are
ESTIMATED from §189 and AU4.4's measured runtimes (SBA out-of-fold up to 1,791 s per split; AU4.4
folds took 40 to 68 min at 1,000,087 query rows including three CatBoost fits) and the 600 s search
budget; not measured at scale. Peak RAM at scale is not measured: expect 10 to 15 GB while the
panel loads.

1. SBA logits: `python -m fintfm.evaluation.credit_suite --model runs/v4-cellattn-labels.pt
   --datasets sba --hybrid --save-oof runs/dr84-credit-oof --out runs/dr84-credit-oof/sba.csv`.
   ESTIMATED 0.5 to 0.8 h.
2. ULB logits: same with `--datasets ulb_fraud` and `--out runs/dr84-credit-oof/ulb_fraud.csv`.
   ESTIMATED 1.2 to 1.6 h.
3. Tuned hybrids: `python scripts/credit_lightgbm_tuned.py --datasets sba ulb_fraud --oof-dir
   runs/dr84-credit-oof --out runs/dr84-credit-tuned --inner-embargo 0 --total-budget 18000`.
   ESTIMATED 2.5 to 3.5 h (12 searches of up to 720 s plus refits).
4. V4FinBench, capped training rows (validation and test untouched, so arms pair with AU4.4's):
   `python -m fintfm.experiments.hybrid_catboost --model runs/v4-cellattn-labels.pt --booster
   lightgbm_tuned --max-train-rows 200000 --inner-folds 5 --tuned-configs 30 --tuned-budget 300
   --out runs/dr84-v4`. ESTIMATED 5 to 6.5 h for five folds (about 25 to 40 min of FinTFM plus about
   35 min of three searches per fold); add `--folds 0,1,2` for about 3 to 4 h. Then
   `scripts/compare_v4_scores.py --control runs/dr84-v4/arms/lightgbm_tuned
   baseline=runs/dr84-v4/arms/lightgbm_tuned_fintfm_baseline
   feature=runs/dr84-v4/arms/lightgbm_tuned_fintfm_feature`. Read AP first.

Caveats to carry into the result: the capped V4 run trains on at most 200,000 rows against AU4.4's
about 600,000, so it is a different control from AU4.4's CatBoost; the within-run tuned-versus-hybrid
pairing is the comparison. SBA's inner tuning split has no embargo, as above.
