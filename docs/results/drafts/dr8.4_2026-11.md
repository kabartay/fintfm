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
