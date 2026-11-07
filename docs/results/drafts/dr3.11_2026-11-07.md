# DR3.11 draft: pre-registration of the remaining partition arms (2026-11-07)

Pre-registered before any arm below runs. Nothing here is measured except the inputs, which are
quoted from FINDINGS (§169, §169 addendum, §196, §208) and the Kaggle logs under `runs/`.
Follows `dr3.11_2026-11.md` (§208, fine-tuning per leaf, null).

## Rule

§163 as DR3.11 applies it, against §169's partition (`runs/dr31/partition`, five-fold mean AP
0.2362): **helps** if the five-fold mean dAP exceeds +0.0073 with at least 4 of 5 folds
positive; **hurts** under the mirror; otherwise **null**. Paired company bootstrap, Holm over
five folds, as §208. Average precision first, ROC-AUC beside it. V4FinBench decides; TabArena-Lite
only after it.

## What the existing evidence already decides

| open item (task text) | verdict | why |
| --- | --- | --- |
| depth 1 and 3 **with fine-tuning** | **dropped** | Without fine-tuning depth does not matter: 0.2347, 0.2361, 0.2357 at depth 1, 2, 3 (§169 addendum, MEASURED), a spread of 0.0014, a fifth of the bar. Fine-tuning per leaf at depth 2 is null (§208, +0.0049). A depth change that moves nothing, added to a lever that is null, has no route to +0.0073. It would also cost about 6.4 GPU-hours per arm (below). |
| fine-tuning per leaf (depth 2) | decided, null | §208. |
| depth 1 and 3, zero-shot | decided, null | §169 addendum. |

Kept: 8 members, bagging over 4 and 8 trees. TabArena-Lite is not a new arm: the partition
kernel `scripts/kaggle/task_lite_partition` (Fix 4, §204) is prepared and unlaunched, and is the
"must not lose on numeric-only datasets" gate for the partition itself.

## Arms, in priority order, with predictions

All zero-shot, published checkpoint, V4FinBench full panel, five folds, `n_ensemble`/bag settings
as the YAML. Predictions are written before running and are not probabilities of passing.

1. **`dr311-bag4`** (4 depth-2 trees on random 60% bags, leaf scores averaged). **Prediction: null,
   mean dAP about +0.002, range -0.003 to +0.006, fewer than 4 folds above the bar.** Reason:
   depth is irrelevant (§169 addendum) because one split on the insolvency gate does the work
   (§168), so bagged trees will mostly share that split and differ little. The only lever left
   is averaging four context draws per leaf, which reduces draw noise (§162's one 500-row draw
   collapsed a fold to 0.0684) but the partition's leaf contexts already hold about ten times the
   positives of a uniform one (§169). Cheapest and the most direct test of tree diversity, so first.
2. **`dr311-ens8`** (8 members per leaf, one tree). **Prediction: null, mean dAP about +0.002,
   folds 0 and 1 at or below zero.** Reason: §196's uniform-context `ens8` gained +0.0058 mean
   AP, from folds 2-4 (+0.0099 to +0.0187) with folds 0-1 negative (-0.0021, -0.0083). Folds 2-4
   were the folds where the uniform context was positive-starved, which the partition already
   fixes (+0.0352 to +0.0625 on those folds). Little variance is left for members to remove.
3. **`dr311-bag8`** (8 trees). **Prediction: within 0.001 of bag4's mean dAP.** Conditional: run
   only if bag4 or ens8 reaches a five-fold mean dAP of at least +0.0036 (half the bar). Below
   that, doubling the trees cannot be expected to cross the bar, and it costs 21 GPU-hours.

Gate for the TabArena-Lite partition run (existing kernel, independent of the order above):
prediction, partition loses on datasets under about 2,000 training rows, where a 5% leaf floor
splits a table already smaller than one context, and ties above that. Not a V4FinBench arm.

## Futility rule (fixed now, folds 2 and 3 first)

Each arm's first kernel scores folds 2 and 3 (`f23`). If the mean dAP on those two folds is
below 0.0, stop the arm: a five-fold mean above +0.0073 would then need the other three folds to
average above +0.0122, which exceeds every fold effect seen for these levers except fold 3 of
§208 (+0.0140). This is a stopping bound, not a null verdict; an arm stopped this way is
reported as two folds scored. Otherwise run the remaining folds and apply the §163 rule.

## Cost, ESTIMATED from `task_dr311_ftleaf` (Kaggle T4, one GPU)

MEASURED per fold of the per-leaf fine-tuning arm: 4,757 and 4,769 s (folds 0-1, kernel
start-up 1,300 s) and 4,461, 4,443, 4,450 s (folds 2-4, start-up 438 s); five folds 22,880 s,
6.4 h. Whole kernels 10,826 s and 13,791 s (6.8 h with start-up).

MEASURED, `task_dr310_ft_ctl` (one fit with 200 fine-tuning steps, uniform context): 2,400 s
per fold.

ESTIMATED zero-shot inference per fold. A partition fits the global model plus about four
leaves, so the per-leaf arm made five 200-step fits against the control's one. The difference,
about 2,050 s for four extra fits, gives about 510 s per fit, and the control's 2,400 s less one
fit leaves about **1,900 s of inference per fold**. A partition without fine-tuning scores each
query once, so it should cost the same. This assumes fit cost is linear in leaves and equal
across leaves; no zero-shot partition fold has been timed on a T4, and the first kernel's log
will replace this figure.

| arm | per fold | five folds | kernels (12 h limit each) |
| --- | --- | --- | --- |
| bag4 (4 x inference) | about 7,600 s, 2.1 h | about 10.6 h | `f23` 4.2 h, `f014` 6.3 h |
| ens8 (8 x inference) | about 15,200 s, 4.2 h | about 21 h | `f23`, `f01` 8.4 h each, `f4` 4.2 h |
| bag8 (8 x inference) | about 15,200 s, 4.2 h | about 21 h | as ens8 |
| depth 1 and 3 with fine-tuning (dropped) | 1.2 h | 6.4 h each | not built |

Add about 0.1 to 0.4 h of start-up per kernel. Priority 1 and 2 together: about 32 GPU-hours at
full length; with the futility rule an arm that fails on folds 2-3 costs 4.2 to 8.4 h.
Kaggle's session limit is 12 h, so the five-fold arms are split as above.

## Prepared, not run

- Configs: `configs/arms/dr311-bag4.yaml`, `dr311-ens8.yaml`, `dr311-bag8.yaml` (existing, checked:
  one changed key each, depth 2, 5% leaf floor, `bag_frac` 0.6, no fine-tuning).
  `dr311-ft200.yaml` annotated as measured null.
- Kernels: `scripts/kaggle/task_dr311_<arm>_<group>/` (8 directories) written by
  `scripts/kaggle/make_dr311_kernels.py`, modelled on `task_dr311_ftleaf`. They keep
  `COMMIT = "UNPINNED"` and refuse to run until regenerated with `--commit <pushed sha>`, and
  refuse a CPU run, the guard `task_au51_*` and `ts_4825`/`ts_dr83` carry. The `ts_4815` and
  `ts_4827` kernels as committed are pinned (`3f16cb1`, `304ea7f`) and carry only the GPU guard,
  so the UNPINNED guard follows the au51/dr83 pattern. The kernels embed the arm YAML plus
  `query_chunk: 512` and `feature_chunk: 8` (identity-preserving, as in `task_dr311_ftleaf`).
- Code paths: `PartitionedTFMClassifier` already accepted depth 1-3 and any number of bags and
  members, but only depth 2, 4 (refused) and 3 bags were tested. New tests in
  `tests/test_large_n.py`: depth 1, 2, 3 (leaf bound, routing), 4 and 8 bags (one tree per bag,
  distinct rows, average equals the mean of the bags), 8 members reach every leaf context.
  `tests/test_dr311_arms.py`: arm settings, config wraps the classifier, kernels equal the
  generator output and cover five folds, embedded config equals the file plus the chunks, the
  unpinned guard.

## Command for the coordinator (not run)

After pushing and regenerating with `--commit <sha>`: push `task_dr311_bag4_f23` first, read its
log for the zero-shot fold time (replaces the estimate), apply the futility rule, then
`task_dr311_bag4_f014`; then ens8 the same way; bag8 only on the condition above. Score each arm
with `scripts/compare_v4_scores.py --cluster --control runs/dr31/partition <arm>=<dir with folds
0-4>` as §208.
