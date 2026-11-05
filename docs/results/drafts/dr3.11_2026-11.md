# DR3.11 draft: fine-tuning per leaf of a partitioned context (2026-11-05)

Draft for FINDINGS.md; not yet merged. Verdict: **null** under §163's rule.

## Question and rule

Does fine-tuning (200 steps, lr 1e-5, batch 2, 512 queries; the §163 recipe) *per leaf* of
§169's depth-2 partitioned context beat §169's partition arm (one 2,000-row context per leaf, no
fine-tuning)? Rule, §163 as DR3.11 applies it: **helps** if the five-fold mean dAP exceeds
+0.0073 with at least 4 of 5 folds positive; **hurts** under the mirror; otherwise **null**.

## Methods

- Arm `FTLEAF` (Kaggle, T4): `method: partition`, `finetune_steps: 200`, `finetune_lr: 1e-5`.
  Folds 0-1: `runs/kaggle-fintfm-task-dr311-ftleaf/` (kernel `scripts/kaggle/task_dr311_ftleaf/`);
  folds 2-4: `runs/kaggle-fintfm-task-dr311-ftleaf-f234/`
  (`scripts/kaggle/task_dr311_ftleaf_f234/`). The two `FTLEAF.yaml` files are identical.
- Reference arm: §169's partition, `runs/dr31/partition`, all five folds present locally; no
  refit. Same checkpoint, same V4FinBench full panel, identical test rows (checked by the script).
- Folds 0-4 were linked into one directory and scored with the script that wrote
  `runs/dr311_ftleaf_cmp.txt` (paired bootstrap, 2,000 resamples, rows and whole companies, Holm
  across the arm's folds):

```
uv run python scripts/compare_v4_scores.py --cluster --control runs/dr31/partition ftleaf=<dir with folds 0-4>
uv run python scripts/dr311_auc_table.py --control runs/dr31/partition <dir with folds 0-4>
```

## Result (MEASURED, 2026-11-05)

Average precision first, ROC-AUC beside it.

| fold | partition AP | per-leaf FT AP | dAP | 95% CI, companies | Holm p, companies | partition AUC | FT AUC | dAUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.2241 | 0.2301 | +0.0060 | [+0.0001, +0.0117] | 0.184 | 0.9899 | 0.9914 | +0.0015 |
| 1 | 0.2262 | 0.2265 | +0.0003 | [-0.0064, +0.0067] | 1.000 | 0.9923 | 0.9924 | +0.0002 |
| 2 | 0.2411 | 0.2463 | +0.0052 | [-0.0045, +0.0154] | 0.810 | 0.9916 | 0.9924 | +0.0008 |
| 3 | 0.2370 | 0.2510 | +0.0140 | [+0.0079, +0.0206] | 0.002 | 0.9906 | 0.9927 | +0.0021 |
| 4 | 0.2524 | 0.2514 | -0.0010 | [-0.0079, +0.0063] | 1.000 | 0.9921 | 0.9920 | -0.0001 |
| **mean** | 0.2362 | 0.2411 | **+0.0049** | | | | | +0.0009 |

Folds 0 and 1 reproduce `runs/dr311_ftleaf_cmp.txt` exactly (+0.0060, +0.0003, same intervals).
Their Holm p-values are larger here (0.184 and 1.000 against 0.092 and 0.983) only because Holm
now runs across five folds instead of two. The row bootstrap gives near-identical intervals.

4 of 5 folds positive; mean dAP +0.0049 (ROC-AUC +0.0009, 4 of 5 positive); one fold (3)
survives Holm (p = 0.002). Fold 4 is slightly negative.

## Verdict

**Null.** The fold-count condition is met (4 of 5) but the mean, +0.0049, is below the +0.0073
bar. The gain is concentrated in fold 3 (+0.0140); the other four average +0.0026. Per-leaf
fine-tuning is not shown to help beyond the partition, and it costs a fine-tuning run per fold.
