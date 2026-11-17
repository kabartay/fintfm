# Task 39.15: observational against interventional queries on a synthetic SCM, DRAFT 2026-11-17

**Status: DRAFT for the owner. Not in FINDINGS.md, not in openspec.** Task 39.15 of
`cell-attention-and-task-inference`. Source of every number below: `runs/causal_interventional/causal_interventional.json`
(written 2026-11-17 02:21) and the printed tables in `runs/local_queue/h-causal.log`.

## Result

Both checkpoints answer closer to the association than to a clear interventional signal; the harness cannot separate them from
a plain associational predictor. **MEASURED**, verification against a closed-form synthetic SCM, not validation against observed data.

- On interventional queries fin00 loses AUC against its own observational AUC by 0.028, 0.041, 0.056 and 0.090 at g = 0, 0.5, 1, 2. The closed-form
  `oracle_do` loses 0.011, 0.038, 0.056 and 0.079 (its AUC on observational queries minus on interventional queries). The fin00 loss therefore follows the
  loss any predictor suffers when X1 is decoupled from X2, and does not show a model that handles the intervention differently.
- "Leans to interventional" for fin00 at g = 0.5 and 1 is a 0.021 and 0.015 gap in mean absolute error (probability units) in favour of the interventional truth. At g = 2 it flips to
  observational (0.089 against 0.058). At g = 0 both truths are identical, so the label is a coin flip by construction. Logistic regression and gradient boosting lean observational at every g > 0, as expected.
- fin10 has mean absolute error 0.155 to 0.196 against either truth at every g (the untrained control is 0.323 to 0.357, logistic regression 0.014 to 0.110). Its "interventional" lean at g > 0 is the closer of two
  distant answers, not an accurate one.

Verify (39.15): "the two conditional distributions are compared directly on held-out interventional data". **Met as written**: both are compared against held-out interventional
queries with closed-form truths, for two checkpoints and four confounding levels. **The motivating question (structure or only association) is not answered**: the effect sizes
are within what miscalibration produces, and there is no spread across seeds to test them (see caveats). No pre-registered rule exists for this task (searched `docs/results/FINDINGS.md` and `docs/results/drafts/preregs_2026-11.md`), so none was applied.

## Run check

| item | value | source |
| --- | --- | --- |
| queue line | `h-causal`, 3 seeds, g in {0, 0.5, 1, 2} | `runs/local_queue/queue.txt` line 118 |
| exit, time | exit 0, 33 min (1,981.89 s), peak RSS 1.0 GB | `runs/local_queue/queue.log` lines 248-249; `time` footer of the log |
| coverage | 12/12 confounding-level-seeds scored, 0 skipped | log; JSON `coverage` |
| NaN, warnings, tracebacks | none found (grep of the log for nan, warn, skip, error) | log |
| commit | not recorded in the JSON or the log | n/a (see caveats) |
| checkpoints | `runs/dl/v4-cellattn-fin00.pt`, `runs/dl/v4-cellattn-fin10.pt` (both 3,584,819 bytes, dated 2026-09-14) | `ls` |

The command differs from the one in the openspec note only in threads (`OMP_NUM_THREADS=4`, `nice -n 10`, not 2 and 15) and in the output going to the queue log; it does not change the numbers.

## Setup (from the module docstring and the JSON `config`)

SCM `Z->X1, Z->X2, X1->Y`, hidden Z, `a = b = 0.8`, `w = 1.5`, `c = -0.5`; hidden `Z->Y` strength g swept over 0, 0.5, 1, 2 (g = 0 is the task's SCM). Per seed: n = 2,000 rows,
1,000 context rows (observational) and 1,000 query rows; `max_context` 1,000. Interventional queries: X1 replaced by a permutation of itself (`do(X1 ~ own marginal)`),
y redrawn from the structural equation. Seeds 0, 1, 2; every figure is the mean of 3 seeds. Arms: fin00, fin10, seeded untrained control, logistic regression, histogram gradient boosting, two closed-form oracles.
Both fin00 and fin10 are FinTFM trained on synthetic prior data only (fin00: pure SCM prior, fin10: pure financial prior, 6,000-step T4 runs, FINDINGS §78); no FinTFM-R arm. Confirm the family before filing.

## Tables (mean of 3 seeds, 1,000 query rows per seed; MAE in probability units)

AUC on observational queries (`auc_obs`) and on interventional queries (`auc_do`):

| g | fin00 obs / do | fin10 obs / do | log. reg. obs / do | grad. boost. obs / do | oracle_do obs / do | untrained obs / do |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.814 / 0.786 | 0.766 / 0.697 | 0.822 / 0.810 | 0.776 / 0.749 | 0.822 / 0.811 | 0.425 / 0.329 |
| 0.5 | 0.852 / 0.811 | 0.791 / 0.713 | 0.858 / 0.820 | 0.819 / 0.766 | 0.859 / 0.821 | 0.433 / 0.362 |
| 1 | 0.875 / 0.819 | 0.810 / 0.730 | 0.878 / 0.812 | 0.843 / 0.770 | 0.879 / 0.823 | 0.443 / 0.411 |
| 2 | 0.912 / 0.822 | 0.838 / 0.748 | 0.913 / 0.805 | 0.887 / 0.779 | 0.913 / 0.834 | 0.455 / 0.471 |

Mean absolute error of the predicted probability on interventional queries against the interventional truth (`MAEdo`) and the associational truth (`MAEobs`), and the lean:

| g | arm | MAEdo | MAEobs | leans to |
| --- | --- | --- | --- | --- |
| 0 | fin00 | 0.080 | 0.080 | tie (truths identical) |
| 0 | fin10 | 0.163 | 0.163 | tie |
| 0.5 | fin00 | 0.050 | 0.071 | interventional |
| 0.5 | fin10 | 0.155 | 0.176 | interventional |
| 1 | fin00 | 0.047 | 0.062 | interventional |
| 1 | fin10 | 0.155 | 0.184 | interventional |
| 2 | fin00 | 0.089 | 0.058 | observational |
| 2 | fin10 | 0.165 | 0.196 | interventional |
| 0.5 / 1 / 2 | logistic regression | 0.028 / 0.059 / 0.110 | 0.015 / 0.014 / 0.020 | observational |
| 0.5 / 1 / 2 | gradient boosting | 0.154 / 0.162 / 0.174 | 0.144 / 0.140 / 0.134 | observational |

Prediction shift from observational to interventional queries (mean predicted P(Y=1)), against the closed-form shifts (trueDo / trueObs): fin00 -0.013, -0.013, -0.012, -0.011 and fin10 -0.008, -0.008, -0.007, -0.004 at g = 0, 0.5, 1, 2;
trueDo 0.000, -0.009, -0.014, -0.014; trueObs 0.000, -0.004, -0.007, -0.009. The true shifts are at most 0.014, and fin00 predicts -0.013 at g = 0 where the truth is 0.000, so the shift does not discriminate.

## Caveats

- 3 seeds, one SCM parameter set, 1,000 query rows per seed. The JSON stores means only, so no spread, interval or paired test is available; the 0.015 to 0.021 MAE gaps behind the "interventional" labels are untested for significance.
- The lean is a two-way comparison of MAE, and at g = 0 the two truths coincide. It does not say the model reasons causally: every arm sees only observational context rows.
- fin00 at g = 0.5 and 1 is not better than logistic regression on `MAEdo` at g = 0.5 (0.050 against 0.028) and is better at g = 1 (0.047 against 0.059); the model is not a better interventional predictor than the linear floor.
- Both checkpoints are the 2026-09-14 §78 runs (cell attention, 6,000 steps); nothing here speaks for other checkpoints, other SCMs or the retrieval-augmented family.
- The run commit is not recorded; the working tree at the time cannot be reconstructed from the record. Record it in the progress note if known.
- Synthetic SCM, closed-form truth: verification, not validation.

## Proposed FINDINGS entry (next free section number at filing; §274 is the last)

> ## §275: on a linear-Gaussian SCM fin00 and fin10 answer interventional queries no differently from an associational predictor: fin00's AUC loss follows the closed-form oracle's, its lean flips with confounding (task 39.15)
>
> **MEASURED**, `fintfm.experiments.causal_interventional`, checkpoints `runs/dl/v4-cellattn-fin00.pt` and `runs/dl/v4-cellattn-fin10.pt` (FinTFM, synthetic-only), SCM `Z->X1, Z->X2, X1->Y` with hidden Z (a = b = 0.8, w = 1.5, c = -0.5), hidden `Z->Y` strength g in {0, 0.5, 1, 2}, 3 seeds (0, 1, 2), 2,000 rows per task split 1,000 context / 1,000 query, 12/12 scored, 0 skipped. Queue job `h-causal`, exit 0, 33 min, CPU. Verification against a closed-form synthetic SCM, not validation against observed data.
> fin00's AUC on interventional queries is below its observational AUC by 0.028 / 0.041 / 0.056 / 0.090 at g = 0 / 0.5 / 1 / 2, against 0.011 / 0.038 / 0.056 / 0.079 for the closed-form interventional oracle. Its mean absolute error is closer to the interventional truth at g = 0.5 and 1 (0.050 against 0.071; 0.047 against 0.062) and to the associational truth at g = 2 (0.058 against 0.089). fin10's error is 0.155 to 0.196 against either truth. Logistic regression and gradient boosting lean associational at every g > 0. The true prediction shifts are at most 0.014 and fin00 shifts -0.013 where the truth is 0.000, so the shift is not diagnostic.
> **Reading:** the Verify (direct comparison on held-out interventional data) is met; the question of structure against association is not settled. Three seeds and means only: no spread, no significance test. Not claimed: causal reasoning, or any property of other checkpoints or SCMs.

## Proposed openspec progress note (task 39.15, leave `[ ]` until the owner decides)

> **Progress (2026-11-17, FINDINGS §275 once filed, MEASURED, synthetic SCM, CPU): the measured run was made (`h-causal`, 33 min, exit 0, 12/12 scored, 0 skipped): fin00, fin10, an untrained control, logistic regression, gradient boosting and two oracles on observational and held-out interventional queries at g in {0, 0.5, 1, 2}, 3 seeds. fin00's AUC loss on interventional queries (0.028 to 0.090) follows the closed-form oracle's (0.011 to 0.079); its lean to the interventional truth holds at g = 0.5 and 1 and flips at g = 2; fin10 is 0.155 to 0.196 from either truth. The Verify's direct comparison is made; whether the prior encodes structure or only association is not settled by 3 seeds and means. Remains: spread across more seeds, then decide whether to close.**
