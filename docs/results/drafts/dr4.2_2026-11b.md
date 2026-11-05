# DR4.2: the Spearman Verify, computed on n = 6 (2026-11-05)

**Result (MEASURED): proxy Elo against TabArena-Lite Elo, Spearman rho = +0.60 (n = 6 distinct checkpoints, exact two-sided p = 0.24, bootstrap-over-checkpoints 95% interval [-0.64, +1.00]). Threshold 0.8: NOT met, and not decidable either way at n = 6, since the interval spans it.** Mean per-task average precision against Lite Elo: rho = +0.03; mean ROC-AUC: rho = -0.31. The proxy therefore stays unvalidated and must not decide anything (DR4.2). This replaces `dr4.2_2026-11.md`, which found only 2 checkpoints with both columns.

## What was run

Proxy runs (MEASURED, CPU, 2 threads, nice 19, one at a time, memory gate of at least 14 GB before each): `python -m fintfm.evaluation.proxy_suite run --model runs/v4-<arm>.pt --name <arm> --tasks 3,15,43,49,3918,7592,9946,9976,9977,9978,10093,14952,14954,125920,146819,146820,167120,167125 --no-baselines --n-jobs 2`, the same 18 tasks and 3 splits (repeat 0, folds 0-2) as `runs/dr73/proxy_binary.csv`. Each wrote 48 scored rows of 54 (16 tasks scorable; `madelon`, 500 features, and one more task are skipped by the 136-feature cap, as for the existing runs). All four loaded and ran under today's code; no load failure. Wall time per arm, MEASURED from `queue.log` under a heavily loaded machine: priorctl 86 min, treeclean 75 min, treeprior 136 min, priorwide 122 min (much longer than the 0.05-0.2 h the earlier draft estimated for a quiet machine; `adult` and `nomao` alone took 400-700 s per split). Output: `runs/agents1104/DR4.2/proxy_<arm>.{csv,json,log}`.

Not run for time, not for failure: `priorctl-s1`, `treeclean-s1`, `widerprior` (about 6 h more at this load). They have Lite results and weights, so n = 9 is one queue away: `runs/agents1104/DR4.2/queue.sh`-style command, arms in `lite_elo.csv`.

## Lite Elo (MEASURED, fork's own `TabArenaContext.compare`, fork untouched, 27 binary datasets, split 0, `fillna=None`)

Script: `runs/agents1104/DR4.2/lite_elo.py` (run with the fork's venv, `-P`, output only under `runs/agents1104/DR4.2/lite_elo.csv`). Each arm was scored twice: on its own stored 95-method results frame (as originally run) and on one common pool (the 95 non-FinTFM methods of `fintfm_lite_nan_1019`).

**Reproduction check passed:** published 755.9 (+126.5/-179.8), 762.8 zero fill, `e1-v480-s0` 841.8 (+139.4/-213.2), identical to TABARENA.md's 756, 763 and 842, 94th of 95 (position 95 of 96 with the extra row).

| arm | Lite Elo, stored pool | Lite Elo, common pool | proxy Elo (RF = 1000, 16 tasks, 48 splits) [95% interval] | proxy mean AP | proxy mean ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| published `v4-cellattn-labels` | 755.9 (NaN passed through) | 755.9 | 574 [318, 742] | 0.6805 | 0.8203 |
| `e1-v480-s0` | 841.8 | 841.8 | 677 [431, 812] | 0.6827 | 0.8197 |
| priorctl | 778.2 | 775.4 | 664 [390, 826] | 0.7001 | 0.8351 |
| treeclean | 842.4 | 840.8 | 721 [461, 885] | 0.6926 | 0.8335 |
| treeprior | 785.7 | 783.0 | 738 [497, 886] | 0.6998 | 0.8375 |
| priorwide | 792.4 | 789.7 | 673 [423, 828] | 0.6817 | 0.8283 |

The stored-pool and common-pool columns differ by 1.6 to 4.5 Elo and give the same ranking. Proxy Elo comes from one joint Bradley-Terry fit of the five baselines (from `proxy_binary.csv`) plus the six checkpoints, `elo_table(..., anchor="RF", n_boot=1000, seed=0)`; the published checkpoint's 574 here differs from §181's 561 because the other five checkpoints are in the fit. Proxy mean AP and AUC are the mean over 16 tasks of the per-task mean over 3 splits. All other Lite Elo values on record (not used, no proxy run): priorctl-s1 736.9, treeclean-s1 825.7, widerprior 761.2.

## Spearman (n = 6; MEASURED; script `spearman.py`, inputs `spearman_inputs.csv`)

| proxy score | against Lite Elo (common pool) | against Lite Elo (stored pool) |
| --- | --- | --- |
| proxy Elo | rho = +0.600, exact two-sided p = 0.242 (one-sided 0.121), bootstrap 95% [-0.640, +1.000] | +0.657, p = 0.175, [-0.500, +1.000] |
| mean AP | +0.029, p = 1.00, [-1.00, +1.00] | +0.086, p = 0.92 |
| mean ROC-AUC | -0.314, p = 0.56, [-1.00, +1.00] | -0.143, p = 0.80 |

p is the exact permutation test over all 720 pairings; the interval is a 5,000-draw bootstrap over the 6 checkpoints (valid draws only, those with at least 2 distinct values in each column). The bootstrap resamples checkpoints, not tasks or datasets, and at n = 6 it is wide by construction. The point estimate 0.60 is below the bar and the lower end of the interval is far below it, so the data cannot show 0.8.

## Reading

- **The Verify, as written (Spearman at or above 0.8 across 6 or more checkpoints), is not met.** At the proxy's headline score (Elo) it is 0.60; on AP, which this project reads first, it is about zero.
- **Where they agree and disagree.** Both put the published checkpoint last and `treeclean` and `e1-v480-s0` above it. They disagree on the middle: Lite common-pool order is e1-v480-s0 841 ~ treeclean 841 > priorwide 790 > treeprior 783 > priorctl 775 > published 756; proxy Elo order is treeprior 738 > treeclean 721 > e1-v480-s0 677 > priorwide 673 > priorctl 664 > published 574; mean AP order is priorctl 0.7001 > treeprior 0.6998 > treeclean 0.6926 > e1-v480-s0 0.6827 > priorwide 0.6817 > published 0.6805.
- **Both columns are noisy relative to the spread.** Lite intervals are +/-115 to 220 Elo wide against a 85 Elo spread between the 6 arms (755.9 to 841.8 stored); proxy intervals are 300-420 Elo wide against a 164 Elo spread (574 to 738). Most pairs are within each other's intervals, so the rank correlation is mostly ranking noise, as the earlier draft warned. A larger n adds arms of the same near-tie kind; it does not widen the spread.
- **Confounders (not removed).** (a) NaN handling: the four older arms were Lite-run with NaN zero-filled, the published and `e1-v480-s0` runs with NaN passed through; the published pair differs by 6.9 Elo (755.9 and 762.8) between the two encodings, small against the spread but not zero. Using the zero-fill value for the published checkpoint (762.8) leaves its rank (lowest of the six) and so every rho unchanged. (b) Checkpoint identity: the Lite runs' logs are not on disk for the older arms; the match of `runs/v4-<arm>.pt` to each Lite directory rests on name and timestamps (checkpoint written before the Lite run started: priorctl 14:06 then 14:13, treeprior 14:06 then 17:43, treeclean 22:34 then 23:04, priorwide 14:06 then 15:57, all 2026-09-22), not on a recorded hash. (c) The proxy and the Lite panel use different datasets (proxy OpenML tasks outside TabArena; Lite 27 TabArena binary), so the correlation also carries the dataset-mix difference, which is what it is meant to measure.
- **What this does not say.** It does not show the proxy is useless: with this spread and n = 6, a true rho of 0.8 would often produce an estimate near 0.6. It shows the Verify cannot be passed with the data in hand, and the one-sided p of 0.12 gives no evidence of any positive rank agreement.

## Left open

- Run the remaining three (priorctl-s1, treeclean-s1, widerprior; about 6 h at current load) for n = 9, and report the interval again. Further arms of the same near-tie kind will not tighten it much; checkpoints that are far apart on both columns (a clearly weak model and a clearly strong one) would.
- DR4.2 stays open: proxy remains unusable for decisions (§181's +168 Elo for E1 stays unvalidated).
