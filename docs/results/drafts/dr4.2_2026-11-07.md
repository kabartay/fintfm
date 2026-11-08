# DR4.2: the n = 9 Spearman run (prepared 2026-11-07, run and read 2026-11-08)

Status: run. Result and verdict in section 0 (added 2026-11-08); sections 1 to 6 are the 2026-11-07 preparation, kept as written (its 'not run' wording is superseded). Numbers are labelled MEASURED, SMOKE-TEST or ESTIMATED.

## 0. Result (2026-11-08, MEASURED)

Proxy suite against TabArena-Lite Elo, Spearman rho, n = 9 distinct checkpoints, from `runs/local_queue/dr42-spearman.log` (script `scripts/dr42_spearman.py`, run by the queue at 20:26 on 7 Nov after the last proxy run finished; I did not rerun it). All three new proxy runs finished with 48 of 48 scored split rows (`proxy_priorctl-s1.csv`, `proxy_treeclean-s1.csv`, `proxy_widerprior.csv`).

| proxy score | vs Lite Elo, common pool | vs Lite Elo, stored pool |
| --- | --- | --- |
| proxy Elo | rho +0.717, exact two-sided p 0.037, bootstrap 95% [0.105, 0.982] | rho +0.750, p 0.026, [0.128, 1.000] |
| mean AP | +0.500, p 0.178, [-0.385, 0.914] | +0.517, p 0.162, [-0.368, 0.947] |
| mean ROC-AUC | +0.267, p 0.493, [-0.670, 0.895] | +0.350, p 0.359, [-0.535, 0.896] |

Checkpoints that entered (9, none dropped; every one has 48 scored rows and a Lite Elo): published, e1-v480-s0, priorctl, priorctl-s1, treeclean, treeclean-s1, treeprior, widerprior, priorwide. Proxy Elo (RF = 1000, refit jointly with the five baselines): published 584.8, e1-v480-s0 691.8, priorctl 674.8, priorctl-s1 606.2, treeclean 736.6, treeclean-s1 697.4, treeprior 760.1, widerprior 561.0, priorwide 688.0. Lite Elo common pool 732 to 841 (published 755.9, e1-v480-s0 841.8, treeclean 840.8).

Against the pre-registration: the bar (rho at or above 0.8 across 6 or more checkpoints) is the same, n = 9 meets the count, and the task set is the same (18 task ids, 16 scorable, 3 splits, 48 rows). Differences: (a) the published row's proxy Elo moved from 574 (n = 6) to 584.8 because the Elo is refit with the added arms; (b) three of the nine are second seeds or near-siblings of arms already in (priorctl and priorctl-s1, treeclean and treeclean-s1, widerprior and priorwide), so n = 9 is nine distinct weight files but not nine independent draws; (c) no spread-widening checkpoint (ens1, linprobe) was added, so the Lite spread is 732 to 841; (d) the checkpoint-to-Lite-directory match still rests on name and timestamps, not a recorded hash.

**Verdict: no.** The headline correlation, proxy Elo vs Lite Elo, is +0.717 (common pool) and +0.750 (stored pool), both below 0.8; the bootstrap interval reaches down to 0.105, so the data do not exclude a weak relation either. The Verify is not met. Under the pre-stated bar the proxy is not validated and decides nothing: it is not usable to choose between checkpoints or to gate a release. What the run does show (MEASURED): the rank relation is positive and unlikely under no association (exact p 0.026 to 0.037), up from +0.600 (p 0.24) at n = 6, so the proxy Elo is informative about Lite Elo but too noisy at this spread. Mean AP and ROC-AUC are weaker (rho 0.27 to 0.52, p above 0.16). Using the proxy for anything would need either a higher rho on a wider Lite spread or a lower bar set before looking, and neither is available now.

Open: the licence half of the Verify for binary and multiclass CC18 tasks (another agent is on it; not touched here).

## 1. Checkpoints (MEASURED, on disk)

All three weights are present in the main checkout, 3.58 MB each, like the six already scored: `runs/v4-priorctl-s1.pt` (22 Sep 22:34), `runs/v4-treeclean-s1.pt` (22 Sep 22:34), `runs/v4-widerprior.pt` (22 Sep 10:34). `widerprior` is a different file from `priorwide` (14:06), which is already in the n = 6.

## 2. Lite Elo (MEASURED, `runs/agents1104/DR4.2/lite_elo.csv`)

All three already have a TabArena-Lite Elo from the fork's own `compare`, so none needs a Lite run: priorctl-s1 736.9 stored pool / 732.4 common pool, treeclean-s1 825.7 / 823.2, widerprior 761.2 / 759.4 (27 binary datasets, split 0). The only missing inputs are their proxy runs. The caveat from the n = 6 draft stands: the match of each checkpoint to its Lite directory rests on name and timestamps, not on a recorded hash.

## 3. The proxy command the n = 6 runs used

From the n = 6 draft and `proxy_<arm>.json`/`.log` (the `queue.sh` itself is not on disk, so `--out` is reconstructed from the output file names). Same 18 task ids, 3 splits (repeat 0, folds 0-2), `--no-baselines`, 2 threads, 48 of 54 rows scored (the 136-feature cap skips two tasks). Serial commands for the coordinator, one at a time, nice 19, with at least 14 GB free before each (as in the n = 6 queue):

```
cd /Users/morganoko/proj/fintfm && PYTHONUNBUFFERED=1 nice -n 19 uv run --frozen --extra bench python -m fintfm.evaluation.proxy_suite run --model /Users/morganoko/proj/fintfm/runs/v4-priorctl-s1.pt --name priorctl-s1 --tasks 3,15,43,49,3918,7592,9946,9976,9977,9978,10093,14952,14954,125920,146819,146820,167120,167125 --no-baselines --n-jobs 2 --out /Users/morganoko/proj/fintfm/runs/agents1104/DR4.2/proxy_priorctl-s1.csv > /Users/morganoko/proj/fintfm/runs/agents1104/DR4.2/proxy_priorctl-s1.log 2>&1; echo EXIT=$?
```

Repeat with `treeclean-s1` (`v4-treeclean-s1.pt`) and `widerprior` (`v4-widerprior.pt`), same task list, same output directory. Then, from a checkout that contains `scripts/dr42_spearman.py` (this branch):

```
nice -n 19 uv run --frozen python scripts/dr42_spearman.py --arms priorctl,priorctl-s1,treeclean,treeclean-s1,treeprior,widerprior,priorwide
```

That gives n = 9 with the published and `e1-v480-s0` rows; it writes `spearman_inputs.csv` beside the proxy runs.

## 4. Smoke test (SMOKE-TEST)

`v4-widerprior.pt`, tasks 9946 (wdbc) and 3 (kr-vs-kp), 1 split, `--n-jobs 2`, nice 19, `/usr/bin/time -l`. The checkpoint loads and scores under today's code: 29.2 s wall, 44.9 s user, peak resident set 5.77 GB. Per-split model time 0.53 s (wdbc) and 9.6 s (kr-vs-kp), against 4.3-4.8 s and 95-115 s for the same tasks in the n = 6 `priorwide` log (about 9 to 10 times faster; the n = 6 runs were on a loaded machine, and the load average was about 26 when the smoke test started, so the ratio is not clean).

ESTIMATE for the full run (18 tasks, 3 splits): the n = 6 runs took 75-136 min each at load (summed fold times in the `priorwide` log: 120 min). Scaling by the 9-10x gives roughly 12 min per arm on a quieter machine, but the large tasks (`adult`, `nomao`, 400-700 s per split at load) were not in the smoke test and may not scale the same. Plan on 0.2 to 2 h per arm, 0.6 to 6 h for the three. Peak RAM at least 5.8 GB per process (small tasks only); keep the 14 GB free-memory gate.

## 5. Candidates to widen the spread

Current spread (MEASURED): Lite common pool 756 to 841 among the six; with the three added, 732 to 841. Proxy Elo 574 to 738. No new-weights checkpoint on disk is far apart from the rest on Lite: every FinTFM Lite run on disk sits at a mean error of 0.207 to 0.225. The widest options are inference variants of the published checkpoint and one different head. Lite Elo below is MEASURED by the fork's `compare` on the stored 95-method pool (a copy of `lite_elo.py` writing to the scratchpad; nothing written to the repo or the fork):

| candidate | what it is | Lite Elo (stored pool) | proxy run on disk | note |
| --- | --- | --- | --- | --- |
| `fintfm_lite_ens1_1104` | published weights, 1 ensemble member | 696.4 | none | lowest Lite; needs a proxy run with `n_ensemble: 1`, about 1/8 of the default cost (ESTIMATED) |
| `fintfm_lite_cshift_1104` | published weights, class-shift ensemble | 716.0 | `inf2611/proxy_au512-ens8-cshift.csv` (48 rows) | no new compute; Lite config match (n_ensemble 8) to confirm from the Lite run config |
| `fintfm_lite_svd16_1104` | published weights, SVD 16 | 782.5 | `inf2611/proxy_au512-ens8-svd16.csv` (48 rows) | no new compute; same confirmation |
| `fintfm_lite_tfsq_1104` | published weights, rank + squash transforms | 807.0 | `dr73/proxy_lever_ens8-tf.csv` uses rank, power, squash: not the same config | no clean match |
| `mixedprior` | older 0.3 checkpoint, mixed prior | 813.0 | none; no `v4-mixedprior.pt` found | not usable without weights |
| `medium-matched`, `medium5m` | older medium checkpoints | 766.9, 750.9 | none | `v4-medium-matched.pt` exists; inside the current range, no widening |
| `fintfm_linprobe_d1fix_1104` | linear-probe head on the published model | 913.8 | none | highest Lite by 70 Elo; a head, not a checkpoint, and I did not check whether the proxy suite can run it |

Reading: ens1 (696) and linprobe (914) are the only points far enough out to widen the Lite range materially, and each needs new proxy work. cshift and svd16 cost nothing but share weights with the published checkpoint, so they are not independent checkpoints and would inflate n. Whether to count them or keep n = 9 as nine distinct checkpoints is the owner's call; I would report both. Proxy Elo for any added arm is refit jointly with the baselines, so the published row shifts again (as the n = 6 draft noted).

## 6. Script

`scripts/dr42_spearman.py` (new, derived from the untracked `runs/agents1104/DR4.2/spearman.py`): takes any number of checkpoints; the exact permutation p-value is vectorised for n up to 9 (the old per-permutation scipy loop would need 362,880 calls at n = 9); above 9 it uses a seeded Monte Carlo and says so; arms with fewer than 48 scored rows or no Lite row are skipped with a message; `--arms` and `--out-dir` options. An arm outside `lite_elo.csv` needs its row added with `lite_elo.py` first. `tests/test_dr42_spearman.py`: 6 tests pass (exact p equals the scipy loop at n = 6; n = 9 identity gives 1/9!; n = 9 equals scipy's `permutation_test`; n = 10 falls back; `correlate` on 9 rows; skip logic on synthetic runs). Run on the existing proxy runs, it reproduces the n = 6 draft: proxy Elo rho = +0.600, two-sided p = 0.2417, bootstrap [-0.640, 1.000] (MEASURED).

## Left open

The three proxy runs, then the script, then the n = 9 write-up. The Verify (rho at or above 0.8) stays unmet and the proxy stays unvalidated until then.
