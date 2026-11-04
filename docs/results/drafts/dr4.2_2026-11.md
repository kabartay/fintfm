# DR4.2: the Spearman Verify, inventory and gap (2026-11-04)

**Result: the Verify cannot be computed yet. 2 distinct checkpoints have both a full proxy run and a TabArena-Lite Elo; it needs 6 or more.** No inference was run for this note; everything below is read from files on disk.

## What the Verify is

DR4.2's Verify (`openspec/changes/design-review-2026-10-16/tasks.md`): "Spearman 0.8 or higher against Lite Elo across 6 or more existing checkpoints before the proxy is used to decide anything." So: per checkpoint, proxy-suite score (Elo or mean score) against TabArena-Lite Elo, rank correlation at least 0.8, n at least 6 distinct checkpoints. Inference-lever arms of one checkpoint are not distinct checkpoints (same weights).

## Full proxy runs on disk (binary, 18 tasks, 16 scorable, 3 splits = 48 scored splits; MEASURED, `runs/dr73/`, `runs/inf2611/`)

Distinct checkpoints (12 with a full 54-row binary run):

| checkpoint | file | proxy CSV |
| --- | --- | --- |
| published `v4-cellattn-labels.pt` | `runs/v4-cellattn-labels.pt` | `dr73/proxy_binary.csv` (with 5 baselines; Elo 561 [316, 726], §181) |
| `e1-v48-s0`, `e1-v48-s1` | `runs/kaggle-ts-e1-v48-s*/` | `dr73/proxy_e1_v48-s0.csv`, `proxy_arm_e1-v48-s1.csv` |
| `e1-v480-s0` | `runs/kaggle-ts-e1-v480-s0/` | `dr73/proxy_e1_v480-s0.csv` (+168 Elo [+50, +326] over `e1-v48-s0`, §181) |
| `dr24-a`, `-b`, `-c`, `-d` (s0) | `runs/kaggle-ts-dr24-*-s0/` | `dr73/proxy_arm_dr24-*-s0.csv` |
| `dr62-induced-s0`, `dr63-cls-s0` | `runs/kaggle-ts-dr6*/` | `dr73/proxy_arm_dr62-*.csv`, `proxy_arm_dr63-*.csv` |
| `dr82-depth8-s0`, `dr82-width192-s0` | `runs/kaggle-*-dr82-*/` | `dr73/proxy_dr82-*.csv` (also 24 multiclass, 27 regression rows) |

Not full: `e1-v480-s1` (24 rows, partial), `lever-ft200` (3 and 16 rows). Multiclass: `v4-multiclass.pt`, `dr74-mc-s0/s1`, `dr74-mc-v480-s0` (24 rows each, 8 tasks).

Inference-lever arms of the published checkpoint (same weights, so not distinct): `lever-ctx8k`, `lever-ens8`, `-ens8-swap`, `-ens8-tf`, `lever-retrieval`, `inf2611/au43-ctx900-ens3`, `au512-ens8-cshift`, `au512-ens8-svd16`.

## TabArena-Lite on the 27 binary datasets (fork `experiments/`, read only; Elo MEASURED, from `docs/results/TABARENA.md`)

| checkpoint | fork directory | Lite Elo on record |
| --- | --- | --- |
| published `v4-cellattn-labels.pt` | `fintfm_lite_nan_1019` | 756 (+126/-180), NaN passed through; 763 with zero fill |
| `e1-v480-s0` | `fintfm_lite_e1v480s0_1024` | 842 (+139/-213), 94th of 95 |
| unpublished linear probe (§132) | `fintfm_linprobe_lite` | 910, 93rd of 95; no proxy run, no checkpoint of the proxy's kind |
| `medium5m`, `medium-matched`, `priorctl`, `priorctl-s1`, `treeclean`, `treeclean-s1`, `treeprior`, `widerprior`, `priorwide`, `mixedprior`, `cat_full`, `deep26m` (8 of 27 datasets only) | same names | results exist (27 datasets each, 8 for `deep26m`); mean ROC-AUC on record for some (`medium5m` 0.7746, `medium-matched` 0.7774, FINDINGS §108); no Elo on record |

## The overlap

Both columns exist for 2 distinct checkpoints: the published one and `e1-v480-s0`. A Spearman on 2 points is undefined in practice. The 12 proxy-run checkpoints other than those two have no Lite run; the old Lite arms have no proxy run.

## Cheapest way to 6

All of these are 3.4 MB checkpoints of the published architecture except `medium-matched` (19 MB) and `deep26m` (10 MB), and their weights are on disk (`runs/v4-priorctl.pt`, `v4-priorctl-s1.pt`, `v4-treeclean.pt`, `v4-treeclean-s1.pt`, `v4-treeprior.pt`, `v4-widerprior.pt`, `v4-priorwide.pt`).

1. Run `proxy_suite run` (binary, same 18 tasks, 3 splits) on 4 or more of those 3.4 MB checkpoints. Cost, ESTIMATED from MEASURED per-split seconds summed over 48 scored splits: published 0.16 h, `e1-v480-s0` 0.04 h, `e1-v48-s0` 0.05 h; call it 0.05 to 0.2 h per checkpoint, 0.2 to 0.8 h for 4, at the batch's 2 threads and nice 19 possibly longer.
2. Compute their Lite Elo from the Lite results already on disk with the fork's own leaderboard code (analysis, no inference; Elo is only defined against the full 95-method board, so it must come from that code and not be re-derived here). Not done: the instruction was to run nothing in the fork.
3. Alternative, dearer: Lite-run the 4 distinct proxy-run checkpoints (`e1-v48-s0`, `e1-v48-s1`, `dr24-*`...). One Lite binary run took about 50 minutes (ESTIMATED from the log's file creation and last-write times, `runs/lite_e1v480s0.log`, 2026-10-24 19:37 to 20:27), so about 3.3 h for 4. Step 1 and 2 is 4 to 5 times cheaper.

With step 1 and 2 on 4 old arms, n = 6 (published, `e1-v480-s0`, 4 old arms); with all 7 available, n = 9.

## Caveats the eventual Spearman must carry

- **Narrow spread.** Lite Elo on record for published and `e1-v480-s0` differ by 86 Elo, inside both intervals (756 +126/-180; 842 +139/-213), and the binary arms sit within 0.7746 to 0.7926 mean ROC-AUC (TABARENA.md). The old arms are near-ties, so a rank correlation across them mostly ranks noise; report an exact or bootstrap interval, not a point.
- **Different NaN handling.** The old Lite arms were run with NaN filled with 0 (TABARENA.md); `e1-v480-s0` with NaN passed through. The two published-checkpoint runs differ by 7 Elo (756 against 763), small beside the spread but it is a confounder. Prefer the zero-fill Elo for the old arms and flag it.
- **The old checkpoints may not load** under today's `from_config` (not tested here; testing needs a run).
- **Elo has two sets.** Proxy Elo is anchored at RF = 1,000 on 16 scorable tasks; Lite Elo on 27 datasets against 95 methods. Compare ranks, not values.
- **Verification, not validation, of the proxy's use.** The proxy compares models on OpenML tasks outside TabArena; it must not decide anything until the Spearman is at or above 0.8, as DR4.2 states. §181 and the E1 reading ("positive on the proxy", +168 Elo) stay unvalidated meanwhile; the Lite pair for the same checkpoints runs the other way in size (+86 Elo, same rank).

## Left open

- The 4 or more proxy runs and the Lite Elo computation above (not started; compute is shared).
- Spearman with a bootstrap over tasks and datasets, then the verdict against 0.8.
