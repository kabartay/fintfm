# Deferred work, 2026-11: what stopped, and how to resume it

## Why

From mid-November 2026 the project is no longer actively supported (README, "Contributing").
Work was in flight when that decision was taken: readout arms on the local queue, GPU arms
waiting on the Kaggle quota, harness defects found by the 2026-11-13 inspection, and owner
decisions still open. Most of it is recorded somewhere already, but spread across FINDINGS
entries, task notes and chat. This change puts it in one place so that anyone resuming can
start from a list rather than a reconstruction.

Nothing here is a claim. Every figure cites the entry that measured it, and every task says what
evidence would complete it.

## What

Two parts.

**New tasks (`tasks.md`, series 49)** for work that is not yet an openspec task:

| task | area | state on 2026-11-13 |
| --- | --- | --- |
| 49.1 | regression readout arms `reg-head`, `reg-ctx4k` | running or queued locally |
| 49.2 | a regression checkpoint on the current recipe | not started; needs GPU |
| 49.3 | multiclass partition bags, `mc-bag8` | queued locally |
| 49.4 | multiclass and binary wide tables | blocked on DR8.3's checkpoint |
| 49.5 | binary readout options, V4FinBench first | `ppfix-v4` folds queued locally |
| 49.6 | the TabArena resubmission package | blocked on the 1,000 bar (DR7.6) |
| 49.7 | out-of-time split leakage | found 2026-11-13, not fixed |
| 49.8 | term-structure coherence leakage | found 2026-11-13, not fixed |
| 49.9 | smaller harness and estimator defects | found 2026-11-13, not fixed |
| 49.10 | owner decisions on Claim 9 and DR2.4 | recommendations made, not decided |
| 49.11 | release 0.10.1 | not started |
| 49.12 | small repository leftovers | found 2026-11-13 |

**Open tasks in other changes**, listed here so the queue is complete. They are not duplicated
and stay where they are; deferred ones stay open (owner, 2026-11-13).

| task | change | blocked by |
| --- | --- | --- |
| DR3.11 | design-review-2026-10-16 | Kaggle GPU quota (`task_dr311_bag4_f23`) |
| DR6.4 | design-review-2026-10-16 | seed-1 arms of the pending pretraining levers |
| DR7.6 | design-review-2026-10-16 | the 1,000 overall Elo bar; 903 at §224 |
| DR8.3 | design-review-2026-10-16 | Kaggle GPU quota (`ts_dr83_width512_s0`) |
| AU5.1 | audit-2026-10-15 | Kaggle GPU quota (`task_au51_*`) |
| AU5.3 | audit-2026-10-15 | Kaggle GPU quota (`ts_au53_cell2_s0`) |
| AU5.9 | audit-2026-10-15 | Kaggle GPU quota (`ts_au59_hazard_*`) |
| AU5.11 | audit-2026-10-15 | Kaggle GPU quota (`ts_au511_*`) |
| AU5.13 | audit-2026-10-15 | deferred by the owner (2026-11-08) |
| AU5.15 | audit-2026-10-15 | deferred by the owner (2026-11-08) |
| AU5.17 | audit-2026-10-15 | Kaggle GPU quota (`ts_au517_lenscale_s0`); the Muon half is harmful (§211) |
| 48.15 | learn-from-peers | Kaggle GPU quota (`ts_4815_maskcol_s0`) |
| 48.16 | learn-from-peers | narrowed by the owner (2026-11-08) |
| 48.24 | learn-from-peers | local queue (`4824-uni4000c`, `4824-compare`) |
| 48.25 | learn-from-peers | Kaggle GPU quota (`ts_4825_headonly_s0`) |
| 48.27 | learn-from-peers | Kaggle GPU quota (`ts_4827_latent_s0`) |
| 48.28 | learn-from-peers | deferred by the owner (2026-11-09): no official data route |

## Non-goals

- No new claims, and no change to any recorded number except through a new FINDINGS entry.
- No closing of deferred tasks "without implementing"; a task closes only when its Verify
  clause is met.
- No rewriting of the git history of this public repository.

## Blocked by

The Kaggle weekly GPU quota (49.2, 49.4 and most tasks in the second table), the owner's
decisions (49.6, 49.10, 49.11), and the local run queue (`runs/local_queue/queue.txt`: 49.1,
49.3, 49.5, 48.24). The machine is shared; a GPU or long CPU job needs the checks in
`CLAUDE.md` ("The machine is shared") first.
