# scripts/kaggle: notes on historical kernels

## Kernel `id` slug and title differ (intentional, historical)

Four kernels were launched with an `id` slug that does not match the `title` in their
`kernel-metadata.json`. The `id` is the record of what ran on Kaggle, so it is kept as launched and
must not be "fixed":

| directory | `id` slug | title |
| --- | --- | --- |
| `task_39_28_replicate_s91` | `fintfm-task-39-28-s91-replicate` | fintfm task 39.28 - s91 seed-1 replication |
| `task_48_3_binned_control` | `fintfm-task-48-3-binned-control` | fintfm task 48.3 - matched binned-head control |
| `task_48_3_quantile_head` | `fintfm-task-48-3-quantile-head` | fintfm task 48.3 - pinball quantile head |
| `task_48_9_schedulefree` | `fintfm-task-48-9-schedulefree` | fintfm task 48.9 - schedule-free optimisation |

## Unpinned kernels

- `task_dr311_*`: in the tree at 554feba the eight `bag4`/`bag8`/`ens8` kernels are already pinned
  at `e89def2` (commit 3d257d1), and `task_dr311_ftleaf*` carry their pin inline (`@027e0cc` in the
  `pip install` line, not a `COMMIT` constant), which is the commit that run used.
- `ts_dr75_width512_mc_s0`: `COMMIT = "UNPINNED"` on purpose. The kernel was prepared, not
  launched (DR7.5 note, launch gate in `docs/results/drafts/dr8.3_2026-11.md`). cfbc845 holds the
  arm config and is on origin/main, but pinning a never-launched kernel to an old sha would freeze
  out later fixes; set `COMMIT` to the then-current pushed sha at launch. The refuse-unpinned guard
  stays until then.
