# Design review of 2026-10-16: integrity, cheap measurements, then volume and v2

## Why

`docs/design/DESIGN_REVIEW_2026-10-16.md` records six design reviews. Three findings change the
plan: V4FinBench's label sits behind one gate a uniform context wastes (3,586 of 3,587
positives have `Insolvency_flag = 1`); §116's tree-prior loss is confounded with the financial
share (§166); and the code cannot yet prove what trained a checkpoint, which is how that
confound went unseen. The audit of 2026-10-15 fixed bugs and built levers; this change orders
what to do with them.

## What this is

A queue in four phases: integrity work that changes no output (DR1), measurements that need no
pretraining (DR3, DR4), throughput and a volume test (DR5), then the prior and architecture v2
bundles (DR2, DR6), each judged under a pre-registered rule.

## What this is not

Not a claim that any change helps. Every estimate in the review is labelled ESTIMATED, and every
lever is scored before it is believed, V4FinBench first.

## Non-goals

- No training on TabArena or BeyondArena datasets, and no choosing arms on them (DR4 removes
  the existing overlap).
- No code, weights or data from any tabular-foundation-model project.
- No resubmission to TabArena before the DR4 proxy clears the bar in DR7.

## Blocked by

- **DR1 is blocked by nothing** and blocks every new GPU checkpoint (provenance).
- **DR3 and DR4 need no GPU pretraining**; DR3's V4 arms need one scoring slot each.
- **DR5 is blocked by DR1** and blocks DR2 and DR6 at volume.
- **DR7 is blocked by DR4.**

## Verify

`uv run python openspec/tools/validate.py` passes, and each task's own Verify line is met
before it is ticked.
