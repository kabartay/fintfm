# Audit of 2026-10-15: bugs, rigour gaps and strength levers

## Why

§163 moved V4FinBench for the first time (+0.0076 mean AP, 5/5 folds), yet FinTFM still sits at
about half of CatBoost's AP (0.2062 against 0.4005) and 94th of 95 on TabArena. Before spending
more GPU time on levers, an independent review of the code, the evaluation and the documents was
run on 2026-10-15: six code and consistency reviews, three literature and implementation surveys,
one re-derivation of §159-§163 from the raw runs, and seven documentation reviews.

Every task below was checked against the code or the runs before it was written down. Items an
agent reported but that did not survive that check are not here. Where a task rests on a figure
the agent could not verify (peer model sizes, for instance), the task says so and its first step
is to verify it.

## What this is

A queue, ordered by phase: correctness first, because a bug can bias every later measurement;
then evaluation rigour, because it decides what the levers' results mean; then documents; then
cheap diagnostics of the CatBoost gap; then the expensive levers.

## What this is not

Not a claim that any lever below will help. Each lever task names its comparator and the
pre-registered rule it will be judged by (fold-paired AP delta above 0.0073 on at least 4 of 5
folds, as in §159), and is written up in FINDINGS whatever the answer.

## Non-goals

- No training on TabArena or BeyondArena datasets.
- No importing the AXIOM harness.
- No GPL or AGPL dependencies.

## Blocked by

- **Phase 0 is blocked by the owner.** Each item is a decision only the project owner can make.
- **Phase 1 and Phase 2 are blocked by nothing.** They are code and evaluation fixes with tests.
- **Phase 4 is blocked by §164 finishing**, so its GPU time does not compete with §164's arms,
  and by AU2.2 for the booster arms.
- **Phase 5 is blocked by Phase 1** (AU1.1 and AU1.13 change what pretraining sees) and by the
  Phase 4 diagnostics, which say which lever the gap points at.

## Verify

`uv run python openspec/tools/validate.py` passes, and every task's own Verify line is met
before it is ticked.
