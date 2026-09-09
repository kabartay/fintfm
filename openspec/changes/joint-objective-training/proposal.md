# Train both objectives, so one checkpoint can serve both paths

## Why

`docs/FINDINGS.md` §34: the training loop picks one objective per step with an `if/else`, so a
hazard checkpoint **never optimises the classification head**. That head keeps its random
initialisation, and `predict_proba` served it — AUC 0.3745 and a stated 69% default
probability against a 4.7% base rate, with no error.

A guard now refuses to serve an untrained head, which converts silent nonsense into a clear
failure. It does not make the checkpoint useful for both paths, and there are two reasons to
want that:

- **Every real evaluation needs both.** The binary path is where three of the four panels can
  be scored at all (the UCI sets carry no firm identifiers, §7), and the survival path is
  where the thesis lives. Two checkpoints per experiment doubles pretraining cost and makes
  every cross-path comparison confounded by which checkpoint it used.
- **The heads share the whole encoder.** The classification loss is signal about the same
  representation the hazard head consumes, and discarding it is discarding gradient for free.
  Whether it *helps* the hazard head is an open question and worth measuring.

## What

- Sum the two losses when a batch carries `period`, with a weight, rather than choosing one.
- Record both objectives in `trained_objectives` so the guard passes for both paths.
- Sweep the weight, at least coarsely. A hazard-only checkpoint is the weight-zero endpoint
  and already measured, which makes this a cheap comparison rather than a new baseline.

## Non-goals

- Not changing either loss. `survival_loss` and `loss` stay as they are.
- Not removing the guard. Even with joint training, a checkpoint should still say what it was
  trained on — the guard is what turned this defect from invisible into obvious.

## Falsified by

If joint training degrades the hazard head's out-of-time AUC below §35's 0.8118 at matched
compute, the objectives genuinely conflict and the right answer is two checkpoints plus the
guard, not one checkpoint. Record that plainly if so.

## Blocked by

Nothing, but it needs a pretraining run, so it is more expensive than anything currently
ahead of it in the queue. `retrieval-context` task 17.6 and `public-benchmark-claim` task 33.1
are both inference-only and come first.
