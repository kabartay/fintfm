# Cheap per-lender adaptation instead of retraining

## Why

A pretrained model that is merely *handed* a lender's table treats every institution
identically. But books differ in ways a context window cannot express: origination policy,
sector concentration, internal definitions of default, recovery practice, the vintage mix.
The obvious response — retrain per customer — destroys the entire economic argument, since
pretraining is the expensive part and doing it per lender costs more than the incumbent
approach it replaces.

The inspiration is FinGPT's position against BloombergGPT: rather than retraining a
foundation model per domain at $2.67M, adapt a base model for roughly $17 with a
low-rank update. That is a *structural* lesson about how a small team competes, not a method
to copy, and the arithmetic transfers directly — the deployment story only works if
specialising to a lender is cheap.

It also connects to the deployability gap (`docs/results/FINDINGS.md` §24): a model a lender can run
*and* tune on their own hardware is a categorically different product from one they may not
legally deploy at all.

## What

- A **low-rank adapter** over the frozen pretrained model, fitted on a lender's own book,
  small enough to train on a single consumer GPU in minutes rather than hours.
- **The base model stays frozen and shared**, so the provenance claim survives: no real data
  ever enters *pretraining*, and an adapter is plainly labelled as fitted to that lender.
- **Adapters are per-lender artefacts**, never merged back. Merging one lender's adapter into
  a shipped base model would leak their book into everyone else's predictions and void §1's
  auditability.
- Measure whether an adapter beats in-context conditioning at all. It may not: the whole
  premise of in-context learning is that conditioning already does this work.

## A concrete recipe exists, and it is cheaper than expected

FinCast (arXiv:2508.19609 §4.2) fine-tunes for **one epoch**, with gradient updates
restricted to the output block and the **last 10% of layers**, describing this as "minimal
task-specific tuning". That is a smaller intervention than a full low-rank adapter and it is
a reasonable first thing to try here: freeze everything except the head and the final block,
one pass over the lender's book. If that is enough, the adapter machinery is unnecessary.

## Non-goals

- Not fine-tuning the whole model per lender; that is the cost this exists to avoid.
- Not federated or multi-lender training. One adapter, one book, no pooling.
- Not a substitute for the term structure or the certificate.

## Falsified by

The cheapest test needs no adapter at all: **does giving the model more of a lender's own
rows already capture what an adapter would?** §23 measured context sweeps on consumer credit
and found AUC flat from 500 to 20,000 rows — if that also holds on corporate panels, then
neither conditioning *nor* adaptation is information-limited, and this proposal is solving a
problem the model does not have.

## Blocked by / blocks

- **Blocked by** a checkpoint worth adapting, and by §23's context sweep being repeated on
  corporate data.
- **Blocks** nothing, but it is a prerequisite for any deployment story beyond "send us your
  table".
