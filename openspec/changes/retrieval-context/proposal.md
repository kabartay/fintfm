# Retrieve the context instead of sampling it

## Why

`docs/FINDINGS.md` §5 established the single most useful engineering fact in this project:
**how the context is built explains more variance than which architecture is used.** Tanna et
al. measured it, and our own results turn on it.

Yet every context strategy implemented here selects rows **without looking at the query**:
`uniform`, `balanced`, `hybrid` all sample from the training pool blind. That is a large
unexplored axis, and it is the one the finding says matters most.

Retrieval-augmented tabular models take the obvious next step: given a firm to score, build
its context from the **most similar firms** rather than a random draw. TabDPT (*Scaling
Tabular Foundation Models*) uses retrieval over real tables with in-context learning and
reports strong results on OpenML-CC18 and CTR23 without task-specific fine-tuning.

Two reasons to expect this matters more here than in general tabular work:

- **§23 showed our model gains nothing from more context** — AUC flat from 500 to 20,000
  rows on Home Credit. If more rows do not help but *better* rows do, retrieval is precisely
  the lever, and the flatness is evidence the model is context-*quality* limited rather than
  context-quantity limited.
- **Credit books are heterogeneous.** A German manufacturer with €40M turnover is not
  informed by a Polish retailer with €2M. Sector, size and country similarity are exactly
  what a credit analyst uses, and a random context dilutes it.

## What

- A `retrieval` context strategy: for each query (or query batch), select the `max_context`
  nearest training rows under a distance on normalised features.
- **Combine retrieval with class balancing**, since §5's finding still holds — retrieving 2,000
  nearest neighbours at a 0.3% base rate could return almost no defaults, which would be a
  regression, not an improvement. Retrieve within class, then balance.
- Cache the index. Retrieval per query batch must not make inference unusable; §23 measured
  20,000-row contexts at 233 seconds and that is already near the limit of practicality.
- Compare against `uniform`, `balanced` and `hybrid` on both corporate panels.

## Non-goals

- Not learned retrieval or a trained retriever. A simple distance first; if that fails, the
  idea is likely wrong rather than under-engineered.
- Not retrieval over an external corpus. Context comes from the lender's own book, which
  keeps the provenance invariant intact.

## Falsified by

Compare retrieval against balanced sampling at matched context size on V4FinBench and the
UCI panels, with the paired bootstrap. If it does not beat blind sampling, then context
*quality* is not the lever either, and §23's flatness means the model is simply weak — which
is a much more important conclusion than this change, and would redirect effort to scale.

## Blocked by / blocks

- **Blocked by** nothing. Implementable today against existing checkpoints.
- **Blocks** nothing, but it is the cheapest remaining idea that could move accuracy, since
  it needs no retraining at all.
