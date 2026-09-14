# Schema robustness, semantic column information, and active context selection

## Why

Every real-data comparison this project has run so far (V4FinBench, Polish, Taiwan) holds the
schema roughly fixed within a comparison: a table's column count and structure do not change
mid-evaluation. `diverse-eval-suite` addresses breadth across *datasets*; nothing in this
repository yet tests robustness to schema *variation* within a controlled setting — the same
underlying task presented through progressively wider, noisier, or differently-typed tables.

Two further gaps, raised in an externally-relayed review (2026-09-14) and not covered by any
existing proposal: whether the model uses **column semantics** (a name like "EBITDA margin,"
or even just that column's own statistical fingerprint) rather than treating every column as
anonymous, and whether **context selection can be active** rather than a fixed sampling
strategy — choosing which historical rows to show the model based on what would be maximally
informative, rather than uniform or blind-retrieval sampling.

## What

1. **Schema-variation sweep at fixed task simplicity.** The same underlying signal
   (`y=f(x0,x1)`), varying column count (5/10/50/200), useful-vs-useless ratio,
   categorical/numeric mix, duplicated and constant columns. The model should learn "3 of these
   200 columns matter," not have signal diluted by width — directly testable with the
   task-family generators `mechanism-diverse-prior` proposes, applied at controlled widths.
2. **Irrelevant-feature robustness curve.** Fixed task, progressively added pure-noise
   dimensions to d=100; plot achieved-AUC-vs-d. Directly comparable to how tree ensembles are
   known to behave here, which is the field's own standing comparison point.
3. **Semantic column information, three modes tested against each other**: (A) anonymous
   numeric columns (current default, everywhere), (B) derived statistical descriptors per
   column made available to the model (mean, std, skew, missing rate, quantiles), (C) column
   name / metadata embeddings. Particularly relevant to transfer onto real company financials,
   where column names carry real information this project currently discards entirely.
4. **Active context selection.** Given a candidate pool of historical rows and a budget, select
   the most informative subset for a query rather than sampling uniformly or retrieving by
   naive distance — an extension of `retrieval-context`'s existing work, conditional on that
   proposal's retrieval mechanism actually being understood (task 38.13) rather than actively
   harmful as currently measured (§70).

## Non-goals

- **Not making natural-language column semantics mandatory.** That would conflict with this
  project's schema-agnostic goal; mode (C) is one candidate among three to be *measured*
  against, not a decision made in advance.
- **Not started before `mechanism-diverse-prior`'s task-family generators exist** for item 1,
  and not before `retrieval-context` task 38.13 resolves the retrieval mechanism for item 4.
  Items 2 and 3 have no such dependency and can start independently.

## Blocked by

Item 1 by `mechanism-diverse-prior` task 40.2. Item 4 by `retrieval-context`/`cross-domain-
coverage` task 38.13.
