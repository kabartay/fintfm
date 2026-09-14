# An explicit task representation, and retrieval over mechanisms instead of rows

## Why

`cell-attention-and-task-inference` fixes how column-level information flows (two-way cell
attention, labels throughout). This proposal is the next architectural step past that, raised
independently by an externally-relayed review and consistent with this project's own framing
of pretraining as amortised inference over `p(y | x, D_context)`: does the model form an
explicit, inspectable representation of *which task it is looking at*, or does prediction
happen without ever separating "what kind of problem is this" from "what is the answer"?

`cell-attention-and-task-inference` task 39.7 already proposes probing for this (a small
classifier on the model's context representation, predicting which of ~20 known task families
generated the context). This proposal is what to build **if that probe finds task
representations are not well separated**: an architecture that is explicitly pushed to form
one, rather than hoping it emerges from the prediction objective alone.

## What

1. **An explicit task token / latent summary `z_task`.** Context rows produce, in addition to
   per-row predictions, a pooled task representation — a natural extension of the pooling
   already used for columns, applied once more across context rows. Query predictions condition
   on `z_task` as well as their own features.
2. **A mixture-of-priors reading of `z_task`.** Rather than one monolithic predictor, route
   through `p(y | x, D) = sum_k p(k | D) p(y | x, D, k)` where `k` indexes coarse task families
   (linear-like, tree-like, interaction-heavy, financial-SCM, ...). This is a mixture over task
   *priors*, not a mixture-of-experts over architecture — the distinction the source review
   draws explicitly, and the more natural fit to this project's own "prior over
   data-generating-processes" framing (`docs/RESEARCH_NOTES.md`).
3. **Retrieval over mechanisms, not rows** — a reframing of what `retrieval-context` already
   builds. Instead of `D_query -> nearest training rows`, `D_query -> task-family embedding ->
   most relevant prior family -> in-context inference`. Worth testing only once (1) and (2)
   exist to retrieve *into*.
4. **Posterior task uncertainty**, not only predictive uncertainty: on a deliberately ambiguous
   context consistent with two plausible task families, `p(k | D)` should spread across both
   rather than commit early. Connects directly to this project's calibration thesis (§12, §73)
   and is the natural complement to `conformal-pd-certificate`'s existing OOD-refusal work.

## Non-goals

- **Not a replacement for `cell-attention-and-task-inference`.** This is what comes after it
  reports, not a parallel track — an explicit task token without labels-throughout-context
  encoding underneath it would be building on the same unresolved foundation §74 measured as
  broken.
- **Not started until task 39.7's probe result exists.** If task representations already
  separate well under the cell-attention architecture, this proposal's premise (the model needs
  to be *pushed* toward one) is weakened and the proposal should be revisited, not assumed.

## Blocked by

`cell-attention-and-task-inference` tasks 39.4 and 39.7.
