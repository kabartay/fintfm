# Two-way cell attention, labels throughout context encoding, and a task-inference diagnostic

## Why

`docs/results/FINDINGS.md` §74 found that training predominantly or exclusively on the financial prior
caps basic signal extraction at ~0.73 AUC regardless of true task difficulty — measured against
an *exactly known* Bayes-optimal AUC, on a probe with a single informative dimension and no
column-identity structure at all. Training on the generic SCM prior instead, with the identical
architecture, tracks the true curve almost exactly (0.997 achieved at 0.999 true).

§76 bisected four candidate content-side causes using checkpoints that already existed — zero
new GPU spend — and none of them closes the gap: balancing the base rate, raising raw
signal-to-noise, swapping to clean uncorrelated features, and breaking accounting identities all
leave the cap at 0.59-0.76, while only discarding the financial generator's structure entirely
(pure SCM) reaches 0.997. That pattern — four different, real content interventions producing
essentially the same capped curve, while only a wholesale generator swap escapes it — is the
signature of a representational limitation in what the current architecture can extract, not a
content property of the prior that the next experiment happens to fix.

`docs/design/DECISIONS.md` D12 already named the unresolved piece when it traded exact column-order
invariance for random per-task identities: *"the cell-level two-way design is the one with
published evidence behind it, and is what a serious version of this model should end up with."*
Random column identities (D12) fixed whether the model can tell `x_0` from `x_1` — the trivial
prior went from AUC 0.713 to 0.997 on that alone (§56). They do not give the model a way to
learn what a column *means* from its own distribution across the context, because the column
stage in `encode_rows` still attends only within a row, never across rows within one feature —
the docstring's own words: "runs on one row at a time and so never sees a column."

An externally-proposed review (relayed by the user, 2026-09-14, reviewing this project's own
decision and findings documents) independently converged on the same diagnosis and adds a
second, related gap: labels currently enter the model once, after column attention and pooling
have already collapsed each row to one vector (`model.py::forward`, the `h = h + y_emb` step).
Every column-level computation happens label-blind. A model whose job is
`p(y_q | x_q, X_c, y_c)` should let column-level representations be conditioned on the context
labels that make one financial ratio's meaning ("high is risky" vs "high is safe," per-task,
per §47) inferable in the first place.

## What

Three pieces, ordered so each is checkable before the next is trusted.

1. **Two-way cell attention.** Keep per-cell representations `(B, N, F, d_cell)` through
   alternating blocks of *row attention within each feature* (a cell attends to the same
   feature across other rows — a data-derived column identity, computed from the column's own
   values rather than a random tag) and *column attention within each row* (the existing
   mechanism, attending across features). Config-gated (`ModelConfig.n_cell_blocks`, default
   `0` reproducing every existing checkpoint exactly) so this is additive, not a replacement —
   matching this project's standing pattern (`pooling`, `column_id_dim`) of never breaking a
   prior comparison silently.
2. **Labels throughout.** Context cells get the label embedding added *before* the cell-attention
   blocks (`h_ij = E_x(x_ij) + E_y(y_i)` for context rows), query cells get a learned mask
   embedding instead of a real label. Row/column attention inside the blocks can then learn
   label-conditioned column meaning, not just label-blind column statistics. Additive: the
   existing post-pooling label injection stays, so `n_cell_blocks=0` checkpoints are unaffected.
3. **A permanent Bayes-ceiling regression test.** §74's probe — exact Bayes-optimal AUC via a
   closed-form Gaussian mean-shift construction, verified numerically before use — becomes a
   committed diagnostic (`experiments/capability.py` or a dedicated module), run on every
   checkpoint compared in this project from now on, not just this one.

## Non-goals, this round

- **Not a DGP/task-representation classification probe, not an interaction-order curriculum,
  not schema-variation or irrelevant-feature-robustness sweeps, not causal/interventional
  tests, not label-permutation-sensitivity re-derivation (already covered, §47/§48).** All
  proposed in the same external review and genuinely useful, tracked as tasks below, explicitly
  sequenced *after* the architecture question this proposal answers — per the reviewer's own
  and the user's explicit priority ordering.
- **Not broadening the synthetic prior, not scaling.** Both are the review's own stated
  "only after" items.
- **Not claiming this will fix §74's cap.** It is the best-supported hypothesis given §76's
  bisection, not a certainty. The experiment is designed to say clearly whether it worked —
  rerunning the exact §74 probe on the new architecture — rather than to assume it did.

## Blocked by

Nothing code-side. Conceptually gated on §76's bisection, which is done.
