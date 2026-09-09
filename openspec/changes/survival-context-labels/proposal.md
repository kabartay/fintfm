# Put the default timing into the context

## Why

The hazard head is asked to produce a six-horizon term structure from a context that contains
**no timing information at all**. `FinancialTFMClassifier.fit` collapses the survival label to
binary `y`, so the context says which firms defaulted and never when. The model must infer the
shape of the whole curve from covariates alone.

The out-of-time AUC column is exactly what that predicts (`docs/FINDINGS.md` §30):

| horizon | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| fintfm hazard | 0.8398 | 0.7559 | 0.6842 | 0.5968 |
| per-horizon logreg | 0.9717 | 0.8908 | 0.8310 | 0.7530 |

**Corrected 2026-09-09 by `docs/FINDINGS.md` §31, before any work started.** This proposal
was originally written claiming the widening gap was "the signature of missing timing
evidence". Decomposition shows otherwise: the baseline decays almost as fast as we do (−0.2187
against −0.2429 across the grid), so the gap is a **constant 0.132 deficit present already at
the first horizon** plus a 0.024 widening. **84% of the deficit is horizon-independent** and
belongs to `retrieval-context`, not here.

What survives is the 16%: our excess decay of 0.0242 over three steps is real, and a context
carrying no timing information is a plausible cause of it. The IFRS 9 lifetime-ECL argument
(D3) does rest on the far end of the curve, so 0.024 there is worth having — but this is a
second-order change on a first-order problem, and it should not be worked before
`retrieval-context`.

Task 31.1's cheap premise test came back **confounded rather than supportive**: `n_observed`
is censored by default itself, so filtering to full-grid observation removes the defaulters
and changes the context base rate 8×. There is no cheap unconfounded test, which is a further
argument for sequencing this behind cheaper work.

## What

- Extend `fit` to accept optional `period` and `n_observed` alongside `y`, and carry them into
  the context rather than discarding them.
- Embed the context's period label so the model reads *when* each context firm defaulted.
  `FinancialTFM` currently embeds context labels through `y_proj` over a one-hot of
  `max_classes`; a period grid of `K + 1` states (default in period `k`, or censored) is the
  natural generalisation, and pretraining already has `period` available per row.
- Keep the binary path working unchanged when `period` is not supplied.
- Preserve censoring semantics: a context firm observed for three horizons must not be
  presented as a five-horizon survivor. The training loss already handles this per row
  (`hazard.HazardHead.loss`); the context embedding must not undo it.

## Non-goals

- Not changing the hazard head or the monotonicity guarantee, which are working (§20, §26).
- Not adding a horizon-specific base-rate correction. That is a real gap — §28's shift is a
  single scalar applied at every horizon because a binary context cannot yield per-horizon
  rates — but it becomes *tractable* only once this change lands, so it is sequenced after,
  not merged in.

## Falsified by

Pretrain one checkpoint with period-labelled contexts and one without, at matched compute and
matched prior, and score both on the V4FinBench out-of-time split. If mean AUC does not
improve, and specifically if the decay across horizons is not flattened, the missing-timing
explanation is wrong and the gap is about capacity or the prior instead.

**Pre-registered prediction, recorded before running:** mean AUC improves by at least 0.03 and
the largest gain lands at horizons 2 and 3. Writing this down first is what makes the result
evidence rather than a story fitted afterwards.

**Left exactly as written, and now expected to fail.** §31 caps the horizon-dependent share of
the gap at about 0.024, below this prediction's 0.03 threshold. The prediction is not being
revised to match later evidence — that would defeat its purpose. If the change lands above
0.03 anyway, §31's decomposition is wrong and that is worth more than the accuracy.

## Blocked by

Nothing. The data supplies `period` and `n_observed` already (`load_v4finbench`), the prior
emits `period`, and the training loss consumes it per row. Only the *context* path discards
it, which is why this is a small change with a large expected effect — and why task 31.1 can
test the premise before any architecture is touched.
