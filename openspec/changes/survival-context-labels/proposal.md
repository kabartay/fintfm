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

The first horizon is respectable and the curve decays toward chance, while the baseline — which
gets a *separate fitted model per horizon, each one seeing that horizon's labels* — holds up.
The gap widens monotonically with the horizon, which is the signature of missing timing
evidence rather than of a weaker model.

This is now the largest identified, unfixed defect on the path the strategy depends on. The
whole IFRS 9 lifetime-ECL argument (D3) rests on the far end of the curve, which is precisely
where this model is currently worst.

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

## Blocked by

Nothing. The data supplies `period` and `n_observed` already (`load_v4finbench`), the prior
emits `period`, and the training loss consumes it per row. Only the *context* path discards
it, which is why this is a small change with a large expected effect — and why task 31.1 can
test the premise before any architecture is touched.
