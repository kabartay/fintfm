# Calibrated prediction

## Purpose

A lender prices, provisions and holds capital against the *level* of a probability, not its
ordering. Measured: gradient boosting's calibration error is 2.3× to 11.7× worse than this
model's at every dataset size, while still winning AUC above a few hundred rows
(`docs/results/FINDINGS.md` §12). The two properties are separable, and the level is the deliverable.

## Requirements

**C1 — `predict_proba` returns probabilities, not scores.** Rows sum to one; the positive
column is interpretable as a default probability.

- *Enforced by:* `tests/test_classifier.py::test_fit_predict_shapes`.

**C2 — Resampled context is base-rate corrected.** An in-context learner reads the class
balance out of its context, so a rebalanced context must be corrected or its probabilities
are inflated (measured at 14.9% predicted against 4.7% actual, `FINDINGS` §6).

- *Enforced by:*
  `tests/test_classifier.py::test_prior_correction_preserves_ranking_but_shifts_probabilities`.

**C3 — The correction must not change ranking.** It is a constant per-class logit shift,
hence monotone, so AUC is provably unchanged.

- *Enforced by:* the same test, which asserts identical orderings.

**C4 — The correction is inert when the context was not resampled.**

- *Enforced by:*
  `tests/test_classifier.py::test_prior_correction_is_inert_when_context_is_not_resampled`.

**C5 — Training uses a proper scoring rule.** Cross-entropy is minimised only by honest
probabilities. A ranking loss would optimise away the one number the buyer uses.

- *Enforced by:* `FinancialTFM.loss`, and review.

## Known gap

The analytic correction *overshoots* at the 1-year horizon (`FINDINGS` §6). The successor is
a fitted correction on a validation split — `changes/fitted-calibration`. Until that lands,
C2 is satisfied but imperfect, and saying so is part of satisfying C6 below.
