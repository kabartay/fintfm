# Fit the calibration correction instead of deriving it

## Why

`docs/results/FINDINGS.md` §6 established that balanced context inflates predicted default rates
threefold and that an analytic label-shift correction fixes it — ECE from 0.102 to 0.007 at
the 3-year horizon. But the same measurement showed the correction **overshooting** at the
1-year horizon, where ECE worsened from 0.0151 to 0.0173.

That is expected. The analytic shift is exact only if the model reads the base rate *purely*
from the context prior, and a real model does something messier. The finding already names
the successor: fit the correction on a validation split rather than assuming the mechanism.

Leaving this unfixed means the headline calibration claim has a known counterexample in its
own evidence, which is exactly the kind of thing a model-risk reviewer finds.

## What

- Fit a one-parameter (temperature) or two-parameter (Platt) recalibration on a held-out
  validation slice of the training data, at `fit()` time, and apply it in `predict_proba`.
- Keep the analytic shift as the default when no validation slice is affordable, and as the
  comparison baseline.
- Report both in the benchmark so the choice is evidence-driven rather than assumed.

## Non-goals

- Not conformal intervals. Those are `conformal-pd-certificate`; this is point calibration.
- Not isotonic regression initially — it needs more data than a small validation slice and
  can overfit at a few-percent base rate.

## Falsified by

If the fitted version does not beat the analytic one on ECE across all horizons and multiple
seeds, keep the analytic correction: it needs no validation data, costs nothing, and is
exactly explicable to a reviewer, which has independent value.

## Blocked by / blocks

- **Blocked by** `phase1-prior-ablation`, only in the sense that calibrating a model that
  cannot discriminate is premature.
- **Blocks** `conformal-pd-certificate`, which should sit on top of a well-calibrated point
  prediction.
