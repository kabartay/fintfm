# Regression and multiclass heads: the missing two thirds of expected credit loss

## Why

`docs/paper/LIMITATIONS.md` has carried the same entry since the project began: **"No LGD, no
EAD, therefore no ECL."** Expected credit loss is PD x LGD x EAD, and this model produces PD
only. Loss given default and exposure at default are **continuous** quantities — regression
targets — so the missing two thirds of the project's own headline claim are blocked on a
capability the model does not have.

That reframes what looked like a benchmarking inconvenience. `experiments/openml_breadth.py`
and the TabArena run measure coverage at **27 of 51 datasets (53%)**, with the excluded 24
being 13 regression and 8 multiclass tasks against only 3 lost to the feature cap. The same
two gaps bound both the benchmark position and the product claim, and the product claim is the
one that matters.

**Multiclass is nearly free and has been switched off deliberately.** `ModelConfig.max_classes`
already defaults to 10, the head is already that wide, and `prior/scm.py` already samples
`n_classes = rng.integers(2, max_classes + 1)`. Every checkpoint in this project passed an
explicit `--max-classes 2`. The financial prior hardcodes `n_classes=2`, so a multiclass
checkpoint needs the SCM prior in the mixture -- which §73/§75 and §96 independently found
helps on real credit panels anyway.

**Regression does not exist**: no head, no loss, no target handling.

## What changes

A **binned distributional** regression head rather than a point estimate.

Bin edges are taken from the context targets' quantiles, per task; the head emits logits over
bins; the loss is cross-entropy on the bin index. Three reasons this is the right shape here
rather than a mean-squared-error scalar:

1. **It is what a PFN is for.** The network approximates a posterior predictive. A point
   estimate discards the distribution it was trained to represent, while a binned head returns
   it directly -- and quantiles, prediction intervals and tail risk all fall out.
2. **Calibration is this project's one consistent strength** (Claim 9, and calibration
   generalises across panels where discrimination does not). A distributional head keeps that
   measurable on regression targets; a scalar head makes it meaningless.
3. **It reuses machinery that already works.** The existing `head`, `y_proj`, per-cell label
   injection and cross-entropy loss all operate on class indices. Binning makes regression a
   classification problem over the target's own quantiles, so the architectural surface added
   is small and the §91 result -- that per-cell labels are load-bearing -- carries over rather
   than needing re-establishing.

LGD in particular is bounded in [0, 1] and famously bimodal, piling up at 0 and 1. A binned
predictive distribution represents that natively; a Gaussian head would not.

## Non-goals

- **Not a claim that this delivers ECL.** It supplies the *capability* to model LGD and EAD.
  Actually producing an ECL figure needs those targets, real data carrying them, and a
  validation protocol -- none of which this proposal provides.
- **Not a benchmark-coverage exercise.** Coverage improves as a side effect (53% to ~69% with
  multiclass, ~94% with regression), but if the only argument were the leaderboard this would
  not be worth the architectural surface.
- **Not a change to the hazard head.** `modeling/hazard.py`'s monotone cumulative PD is a
  separate construction and stays as it is.

## Blocked by

- Nothing. The multiclass half needs only a training run and is already queued; the regression
  half is new code that can be written against the existing tests.
