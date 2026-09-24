# Test the regime the thesis actually depends on

## Why

**Every real-data measurement in this repository was taken in the regime the literature says
we lose.** `bench.py` evaluates the full panel — 6,000 to 10,500 rows for the UCI sets.
Baesens et al. (arXiv:2605.18147) put the crossover where tuned gradient boosting catches up
at roughly **8,000 observations**, with substantial TFM advantage **below 1,000**. So the
benchmark has been measuring the wrong end of the size axis since it was written, and the
gradient-boosting dominance we recorded in `docs/results/FINDINGS.md` §5 is partly a consequence of
that choice rather than only of a small checkpoint.

This is not a refinement. If the model never wins at any size, the small-data thesis is dead
and no amount of pretraining scale rescues it, because Beyond IID (arXiv:2606.30410) finds
TFMs lose on large, wide and non-IID data regardless.

## What

- A sample-efficiency probe (Marconi's term, arXiv:2507.07296) sweeping training-set size
  from a few hundred rows upward against a **fixed** held-out test set, so the only variable
  is how much data each model learns from.
- Stratified subsampling, so a 250-row training set still contains defaults at all.
- Report the **crossover**: the largest training size at which the in-context model still
  leads gradient boosting.
- Report calibration at every size, not just AUC. Small-n is where a point estimate is least
  trustworthy and where an honest interval is worth most.
- An **untrained control** in the prior ablation, so a tie between priors can be read at all.

## Non-goals

- Not tuning gradient boosting harder at each size. The comparison is deliberately
  out-of-the-box on both sides, matching how Baesens et al. framed it; a tuned-GBM comparison
  is a separate, harder claim.
- Not a scaling law in model size. That is the demoted `scaling-curve` change.

## Falsified by

Itself, cleanly. If the crossover is `None` — the model leads at no size — the thesis fails
its most important test and the strategy pivots to the validation layer.

## Blocked by / blocks

- **Blocked by** a trained checkpoint from `phase1-prior-ablation`.
- **Blocks** any claim that this project has a defensible accuracy regime, and therefore
  Phase 2 onward.
