# Compare against the gradient boosting family that actually matters

## Why

`docs/results/FINDINGS.md` §25: every "gradient boosting beats us" statement in this project was
measured against **sklearn's `GradientBoostingClassifier`**, the weakest member of the
family. LightGBM was silently skipped on every run for want of `libomp`, and CatBoost and
XGBoost were never installed.

The standard failure mode in foundation-model work is demonstrating `FM > poorly-tuned
baseline` when the question is `FM > tuned CatBoost`. We have been committing it.

Installing `libomp` then exposed a **segfault**: PyTorch bundles its own OpenMP runtime and
LightGBM loads Homebrew's, and two OpenMP runtimes in one process crash it. Confirmed by
bisection, and `KMP_DUPLICATE_LIB_OK=TRUE` does not help.

## What

- **Fit the boosting baselines in a subprocess that never imports torch**, exchanging arrays
  through a temporary file. Heavy-handed, and the only reliable fix for a native-library
  conflict we do not control.
- Add **CatBoost** and **XGBoost** beside LightGBM, with the sklearn GBM demoted to a
  reference row rather than the headline comparison.
- **Announce every skipped baseline.** A silently absent baseline flatters us, which is how
  §25 happened.
- Re-run findings 12, 16 and 17 against the strong family and **restate their margins**.

## Non-goals

- Not hyper-parameter tuning the baselines. Out-of-the-box on both sides, matching how
  Baesens et al. framed it; a tuned-GBM comparison is a separate and harder claim.
- Not adding tabular-neural or AutoML baselines yet, though `docs/research/RESEARCH_NOTES.md` lists
  them as the eventual comparison set.

## Falsified by

Nothing — this is a correctness fix. The open question is only how much worse our position
becomes, and that is the point of running it.

## Blocked by / blocks

- **Blocked by** nothing.
- **Blocks** any external claim involving gradient boosting, which is most of them.
