# Does performance scale with pretraining?

## Why

Phase 2 of `docs/roadmap/STRATEGY.md`. A model that does not improve with more pretraining is a
neural network with good marketing, not a foundation model. The scaling relationship is the
scientific result worth publishing, and it is the claim that distinguishes this from an
AutoML wrapper.

It is also the decision procedure for where to spend money: if the curve is flat, compute
should go into prior richness (`temporal-financial-prior`) instead of scale, and that is a
cheaper thing to learn early than late.

## What

Train at 100k, 1M and 10M synthetic tasks — and at more than one model size — then plot
performance on held-out **real** credit panels against pretraining scale. The x-axis is
pretraining volume; the y-axis must be real-data performance, not synthetic held-out
accuracy, which can improve while transfer does not.

## Non-goals

- Not a compute-optimal scaling law. Three points and two sizes indicate a direction; they do
  not fit an exponent, and presenting them as one would be overclaiming.
- Not matching anyone's published scale. Google TabFM used hundreds of millions of synthetic
  datasets; that is not the comparison being drawn.

## Falsified by

A flat or saturating curve. That is a publishable negative result and it redirects the
roadmap rather than ending it.

## Blocked by / blocks

- **Blocked by** `phase1-prior-ablation`. There is no point plotting a curve for an effect
  that does not exist.
- **Blocked by** compute: this is the change that justifies renting NVIDIA
  (`docs/infra/COMPUTE.md`), because the run count multiplies.
- **Blocks** any external claim that this is a foundation model rather than a model.
