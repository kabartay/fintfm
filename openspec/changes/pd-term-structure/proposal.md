# Predict the PD term structure, not one horizon at a time

## Why

"Financial tabular data" hides three different problems, and the project has silently been
solving the least useful one.

| problem | object predicted | rows exchangeable? | who owns it |
| --- | --- | --- | --- |
| classification | default within a fixed horizon | yes | crowded; TFMs win only on small data (`FINDINGS` §9) |
| forecasting | a future value of a series | no, order is the signal | Google (TimesFM-3: 330M params, 10^12 time points, in BigQuery) |
| **panel hazard** | **probability of an event over time, per entity, with time-varying covariates** | **neither** | **nobody** |

Credit risk is the third. It has been the third in the academic literature since discrete-time
hazard models (Shumway; Campbell, Hilscher & Szilagyi), and it is the third in regulation.
**IFRS 9 requires lifetime expected credit loss**, which is a *term structure* of PD rather
than a single number, and Basel PD is quoted at multiple horizons. A 12-month classifier is
structurally insufficient for the provisioning rule every regulated lender is bound by.

Neither foundation-model camp serves this. TFMs treat rows as IID and Beyond IID
(arXiv:2606.30410) finds they fail on temporal and grouped data. TSFMs forecast a series,
which is the wrong object — the target is a hazard path conditioned on covariates, not the
continuation of a numeric sequence.

**And the datasets already have the shape we are discarding.** UCI Polish ships five
horizons; V4FinBench six. `evaluation/datasets.py` loads them as unrelated tasks and
`bench.py` evaluates them independently, so the model never learns that a firm's 1-year and
5-year default probabilities are the same firm's, monotonically related, and jointly
constrained.

## What

- A multi-horizon output: predict a **discrete-time hazard path** (h_1 … h_K) per firm, from
  which cumulative PD at any horizon follows, rather than one binary label.
- Enforce or at least measure **monotonicity** of cumulative PD across horizons. A model
  claiming a lower 5-year than 1-year default probability is incoherent, and that is
  checkable without any new data.
- Extend the financial prior to emit hazard paths rather than a single label, sampling a
  survival process instead of one Bernoulli draw.
- Report the term structure as the deliverable: PD at 12 and 24 months plus lifetime, which
  is what a provisioning model consumes.

## Non-goals

- **Not a time-series model.** Rows stay exchangeable across *firms*; the time axis lives in
  the target (a hazard path) and in trajectory features, not in a sequence the model decodes.
  This keeps the architecture tabular and keeps us out of Google's lane.
- Not full survival analysis with censoring theory in v1. Discrete-time hazard on a fixed
  horizon grid first, because that is what the datasets and the regulation both use.
- Not LGD or EAD. PD term structure only; expected credit loss needs all three, later.

## Falsified by

Cheapest test, needing no new modelling: check whether cumulative PD from the **existing**
per-horizon models is already monotone across the five UCI horizons. If independently
trained per-horizon models happen to produce a coherent, well-ordered term structure, then
joint prediction buys coherence we already have and the case rests on efficiency alone. If
they contradict each other — which is the expectation — that incoherence is the argument,
and it is a defect visible in the current benchmark output.

Second test: does a jointly trained multi-horizon head beat independent per-horizon models at
matched compute, on AUC *and* on term-structure coherence?

## Blocked by / blocks

- **Blocked by** `phase1-prior-ablation` for sequencing only; the falsification test above can
  run immediately against existing checkpoints.
- **Partly blocked by** `second-credit-panel` task 7.5, since V4FinBench's six horizons and
  real dates make it the proper testbed.
- **Blocks** nothing, but it is the most credible route to a product a regulated lender must
  buy rather than merely likes, because IFRS 9 makes the term structure mandatory.
