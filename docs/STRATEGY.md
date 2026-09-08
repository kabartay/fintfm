# Strategy

**Revised 2026-09-08**, after reading the credit-risk and tabular-foundation-model literature
properly. The first version of this document survived nine hours. What changed is recorded in
"What this revision overturns" below, because a plan that quietly rewrites itself is not a
plan.

Opinionated on purpose. Where a claim rests on evidence, the evidence is in
`docs/FINDINGS.md`; where it rests on judgment, it says so.

## The thesis, in one sentence

**A probability-of-default term-structure model for small and low-default credit portfolios,
pretrained entirely on synthetic company financials, shipping calibrated hazard paths with
their own validation evidence.**

Four claims, each load-bearing and each falsifiable:

1. **Term structure, not a label.** The object is a hazard path — PD at 12 months, 24 months,
   lifetime — because IFRS 9 requires lifetime expected credit loss. A single-horizon
   classifier is structurally insufficient for the rule every regulated lender is bound by.
2. **Small and low-default portfolios.** Tabular foundation models measurably beat gradient
   boosting below roughly 8,000 observations, and the advantage grows as data shrinks.
   Above that, they lose.
3. **Synthetic-only pretraining.** Not thrift: firm-level financial data sits behind
   commercial licences, so it is the only licence-clean route, and it makes benchmark
   contamination impossible to commit rather than merely unlikely.
4. **The evidence is the product.** Calibration, conformal coverage, honest refusal, and a
   forward track record — because in this market the certificate is what is bought.

**Stated as a frontier rather than a win** (`FINDINGS` §12, §13). At roughly 100 obligors we
dominate gradient boosting on both discrimination and calibration, a genuine Pareto
improvement. Above ~500 it is a trade-off: they rank better, we calibrate 2-6× better. For
provisioning and low-default portfolios the level is the deliverable and a lender cannot
provision against an ordering — but claiming only the favourable half is the failure this
project's own conventions exist to prevent.

**Two things must be established before any pitch rests on calibration**, and neither is:
whether a *calibrated* gradient boosting closes the gap, and whether the property is generic
to prior-fitted networks rather than ours. See `changes/calibration-mechanism`.

## What this revision overturns

| the first version said | the evidence says | source |
| --- | --- | --- |
| V4FinBench is "the dataset this project needs" | 1.1M rows and 131 features are exactly where TFMs lose to trees. It is a **validation instrument** for temporal work, not the target market | `FINDINGS` §9 |
| Phase 1 targets 10-50M parameters | Size is not the lever. Beyond IID finds TFMs lose on large/wide/non-IID data regardless of scale | `FINDINGS` §9 |
| Beat gradient boosting on the panels we hold | Our panels (6,000-10,500 rows) sit **at or above** the crossover. Every measurement so far was taken in the regime we lose | `FINDINGS` §9 |
| The product is single-horizon PD plus a certificate | Single-horizon PD does not satisfy IFRS 9. The term structure is the object, and no camp predicts it | `FINDINGS` §9, `LANDSCAPE.md` |
| The exit condition is "financial beats generic" | A win count across cells manufactures winners. Only 22 of 406 pairwise comparisons were significant in this domain | `FINDINGS` §9 |

## Why the obvious framings are dead

Every *mechanism* claim is owned (`docs/LANDSCAPE.md`): "no feature engineering" by Kumo in
AP and Forbes, "in-context, no training on your data" by KumoRFM in NVIDIA's own
documentation, synthetic-prior TFMs by Google TabFM and Prior Labs and Neuralk, "universal
tabular FM for financial risk" by Feedzai. Competing on mechanism means competing on capital
and distribution against a $255M Series A and a BigQuery integration. Do not pick that fight.

Every *size* claim is also dead. Google ships TabFM into BigQuery for warehouse-scale
tables and TimesFM-3 at 330M parameters over 10^12 time points. The large end is not
winnable, and Beyond IID says it is not winnable by a TFM by anyone.

What is left is a specific object, in a specific regime, with specific evidence. That is
narrower than "a foundation model for financial data" and it is the only version that
survives contact with the literature.

## The gap, stated precisely

Sort the field by *what object each model predicts* and the hole is obvious:

| camp | object | who |
| --- | --- | --- |
| time series | future values of a sequence | TimesFM-3, Chronos, t0-alpha, Toto |
| tabular | a label for one row | TabPFN, TabFM, NEXUS, Seldon, RiskFM |
| relational | a label for a node in a graph | KumoRFM, GraphPFN |
| **panel hazard** | **an event-probability path per entity over time** | **nobody** |

Time-series models forecast the covariates, not the event, and have no notion of an absorbing
state or a cumulative probability that must not decrease. Tabular models treat horizons as
unrelated tasks with no coherence constraint. Graph models still emit a label.

**And the datasets already carry it.** UCI Polish ships five horizons, V4FinBench six, and we
have been evaluating them independently — which means our own output is probably already
internally incoherent, and that is checkable today with no new modelling
(`openspec/changes/pd-term-structure` task 11.1).

## Who this is for

Not a data scientist wanting better AutoML; that buyer compares AUC and is served by
gradient boosting.

**A credit risk function under supervisory obligation, with a small book.** SME lenders,
specialty and trade-credit finance, regional banks, credit insurers, and anyone managing a
**low-default portfolio** — a named Basel category where a handful of defaults makes point
estimation unreliable by any method, so supervisors demand uncertainty and conservatism.
That is the one place where the hardest statistics and the highest willingness to pay for a
certificate coincide.

The incumbent is not gradient boosting alone. Baesens et al. name the quasi-standard as
**gradient boosting paired with SHAP**, so the comparison includes explanations, and a model
that wins on AUC but cannot say why loses anyway.

**A competitor states the case for us.** Neuralk's financial services page opens by naming
the two hard constraints as extreme accuracy *and tight regulatory scrutiny*, then addresses
neither calibration, validation, PD term structure, IFRS 9, Basel nor low-default portfolios.
Their own demo shows a bare "Default risk 6%" with "Approve at 4.9% APR" attached — a point
estimate driving a pricing decision, with no interval, no horizon and no refusal path
(`docs/LANDSCAPE.md`). The distance between naming regulatory scrutiny and shipping something
a model-risk function can validate is the product.

`finkele-axiom` transfers directly here: a validation protocol producing a certificate with
conformal coverage and honest refusals is the same machinery pointed at a different domain.

## Phases, each with a falsifiable exit condition

### Phase 0 — foundations. **Done.**
Priors, permutation-invariant architecture, training, in-context classifier, two real panels,
calibration metrics, base-rate correction, paired significance testing, untrained control,
sample-efficiency probe. 35 tests.

### Phase 1 — does the prior transfer, and in which regime?
Two questions, not one, and the second was missing until today.

- **Exit condition A:** the financial prior beats a generic one at matched compute, and the
  difference survives the paired bootstrap with family-wise correction. A win count is not a
  verdict.
- **Exit condition B:** the model beats gradient boosting **somewhere on the size sweep** —
  most plausibly below 1,000 rows. If it loses at every size, the small-data thesis is dead
  regardless of what the prior ablation says.
- **The untrained control decides how to read a tie.** "The domain prior adds nothing" and
  "no pretraining adds anything" are different results and only the control separates them.
- **If A fails but B holds:** keep the model, drop the domain-prior story, compete on
  calibration and the certificate.
- **If B fails:** the model is not the product. Pivot to the validation layer, which needs no
  model of our own, and say so publicly.

### Phase 1.5 — term-structure coherence. **Done 2026-09-08. Exit condition met.**
Measured on 3,151 held-out rows across all five horizons (`FINDINGS` §11):

- **11.0% violation rate per step; only 60.6% of firms get a fully monotone curve.**
- Mean predicted PD *is* monotone at every step, so a portfolio-level report looks correct
  and conceals it entirely. That is why the diagnostic must be permanent.
- Two firms in five therefore receive a term structure where a longer horizon carries a
  lower default probability, which an IFRS 9 lifetime ECL calculation consumes directly.
- **Nothing in the architecture or the loss forbids a violation**, so training cannot drive
  this to zero. The term-structure direction is founded rather than speculative.

### Phase 2 — the term structure
Hazard-path output, a survival process in the prior, monotonicity enforced or measured,
joint versus per-horizon at matched compute.

- **Exit condition:** joint prediction beats independent per-horizon models on AUC per
  horizon *and* on coherence, and the output is what an IFRS 9 provisioning calculation
  consumes.

### Phase 3 — the certificate
Conformal PD intervals, coverage under regime shift, out-of-distribution refusal, and the
supervisory coverage tests the finance literature uses: Kupiec unconditional coverage,
Christoffersen conditional coverage.

- **Exit condition:** a document a model-risk reviewer reads without us in the room.
- This is the product. Everything above exists to make it cheap to produce.

### Phase 4 — attribution
PD and feature attributions from one forward pass, to displace GBM+SHAP rather than half of it.

- **Exit condition:** rank agreement with SHAP-on-GBM where both apply, plus a stability
  advantage across context resamples.

### Phase 5 — the forward register. **Start now, out of order.**
A pre-registered cohort with hashed, timestamped predictions.

- **Exit condition:** twelve months of elapsed, unfalsifiable record.
- Its entire value is elapsed time and it cannot be bought or back-dated, so the cost of
  delay is the whole asset. Begin task 6.1 in parallel with Phase 1.

### Demoted: the scaling curve
Was Phase 2. Beyond IID finds TFMs lose on large, wide, non-IID data, so scale does not buy
the regime we need. Worth measuring eventually to know the shape; no longer on the critical
path, and no longer the justification for renting NVIDIA.

## What we are deliberately not building

- A general tabular foundation model. Contested on every axis by better-funded teams.
- Fraud. Owned by Kumo and Feedzai, and needs real-time serving infrastructure.
- Time-series forecasting. Google has 330M parameters and 10^12 time points in BigQuery.
- Market or return prediction. Harder science, severe validation traps, sophisticated
  proprietary competition.
- Systemic risk and contagion networks. A different problem (and a graph one).
- Anything with a natural-language interface, an agent, or a warehouse connector. That is a
  distribution game.
- **Accuracy claims on large national panels.** The literature says we lose there. Do not
  publish a headline that invites the comparison.

## How it kicks off

1. **Finish Phase 1** — the run in flight, then the sample-efficiency probe on the winning
   checkpoint. The probe matters more than the ablation, because it tests the regime the
   thesis needs.
2. **Run the Phase 1.5 coherence test.** Hours of work, no training, and it either founds or
   deflates the term-structure direction.
3. **Start the forward register** (task 6.1: establish a licensed source with observable
   outcomes). Parallel, because it only compounds.
4. **Get V4FinBench** for temporal validation and regime-shift testing — not to chase
   accuracy on it.
5. **Read two full papers**: Baesens et al. for the exact size crossover and which TFMs, and
   ExplainerPFN for how attribution targets are generated. Both are currently known only from
   abstracts, and both bear directly on scope.

Everything before a Phase 1 result is positioning, and positioning is not evidence.

## The honest risk register

- **The prior may not transfer.** Unmeasured. Phase 1 exists to kill it fast.
- **The margins in this domain are small.** Baesens et al. found significance in 22 of 406
  comparisons. Even a positive Phase 1 may be a small effect, and a small effect cannot carry
  a pitch built on accuracy — which is another reason the certificate is the product.
- **Low-default-portfolio benefit is conjecture.** The field's own authoritative benchmark
  named it and did not test it (mean default rate 22%). We would be first, which is the
  opportunity and the risk in one sentence.
- **Fundamental verticalises into finance.** A go-to-market motion for them, not research.
  Only regulatory depth and elapsed track record answer it.
- **Data access is the binding constraint**, and it may cap what can be shown without capital.
- **Homogenisation is a real risk of the category**, not a talking point: if many lenders
  score with one model their failures correlate, which is systemic risk. Say it to a
  supervisor before they say it to us.
- **Solo founder against funded teams with distribution.** Unfixable head-on; the whole plan
  competes on axes where that asymmetry does not decide the outcome.
