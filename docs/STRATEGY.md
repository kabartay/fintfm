# Strategy

Written 2026-09-08, after a day of surveying the field. This is the plan of record. It is
opinionated on purpose: a plan that keeps every option open is not a plan. Where a claim
here rests on evidence, the evidence is in `docs/FINDINGS.md`; where it rests on judgment,
the judgment is labelled as such.

## The one-sentence thesis

**A credit-risk model that arrives with its own validation evidence, built on a tabular
foundation model pretrained entirely on synthetic company financials.**

The foundation model is *how* it is built cheaply. The evidence is *what* is sold.

## Why not the obvious framing

The obvious framing — "a tabular foundation model for finance, no training on your data" — is
dead on arrival, and it took one day of reading to establish why (`docs/LANDSCAPE.md`):

| the pitch | who already owns it |
| --- | --- |
| "no feature engineering" | Kumo, in AP and Forbes |
| "in-context, no training on your data" | KumoRFM, documented on docs.nvidia.com |
| "tabular foundation model, synthetic priors" | Google TabFM, Neuralk, Prior Labs |
| "universal tabular FM for financial risk" | Feedzai RiskFM |
| vertical specialisation of a general TFM | Fundamental, already shipping into oil & gas |

Every mechanism claim is taken. Competing on mechanism means competing on capital and
distribution against a company with a $255M Series A and another inside NVIDIA's docs. That
is a losing fight and should not be picked.

## What is actually defensible

Three things, in increasing order of durability.

**1. A financial prior, not a generic one.** Weak on its own — Kumo and TabFM also train on
synthetic data, so this is a difference of degree. It has to be *demonstrated* on credit
panels, never asserted. But there is a real asymmetry underneath it (§4 of FINDINGS):
firm-level financial data is absent from every major foundation model's pretraining corpus
because it sits behind Bloomberg, S&P and Moody's. Energy got foundation models because its
data was free. Finance did not. **Synthetic generation is not a budget substitute here; it is
the only licence-clean route into the domain**, which is why the space is empty.

**2. Auditable provenance.** Nothing real touches pretraining, verified by inspection and
enforced as an invariant (§1). Every competitor training on real tables has public-benchmark
numbers open to the contamination critique that Meyer et al. quantify at up to 32 points of
MAPE. A model that *cannot* have memorised the benchmark is a claim an auditor can check.
This is cheap to hold and expensive to acquire later.

**3. A pre-registered forward track record.** The only evaluation immune to both leakage
modes is predicting outcomes that do not yet exist (§3). Corporate credit suits this better
than forecasting does: the horizon is already 12-24 months, defaults are publicly observable
in insolvency registers, and the cohort can be fixed today from public filings. **A track
record cannot be bought or back-dated — it has to be lived.** This is the one axis where
starting now beats being funded later, and it is the reason to start the clock before the
model is good.

## Who this is for

**Not** a data scientist who wants a better AutoML. That buyer compares AUC, is served by
gradient boosting, and has no reason to switch.

**The buyer is a credit risk function under supervisory obligation** — a bank, a lender, an
insurer, a rating or trade-credit business. Their binding constraint is not accuracy. Their
scorecards are logistic regression *because regulators demand interpretability*, which is why
Fundamental's published benchmark beats linear regression rather than gradient boosting (§2).
The accuracy bar is low. **The barrier is model risk management**: calibrated PD, stability
across regimes, documented out-of-time backtesting, and evidence a validation committee
accepts. Vendors selling "upload data, get predictions" structurally cannot ship that,
because there the certificate is an afterthought and here it is the product.

This is also where `finkele-axiom` transfers directly: a validation protocol producing a
certificate with conformal coverage and honest refusals is the same machinery pointed at a
different domain.

## What we are building, in order

Each phase has a **falsifiable exit condition**. If the condition fails, that is a result, not
a failure — and the next phase changes.

### Phase 0 — foundations (done, 2026-09-08)
Priors, permutation-invariant architecture, training loop, sklearn-compatible in-context
classifier, real credit evaluation on UCI Polish bankruptcy, calibration metrics, prior
correction. 24 tests. **Exit: reached.**

### Phase 1 — does the prior transfer at all?
The single experiment that decides whether this is a company.

Pretrain at real scale (millions of synthetic tasks, 10-50M parameters) and measure on
held-out real credit panels, with time-based splits, against logistic regression, gradient
boosting, LightGBM and CatBoost.

- **Exit condition:** the financial prior beats a generic SCM-only prior of identical size and
  compute on credit tasks, and the model is within reach of gradient boosting on AUC while
  beating it on calibration.
- **Harness: built and tested** (`src/fintfm/experiments.py`, `fintfm-ablate`). It trains one
  model per prior mixture holding architecture, parameter count, optimiser, steps, batch,
  seed and evaluation identical, refuses to report if parameter counts diverge, scores every
  variant on the same paired splits, and writes `results.json` with the git commit so any
  number is re-derivable. The run is one command:

  ```bash
  uv run fintfm-ablate --steps 20000 --out runs/phase1 --threads 8
  ```
- **If the financial prior does *not* beat the generic one**, the domain-specialisation thesis
  is dead and the honest move is to say so publicly and pivot to the validation layer alone,
  which does not require owning a model at all.
- **Blocked on:** compute, not code. This machine shares 16 cores with a genomics pipeline
  and was at load average 145 when the harness was finished (2026-09-08), which is the
  condition `CLAUDE.md` records as having frozen it before. Run on a rented GPU, or on this
  machine once the pipeline is idle and with `--threads` set below the free core count.

### Phase 2 — does it scale?
Train at 100k, 1M, 10M synthetic tasks. Plot real-data performance against pretraining scale.

- **Exit condition:** a monotone, non-saturating curve. That is the scientific result worth
  publishing and the thing that makes this a foundation model rather than a neural network
  with good marketing.
- **If flat:** stop scaling, spend the compute on prior richness instead.

### Phase 3 — the certificate
Port the axiom protocol: conformal PD intervals, coverage under regime shift, out-of-
distribution detection with refusal, calibration stability across economic cycles.

- **Exit condition:** a document a model-risk reviewer reads without needing us in the room.
- This is the product. Phases 1-2 exist to make it cheap to produce.

### Phase 4 — the forward register
Fix a public cohort, publish hashed timestamped PD predictions, wait.

- **Exit condition:** twelve months of elapsed, unfalsifiable track record.
- **Start it during Phase 1, not after.** It costs almost nothing and only time makes it
  valuable. Sizing matters: at a few percent base rate, a few thousand firms yield only tens
  of defaults a year, so state the statistical power up front rather than discovering later
  that a null result was underpowered.

## What we are deliberately not building

- **A general tabular foundation model.** Contested by better-funded teams on every axis.
- **Fraud.** Kumo and Feedzai own it, and it needs real-time serving infrastructure we would
  have to rebuild from nothing.
- **Market or trading prediction.** Harder science, severe temporal-validation traps, and
  sophisticated proprietary competition.
- **Time-series forecasting.** A different architecture family. The Forecasting Company's
  argument that time series are not tables is correct, and corporate default on an annual
  panel is genuinely a tabular problem — but only while we stay on that side of the line.
- **Anything with a natural-language interface, an agent, or a warehouse connector**, until
  the model is worth querying. That is Kumo's game and it is a distribution game.

## How it kicks off

The next action is unchanged and singular: **run Phase 1**. Everything else is preparation
that has now been done.

Concretely, in order:

1. Rent GPU time. A single A100 for a few days is enough for a 10-50M-parameter model on
   millions of synthetic tasks; this does not need a cluster.
2. Run the generic-versus-financial prior comparison at matched compute. **This is the
   experiment that decides whether the company exists**, so run it before anything cosmetic.
3. Acquire a second, independent credit panel so results do not rest on one dataset. Check
   licences for commercial use before ingesting anything (`CLAUDE.md`).
4. Start the forward register in parallel, because its value is purely a function of elapsed
   time.

Everything before a Phase 1 result is positioning, and positioning is not evidence.

## The honest risk register

- **The prior may not transfer.** The core scientific bet, unmeasured. Phase 1 exit condition
  exists to kill it fast rather than slowly.
- **Fundamental verticalises into finance.** Entering is a go-to-market motion for them, not
  research. Mitigated only by regulatory depth and an accumulated track record, never by
  being first.
- **Data access is the binding constraint.** Good public credit panels are few and the good
  ones are licensed. This may cap what can be demonstrated without capital.
- **Homogenisation is a real risk of the category, not just a talking point.** Bommasani et
  al. warn that a foundation model's defects are inherited downstream; if many lenders score
  with one model their failures correlate, which is systemic risk. Say this to regulators
  before they say it to us — a vendor who has thought about it is more credible, and it is
  also the argument for why per-deployment validation must exist.
- **Solo founder against funded teams with distribution.** Unfixable head-on. The whole plan
  above is an attempt to compete on an axis where that asymmetry does not decide the outcome.
