# Strategy

**Revised 2026-09-08**, after reading the credit-risk and tabular-foundation-model literature
properly. The first version of this document survived nine hours. What changed is recorded in
"What this revision overturns" below, because a plan that quietly rewrites itself is not a
plan.

Opinionated on purpose. Where a claim rests on evidence, the evidence is in
`docs/FINDINGS.md`; where it rests on judgment, it says so.

## The thesis, in one sentence

**A coherent, auditable probability-of-default term structure for small and low-default
credit portfolios — pretrained entirely on synthetic company financials, and shipped with the
validation evidence a model-risk committee needs.**

**Revised again 2026-09-08 (second revision of the day) on the strength of measurement, not
argument.** The previous version led with accuracy and calibration. Both were measured and
both are *small* (`docs/FINDINGS.md` "Where we stand"): +0.033 AUC over logistic regression,
~2× calibration over a *calibrated* incumbent and only below ~250 rows, 1-2% Brier skill over
a feature-free predictor. A sophisticated buyer finds that ceiling in an afternoon.

**What is large is the incoherence.** 39% of firms receive a term structure where a longer
horizon carries a *lower* cumulative default probability, and the portfolio aggregate looks
monotone and conceals it entirely (§11). That is not a ranking imperfection; it is an
incoherent object that IFRS 9 lifetime expected credit loss consumes directly. So the thesis
now leads with it.

Four claims, ordered by how much evidence each carries:

1. **Coherence, and it is the strongest card — and it is free.** The object is a hazard path
   — PD at 12 months, 24 months, lifetime — because IFRS 9 requires lifetime ECL. Independent
   per-horizon models contradict themselves 39% of the time on real data (§11) and 12-29% on
   synthetic (§21), while a hazard parameterisation is monotone **by construction** (§20).
   Measured across three seeds (§21): at **equal compute** joint prediction costs nothing in
   discrimination (+0.0008 ± 0.0025) while eliminating incoherence entirely, using **one
   model instead of five**. Training five models at **5× compute** buys about **+0.008 AUC**
   and returns curves that contradict themselves for two firms in three — a poor trade for a
   regulated PD, and one that can now be *quantified* rather than asserted. No camp in the
   field predicts this object at all (`docs/LANDSCAPE.md`).
2. **Auditable provenance.** Nothing real reaches pretraining, checkable by grep and enforced
   in CI-in-waiting. Competitors training on real tables cannot retrofit this, and the
   leakage literature prices contamination at up to 32 MAPE points (§1).
3. **Small and low-default portfolios**, but for a sharper reason than accuracy. It is where
   the *incumbent's remedies fail*: post-hoc calibration needs held-out defaults and Platt
   scaling at n=100 dropped gradient boosting's AUC from 0.632 to 0.545 (§16), and
   per-horizon models multiply incoherence when each horizon has few events.
4. **Synthetic-only pretraining as the only clean route**, since firm-level financial data is
   licence-locked (§4, §8) — and pretraining is what buys calibration in the first place, 7×
   on Brier over an untrained model (§15).

**What we do not claim *yet*:** more accurate than gradient boosting, best-calibrated as a
standalone virtue, or calibration as a differentiator against other foundation models.

The distinction between "not yet" and "not ever" is load-bearing (`FINDINGS` "Small because
early, or small because structural?"). The accuracy gap is **provisional**: this checkpoint is
2.2M parameters at 5,000 steps against a 10-50M target, the prior was 3-5× too narrow until
the day these numbers were taken, AUC is flat across dataset sizes (the signature of a model
that cannot yet use more data), and **Baesens et al. found properly trained TFMs beating 29
competitors including tuned gradient boosting on real credit data.** The approach works at
scale; ours is not there.

What is *not* provisional is coherence, the metric-range problem, and calibration being a
method property rather than a domain one. Those are design and measurement facts that scale
does not touch — which is precisely why the thesis leads with the one of them that is a
product opportunity rather than a constraint.

**Stated at the strength the evidence supports** (`FINDINGS` §16, which tempers §12 and §13):

> Below a few hundred obligors we produce the best probability estimates available on a
> proper scoring rule, and at every portfolio size the best-calibrated ones. **Above roughly
> a thousand rows a calibrated gradient boosting is the better model overall, and we say so.**

The earlier 11.7× calibration figure was measured against an *uncalibrated* incumbent and
**must not be used**. Against a calibrated one the gap at n = 100 is about 2×. What survives:
best ECE at every size by 2-4×, and best Brier below ~250 rows.

One real asymmetry remains in our favour: post-hoc calibration needs held-out events, and
Platt scaling at n = 100 dropped gradient boosting's AUC from 0.632 to **0.545** — it damaged
the ranking it was fixing. Thin books punish the incumbent's remedy.

**One of those two questions is now answered, against us** (`FINDINGS` §14). Calibration is
*not* a property of the financial prior — pure financial is the worst-calibrated variant, and
calibration tracks prior *breadth* instead. So it is a property of the method, and TabPFN,
TabICL and TabFM very likely share it. **Calibration alone is not a differentiator.** The
defensible position is the combination: the domain prior for discrimination, a prior mixture
for calibration, and the certificate for evidence. A head-to-head against TabPFN is required
before any calibration claim is made against other foundation models.

Still open: whether a *calibrated* gradient boosting closes the gap in the small-n regime.
See `changes/calibration-mechanism`.

## What this revision overturns

| the first version said | the evidence says | source |
| --- | --- | --- |
| V4FinBench is "the dataset this project needs" | 1.1M rows and 131 features are exactly where TFMs lose to trees. It is a **validation instrument** for temporal work, not the target market | `FINDINGS` §9 |
| Phase 1 targets 10-50M parameters | **Withdrawn 2026-09-22.** Size is not the lever, now measured directly rather than inherited: 5.7x the parameters at *matched* tasks scores **-0.0049** (§114), volume is null at 5x (§93) and +0.0028 at 2x (§114), and a peer's published curve returns **+0.0049 R² for 16.7x** against our 0.035 deficit. The deep-narrow alternative could not complete the benchmark at all. Three independent lines, and "we are 200x under the field norm" does not survive any of them | `FINDINGS` §9, §93, §110, §114 |
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
**Exit condition A met 2026-09-08** (`FINDINGS` §14, **amended by §15**): domain-specific
pretraining transfers. The financial prior beats a generic one at matched compute, 3 of 6
cells significant after correction with none against, and the mixture at p = 0.7 is the
better operating point on Brier.

**But the untrained control reframed what that means** (§15). A random-weight model already
ranks at AUC 0.726, so **ranking is largely architectural**; what pretraining buys is
calibration, 7× on Brier and up to 130× on ECE. The domain prior adds a modest +0.031 AUC
over random init. On Brier the three trained variants sit within 0.7% of each other, so
meeting the exit condition matters less than the AUC table suggested.

Exit condition B was answered in §12: the model beats gradient boosting only below a few
hundred rows.

Remaining: three seeds, independent panels, and — newly more interesting than another prior
variant — an ablation of the architecture against a trivial baseline on the same normalised
features, since the architecture may be carrying more than the prior.

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

### Phase 2 — the term structure. **Promoted to the critical path.**
Hazard-path output, a survival process in the prior, monotonicity enforced or measured,
joint versus per-horizon at matched compute.

- **Exit condition:** joint prediction beats independent per-horizon models on AUC per
  horizon *and* on coherence, and the output is what an IFRS 9 provisioning calculation
  consumes.
- **Why it moved up:** §11 measured a 39% incoherence rate, which is an order of magnitude
  larger than any accuracy or calibration edge this project has measured. It is also the only
  finding where the gap is structural rather than incremental — nothing in the architecture
  or loss forbids a violation, so no amount of training or scale closes it.

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

## Two structural plays, borrowed as inspiration rather than method

Neither of these is about tabular models, and neither involves using anyone's code. Both are
about *how a small team competes with a frontier lab*, which is our actual situation.

**Compete on cost and access, not capability.** FinGPT's published benchmark leads on price,
not quality: 0.882 weighted F1 for **$17** of GPU time on one consumer card, against
BloombergGPT's **$2.67M** and Llama-2-7B's **$4.23M**. They did not beat Bloomberg by having
more compute; they made the compute irrelevant. Our position is the same shape — Google
pretrains TabFM on hundreds of millions of synthetic datasets and we have a laptop GPU — so
"as good, vastly cheaper, and deployable" is a more winnable claim than "better".

**Out-of-time evaluation is rarer than it should be, even at the top.** BloombergGPT's corpus
is timestamped 2007-2022 and the paper states plainly that "we do not utilize date
information in this work", deferring temporal evaluation to future work. FinCast evaluates
zero-shot but splits 7:1:2 rather than by date. A field where the best-resourced model had
forty years of timestamped data and did not split on it is a field where **"validated out of
time, across the 2008 and 2020 regimes" is a differentiator rather than table stakes** — and
it is precisely what a supervisory validator asks for first.

**Deployability is the gap the frontier labs are opening themselves** (`FINDINGS` §24).
Google ships Apache-2.0 code with non-commercial weights for both TabFM and TimesFM 3.0, and
TimesFM's weights were permissive through 2.5 and are not at 3.0. A regulated lender
therefore **cannot deploy the best available models at all**. That is not a quality gap and
no amount of their accuracy work closes it. It is the same opening FinGPT took against
Bloomberg's privileged data and closed APIs.

Combined with the coherence guarantee, the honest pitch shape is: *a model a lender can
actually run in production, whose PD curve cannot contradict itself, with the evidence to
show a validator* — rather than a claim to be more accurate than Google.

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
