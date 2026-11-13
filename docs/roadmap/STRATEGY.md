# Strategy

**Revised 2026-09-08**, after reading the credit-risk and tabular-foundation-model literature
properly. The first version of this document survived nine hours. What changed is recorded in
"What this revision overturns" below, because a plan that quietly rewrites itself is not a
plan.

Opinionated on purpose. Where a claim rests on evidence, the evidence is in
`docs/results/FINDINGS.md`; where it rests on judgment, it says so.

**Checked against `FINDINGS` §195-§224 and `docs/paper/CLAIMS.md` on 2026-11-13.** The thesis
and the term-structure evidence stand. What moved since this file was written is the
general-tabular standing (now 903 Elo, 107th of 110 over all 51 TabArena-Lite datasets, §224,
"FinTFM with a trained linear readout"; the zero-shot entry was 94th of 95 on 27 datasets, §122)
and the credit comparison (§216: FinTFM is below logistic regression on all four admitted credit
datasets). Dated notes below mark each place. Two commercial entrants were removed from this
file by owner decision (2026-11-13), together with the points that rested on their own
publications.

## The thesis, in one sentence

**A coherent, auditable probability-of-default term structure for small and low-default
credit portfolios — pretrained entirely on synthetic company financials, and shipped with the
validation evidence a model-risk committee needs.**

**Revised again 2026-09-08 (second revision of the day) on the strength of measurement, not
argument.** The previous version led with accuracy and calibration. Both were measured and
both are *small* (`docs/results/FINDINGS.md` "Where we stand"): +0.033 AUC over logistic regression,
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
   field predicts this object at all (`docs/competition/LANDSCAPE.md`).
2. **Auditable provenance.** Nothing real reaches pretraining, checkable by grep and enforced
   in CI-in-waiting. (CI now exists, `.github/workflows/ci.yml`, but has no step that checks
   this; and since D17 it holds for FinTFM, not FinTFM-R.) Competitors training on real
   tables cannot retrofit this, and the leakage literature prices contamination at up to 32
   MAPE points (§1).
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
scale; ours is not there. (Superseded: §114 closed scale as a lever for this model; see
"What this revision overturns" below.)

(2026-11-13, §216: the accuracy gap has since been measured on credit data and is against us on
all four admitted credit datasets, FinTFM below logistic regression; §211: no inference or
pretraining lever tried on V4FinBench clears the bar of +0.0117 mean AP with 4 of 5 folds.
§169's partitioned contexts, +0.0375 mean AP to 0.2362 on 5 of 5 folds, remain the largest
zero-shot gain, against CatBoost's 0.4005, §167. §114 closed scale for this model.)

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
See `openspec/changes/calibration-mechanism`.

## What this revision overturns

| the first version said | the evidence says | source |
| --- | --- | --- |
| V4FinBench is "the dataset this project needs" | 1.1M rows and 131 features are exactly where TFMs lose to trees. It is a **validation instrument** for temporal work, not the target market | `FINDINGS` §9 |
| Phase 1 targets 10-50M parameters | **Withdrawn 2026-09-22.** Size is not the lever, now measured directly rather than inherited: 5.7x the parameters at *matched* tasks scores **-0.0049** (§114), volume is null at 5x (§93) and +0.0028 at 2x (§114), and a peer's published curve returns **+0.0049 R² for 16.7x** against our 0.035 deficit. The deep-narrow alternative could not complete the benchmark at all. Three independent lines, and "we are 200x under the field norm" does not survive any of them | `FINDINGS` §9, §93, §110, §114 |
| Beat gradient boosting on the panels we hold | Our panels (6,000-10,500 rows) sit **at or above** the crossover. Every measurement so far was taken in the regime we lose | `FINDINGS` §9 |
| The product is single-horizon PD plus a certificate | Single-horizon PD does not satisfy IFRS 9. The term structure is the object, and no camp predicts it | `FINDINGS` §9, `docs/competition/LANDSCAPE.md` |
| The exit condition is "financial beats generic" | A win count across cells manufactures winners. Only 22 of 406 pairwise comparisons were significant in this domain | `FINDINGS` §9 |

## Why the obvious framings are dead

Every *mechanism* claim is owned (`docs/competition/LANDSCAPE.md`): "no feature engineering" by Kumo in
AP and Forbes, "in-context, no training on your data" by KumoRFM in NVIDIA's own
documentation, synthetic-prior TFMs by Google TabFM and Prior Labs, a tabular foundation model
for financial crime by Feedzai (announced 2026-03-24). Competing on mechanism means competing on
capital and distribution against a BigQuery integration and an NVIDIA documentation listing. Do
not pick that fight. (The NVIDIA listing is a 2026-09-08 reading, not re-confirmed 2026-11-13;
`docs/competition/LANDSCAPE.md`.)

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
| tabular | a label for one row | TabPFN, TabFM, RiskFM |
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

A point estimate driving a pricing decision, with no interval, no horizon and no refusal path,
is the object this project argues is insufficient: a model-risk function cannot validate a
single 6% or provision against it. That distance, between a bare number and something a
model-risk function can validate, is the product. (2026-11-13: the earlier paragraph here cited
a competitor's own page for it; the point stands without the citation.)

`finkele-axiom` transfers directly here: a validation protocol producing a certificate with
conformal coverage and honest refusals is the same machinery pointed at a different domain.

## IFRS 9 framing: what lifetime ECL needs, what fintfm does not supply, what it does

Written 2026-11-11 for `openspec/changes/pd-term-structure` task 11.6, to be read before any
customer conversation. **Gaps first.** Every capability below carries its `FINDINGS` section
and its label (MEASURED, PROVED, ESTIMATED, SMOKE-TEST). Where a number was measured before
the §54 column-identity fix, it is marked, because §56 lists the survival and hazard work of
§26-§40 for re-measurement.

### What IFRS 9 requires

Source: IFRS Foundation, *IFRS 9 Financial Instruments, Project Summary*, July 2014
(`https://www.ifrs.org/content/dam/ifrs/project/fi-impairment/ifrs-standard/published-documents/project-summary-july-2014.pdf`,
pp. 15-19). The full standard text is behind an IFRS Digital subscription and was **not**
read, so this section cites no paragraph numbers. Read the standard itself before quoting a
paragraph to a customer.

- **Expected credit loss (ECL), not incurred loss.** Credit losses are "the present value of
  all cash shortfalls"; ECL is an estimate of them over the life of the instrument.
- **Two measurement bases, set by staging.** Stage 1: 12-month ECL from origination. Stage 2:
  lifetime ECL once credit risk has increased significantly since initial recognition (and
  the asset is not low credit risk). Stage 3: credit-impaired, lifetime ECL continues.
- **12-month ECL is not a 12-month cash shortfall.** It is the lifetime ECL from default events
  possible in the 12 months after the reporting date, "weighted by the probability that this
  loss will occur in the next 12 months".
- **Significant increase in credit risk (SICR)** is judged on the likelihood or risk of default
  since initial recognition. The summary says IFRS 9 "does not mandate the use of an explicit
  probability of default" for this, and describes a rebuttable presumption at more than 30
  days past due as a backstop.
- **The estimate must reflect** (a) a probability-weighted outcome, neither best nor worst
  case; (b) the time value of money, discounted to the reporting date; and (c) reasonable and
  supportable information, available without undue cost or effort, covering past events,
  current conditions and forecasts of future conditions.
- **No prescribed method.** "IFRS 9 does not prescribe particular measurement methods."
  PD x LGD x EAD is the usual way lenders build the number, not a requirement of the
  standard. The summary does not name the discount rate; discounting at the effective
  interest rate is **to be confirmed against the standard text**, not asserted here.

### What fintfm does not supply

1. **No LGD and no EAD.** `docs/paper/LIMITATIONS.md` ("ECL needs three inputs") states it.
   A binned regression head now exists (§106 on synthetic probes; §121 on 12 TabArena
   regression datasets, where it ranks last on 5 of 12, mean rank 93.1, MEASURED). **No LGD
   or EAD panel has been scored**, so there is no LGD or EAD number to offer. (2026-11-13, §220:
   a ridge readout on `v4-regression.pt` beats §121's entry on 12 of 12 paired TabArena-Lite
   regression datasets, geometric-mean RMSE ratio 0.832; §223 gives 578 Elo on 13. These are
   general regression datasets, not LGD or EAD.)
2. **No discounting.** The model has no cash-flow schedule, contractual terms, repayment
   profile or effective interest rate. It emits a probability curve, not a present value.
3. **No ECL.** Nothing here multiplies PD, LGD, EAD and a discount factor. Any statement that
   fintfm "computes IFRS 9 ECL" is false. It supplies a PD input.
4. **No staging and no SICR test.** SICR compares credit risk now with credit risk at
   origination. The model scores a row as presented and stores no origination PD. Staging
   rules, the 30-day backstop and low-credit-risk exemptions sit outside it.
5. **No forward-looking scenarios.** There are no macroeconomic inputs and no scenario
   weights. The output is one curve per row, not a probability-weighted set of scenario
   curves. How a lender would condition it on forecasts is unbuilt and unmeasured.
6. **Point-in-time versus through-the-cycle is undetermined.** `docs/paper/LIMITATIONS.md` lists it as
   **[verify]**. The prior samples a parametric macro regime, and the base-rate correction
   (D5) shifts the curve to the population's default rate, but which of the two the output
   corresponds to has not been tested. Do not describe it as either.
7. **A short, fixed horizon grid, not a lifetime.** The head predicts K discrete periods
   (`HazardHead(n_horizons=K)`). The panels are annual: UCI Polish has five horizons and
   V4FinBench six. Lifetime ECL on a multi-year amortising loan needs the curve to the
   instrument's maturity. The model supplies nothing beyond the grid it was trained on.
8. **The 12-month point is not yet mapped to the standard's definition.** The first grid
   point equals "PD over the next 12 months" only if the label is default within 12 months
   of the scoring date. That has not been checked against each panel's label definition.
9. **No valid real-data accuracy or calibration number for the term structure today.** The
   out-of-time results on V4FinBench (§26, §30, §32, §33) were measured through the §54
   encoder ceiling and §56 lists them for re-measurement. They are not quoted here. §26's
   headline level error was also our own bug (§28), and a hazard checkpoint serves the
   survival path only (§34).
10. **No IFRS 9 validation evidence.** Coverage and the certificate (Phase 3) are not built
    for the term structure. Nothing here has been reviewed by a model-risk function or an
    auditor.
11. **Marginal PD is not exposed.** `predict_term_structure` returns cumulative PD, shape
    `(n, n_horizons)`. Period PD is the difference of adjacent columns.

So the accurate sentence for a customer is: *fintfm produces a coherent cumulative PD curve
over a short annual grid, which is one input to a lifetime ECL calculation. It does not
produce LGD, EAD, discounting, staging or scenarios.* This is the sentence `docs/paper/LIMITATIONS.md`
already requires beside any use of the words "IFRS 9".

### What fintfm supplies

Each claim is tied to its evidence. All are about the PD term structure only.

| claim | evidence | label |
| --- | --- | --- |
| Cumulative PD is non-decreasing in the horizon for any weights, by construction (`F_k = 1 - prod(1 - h_j)`, hazards in (0, 0.999]). Public path: `FinancialTFMClassifier.predict_term_structure`; the base-rate correction keeps it monotone because the shift is strictly increasing | §20; `tests/test_hazard.py`, including weights scaled 500x | **PROVED** (structural) |
| 0 violations in 12,000 horizon steps after training on synthetic survival tasks | §20 | MEASURED, synthetic |
| The same guarantee held out of time on real data: 0.00% violations and 100% monotone curves, against 39.06% and 1.4% for per-horizon logistic regression (V4FinBench, 72,622 train rows to 2016, 47,378 test rows from 2017) | §26 | MEASURED, **before the §54 fix**; the coherence result is structural, the same run's AUC and ECE are not quotable |
| Independently trained per-horizon models contradict themselves: 11.0% of horizon steps and 39.4% of firms (60.6% fully monotone) on 3,151 held-out UCI Polish rows, five horizons, while the portfolio mean stays monotone | §11 | MEASURED, 5,000-step checkpoint, **before the §54 fix**; the UCI files have no firm identifiers, so this is not a nested panel |
| Joint hazard prediction costs no discrimination at matched total compute: AUC +0.0008 +/- 0.0025 over 3 seeds; at 5x compute the per-horizon baseline is ahead by 0.0084 +/- 0.0032 AUC and still gives incoherent curves (22.81% of steps violated) | §21 | MEASURED, synthetic survival tasks, 3 seeds, **before the §54 fix**; across-seed means only, Holm-corrected paired bootstrap still owed (task 11.8) |
| One model serves the whole curve, against five per-horizon models | §21 | MEASURED (a count, not a quality claim) |

What this table does not claim: that the curve is accurate or calibrated on any real book,
that it beats a calibrated per-horizon baseline on discrimination (at 5x compute it does
not, in synthetic tests), or that the hazard construction is new. Discrete-time hazard
models are decades old, and `docs/paper/CLAIMS.md` Claim 1 frames this as a measurement of
a live defect in the field's standard construction, not a new method.

### Open before the first customer conversation

- Re-measure the real-data term structure (§26-§33) on a post-§54 checkpoint, with the level
  printed beside the observed rate. Until then gap 9 stands.
- Read the standard text itself (IFRS Digital) and replace the summary-level citations with
  paragraph references, including the discount rate.
- Settle point-in-time versus through-the-cycle by test, then say so in `docs/paper/LIMITATIONS.md`.
- Score an LGD or EAD panel before the words "three inputs" are said in the present tense.

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
(2026-11-13: not started. Only task 6.2, the power calculation, is done, 2026-11-11,
`scripts/register_power.py`; 6.1, 6.3, 6.4 and 6.5 are open in
`openspec/changes/forward-prediction-register/tasks.md`, so the elapsed-time asset is still not
accruing.)
A pre-registered cohort with hashed, timestamped predictions.

- **Exit condition:** twelve months of elapsed, unfalsifiable record.
- Its entire value is elapsed time and it cannot be bought or back-dated, so the cost of
  delay is the whole asset. Begin task 6.1 in parallel with Phase 1.

### Demoted: the scaling curve
Was Phase 2. Beyond IID finds TFMs lose on large, wide, non-IID data, so scale does not buy
the regime we need. Worth measuring eventually to know the shape; no longer on the critical
path, and no longer the justification for renting NVIDIA.

## What we are deliberately not building

- A general tabular foundation model. Contested on every axis by better-funded teams. **This
  was a position when written; it is now measured, on two independent populations.** TabArena
  places this project 94th of 95 (`docs/results/FINDINGS.md` §122); a private, unpublished
  comparison on the TabBench benchmark places it at a mean rank of 15.1 of 16 against the
  same field on a differently-constructed 55-dataset population, and checked that this is not
  a preprocessing artefact -- fintfm loses to matched-preprocessing gradient-boosted trees by
  a mean AUC of −0.0815 (§134). Scale (§114) and depth (§123/§124/§126) were each tried and
  came back null. **The conclusion this
  supports is not "stop measuring general-tabular standing" -- `docs/roadmap/ROADMAP.md`'s
  Phase A' and Phase C were still worth running, because a differently-shaped decision layer
  (§130-§133) moved an unpublished exploratory variant up one position (§132, 94th to 93rd [§225: one place lower] of
  95) and prior diversity was untested (since run: Phase A' closed at §136, and Phase C's
  combined prior measured negative on V4FinBench, §141) -- it is
  "the credit-panel thesis below is the actual bet, and every general-tabular result is a
  calibration check on that bet's cost, not a competing goal in its own right."**
  (2026-11-13, §224: the standing since §122 is a linear readout on the network's
  representation, not the zero-shot model: 903 Elo, 107th of 110 over all 51 TabArena-Lite
  datasets, against 1,000 for RandomForest (default). By problem type: binary 974 on 30 datasets
  (§220), multiclass 865 on 8, regression 578 on 13 (§223, §224). Owner decision DR7.6
  (2026-11-11): hold the TabArena PR until multiclass and regression improve; the target is about
  1,000 overall Elo. The pools differ from §122's, so 94th of 95 and 107th of 110 are not
  comparable ranks.) (2026-11-13: an earlier sentence here cited a competitor's own
  published account of TFM scaling as corroboration of §112; it was removed with that
  competitor, and §112 and §114 stand on this project's own measurements.)
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

Status 2026-11-13: items 1 and 2 are done (Phase 1 exit condition A, §14, amended by §15;
coherence, §11); item 4 is done (V4FinBench ingested, §22); item 3 is open (task 6.1); item 5
(the two full papers) has no record in `FINDINGS` of being done, so it is unverified here.

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

- **The credit result may stay below logistic regression.** (Added 2026-11-13, §216.) On the
  four admitted credit datasets FinTFM is below logistic regression, and on V4FinBench it is
  below CatBoost (§167). The thesis does not depend on beating either on accuracy, but a
  buyer will ask, and the answer today is the one the evidence gives.
- **The prior may not transfer.** Unmeasured. Phase 1 exists to kill it fast. (Since measured:
  exit condition A met, §14, amended by §15; see Phase 1 above.)
- **The margins in this domain are small.** Baesens et al. found significance in 22 of 406
  comparisons. Even a positive Phase 1 may be a small effect, and a small effect cannot carry
  a pitch built on accuracy — which is another reason the certificate is the product.
- **Low-default-portfolio benefit is conjecture.** The field's own authoritative benchmark
  named it and did not test it (mean default rate 22%). We would be first, which is the
  opportunity and the risk in one sentence.
- **A funded general-tabular team verticalises into finance.** A go-to-market motion for
  them, not research. Only regulatory depth and elapsed track record answer it. (A judgement,
  not an observation of any named company; 2026-11-13.)
- **Data access is the binding constraint**, and it may cap what can be shown without capital.
- **Homogenisation is a real risk of the category**, not a talking point: if many lenders
  score with one model their failures correlate, which is systemic risk. Say it to a
  supervisor before they say it to us.
- **Solo founder against funded teams with distribution.** Unfixable head-on; the whole plan
  competes on axes where that asymmetry does not decide the outcome.
