# Findings

Numbered, dated, with the source or command that produced each. A finding that lives only in
a conversation is lost when the conversation compacts.

## Where we stand — the honest scorecard

Updated 2026-09-10 after forty-two findings. **Read this before quoting any number below**,
because several findings temper, amend or outright retract earlier ones and the amendments
matter more than the originals. §28 retracts §26's headline as our own bug.

### Read this first

**§42 puts every accuracy number in this repository in question.** Against an untrained model of
the same architecture, pretraining is worth **+0.015 AUC** on a clean linear task that logistic
regression solves at 0.9997; four times the context buys 0.014; and the training loop's only
quality signal was *accuracy*, which a constant predictor beats at these base rates. The likely
cause is that the prior was deliberately tuned to be hard (§19) and so contains almost no
learnable tasks. Coherence (§11, §20, §26) is structural and unaffected. Everything about
accuracy below should be read as provisional until the prior and the training metric are fixed.

### What is measured and large

| claim | evidence | § |
| --- | --- | --- |
| **PD term structures are incoherent** — 39% of firms get a curve where a longer horizon carries *lower* default probability, while the portfolio aggregate looks monotone and hides it | measured, real data | §11 |
| **Pretraining buys calibration** — 7× on Brier, up to 130× on ECE against an untrained model that predicts a 39% default rate against a 4.7% base | measured, untrained control | §15 |
| **Firm-level financial data is licence-locked**, so synthetic pretraining is the only clean route into this domain, which is why the space is empty while energy is crowded | dataset census | §4, §8 |
| **The prior matches real task difficulty** — logistic-regression AUC 0.743 synthetic against 0.769 real | measured, held-out seed | §18, §19 |
| **Coherence transfers to real out-of-time data** — 0% violations against 39% for per-horizon models, 98.6% of firms affected | measured, V4FinBench | §26 |
| **A base-rate error is invisible to every guard we had** — a 27× level error passed AUC, passed the coherence check, and was not printed; the fix cut fourth-horizon ECE 32× | measured, same checkpoint | §28 |
| **Context construction dominates**: the best construction beats the worst by ~0.22 AUC out of time (0.8143 against 0.5938 for balanced), which is larger than any architectural change measured here | measured, 3 seeds | §29, §33, §35, §38 |
| ~~Query-conditioned retrieval is our accuracy contribution~~ — **retired**: the published prototype context matches it on AUC, beats it significantly at horizon 0, is 1.6× better calibrated, 4× cheaper, and stays batch-independent | measured, 3 seeds, paired bootstrap | §32 → §36, §38 |
| **Rank-transforming features is worth +0.086 AUC** — 110 of 136 features have a standard deviation above 10× their interquartile range, which the model's mean/std normalisation cannot survive | measured, 3 seeds, 3 panels | §35 |
| **Out-of-time mean AUC 0.5869 → 0.8143 in one day**, closing the gap to the incumbent from 0.142 to 0.047, from three inference-time changes and **no retraining** | measured | §35, §38 |
| **Balanced context sampling costs 10-12 AUC points on the survival path**, reversing the default adopted from the literature; 12 in-context defaults outrank 1,122 | measured, 3 context sizes | §29 |
| ~~The prior cannot generate the low-default regime~~ — **retracted**: the floor was 1% and is now 0.195%, but it was never what caused §26's failure | superseded | §26 → §28, §30 |

### What is measured and small

| claim | size | § |
| --- | --- | --- |
| Domain prior beats a generic one | **+0.049 AUC**, 3 of 6 cells significant | §14 |
| **Gap to a competent gradient booster** | **CatBoost 40-60% Brier skill against our 1-2%** | **§25** |
| Model beats logistic regression | **+0.033 AUC** | §17 |
| Model beats an untrained model of the same architecture | **+0.031 AUC** | §15 |
| Calibration advantage over a *calibrated* gradient boosting | **~2×**, and only below ~250 rows | §16 |
| Low-default retrain, under the configuration that works | **+0.023 AUC, 2.7× ECE** | §30 |
| Brier skill over a feature-free constant predictor | **1-2%** | §17 |

### What has been ruled out

- **"More accurate than gradient boosting."** False above a few hundred rows; above ~1,000 a
  calibrated gradient boosting wins on both AUC and Brier (§16).
- **"Best calibrated" as a standalone claim.** A constant base-rate predictor beats every
  model here on ECE, so calibration numbers cannot carry an argument alone (§17).
- **"Ahead of the incumbent out of time."** False. On V4FinBench out of time, per-horizon
  logistic regression leads on mean AUC 0.8616 to our best 0.8143 and on calibration by
  roughly fourfold. The gap is now **a third** of what it was this morning and it is not
  closed (§35). Only coherence favours us, and by construction.
- **Single-seed comparisons between context strategies.** Hybrid at 1,000 rows spreads ±0.046
  across seeds, wide enough that any single-draw comparison between blind strategies was
  never safe (§33). A single-seed sweep also manufactured a +0.008 "gain" from tightening
  retrieval groups that three seeds erased entirely (§38).
- **Calibration as a differentiator against other foundation models.** It tracks prior
  *breadth*, not our domain prior, so TabPFN and TabFM very likely share it (§14, §15).
- **The 11.7× calibration figure.** Measured against an uncalibrated baseline. Do not use it.
- **Scale as the lever.** Beyond IID finds TFMs lose on large, wide, non-IID data regardless
  of size (§9).

### Small because early, or small because structural?

**This distinction matters more than the sizes**, and an earlier version of this scorecard
ran the two together, which was itself a distortion. Not everything small is a ceiling.

**Likely to improve with scale and work** — treat these as provisional, not as verdicts:

- Every AUC comparison against gradient boosting and logistic regression. The checkpoint is
  **2.2M parameters at 5,000 steps** against a stated target of 10-50M, in a field that
  pretrains on hundreds of millions of synthetic datasets. **Baesens et al. found properly
  trained TFMs beating 29 competitors including tuned XGBoost, LightGBM and CatBoost on real
  credit data** (§9), and TabPFN v2 is in *Nature*. So the approach works at scale, and our
  failure to beat gradient boosting above ~500 rows is evidence about **this checkpoint**,
  not about the thesis.
- The crossover point (~100-250 rows here against ~8,000 reported by Baesens). AUC is **flat
  at 0.67-0.72 across every dataset size**, which is the signature of a model that cannot yet
  exploit more data rather than one at its ceiling (§13).
- Anything measured before 2026-09-08 evening: the prior was **3-5× too narrow** until then
  and the width fix's transfer impact is **unmeasured** (§19).
- Training loss was still declining when the 5,000-step run ended.

**Will not improve with scale** — these are design or measurement facts:

- **Term-structure incoherence (§11).** Nothing in the architecture or the loss forbids a
  violation, so no amount of training closes it. It needs a design change, which is exactly
  why it is now the plan rather than a complaint.
- **A constant predictor beating every model on ECE (§17).** A property of the metric at a
  4-7% base rate, not of the model.
- **Calibration tracking prior *breadth* rather than the domain prior (§14, §15).**
  Mechanistic, so TabPFN and TabFM plausibly share it however large ours gets.
- **TFMs losing on large, wide, non-IID data (§9).** A field-wide result measured at full
  scale by BeyondArena, not an artefact of our size.

### What this implies

**Lead with coherence, but not because the accuracy numbers are hopeless — because coherence
is where a *design* advantage exists rather than a scale one.** A scale advantage has to be
bought and defended against better-funded teams; a design advantage on an object nobody else
predicts does not. The accuracy work stays on the roadmap and is expected to improve; it just
should not be the pitch while it is unproven.

**And do not quote the small numbers as ceilings.** They are one under-trained checkpoint on
one panel family with a prior that was too narrow at the time. The honest form is "not yet
demonstrated", not "does not work". See `docs/STRATEGY.md`.

## 1. Synthetic-only pretraining is an auditability property, not just a cost saving

**Date:** 2026-09-08. **Status:** design invariant.

Pretrained models can score well on a benchmark by having memorised it. Meyer, Kaltenpoth,
Zalipski & Müller, *"Rethinking Evaluation in the Era of Time Series Foundation Models:
(Un)known Information Leakage Challenges"* (arXiv:2510.13654), traced the data lineage of 15
prominent time-series foundation models and measured the effect. **Verified from the paper**,
not from secondary coverage:

| condition | effect |
| --- | --- |
| Moirai, 0.1% test data contaminating pretraining, short horizon | 7.6 pp lower MAPE |
| same, medium horizon | 32 pp lower MAPE |
| same, long horizon | 29 pp lower MAPE |
| real-world case, models pretrained on leaking datasets | 47%–184% lower MSE than the best clean model |

Larger models are the more affected: *"bigger models tend to memorize rather than
generalize."* They identify **two** leakage modes — direct test-set contamination through
multi-purpose reuse of public datasets, and **memorisation of global patterns** induced by
external shocks (crises, pandemics) that correlate series across unrelated domains.

**Why this matters here.** Verified 2026-09-08 by inspection: `train.py` imports only
`fintfm.prior`, the two generators are pure NumPy, and the single real-data call
(`fetch_openml`) exists solely in `bench.py`, the evaluation harness. **No real dataset can
reach the pretraining path.**

```bash
grep -rn "fetch_openml\|read_csv\|requests\|urllib\|load_dataset" src/fintfm/   # bench.py only
```

So a benchmark result from this model cannot be inflated by memorisation, and that is
*checkable by a third party* rather than asserted. Competitors pretrain on real tables
(Fundamental: "billions of tables") or real relational data (Kumo: "real and synthetic"),
which leaves their public-benchmark numbers open to exactly this critique. In a market where
the deliverable is validation evidence (see `docs/LANDSCAPE.md`), this is a rare defensible
property.

### A published paper now uses provenance as a competitive claim (added 2026-09-09)

FinCast (arXiv:2508.19609 §4.1) excludes its benchmark from pretraining and then says of its
competitors:

> "existing general-purpose time series models may benefit from inadvertent overlap between
> their pretraining datasets and our benchmark, potentially inflating their performance due
> to information leakage."

That is this finding's argument, deployed in publication as a competitive differentiator
against better-resourced labs. It confirms provenance is a *claimable* advantage rather than
mere hygiene.

**And our version is strictly stronger.** FinCast trains on 20B real financial time points
and must therefore *curate* an exclusion, which is a promise about their pipeline that a
reader has to take on trust. This project trains on **no real data at all**, so contamination
is not excluded — it is impossible, and checkable by grep in seconds
(`openspec/tools/validate.py --provenance`). The distinction is between "we were careful"
and "we could not have done it even by accident", and only the second survives an adversarial
reviewer.

That is worth stating plainly in any external material, because it is one of the few claims
here that does not depend on model scale.

**The caveat, and it is the important half.** The paper's authors are only *cautiously*
optimistic about synthetic pretraining, and they name the tension this project's roadmap
walks straight into: *"as the generation process becomes more similar to actual historical
data, does the risk of information leakage or memorizing global patterns increase?"* They
also note synthetic data's external validity is questionable and may be reverse-engineerable.
Their own conclusion is appropriately hedged: contamination effect sizes are *"proven to be
significant in isolated cases so far"* and global-pattern memorisation magnitude is
*"largely unknown."* Do not overclaim this in a pitch — state it as a structural property of
the training regime, which is checkable, not as a proven accuracy advantage, which it is not.

**The invariant that follows.** `prior/financial.py` samples a macro regime (`rate`, `cycle`)
as a *parameter* rather than learning real crisis history. That is what preserves the
property. **Fitting the prior to real company panels to make it more realistic would destroy
it silently** — nothing would fail, the benchmarks would improve, and the auditability claim
would quietly become false. If realism ever has to be increased, do it by enriching the
generative structure (more accounting identities, more regimes), never by conditioning the
generator on a real dataset.

## 2. The enterprise competitive baseline is linear regression, not gradient boosting

**Date:** 2026-09-08. **Status:** MEASURED by others (Fundamental's own published figure);
our reading of what it implies is inference, not measurement. See `docs/LANDSCAPE.md`.

Fundamental's published oil & gas result reports NEXUS beating **linear regression** by 75%
MAE / 43% RMSE across 13 regional markets. That is the comparison a funded competitor chose
to publish, so it indicates what enterprise buyers actually replace. Bank credit scorecards
are logistic regression for regulatory-interpretability reasons, so the same holds in this
domain. Consequence for `bench.py`: keep both baselines and read them differently — logistic
regression is the *commercial* comparison, gradient boosting the *scientific* one. Reporting
only the flattering one is the failure this file exists to prevent.

## 3. A pre-registered forward prediction record is the only leakage-proof evidence, and it is buildable now

**Date:** 2026-09-08. **Status:** proposed strategy, not started. **Provenance:** reasoning
from Meyer et al. (arXiv:2510.13654) and the M6 design (Makridakis et al., 2024); no part of
this has been measured.

Finding §1 establishes that this model's benchmark numbers cannot be inflated by
memorisation, because nothing real reaches pretraining. That is a property of *this* model.
It does not solve the buyer's problem, which is harsher: **a model risk committee has no way
to distinguish a genuinely skilful model from a well-tuned backtest**, and every vendor
arrives holding excellent retrospective numbers.

Meyer et al. conclude that the design which escapes both leakage modes is evaluation on data
that did not exist when the prediction was made. M6 implemented it: live financial assets,
predictions registered into the real future. Their own criticism of it is administrative cost
and long waiting periods.

**Corporate credit suits this design better than forecasting does**, for four reasons:

1. The natural horizon is already 12–24 months, so the waiting period is the product's own
   prediction horizon rather than an artificial delay.
2. Outcomes are **publicly and unambiguously observable** — insolvency filings and company
   registers — so scoring needs no customer's private data.
3. The cohort can be fixed *today* from public company financials. Nothing has to be
   negotiated first.
4. It is the exact artefact supervisory validation asks for: genuine out-of-time performance,
   not a resampled backtest.

**Why this is a small team's asset rather than a large one's.** A track record cannot be
bought, back-dated, or acquired with a funding round; it has to be lived. A well-capitalised
competitor entering credit in a year starts its clock then. Starting the clock costs almost
nothing now — publish a hashed, timestamped cohort and its predictions — and the evidence
compounds monthly while everything else in `docs/LANDSCAPE.md` says distribution and capital
favour the incumbents. This is the one axis where being early beats being funded, and it
converts the current weaknesses (no customers, no proprietary data, no compute) into the one
asset nobody else can hold.

It is also the same discipline as `finkele-axiom`'s pre-registration freeze — commit the
splits, the config and the predictions *before* the outcome is knowable — applied to a market
where the certificate is the product.

**The honest costs, none of which are avoidable:**

- **It is slow.** The record says nothing for twelve months. It is a compounding asset, not a
  demo, and it cannot be the only evidence in the meantime.
- **Statistical power needs a large cohort.** Annual corporate default rates run low single
  digits, so a few thousand firms yield only tens of defaults in a year — enough to separate
  models that differ substantially, not enough to resolve small AUC differences. Size the
  cohort deliberately and state the power up front; a null result from an underpowered cohort
  is not evidence of parity.
- **It needs licensed company financials with observable outcomes.** Check commercial-use
  terms before ingesting anything (`CLAUDE.md`), and expect this to be the binding
  constraint.
- **The commitment mechanism must be independently checkable**, or it proves nothing. A
  public hash with a timestamp nobody can backdate — a signed git tag pushed to a public
  repository, or an OSF/arXiv registration — not a local file.

**Nothing here has been done.** No cohort exists, no predictions are registered, and the
model has not been pretrained. Recorded because the cheapest moment to start a clock is
always now, and because this is the one strategy identified this session that a funded
competitor cannot simply outspend.

## 4. Energy dominates TSFM pretraining because its data is free; finance is empty because its data is not

**Date:** 2026-09-08. **Status:** the absence is MEASURED (read directly from a published
table); the four-cause explanation for it is inference and labelled as such below.
**Source:** Table 5 of Meyer et al. (arXiv:2510.13654), which catalogues pretraining (P),
train/test (T/T) and zero-shot (ZS) dataset use across 15 TSFMs.

**The fact.** Energy is the most heavily represented domain in TSFM pretraining corpora.
BuildingsBench alone contributes ~10 energy datasets (BDG-2 Bear/Fox/Panther/Rat, Borealis,
Buildings900K, IDEAL, Low Carbon London, Sceaux, SMART), marked `P` across Moirai,
Moirai-MoE and Time-MoE; ETT1/ETT2 appear in nearly every model in the table.

**Why (inference, four compounding causes):**

1. **Volume and regularity.** Smart meters emit gap-free series at 15-minute or hourly
   resolution; one utility rollout yields millions of well-formed series.
2. **It is publicly releasable.** Consumption data is far less sensitive than health or
   financial records, and governments fund open trials (Low Carbon London) or simulate stock
   at scale (Buildings900K). The open data exists because releasing it harms nobody.
3. **Strong learnable structure.** Demand is driven by temperature, daylight and human
   schedules — clean daily/weekly/annual seasonality, ideal pretraining signal.
4. **Benchmark path dependence.** ETT (Electricity Transformer Temperature) came from the
   Informer paper and became *the* long-sequence benchmark, so every subsequent model
   included it — convergence by momentum rather than merit.

**The finding that matters here is the absence.** The `Econ/Fin` rows in that table are Air
Passengers, Aus Beer, Gas Rate CO2, Monthly Milk, Wooly (Darts — small classic teaching
series), FRED (macro aggregates) and one stock ticker. **No firm-level financial data appears
in any of the 15 models' pretraining corpora.**

That is market structure, not oversight: company financials and default outcomes sit behind
Bloomberg, S&P Capital IQ and Moody's — priced, licence-restricted, and with default labels
commercially guarded. **Energy got foundation models because its data was free. Finance did
not because its data is not.**

**Three consequences for this project:**

- It explains **why Fundamental went to oil & gas before finance** (`docs/LANDSCAPE.md`):
  vendors follow available data, not the largest market.
- **The synthetic prior is not a substitute for real data in this domain — it is the only
  affordable, licence-clean route.** That is why this space is empty while energy is crowded,
  and it converts the data scarcity from an obstacle into the reason the approach is correct.
  This is the strongest available framing of the thesis; it is also the one that must not be
  overstated, because "nobody has done it" is not evidence that it works (see §1).
- It sharpens §1. The few finance series that *do* appear (FRED macro aggregates, a stock
  ticker) are precisely those most exposed to **global-pattern memorisation** across shared
  crises — the second leakage mode. A financial model pretrained on real panels spanning 2008
  and 2020 inherits exactly that problem; one pretrained on a sampled macro regime does not.

## 5. Context-construction strategy affects credit AUC more than expected; verified on real data at smoke scale

**Date:** 2026-09-08. **Status:** SMOKE-TEST-SCALE MEASURED. Real data
(`data.py::load_polish_bankruptcy`), real benchmark run, but the checkpoint is a 400-step,
64-dim, 2-layer toy — this is a directional check, not a claim about the architecture.

**Source of the design.** Tanna, Solanki, Bouadi, Bouarour, Seth & Sankarapu (2026), *Data
Presentation Over Architecture: Resampling Strategies for Credit Risk Prediction with Tabular
Foundation Models* (arXiv:2605.18635) — verified from the abstract. They benchmark seven
context-construction strategies across four classical models and five TFMs on Home Credit and
Lending Club, and find balanced/hybrid sampling adds 3-4 AUC points over uniform, a gap wider
than the spread between TFM families. With a balanced context of 5k-10k, the strongest TFMs
match classical baselines trained on the full data.

**What this repository had before today:** `classifier.py`'s `fit()` subsampled a table
exceeding `max_context` uniformly at random. On a ~4% default rate, a 2000-row uniform sample
keeps roughly 80 positives — most of the signal a credit model needs is discarded before the
model ever sees it. This was a real defect, not a design choice; it existed only because the
first pass never faced real class imbalance.

**Fix:** `_select_context()` in `classifier.py` adds `"balanced"` (water-fill quotas so every
class gets as even a share as its size allows — a rare class this small is kept in full),
`"hybrid"` (half the budget balanced, half uniform, to retain some of the true base rate),
and keeps `"uniform"` for comparison. Default is `"balanced"`.

**Measured**, `uv run fintfm-bench --model runs/credit-smoke.pt --credit`, single seed:

| horizon | default rate | uniform AUC | hybrid AUC | balanced AUC | defaults kept (uniform → hybrid/balanced) |
| --- | --- | --- | --- | --- | --- |
| 1 year | 3.86% | 0.596 | 0.638 | **0.645** | 71 → 190 |
| 3 year | 4.71% | 0.598 | **0.680** | 0.603 | 90 → 346 |
| 5 year | 6.94% | 0.486 | 0.572 | **0.596** | 150 → 287 |

Read honestly rather than cherry-picked: **uniform is worst on all three horizons**, matching
Tanna et al.'s direction. Balanced is not uniformly best, though — hybrid wins on the 3-year
horizon by a wide margin (0.680 vs 0.603), which balanced sampling cannot explain by defaults
kept alone (both keep the same count once the water-fill saturates). One seed, one toy
checkpoint; do not conclude "balanced beats hybrid" or the reverse from this — rerun with
multiple seeds once a real checkpoint exists before trusting the ranking between the two.

**What this does not show:** gradient boosting (AUC 0.86-0.96) and random forest (0.85-0.94)
dominate fintfm (0.60-0.65) by a wide margin at this scale, exactly as expected from a
400-step model with no real pretraining. **The comparison that matters — fintfm at real
pretraining scale against gradient boosting — has not been run.** This finding is about
context construction being a real, measurable lever independent of that question, confirmed
directionally on real data rather than merely cited from a paper.

## 6. Balanced context buys ranking at the cost of calibration; a prior correction recovers both

**Date:** 2026-09-08. **Status:** SMOKE-TEST-SCALE MEASURED (200-step, 127k-parameter
checkpoint on real data). The *mechanism* is confirmed and the correction is exact under a
stated assumption; the *magnitudes* at real pretraining scale are unknown.

Finding §5 adopted balanced context sampling on Tanna et al.'s AUC evidence. Adding
calibration metrics (`metrics.py`) immediately showed what an AUC-only comparison cannot:

| horizon | strategy | AUC | ECE | predicted mean | actual rate |
| --- | --- | --- | --- | --- | --- |
| 3 year | uniform | 0.652 | 0.020 | 2.9% | 4.7% |
| 3 year | balanced, raw | 0.646 | **0.102** | **14.9%** | 4.7% |
| 3 year | balanced, corrected | 0.646 | **0.007** | 4.0% | 4.7% |

**The mechanism.** An in-context learner reads the class balance out of its context, because
that context is the only evidence it has about how common the positive class is. Rebalancing
the context therefore tells the model that defaults are roughly ten times commoner than they
are, and its probabilities come out inflated by about that factor. Ranking survives, since
every prediction is inflated alike — which is exactly why AUC cannot see the damage, and why
a paper optimising AUC would not report it. For credit risk this is the wrong thing to
break: a bank prices, provisions and holds capital against the *level*, not the ordering.

**The correction.** Shift the logits by ``log P_true(y) − log P_context(y)``. This is exact
under the label-shift assumption that ``P(x | y)`` is unchanged by resampling — which holds
*by construction* here, since `_select_context` selects on ``y`` alone and never looks at
``x``. It is a constant per-class shift, hence a monotone transform of the binary score, so
AUC is provably unchanged (asserted in `tests/test_classifier.py`). Default is on.

**Measured across all three horizons**, balanced context, raw → corrected:

| horizon | ECE raw → corrected | Brier raw → corrected | predicted mean → actual |
| --- | --- | --- | --- |
| 1 year | 0.0151 → 0.0173 | 0.0371 → 0.0371 | 5.36% → 2.13% (actual 3.86%) |
| 3 year | 0.1021 → **0.0071** | 0.0556 → **0.0449** | 14.94% → 4.02% (actual 4.71%) |
| 5 year | 0.0350 → **0.0225** | 0.0642 → 0.0642 | 9.90% → 4.69% (actual 6.94%) |

**Read honestly, including the case that disagrees.** The correction is a large win where the
distortion is large (3-year: ECE improves 14×) and a modest one at 5-year. At 1-year it
slightly *overcorrects*: ECE worsens from 0.0151 to 0.0173 and the predicted mean undershoots
the true rate. So this is not a free lunch — where the raw distortion was already small, the
shift can overshoot. The theory is exact only if the model reads the base rate *purely* from
the context prior, and a 200-step model plainly does something messier. **Re-measure at real
pretraining scale before treating "always correct" as settled**, and consider fitting the
shift on a validation split instead of deriving it analytically, which would absorb whatever
the model actually does rather than assuming.

**Consequence for the project.** This is a small worked example of the thesis in
`docs/LANDSCAPE.md`: the deliverable in regulated credit is the validation evidence, and the
evidence only exists if the metrics can see the failure. Nothing here was visible in AUC.

## 7. The only real dataset has no dates and no company identifiers, which blocks two changes

**Date:** 2026-09-08. **Status:** MEASURED by inspection of the source files. Resolves
`openspec/changes/time-based-evaluation` task 3.1 in the negative.

The UCI Polish bankruptcy panels contain **64 anonymous numeric attributes** (`Attr1` to
`Attr64`) plus a binary `class`. There is no date, no reporting period, and **no company
identifier**. Re-derivable:

```bash
uv run python -c "
import io,zipfile;from scipy.io import arff
z=zipfile.ZipFile('data/cache/polish_bankruptcy.zip')
raw=z.read('3year.arff').decode('utf-8','replace')
print([l for l in raw.splitlines() if l.startswith('@attribute')][:3])"
```

**Consequence 1: time-based evaluation is impossible on this dataset.** Splitting train and
test by observation period requires period labels. The dataset description gives ranges for
the corpus as a whole (bankrupt firms 2000-2012, operating firms 2007-2013) but nothing
per row. So `time-based-evaluation` cannot be done here at all, and is now **blocked on**
`second-credit-panel` rather than merely sequenced after it.

**Consequence 2: firm trajectories cannot be built from the five horizon files.** The
`temporal-financial-prior` falsification test (task 2.1) planned to join them per company.
Without identifiers there is nothing to join on.

**Consequence 3, and the uncomfortable one: horizon independence is unverifiable, not
verified.** Comparing row fingerprints across files shows near-zero overlap (1 shared
fingerprint between `1year` and `3year`), but that is exactly what one expects *either* way
— the same firm measured in a different year has different ratios. So the three horizons
**may or may not** share companies, and nothing in the data can settle it.

Every existing result reports all three horizons as if they were three evaluations. That is
not wrong, but the independence it implies is an assumption, and any external presentation
of those numbers must say so rather than implying three independent confirmations.

**Also noted, minor:** the internal `@relation` names are unreliable — `2year.arff`,
`3year.arff` and `4year.arff` all declare `'1year'`. Row and positive counts match the
published per-horizon description exactly (e.g. 3-year: 10,503 rows, 495 positives), so the
filenames are trustworthy and the relation strings are careless authoring in the source. The
files also carry a Weka `SubsetByExpression` filter in that header, so the published data is
already a filtered subset rather than the raw panel.

## 8. Every free bankruptcy dataset is a cross-sectional ratio table, except one — and it is licensed

**Date:** 2026-09-08. **Status:** MEASURED by inspection and licence verification. Resolves
`second-credit-panel` tasks 7.1 and 7.2, and **unblocks** `time-based-evaluation` and
`temporal-financial-prior`.

Finding §7 established that the UCI Polish panels have no dates or identifiers. Checking the
other candidates showed this is the norm rather than an accident of that dataset:

| dataset | rows | features | positives | dates? | firm id? | licence |
| --- | --- | --- | --- | --- | --- | --- |
| UCI Polish bankruptcy | 5,910-10,503 per horizon | 64 | 3.9-6.9% | **no** | **no** | CC BY 4.0 |
| UCI Taiwanese bankruptcy | 6,819 | 95 | 3.23% | **no** | **no** | CC BY 4.0 |
| **V4FinBench** | **1,106,879 company-years** | **131** | **0.19-0.36%** | **yes (2006-2021)** | yes (company-year) | **CC BY 4.0** |

Both UCI sets are anonymised cross-sectional ratio tables — usable for discrimination and
calibration work, useless for anything temporal. Verified for Taiwan by download and column
inspection; its only date-shaped column names are ratios containing the word "times".

**V4FinBench is the dataset this project needs.** Tomczak et al., *V4FinBench: Benchmarking
Tabular Foundation Models, LLMs, and Standard Methods on Corporate Bankruptcy Prediction*,
[arXiv:2605.10896](https://arxiv.org/abs/2605.10896), May 2026. Visegrád Group economies
(Czech Republic, Hungary, Poland, Slovakia), 2006-2021, six prediction horizons, a composite
distress criterion covering solvency, profitability and liquidity. Code at
[github.com/genwro-ai/V4FinBench](https://github.com/genwro-ai/V4FinBench) under MIT; **the
data is CC BY 4.0**, hosted on Kaggle — verified from the repository's separate
`DATA_LICENSE.md`, not assumed from the code licence. That distinction matters: Google's
TabFM ships Apache-2.0 code with **non-commercial** weights, so "the repo is permissive" is
never sufficient.

**Why it changes the plan:**

- **Temporal work becomes possible.** Company-year rows spanning 2006-2021 cover the
  financial crisis *and* COVID, so out-of-time validation across genuine regime shift — the
  thing a supervisor actually asks for — can finally be done.
- **It carries published TabPFN reference evaluations**, giving a directly comparable
  baseline without running anyone else's weights.
- **The imbalance is an order of magnitude harsher**: 0.19-0.36% positives against 3-7% in
  the UCI sets. At a 2,000-row context and 0.3% positives, *uniform* sampling would supply
  about six defaulters. That makes the context-construction and base-rate-correction work
  (§5, §6) load-bearing rather than a refinement, and it is the regime where those findings
  should be re-measured.

**Two practical consequences before it can be used:** the data is on Kaggle and needs
credentials, and 131 features exceeds the current `max_features=64`, so a wider model must be
pretrained rather than reusing existing checkpoints.

## 9. The winning niche is small and low-default portfolios, and this reverses §8's prioritisation

**Date:** 2026-09-08. **Status:** MEASURED by others, verified to abstract level. Full PDFs
not yet read, so the specifics flagged below are genuinely open.

Two papers, read together, answer the question of where a specialist can beat both gradient
boosting and a general TFM. They point the same way, and it is not where §8 was heading.

**Baesens, Goethals, Lessmann, De Vos, Bravo, Martens, Medina-Olivares, Mues, Oskarsdóttir,
vanden Broucke, Van Gestel, Verdonck & Verbeke (2026).** *Foundation Models for Credit Risk
Prediction: A Game Changer?* [arXiv:2605.18147](https://arxiv.org/abs/2605.18147). An
authoritative team in credit-risk benchmarking, testing TFMs against established and
advanced ML on **PD and LGD**, out-of-the-box with no tuning:

> "tabular foundation models generally perform best across datasets and tasks. Moreover, they
> offer significant improvement in predictive performance **as dataset size shrinks**."

They name the niche explicitly: *"small-data settings, such as SME lending or specialized
corporate portfolios"*, and *"longstanding challenges including low default portfolios and
class imbalance."* They also name the incumbent to displace: the quasi-standard is
**gradient boosting paired with SHAP**.

**Purucker, Tschalzev, Erickson et al. (2026).** *Beyond IID: How General Are Tabular
Foundation Models, Really?* [arXiv:2606.30410](https://arxiv.org/abs/2606.30410). Introduces
BeyondArena, covering IID, temporal and grouped tasks:

> "existing tabular foundation models excel on tiny- to medium-sized IID data, while
> traditional tree-based and deep learning models still dominate on **non-IID, large, and
> high-dimensional** datasets."

**The two together define the winnable ground.** TFMs win where data is small; they lose
where it is large, wide, or shifted. Credit risk contains both regimes, and the project must
choose one:

| regime | who wins today | our position |
| --- | --- | --- |
| SME / specialty / low-default portfolios: thousands of rows, few defaults | **TFMs, and the margin grows as data shrinks** | **target this** |
| Large national panels: 10^6 rows, 131 features, 15 years | gradient boosting | do not fight here |

**This reverses §8's framing.** §8 called V4FinBench "the dataset this project needs" on the
strength of its 1.1M company-year rows and 131 features. Per Beyond IID those are precisely
the three conditions under which TFMs lose. V4FinBench remains valuable — it is the only
licensed panel with dates, so it is the only way to do out-of-time validation and to test
robustness under regime shift — but it is **a validation instrument, not the target market**.
Chasing accuracy on it would be picking the fight the literature says we lose.

**It also reframes our existing data as adequate rather than limiting.** The UCI Polish and
Taiwanese panels are 6,000-10,000 rows with 3-7% default rates. That is the home turf, not a
compromise forced by having no better source.

**And it explains our own smoke-scale results.** Gradient boosting beat the model 0.86-0.96
against 0.60-0.65 (§5). That was read as "expected from a 400-step model", which is true, but
Beyond IID says the gap is also the field-wide pattern in this regime. **Do not assume scale
alone closes it.** The honest test is whether the model wins in the *small*-data regime, and
`bench.py` currently evaluates on the full panel every time — it never actually tests the
condition where TFMs are supposed to win.

**Why this is defensible rather than merely a smaller ambition:**

- Google's TabFM ships inside BigQuery, aimed at enterprise warehouse scale. That is the
  opposite end of the size axis from an SME lender with 2,000 obligors, and Beyond IID says
  the big end is not winnable by a TFM anyway.
- **Low Default Portfolios are a named regulatory category**, not just a small dataset. With a
  handful of defaults you cannot estimate PD reliably by any method, so supervisors demand
  uncertainty quantification and conservatism. That is exactly the conformal-certificate work
  in `docs/DECISIONS.md` D3 — the niche where accuracy is hardest is the niche where the
  certificate is worth most.
- The incumbent is gradient boosting **plus SHAP**, so the comparison is not accuracy alone.
  A single in-context model that returns PD *and* attributions would replace both halves; see
  `openspec/changes/zero-shot-attribution`.

### The specifics, from the full text — and they cut the headline down

Read 2026-09-08 from [the HTML version](https://arxiv.org/html/2605.18147v1). Five TFMs
(TabPFN, TabPFNv2, TabPFN-Real, MITRA, TabICL) against 29 PD methods and 22 LGD methods,
including tuned XGBoost, LightGBM, CatBoost, FT-Transformer, TabNet and logistic regression.
14 PD datasets (1,000 to 532,428 rows) and 7 LGD datasets (594 to 57,931 rows).

**The crossover is roughly 8,000 observations.** For LGD, TabPFNv2 *"leads clearly at small
sample sizes"* before tuned GBMs converge *"around 8,000 observations, where TabPFNv2's
performance declines."* Learning curves show substantial TFM advantage **below 1,000
observations**.

Three caveats that change how much weight this can carry, none of them in the abstract:

1. **The margins are small and mostly not significant.** *"Performance differences among top
   methods were small in absolute terms"*, with statistical significance in only **22 of 406**
   pairwise PD comparisons. TFMs won 44.3% of PD folds collectively; TabICL alone 25.7%. So
   "foundation models generally perform best" means "win more often by a little", not
   "dominate". Any pitch built on it must say so.
2. **The low-default-portfolio benefit is conjectured, not demonstrated.** Their PD datasets
   have default rates from 6.7% to 40%, **mean ≈22%** — consumer lending, not low-default
   corporate books. The paper itself notes direct empirical validation of LDP and imbalance
   *"wasn't extensively detailed"*. The niche they name is not the niche they tested.
3. **Explainability is named as their future work**: comparing *"feature attributions derived
   from PFN and GBM pipelines using SHAP"*. Open, by their own account.

### Two consequences, one of which is a defect in our own harness

**Our panels sit at or above the crossover, so we have not been testing the regime where TFMs
win.** UCI Polish is 6,027-10,503 rows and Taiwan is 6,819 — right where Baesens et al. find
GBMs converging and TFM advantage decaying. `bench.py` evaluates on the *full* panel every
time. It therefore measures the regime the literature says we lose, and never the one where
the thesis lives. **Fix: a learning-curve evaluation that subsamples training data** to a few
hundred rows and up, and reports where the crossover falls for *this* model.

**The gaps are the opportunity, and they are specific.** Not "a better TFM" but: does the TFM
advantage hold on genuinely low-default corporate books (a named Basel category the field's
own benchmark did not test), does it hold *calibrated* rather than merely ranked when there
are twenty defaults, and can one in-context model return PD and attribution together. All
three are open by the authoritative team's own admission, all three matter to a supervisor,
and none is a size fight against BigQuery.

## 10. Domain-specific pretraining beats general pretraining in finance, in an adjacent modality

**Date:** 2026-09-08. **Status:** MEASURED by others, abstract level. Suggestive for this
project rather than direct evidence, because the modality differs.

Rahimikia, Ni & Wang (2025), *Re(Visiting) Time Series Foundation Models in Finance* (SSRN,
138 pp.), ran the first comprehensive study of TSFMs on global financial markets using daily
excess returns, comparing zero-shot inference, fine-tuning, and pretraining from scratch:

> "off-the-shelf pre-trained TSFMs perform poorly in zero-shot and fine-tuning settings,
> whereas models **pre-trained from scratch on financial data achieve substantial forecasting
> and economic improvements**, underscoring the value of domain-specific adaptation.
> Increasing the dataset size, incorporating **synthetic data augmentation**, and applying
> hyperparameter tuning further enhance performance."

**Independently corroborated again 2026-09-09** by FinCast (arXiv:2508.19609), which
motivates a purpose-built financial time-series foundation model on precisely the grounds
that general ones "do not specifically address the idiosyncrasies of financial data", and
reports 20-23% error reductions over TimesFM, Chronos and TimesMoE. Two independent groups
in the adjacent modality now report what §14 measured for tabular credit.

**This is the Phase 1 hypothesis, confirmed in a neighbouring modality.** Our bet is that a
financial prior beats a generic one; they found general pretraining insufficient for finance
and domain pretraining substantially better, with synthetic augmentation helping. Two reasons
not to overweight it: time series are not tables (the failure modes differ, and The
Forecasting Company's argument on that is sound), and "pretrained from scratch on financial
data" means real financial data, whereas this project deliberately uses none. So it supports
the *domain-specificity* half of the thesis, not the *synthetic-only* half.

**Architecture convergence, noted in passing.** TimesFM-3 (Google, Aug 2026) alternates
causal temporal attention with full variate attention; TabFM alternates row and column
attention; TabPFN v2 and TabICL do likewise in tabular form. This repository's design —
attention across columns within a row, then across rows — is the same pattern arrived at
independently. Two consequences: the design is unremarkable rather than novel, so **never
claim novelty on it**, and it is unlikely to be the thing that wins or loses.

**Also noted:** TimesFM-3 outputs nine quantiles by default. Probabilistic output is the
field standard now, not a differentiator, which raises the bar for what the certificate work
must deliver — coverage guarantees and honest refusal, not merely intervals.

## 11. Two firms in five get an incoherent PD term structure, and the aggregate hides it

**Date:** 2026-09-08. **Status:** MEASURED on real data with a real (if under-trained)
checkpoint. Resolves `openspec/changes/pd-term-structure` task 11.1 — **in favour of the
change**. Re-derivable:

```bash
uv run fintfm-ablate --coherence runs/phase1-5k/financial.pt --out runs/coherence
```

Cumulative default probability must not decrease with the horizon: a firm that has defaulted
by year 3 has defaulted by year 5. Measured on 3,151 held-out rows, scored under contexts
drawn from each of the five UCI horizons in turn:

| context horizon | observed base rate | mean predicted PD |
| --- | --- | --- |
| 1 year | 3.857% | 3.278% |
| 2 year | 3.932% | 3.354% |
| 3 year | 4.713% | 3.848% |
| 4 year | 5.259% | 4.172% |
| 5 year | 6.937% | 5.588% |

**The aggregate is monotone and that is the trap.** Mean predicted PD rises at every step, so
any report at portfolio level looks correct and this defect would never surface. Per firm:

- **violation rate per step: 11.0%**
- **fully monotone curves: 60.6%**

So **two firms in five receive a term structure in which a longer horizon carries a lower
default probability.** That is not a ranking imperfection; it is an incoherent object. An
IFRS 9 lifetime expected-credit-loss calculation consumes this curve directly, and a
provisioning number built on a decreasing cumulative hazard is indefensible in front of a
reviewer.

**Why it happens.** Nothing ties the horizons together. Each is an independent in-context
prediction from an independent labelled context, with no shared parameters and no
monotonicity constraint. The model is not doing anything wrong by its own objective — the
objective simply never mentioned coherence. This is the defect
`openspec/changes/pd-term-structure` predicted before it was measured.

**Caveats, and one is substantial:**

- The checkpoint is from a 5,000-step run that had not finished when this was measured, so
  the *level* of these numbers is not a quality claim. The violation rate may improve with
  training. **It cannot be driven to zero by training**, though, because nothing in the
  architecture or loss forbids a violation — which is the point.
- The horizon files are different firm samples (no identifiers, `FINDINGS` §7), so contexts
  differ in composition as well as in label meaning. Base rates rise monotonically with
  horizon, so the expected direction is unambiguous, but this is not a nested panel and a
  clean version of this test needs V4FinBench.
- Query rows are fixed and only the context varies, which is the strongest design available
  given no identifiers, and it is stated in the function's docstring rather than buried.

**Consequence:** the term-structure direction is founded rather than speculative, and the
diagnostic should be permanent — `pd-term-structure` task 11.2 — so that a coherence
regression cannot pass unnoticed behind a healthy-looking aggregate.

## 12. The advantage is calibration, not ranking — and it holds at every dataset size

> **ALSO WEAKENED by §17:** a feature-free constant predictor beats every model here on ECE
> (0.0002 vs our 0.00255), so a low ECE is not by itself evidence of a good model. Report
> Brier *skill* against that baseline, which for us is 1-2%.
>
> **NARROWED by §27:** restated against CatBoost, the usable window is **below ~200
> obligors**, not below 1,000 — but within it we win on AUC, ECE *and* Brier skill, and the
> best-ECE-at-every-size claim survives.
>
> **TEMPERED by §16 (same day).** The 2.3-11.7× ratios below are against an **uncalibrated**
> gradient boosting. Against a calibrated one the gap at n = 100 is roughly **2×**, and above
> ~1,000 rows calibrated gradient boosting is the better model on both AUC and Brier. **Do
> not quote 11.7× anywhere.** The surviving claim is in §16.

**Date:** 2026-09-08. **Status:** MEASURED, 5 seeds, real data. The checkpoint is a
5,000-step 2.2M-parameter model, so **absolute AUC is not a quality claim**; the
model-versus-model *comparison* at matched conditions is. Resolves
`sample-efficiency-regime` tasks 13.3-13.5. Re-derivable from
`runs/probe-financial/sample_efficiency.json` and the seed sweep in this section.

Finding §9 said our benchmark had only ever measured the regime the literature says we lose.
Measuring the other end changes the thesis — not by confirming it, but by relocating it.

| n train | defaults | AUC delta (fintfm − gboost) | verdict | fintfm ECE | gboost ECE | ECE ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 5 | **+0.069 ± 0.040** | **TFM wins** | 0.0066 | 0.0781 | **11.7×** |
| 250 | 12 | +0.024 ± 0.036 | inconclusive | 0.0050 | 0.0505 | **10.0×** |
| 500 | 24 | −0.039 ± 0.035 | GBM wins | 0.0068 | 0.0415 | 6.1× |
| 1,000 | 47 | −0.096 ± 0.015 | GBM wins | 0.0077 | 0.0303 | 3.9× |
| 2,000 | 94 | −0.126 ± 0.018 | GBM wins | 0.0072 | 0.0168 | 2.3× |

### The ranking advantage is real but narrow

It survives five seeds only at **n = 100**, is inconclusive by 250, and is gone by 500. The
crossover is therefore around 100-250 observations, **far below the ~8,000 that Baesens et
al. report** for LGD. Two candidate explanations and they are not distinguishable yet: our
checkpoint is heavily under-trained (AUC is roughly flat at 0.67-0.72 across every size,
which is what a model that cannot use more data looks like), or corporate default at a
3-year horizon simply crosses over earlier than consumer LGD. **Do not quote 100-250 as the
product's crossover** until a properly trained model has been measured.

### The calibration advantage is the finding

Gradient boosting's expected calibration error is **2.3× to 11.7× worse at every size
tested**, and worst exactly where it matters: 0.078 at n = 100, meaning a stated probability
is off by nearly 8 percentage points on average. Our ECE is flat at 0.005-0.008 across all
sizes with tiny variance, so this is a stable property rather than a lucky draw.

The mechanism is not mysterious and that is what makes it credible: gradient boosting on 100
rows with 5 defaults overfits into overconfident probabilities, while this model is trained
under cross-entropy (a proper scoring rule), receives a balanced context, and has its base
rate corrected analytically (§6). Note that **gradient boosting still wins AUC decisively at
n = 2,000 while being 2.3× worse calibrated** — the two properties are genuinely separable,
which is the whole reason §9's AUC-only framing was insufficient.

### Consequence: the pitch is calibration, not accuracy

This is the strongest evidence yet for `docs/DECISIONS.md` D3, and it sharpens it. The
defensible claim is **not** "more accurate than gradient boosting" — that is false above a
few hundred rows and would be caught immediately. It is:

> Comparable ranking below a few hundred obligors, and a probability that means what it says
> at every portfolio size, where the incumbent's is off by up to 8 points.

For low-default portfolios, provisioning and capital, the level is the deliverable and the
ranking is secondary. A lender cannot provision against an ordering. This also explains why
Neuralk's demo showing a bare "Default risk 6%" (`LANDSCAPE.md`) is a weaker product than it
appears: nothing in it establishes that 6% means 6%.

### What would falsify or complicate this

- **A tuned gradient boosting baseline with calibration applied.** Ours is out-of-the-box,
  matching how Baesens et al. framed it, but Platt scaling or isotonic regression on a
  validation split is cheap and standard practice, and it is the obvious counter. **Until
  that comparison is run, the 11.7× is against an uncalibrated incumbent and must be
  described that way.** This is the single most important follow-up.
- One panel, one horizon, one architecture. Needs Taiwan and V4FinBench.
- ECE with 5 defaults in the training set and ~3,150 test rows is measurable, but the
  calibration of a 3% base rate estimated from 5 events deserves a conformal interval rather
  than a point ECE.

## 13. The calibration advantage is Bayesian shrinkage, and it is partly conservatism

**Date:** 2026-09-08. **Status:** MEASURED (single seed, 3-year panel, under-trained
checkpoint) plus a **theoretical mechanism that is inference, labelled as such**. Re-derivable
from the script in this section's history; sweeps n against a fixed 3,151-row test set.

§12 measured a calibration advantage of 2.3× to 11.7×. This asks *why*, because a mechanism
that is understood can be defended and one that is a coincidence cannot.

### The hypothesis

Müller et al. (arXiv:2112.10510) show a prior-fitted network approximates the **posterior
predictive** under its prior. If that is what this model does, two things follow as theory
rather than tuning: a posterior predictive is **calibrated by construction** when the
data-generating process lies in the prior's support, and at small *n* the posterior is
**dominated by the prior**, so predictions shrink toward the prior's base rate.

That predicts the advantage should **decay monotonically with n**. Measured ECE ratio
(gboost/fintfm): 11.7×, 10.0×, 6.1×, 3.9×, 2.3× at n = 100, 250, 500, 1000, 2000. Monotone
decay. This is what shrinkage predicts and not what "we happened to tune better" predicts.

### The direct test, and it confirms the mechanism

Panel base rate 4.713%. Distribution of predicted probabilities on a fixed test set:

| n | model | pred mean | pred sd | p99/p50 | \|mean − base\| |
| --- | --- | --- | --- | --- | --- |
| 100 | fintfm | 5.03% | 0.024 | 2.8 | 0.32% |
| 100 | gboost | 3.48% | 0.164 | **396,829** | 1.23% |
| 250 | fintfm | 4.52% | 0.023 | 3.0 | 0.19% |
| 250 | gboost | 1.95% | 0.104 | 1,609 | 2.76% |
| 1000 | fintfm | 4.19% | 0.027 | 4.8 | 0.52% |
| 1000 | gboost | 3.37% | 0.113 | 124 | 1.34% |
| 4000 | fintfm | 3.98% | 0.017 | 3.2 | 0.73% |
| 4000 | gboost | 4.14% | 0.103 | 49 | 0.58% |

**Our mean tracks the base rate closely and our spread is small.** Gradient boosting at small
n is not merely miscalibrated, it is **pathological**: a p99/p50 ratio of 396,829 means the
median firm receives a probability near zero while the top percentile receives an enormous
one. It says "almost nobody defaults, except these few who certainly will." For provisioning
that is precisely the wrong shape, and it is why its ECE reaches 0.078.

Both converge as n grows, and by n = 4,000 gradient boosting's mean is *closer* to the base
rate than ours. The advantage is a small-sample phenomenon, exactly as the mechanism says.

### The honest caveat, and it is substantial

**Part of our calibration advantage is conservatism.** Our predicted spread is 0.013-0.028
against gradient boosting's 0.084-0.164 — four to eight times narrower. A model that predicts
the base rate for every firm is perfectly calibrated and completely useless, and we are closer
to that end of the axis than the incumbent is. Our flat AUC across sizes (0.67-0.72) is the
same fact seen from another direction: a model that cannot exploit more data.

So the claim must be stated as a **frontier, not a win**:

| regime | discrimination | calibration | honest verdict |
| --- | --- | --- | --- |
| n ≈ 100 | ours 0.689 vs 0.632 | ours 0.007 vs 0.078 | **we dominate on both** — a genuine Pareto improvement |
| n ≥ 500 | theirs, decisively | ours, by 2-6× | **a real trade-off**, and which side matters depends on the use |

For low-default portfolios, provisioning and capital, the level is the deliverable and a
lender cannot provision against an ordering. For a ranking application, theirs is better.
**Saying only the first half would be the kind of claim this repository exists to prevent.**

### The strategic consequence: the obvious counter fails where we target

The rebuttal to §12 is "just Platt-scale the gradient boosting", and it is correct in
general. But **post-hoc calibration requires held-out data containing events.** At n = 100
with 5 defaults, fitting a two-parameter calibration map means estimating it from a handful of
positives, and splitting a calibration set off makes the base model worse. So:

> Post-hoc calibration needs the one thing a low-default portfolio does not have: defaults to
> calibrate on.

Our calibration comes from the prior and costs no data. Theirs requires data they do not have
in the regime we target. If that survives measurement it is the most defensible thing in the
project — **and it is not yet measured**, which is why `sample-efficiency-regime` task 13.7
is the highest-priority follow-up.

### The discriminating test that must be run before any of this is claimed

**Is this calibration property generic to prior-fitted networks, or specific to our financial
prior?** If the `generic` variant from the Phase 1 ablation is equally well calibrated, then
calibration is a property of the *method* — which means TabPFN, TabICL and TabFM have it too,
and our differentiator is the domain prior plus the certificate, **not** calibration itself.
That would be a materially weaker position than §12 suggests, and it is answerable from the
run currently in flight. Do not build a pitch on calibration until that comparison exists.

## 14. Phase 1 verdict: the domain prior works, but pure financial is not the optimum

> **ALSO WEAKENED by §17:** the Brier differences between variants here sit inside a 1-2%
> band above a feature-free constant predictor, so "best Brier" carries far less weight than
> written. The AUC result is unaffected.
>
> **AMENDED by §15 (same day).** This finding's framing — "financial buys discrimination,
> generic buys calibration" — is **wrong**. An untrained control shows *all* pretraining buys
> calibration, and that a random-weight model already ranks at AUC 0.726, so ranking is
> largely architectural. The exit-condition verdict below stands; the mechanism explanation
> does not. Read §15 with it.

**Date:** 2026-09-08. **Status:** MEASURED. 5,000 steps × 3 variants at matched compute
(2,175,234 parameters, identical across variants, enforced), Metal, seed 0, evaluated on
three horizons × two context strategies of the UCI Polish panel. Resolves
`phase1-prior-ablation` tasks 1.4-1.5 and the discriminating test in §13.

Re-derivable from `runs/phase1-5k/results-paired.json`:

```bash
uv run fintfm-ablate --steps 5000 --device mps --out runs/phase1-5k
```

### Exit condition A: met

Mean over six cells, and **all three metrics matter**:

| variant | p_financial | mean AUC | mean ECE | mean Brier |
| --- | --- | --- | --- | --- |
| financial | 1.0 | **0.7568** | 0.01060 | 0.04844 |
| **mixed** | **0.7** | 0.7509 | **0.00424** | **0.04818** |
| generic | 0.0 | 0.7081 | 0.00656 | 0.04850 |

Paired bootstrap on AUC with Holm-Bonferroni across cells:

- **financial − generic:** all six point estimates positive; **3/6 significant** after
  correction (up to +0.070), **0/6 favouring generic**. Two further cells were significant
  before correction.
- **mixed − generic:** **4/6 significant**, all favouring mixed.
- **financial − mixed:** **1/6 significant**. Five of six cells have intervals straddling
  zero, so on discrimination these two are **statistically indistinguishable**.

**Domain-specific pretraining transfers.** That is the central bet of the project and it is
now measured rather than argued, with no cell pointing the other way.

### The more useful finding: the two priors contribute different things

**The financial prior buys discrimination. The generic prior buys calibration.**

- Pure financial has the **worst ECE of the three** (0.0106) despite the best AUC.
- Pure generic calibrates better (0.0066) while discriminating far worse.
- The mixture at p = 0.7 gets **both**: AUC statistically tied with pure financial, ECE 2.5×
  better than it, and the **best Brier score** — which matters most, because Brier is a
  proper scoring rule and therefore the single number that respects both properties at once.

So the library default of 0.7, chosen arbitrarily on day one, turns out to be the best of the
three tested. **The operating point is a mixture, not a pure domain prior**, and the
plausible reason is that prior breadth regularises the posterior: a narrow prior fits the
domain and is overconfident off it.

### An independent team landed on almost the same mixture ratio

**Added 2026-09-09.** FinCast's pretraining corpus (arXiv:2508.19609, Table 1) is 20B time
points, of which **22.48% is deliberately non-financial** — 4.61B points from general
time-series sources — with the stated reason that "high-quality financial data is scarce".
So their mix is roughly **78% domain / 22% general**.

Ours, chosen arbitrarily on day one and then *measured* as the best of three on Brier, is
**70% financial / 30% generic** (`p_financial = 0.7`).

**BloombergGPT is a third data point** (arXiv:2303.17564): 363B financial tokens against
345B general-purpose tokens, i.e. **51% domain / 49% general**, with the stated result that
mixed training "outperforms existing models on financial tasks by significant margins
**without sacrificing performance on general LLM benchmarks**".

| project | modality | domain share | reason given |
| --- | --- | --- | --- |
| BloombergGPT | language | **51%** | preserve general capability |
| **this project** | **tabular** | **70%** | **measured best of three on Brier** |
| FinCast | time series | 78% | domain data is scarce |

Three teams, three modalities, three different reasons — preserving general ability, an
ablation, and scarcity — and **not one went pure-domain.** The range is wide (51-78%), so
this says nothing about an optimum and none of us searched finely. What it does support is
the counter-intuitive half of this finding, and the half most likely to be argued with:
**a purely domain-specific pretraining mixture is the wrong choice.** Our own §15 gives the
mechanism for the tabular case — the generic prior is where calibration comes from.

It also echoes §4 and §8 from the other modality: they call financial data scarce for exactly
the reason we found firm-level panels licence-locked.

### It also settles §13's discriminating test, against us

§13 asked whether the calibration advantage is ours or generic to prior-fitted networks.
**It is not the financial prior's** — financial is the worst-calibrated variant here. Good
calibration tracks prior *breadth*, which means it is a property of the method, so **TabPFN,
TabICL and TabFM very likely share it.**

Consequence, and it is a demotion: **calibration alone is not a differentiator.** The
defensible position is the *combination* — domain prior for discrimination, mixture for
calibration, and the certificate for evidence — not calibration as such. Any pitch resting
on calibration versus other foundation models needs a head-to-head against TabPFN before it
can be made, and `docs/STRATEGY.md` has been amended accordingly.

### Limitations, and two are serious

- ~~**No untrained control.**~~ **Resolved by §15**, and it materially changed the reading:
  the untrained model reaches AUC 0.7262, above the generic variant, so the
  financial-versus-generic gap partly measures generic trading ranking for calibration
  rather than financial being good. The load-bearing comparison is against the control.
- **The six cells are not six independent tests.** They are three horizons × two context
  strategies of one panel, and the two strategies on a given horizon **share the same test
  rows**. Holm-Bonferroni treats them as a family of independent comparisons, which they are
  not, so "3/6 significant" is a rough guide and not a clean family-wise statement. A proper
  version needs independent panels — Taiwan and V4FinBench.
- **Single seed.** Task 1.7 requires at least three before any number leaves the repository.
- **Under-trained by 5×.** 2.2M parameters and 5,000 steps against a stated target of 10-50M.
  The *ranking* of variants at matched compute is the claim here; the levels are not.
- Horizon independence within the panel is unverifiable (§7).

## 15. The untrained control reframes §14: pretraining buys calibration, and ranking is largely architectural

**Date:** 2026-09-08. **Status:** MEASURED. Resolves `phase1-prior-ablation` task 1.8.
**This finding corrects §14's central framing** — see the amendment note there.

An untrained control (random initialisation, identical architecture and parameter count,
2,175,234, zero gradient steps) was evaluated on the same six cells. Re-derivable from
`runs/phase1-5k/results-paired.json`.

| variant | steps | mean AUC | mean ECE | mean Brier | mean pred sd |
| --- | --- | --- | --- | --- | --- |
| **untrained** | **0** | **0.7262** | **0.53231** | **0.34663** | 0.0109 |
| generic | 5,000 | 0.7081 | 0.00656 | 0.04850 | 0.0115 |
| mixed | 5,000 | 0.7509 | 0.00424 | **0.04818** | 0.0117 |
| financial | 5,000 | **0.7568** | 0.01060 | 0.04844 | 0.0146 |

### Two facts that change the story

**A randomly initialised model already ranks at AUC 0.726.** That is *better than the
generic-prior model* (0.7081), and within 0.03 of the best trained variant. Ranking on this
task is therefore substantially **architectural**, not learned: an in-context transformer
over context-normalised features behaves like a similarity method even with random weights,
which is unsurprising once stated — random projections approximately preserve distances.

**Its probabilities are worthless.** ECE 0.532 and a predicted mean of **39.2% against an
actual base rate of 4.7%**. Pretraining moves ECE from 0.532 to 0.004-0.011, a **50-130×
improvement**, and Brier from 0.347 to 0.048, a **7× improvement**.

### So what pretraining actually buys

**Calibration, overwhelmingly.** Every trained variant is ~7× better than untrained on
Brier, the proper scoring rule that respects level and ordering together. That is the real
return on pretraining, and it is large and unambiguous.

**§14 said "the financial prior buys discrimination, the generic prior buys calibration."
That is wrong.** *All* pretraining buys calibration; the generic prior is not special in
that respect. What the financial prior adds is a modest amount of **extra discrimination on
top of an architecture that already ranks**: +0.031 mean AUC over untrained (2/6 cells
significant) and +0.049 over generic.

**And generic pretraining is not "worse than nothing".** It loses 0.018 AUC against
untrained while gaining 80× on ECE and 7× on Brier. That is a trade, not a failure, and
reading the AUC column alone would have mis-called it.

### The deflating part, stated because it is true

**On Brier, the three trained variants are within 0.7% of each other** — 0.04818, 0.04844,
0.04850. The domain prior's advantage is real and it is measurable on AUC, but on the single
metric that respects both properties the choice between financial, mixed and generic is
nearly immaterial at this scale. §14's exit condition is still met, and the practical
consequence of meeting it is smaller than that finding implied.

The honest one-line summary of Phase 1 is therefore: **pretraining matters enormously and
mostly through calibration; which prior you pretrain on matters much less, though the
financial prior does measurably help ranking.**

### What this implies for the next experiments

- **Never report AUC without Brier again.** The untrained control would have looked
  competitive on a discrimination-only table, and §14 nearly did read that way.
- **The architecture may be carrying more than the prior.** An ablation of the architecture
  against a trivial baseline (nearest neighbour, logistic regression on the same normalised
  features) is now more informative than another prior variant, and is not yet proposed.
- **Scale may change the ranking.** At 5× the parameters and 4× the steps the priors may
  separate further on Brier, or may not. That is now a more interesting question than it was.

## 16. Against a *calibrated* incumbent the advantage shrinks from 11.7× to ~2×, and holds only below a few hundred rows

**Date:** 2026-09-08. **Status:** MEASURED, single seed, 3-year panel, `mixed` checkpoint.
Resolves `sample-efficiency-regime` task 13.7 and `calibration-mechanism` task 14.1.
**This finding tempers §12**, whose comparison was against an uncalibrated baseline — a
limitation §12 itself flagged as the most important follow-up. It was right to.

Gradient boosting with Platt (`sigmoid`) and isotonic calibration, fitted by
`CalibratedClassifierCV(cv=3)` on the training data:

| n (defaults) | model | AUC | ECE | Brier |
| --- | --- | --- | --- | --- |
| 100 (5) | fintfm | 0.6945 | **0.0070** | **0.0441** |
| | gboost raw | 0.6315 | 0.0719 | 0.0686 |
| | gboost Platt | **0.5446** | 0.0149 | 0.0451 |
| | gboost isotonic | 0.6475 | 0.0127 | 0.0445 |
| 250 (12) | fintfm | **0.6915** | **0.0028** | **0.0442** |
| | gboost Platt | 0.6644 | 0.0117 | 0.0455 |
| | gboost isotonic | 0.6688 | 0.0330 | 0.0480 |
| 1,000 (47) | fintfm | 0.7062 | **0.0080** | 0.0446 |
| | gboost isotonic | **0.8102** | 0.0137 | **0.0417** |
| 4,000 (189) | fintfm | 0.7159 | **0.0017** | 0.0445 |
| | gboost raw | **0.8856** | 0.0079 | **0.0333** |

### What has to be conceded

**Post-hoc calibration works, and the 11.7× headline does not survive it.** At n = 100 the
calibration gap against the best calibrated arm is roughly **2×** (0.0070 against 0.0127),
not tenfold. Any external use of the 11.7× figure would have been against a straw baseline.

**Above roughly 1,000 rows, calibrated gradient boosting is simply the better model** — better
AUC and better Brier. At n = 4,000 it wins Brier 0.0333 to 0.0445, which is not close.

### What survives, and it is narrower but real

**We hold the best ECE at every single size**, by 2× to 4× against the calibrated arms
(0.0017-0.0080 against 0.0099-0.0330). Calibration remains our strongest metric.

**On Brier — the proper scoring rule, and therefore the honest single number — we win below
about 250 rows** and lose above about 1,000. The crossover sits in between, and pinning it
needs seeds.

**Post-hoc calibration has a real cost at very small n, and it is dramatic.** Platt scaling
at n = 100 dropped AUC from 0.6315 to **0.5446** — barely above chance. Fitting a
two-parameter map on five positive events distorted the ranking it was calibrating. So the
§13 argument that "post-hoc calibration needs the events a low-default portfolio does not
have" is **partly** vindicated: it does not fail outright, but it damages discrimination
exactly where the portfolio is thinnest, and isotonic (0.6475) degraded less than Platt.

### The claim, restated at the strength the evidence supports

> Below a few hundred obligors we produce the best probability estimates available on a
> proper scoring rule, and at every portfolio size the best-calibrated ones. Above roughly a
> thousand rows a calibrated gradient boosting is the better model overall, and we should say
> so.

That is a smaller product than §12 implied and it is still a real one, because SME books,
specialty portfolios and low-default portfolios live at the small end (§9). What it forbids
is any general claim to beat gradient boosting, and any use of the 11.7× number.

### Open

- Single seed; the 250-1,000 crossover on Brier needs at least three.
- The `mixed` checkpoint is under-trained by ~5×; a properly trained model may push the
  crossover up. It may also not, since AUC is flat across sizes (§13).
- Isotonic beat Platt on discrimination at n = 100 but was worse on ECE at n = 250. Neither
  is uniformly the right incumbent baseline, so report both rather than the flattering one.

## 17. A feature-free constant predictor is within 1-2% of our best model on Brier, and beats every model on ECE

**Date:** 2026-09-08. **Status:** MEASURED. Resolves `calibration-mechanism` task 14.4.
**This finding weakens the metric basis of §12, §14, §15 and §16** and is the most important
methodological correction so far. Amendment notes added to each.

A predictor that ignores every feature and returns the training base rate for all firms:

| horizon | constant@base Brier | mixed Brier | Brier skill | constant ECE | mixed ECE |
| --- | --- | --- | --- | --- | --- |
| 1 year | 0.03693 | 0.03653 | **1.1%** | **0.00023** | 0.00255 |
| 3 year | 0.04505 | 0.04466 | **0.9%** | **0.00022** | 0.00255 |
| 5 year | 0.06456 | 0.06317 | **2.2%** | **0.00000** | 0.00255 |

The constant predictor has **AUC exactly 0.5000** — no discriminative content whatsoever.

### Two conclusions, both uncomfortable

**Our best model improves on a feature-free baseline by 1-2% of Brier.** Brier on a 4-7%
base rate is dominated by the mass of negatives, so the *achievable range* of the metric is
narrow and a trivial baseline occupies most of it. Every Brier comparison in §14, §15 and
§16 lives inside that 1-2% band. Those comparisons are not wrong, but the effect sizes are
far smaller relative to the achievable range than the tables implied, and describing Brier
as "the honest single number" (§14) was itself misleading.

**A constant predictor is better calibrated than every model here** — ECE 0.0002 against our
best 0.00255, roughly 12× better. That is not a paradox, it is the definition: predicting the
base rate is perfectly calibrated and completely useless. **So a low ECE is not evidence of a
good model**, and §12's and §16's "best ECE at every size" claims cannot carry weight on
their own. §13 flagged that our advantage was partly conservatism; this quantifies how far
that goes.

### What must change, and it is a real change

1. **Report skill, not raw scores.** Brier skill score against the constant-base-rate
   predictor, `1 − Brier_model / Brier_reference`, is the number that means something on an
   imbalanced problem. Ours is 1-2%.
2. **Never report calibration without discrimination.** A scorecard showing ECE and not AUC
   would rank a useless model first. Enforced by spec E2, but E2 did not anticipate the
   degenerate case, and the spec is being extended.
3. **Include the constant baseline in every benchmark**, permanently, as the reference row.
   Its absence is why four findings overstated their case.

### What this does *not* overturn

- §14's exit condition: the financial prior beats the generic one **on AUC**, and AUC is
  immune to this critique, since the constant predictor scores 0.5 there. The domain prior
  result stands.
- §15's finding that pretraining buys calibration: the untrained model's ECE of 0.53 and
  predicted mean of 39% against a 4.7% base rate are catastrophic by any reference.
- §16's finding that calibrated gradient boosting wins above ~1,000 rows: that was decided
  on AUC as well as Brier (0.8856 against 0.7159).

**The honest summary is that our discrimination results are sound and our calibration results
were measured on a scale too narrow to support the weight put on them.**

## 18. The prior matches real difficulty but is 3-5× too narrow, and that is the biggest fixable gap

**Date:** 2026-09-08. **Status:** MEASURED. Twelve sampled financial tasks against four real
panels, probing each with the same logistic-regression pipeline as a difficulty yardstick.

| property | real panels | synthetic prior | verdict |
| --- | --- | --- | --- |
| logistic-regression AUC | 0.7687 mean (0.697-0.899) | 0.7610 mean (0.565-0.981) | **well matched** |
| **feature count** | **64, 64, 64, 95** | **9-21 (mean ~16)** | **3-5× too narrow** |
| default rate | 3.2-6.9% | 1.0-16.4% | good coverage |
| mean \|feature correlation\| | 0.081-0.110 | 0.096-0.220 | **too correlated** |
| missingness | 0-1.5% | 0.08-13.0% | too much, probably harmless |

### The good news, and it refutes the obvious worry

**The prior is not too easy.** Mean logistic-regression AUC on synthetic tasks is 0.761
against 0.769 on real panels — 0.008 apart. The hypothesis that transfer is modest because
the model only ever saw easy problems is **refuted**. Difficulty is essentially calibrated,
which is a non-trivial thing to have got right by construction.

The spread is wider than real, though: synthetic tasks range from near-noise (0.565) to
near-separable (0.981) where real panels sit in 0.697-0.899. Some breadth is deliberate and
useful; tasks at 0.98 are probably teaching very little.

### The defect: the prior structurally cannot generate a wide table

Every synthetic task had **9 to 21 features even though `max_features=64` was requested.**
`prior/financial.py` draws from a fixed dictionary of about 20 named quantities plus at most
4 redundant or noise columns, so it is **hard-capped near 24 columns** regardless of the
configuration. The model is therefore pretrained on ~16-feature problems and evaluated on
64-95 feature ones.

That is a 3-5× width mismatch on the single axis the architecture is most sensitive to, and
it is the most plausible remaining explanation for why the trained model beats logistic
regression by only ~0.033 AUC (§17's ablation).

**The fix is also the realistic one.** Real credit datasets are wide *because they compute
many ratios from a few underlying accounts* — the Polish panel's 64 features are ratios over
one balance sheet and P&L. Our prior generates the accounts already; it simply does not
derive the ratio family from them. Generating tens of ratios per synthetic firm is both the
width fix and a more faithful model of how such datasets are actually built.

### The secondary defect: over-correlated features

Synthetic mean absolute inter-feature correlation runs 0.096-0.220 against 0.081-0.110 real,
with most synthetic tasks above the entire real range. Our exposed features derive from a
small set of latents (assets, revenue, debt, cash), so they are more mutually dependent than
real ratios. Expanding the ratio family will change this, in an unknown direction, and it must
be re-measured after — a wider set of ratios over the same latents could easily make it
worse.

### Consequence

This is now the highest-value R&D direction: it is our own component, it is measurably
mismatched to the target on a specific axis, and the fix is principled rather than a guess.
See `openspec/changes/prior-width-and-fidelity`.

## 19. Widening the prior fixed the width gap and broke the difficulty match; both now hold

**Date:** 2026-09-08. **Status:** MEASURED, verified on a seed not used during tuning.
Implements `prior-width-and-fidelity` tasks 15.1-15.3.

§18 found the prior structurally capped near 24 columns against real panels of 64-95.
`prior/financial.py` now generates a full set of **accounts obeying accounting identities**
(assets = liabilities + equity, current assets = cash + receivables + inventory, EBIT =
EBITDA − depreciation, net profit = EBIT − interest − tax) and derives a **ratio family** of
sampled numerator/denominator pairs over them, which is how real panels become wide.

| property | before | after (held-out seed) | real panels |
| --- | --- | --- | --- |
| features | 9-21 | **9-59** | 64-95 |
| logreg AUC mean | 0.7610 | **0.7429** | 0.7687 |
| logreg AUC range | 0.565-0.981 | 0.532-0.866 | 0.697-0.899 |
| mean \|corr\| | 0.096-0.220 | **0.084-0.185** | 0.081-0.110 |
| missingness | 0.1-13.0% | 0.2-15.3% | 0-1.5% |

### The regression, and the fix

Widening **broke the difficulty match**: more ratios gave a linear model more views of the
same distress signal, pushing synthetic logistic-regression AUC to **0.815** against 0.769
real. Task 15.3 says a lost difficulty match is a regression, so it was treated as one.

The knob is label sharpness — how deterministic default is given fundamentals. Lowering it
is **economically correct**, not a fudge: management quality, fraud, litigation and customer
concentration drive real defaults and appear in no ratio, so financials should explain only
part of the outcome. Swept (0.8, 3.0) → 0.835, (0.5, 2.0) → 0.716, (0.7, 2.7) → 0.764, and
kept (0.7, 2.7).

### Provenance note, because this deserves declaring

**The difficulty target was checked against our own evaluation panels, which is a mild use of
target-domain information to set a prior hyper-parameter.** It is not a P2 violation — no
real data enters training, nothing is conditioned on a real dataset, and a single aggregate
scalar cannot carry test instances — but it is not nothing either, and pretending otherwise
would be the kind of quiet erosion `openspec/specs/pretraining-provenance` exists to prevent.

The defensible version, and the one now written into the code comment: **published credit
scorecard performance sits around Gini 0.4-0.6, i.e. AUC 0.70-0.80**, which is domain
knowledge rather than our held-out data, and it brackets the chosen setting. Future
difficulty targets should be justified from the literature, never from the evaluation panels.
**Matching row-level or per-feature statistics to a real panel remains forbidden.**

### Honest residuals

- **Tuning was partly seed-specific.** The chosen range gave 0.764 on the tuning seed and
  **0.743** on a held-out seed, against 0.769 real. Both are far better than 0.815 and within
  0.026 of target, but the point estimate should not be quoted tightly.
- **Width still skews low**: 9-59 against 64-95, because the exposed count is sampled from a
  wide range. The cap is gone; the *distribution* is not yet centred on the target.
- Missingness remains higher than real, deliberately, for robustness.
- A test began failing and was **diagnosed, not loosened**: the model still learns, needing
  60 steps rather than 30 with harder tasks, so the step count was raised and the assertion
  left untouched.
- **Whether any of this improves transfer is unmeasured.** Task 15.5 retrains at matched
  compute against the old prior. Until then this is a fidelity improvement, not a
  performance one.

## 20. The incoherence is fixed by construction, not by training

**Date:** 2026-09-08. **Status:** MEASURED (0 violations in 12,000 horizon steps) and
**PROVED** (structural, see below). Implements `pd-term-structure` tasks 11.2-11.4.

§11 measured the defect: **11.0% of horizon steps and 39% of firms** received a cumulative
PD that *fell* as the horizon grew, invisible behind a monotone portfolio aggregate. §19's
"small because early or structural?" placed it firmly in the structural column — nothing in
the architecture or loss forbade a violation, so no amount of training or scale would close
it.

`fintfm.modeling.hazard.HazardHead` closes it by changing what is predicted. Instead of a
probability per horizon, the model emits a **per-period hazard**

    h_k = P(default in period k | survived to k)  in (0, max_hazard]

and the cumulative default probability is derived:

    F_k = 1 - prod_{j<=k} (1 - h_j).

Every factor `(1 - h_j)` lies in `(0, 1]`, so the survival function is non-increasing and
`F_k` is **non-decreasing in k for any parameters whatsoever**.

| | per-horizon (§11, real data) | hazard head |
| --- | --- | --- |
| step violations | **11.0%** | **0 / 12,000 (0.0%)** |
| fully monotone firms | **60.6%** | **100.0%** |
| guarantee | none | **structural** |

### Why "proved" is the load-bearing word

The guarantee does not depend on the weights being sensible. `test_monotone_even_with_adversarial_weights`
scales the projection weights by 500 and spreads biases across ±400, then checks 400 rows:
zero violations. A learned monotonicity *penalty* fails exactly there, and would also need
tuning, and would still leave a residual violation rate to explain to a reviewer. A
cumulative-product parameterisation has nothing to tune and nothing to explain.

This matters strategically because it is the project's **first advantage that is not a
scale advantage** (see §19's framing). A scale advantage must be bought and then defended
against better-funded teams. This one is arithmetic.

### The training objective changed too, and that is the deeper part

The loss is now the discrete-time survival likelihood: `h_t · prod_{j<t}(1 - h_j)` for a
firm defaulting in period `t`, and `prod_{j<=K}(1 - h_j)` for one that survives the grid.
That fits **the whole curve at once** rather than each horizon separately, so the horizons
become mutually *consistent* rather than merely non-contradictory. Independent per-horizon
models cannot do this even in principle, because they share no parameters.

### Now trainable end to end (added 2026-09-08, same day)

Task 11.3 is done: `prior/financial.py` samples a **default period** rather than only a
binary label. Per-period hazards are `sigmoid(b_k + shape_k + scale · distress)`, with the
profile sampled per task from rising, falling, hump-shaped or flat — real credit hazards bend
with seasoning, refinancing walls and cyclical exposure, and which way depends on the book.
The intercept is solved so the *cumulative* rate over the grid still hits the sampled 1-30%
target, so the base-rate range means what it did before.

`y` is exactly `period != CENSORED` by construction, asserted in tests, so a survival task
still trains a classifier unchanged. Measured end to end: **survival loss 2.091 → 0.530 over
60 steps, with 0 coherence violations in 1,120 horizon steps after training.**

Two guards that matter more than they look:

- **`collate` refuses to mix survival and binary-only tasks.** A padded period is
  indistinguishable from a real one and would train the likelihood against a fabrication.
- **`n_horizons` requires `p_financial = 1.0`.** The generic SCM prior has no time axis, so a
  mixed batch cannot carry a coherent survival likelihood. Refused with an explanation rather
  than silently degraded.

`fintfm-train --n-horizons 5` now trains the term structure.

### What is still not true
- **Untrained curves are not calibrated.** The 0.44 → 0.95 demonstration above is a
  random-initialisation artefact and says nothing about levels. Coherence and calibration are
  independent properties and this finding is only about the first.
- **No comparison against per-horizon models on accuracy.** Task 11.5. The survival objective
  is now trainable, so this is the next measurement, and it is the one that decides whether
  fitting the whole curve *also* helps discrimination or merely makes it coherent.
- **No real-data evaluation of the term structure.** The UCI panels have no firm identifiers
  (§7), so a per-firm hazard path cannot be scored against them. This needs V4FinBench, which
  is the only licensed panel with company-year rows.

## 21. Coherence is free: joint prediction matches per-horizon on AUC and eliminates incoherence

**Date:** 2026-09-08. **Status:** MEASURED on held-out synthetic survival tasks, single seed.
Resolves `pd-term-structure` task 11.5. Re-derivable:

```bash
uv run fintfm-termstruct --steps 1500 --device mps --out runs/term-structure
```

§20 proved a hazard parameterisation makes the term structure monotone by construction. That
left the question this experiment answers: **does fitting the whole curve cost discrimination?**

| arm | models | total steps | mean AUC | violations | fully monotone |
| --- | --- | --- | --- | --- | --- |
| **joint** (hazard head, survival loss) | 1 | 1,500 | 0.7693 | **0.00%** | **100.0%** |
| per-horizon, matched **total** compute | 5 | 1,500 | 0.7654 | 29.28% | 11.6% |
| per-horizon, matched **per-model** compute | 5 | **7,500** | 0.7725 | 11.88% | 56.4% |

### Discrimination: a tie, and that is the good news

Joint beats per-horizon by **+0.0039** at equal budget and loses by **−0.0032** when the
baseline is given **five times the compute**. Both differences are tiny and neither was
significance-tested, so the honest statement is **no measurable difference in
discrimination**.

That is the ideal shape for this thesis. Coherence is not a trade-off that has to be argued
against accuracy — it is **free**. And the joint model gets there with one model instead of
five, so it is also five times cheaper to train and to serve.

### Coherence: the gap is enormous and it worsens as budget shrinks

Zero violations against 11.88% and 29.28%. Note the direction: **per-horizon incoherence gets
worse as per-model compute falls** — 11.88% at 1,500 steps each, 29.28% at 300 steps each,
with only 11.6% of firms getting a coherent curve in the cheaper arm. So the defect is worst
exactly in the small-portfolio, limited-budget regime this project targets.

### The synthetic prior reproduced the real defect, which is independent evidence for it

The per-horizon arm at matched per-model compute measured **11.88% violations and 56.4%
fully-monotone firms**. §11 measured **11.0% and 60.6%** on real UCI data with a real
checkpoint. Those are close enough to be striking.

The prior was not built to reproduce this and no part of it was tuned toward it, so this is
**out-of-sample evidence that the synthetic prior captures a real structural property of
credit data** — a form of prior validation that owes nothing to AUC, and one of the few
positive signals today that is not about size.

### Confirmed across three seeds (added 2026-09-09), and it sharpens the claim

Task 11.7. Three seeds, 1,200 steps each, same held-out protocol:

| arm | mean AUC | violations | fully monotone |
| --- | --- | --- | --- |
| joint | 0.7653 ± 0.0033 | **0.00% ± 0.00%** | **100.0%** |
| per-horizon, matched total | 0.7645 ± 0.0008 | 19.05% ± 1.50% | 34.6% |
| per-horizon, matched per-model (5× compute) | **0.7737 ± 0.0002** | 22.81% ± 4.26% | 32.0% |

- **joint − matched total: +0.0008 ± 0.0025 → no difference.** Confirmed, not inferred.
  At equal budget, coherence is genuinely free.
- **joint − matched per-model: −0.0084 ± 0.0032 → the baseline is better**, and the tiny
  standard deviation makes this a real effect rather than noise. The single-seed run had
  shown −0.0032 and under-stated it.

**So the honest trade is now quantified, and it is more useful than "a tie".** Training five
models instead of one, at **five times the compute**, buys about **+0.008 AUC** — and returns
a curve that contradicts itself for **two firms in three**. For a regulated PD used in
provisioning that is a poor trade, and the point is that it can now be stated as a trade
rather than asserted as a win.

### What this still does not establish
- **Synthetic evaluation only.** The term structure cannot be scored on the UCI panels at all
  (no firm identifiers, §7). Real validation needs V4FinBench; the loader and a
  `fintfm-fetch v4finbench` path now exist and the data needs Kaggle credentials.
- **No paired bootstrap on the AUC differences**, only across-seed means and standard
  deviations. That is weaker than the Holm-corrected test used in §14 and should be upgraded
  before any external use.
- **Small models near a ceiling.** Giving the baseline 5× compute bought +0.003 AUC, which
  suggests every arm is close to what this size can do. The comparison may look different at
  scale, in either direction.
- **Nothing here is about calibration of the curve levels.** §17's warning applies: coherence
  and calibration are independent, and a monotone curve can still state the wrong numbers.

## 22. V4FinBench ingested, and its shape corrected two assumptions in our own code

**Date:** 2026-09-09. **Status:** MEASURED by inspection of the real files. Resolves
`second-credit-panel` task 7.5. Data CC BY 4.0, 4.8 GB, fetched with
`uv run fintfm-fetch v4finbench`.

| property | value |
| --- | --- |
| company-years (h=0) | **1,000,087** |
| unique companies | **188,338** |
| numeric features | **136** |
| years | **2006-2020** |
| per-horizon positive rate | 0.36% (h=0) to 0.19% (h=5) |
| cumulative default rate | ~1.4% across the grid |

**This is the first panel we hold with dates and firm identifiers**, so it is the first that
supports out-of-time validation and per-firm hazard paths at all (§7 established the UCI
panels support neither).

### Two bugs it caught in code already written

**The horizon files must be joined, not stacked.** Row counts fall from 1,000,087 at h=0 to
598,832 at h=5, because a five-year-ahead label requires five further years of data. The
first loader stacked them positionally after a sort, which would have **silently misaligned
companies** — every feature row paired with another firm's label. Caught only by inspecting
the real files rather than trusting the documented schema.

**Missing horizons are administrative censoring, not survival.** A company-year near the end
of the panel simply has fewer future labels. Measured on a 60,000-row sample, the
observation counts are `[0, 212, 5923, 6341, 5595, 6069, 35860]` for 0 through 6 horizons —
so only 60% of rows carry the full grid. Scoring the other 40% as long-run survivors would
bias every hazard downward. `HazardHead.loss` now takes a per-row `n_observed`, and a test
asserts that surviving five observed periods costs more likelihood than surviving two.

Neither error would have failed loudly. Both would have produced a trained model and
plausible numbers.

### Also worth noting

The real feature names are ratios — `Working_capital/total_assets`,
`Current_assets/short_term_liabilities`, `Equity/long_term_liabilities`. That is exactly the
account-and-ratio structure §19 rebuilt the prior around, arrived at independently from
reasoning about how such panels are constructed. Modest corroboration that the reasoning was
right.

The observed period histogram on that sample is `[212, 192, 134, 121, 82, 72]` — a declining
hazard, which is the seasoning shape `_sample_survival` already samples among its profiles.

### What this unblocks

- **Real-data evaluation of the term structure** (`pd-term-structure`), which was impossible
  on any panel we held.
- **`time-based-evaluation`**, blocked since §7 for want of dates.
- A second economy and accounting regime for every existing result.

## 23. Context size is not the constraint; domain match is. Home Credit scores 0.550

**Date:** 2026-09-09. **Status:** MEASURED, external leaderboard. This finding **refutes a
hypothesis I stated before testing it**, which is why it is recorded rather than quietly
dropped.

Submitted to Kaggle's Home Credit Default Risk (7,176 teams, closed 2018) using only
`application_train`/`application_test` — no auxiliary tables, no feature engineering, no
gradient step on competition data.

| | score |
| --- | --- |
| our held-out estimate | 0.5525 |
| **Kaggle public** | **0.54924** |
| **Kaggle private** | **0.55020** |
| competition winner | ~0.805 |
| random | 0.500 |

A bad result, and the predicted one. Note the held-out estimate landed within 0.003 of the
leaderboard, so the evaluation pipeline is at least honest about what it is producing.

### The hypothesis, and its refutation

I proposed that the model was **information-starved**: gradient boosting sees 267,000
training rows and we hand the model 2,000, a 130× disadvantage, with `max_context` set at
2,000 for CPU cost long before Metal was available. TabPFNv2 uses ~10K and TabICL scales to
500K, so the reasoning seemed sound.

Measured on 10,000 held-out rows, sweeping context with no retraining:

| context | AUC (balanced) | AUC (uniform) | seconds |
| --- | --- | --- | --- |
| 500 | 0.5634 | 0.5647 | 4 |
| 2,000 | 0.5606 | 0.5656 | 9 |
| 8,000 | 0.5606 | 0.5655 | 37 |
| 20,000 | 0.5608 | 0.5663 | 233 |

**A 40× increase in context buys 0.0016 AUC at 33× the inference cost.** Completely flat.
The model is not starved; it cannot extract signal from these features at any context size.

### What it is instead

Domain mismatch, and the contrast is stark. The same architecture and prior reach **0.78 AUC
on UCI corporate panels** (§14) and **0.77 on synthetic corporate tasks** (§21), against
**0.56 here**. Home Credit is consumer credit — bureau scores, employment, housing,
demographics — while the prior generates corporate balance sheets and P&L. There is no
leverage ratio, no interest coverage, no working capital. The relationships the prior teaches
do not exist in this data.

This is **evidence for the domain-prior thesis**, arriving from the unflattering direction: a
prior that matters is a prior whose absence hurts.

### Two side notes

- **Uniform context beat balanced here** (0.5663 against 0.5608) with better calibration, the
  reverse of §5's finding on 4% base rates. At Home Credit's 8.07% rate, balancing distorts
  more than it helps. Context strategy is base-rate dependent and should not be a fixed
  default.
- **`max_context = 2000` is not costing us anything** on this evidence, so raising it is not
  the improvement it appeared to be. Retest on corporate data before concluding generally.

### The honest answer to "how do we improve this number"

Build a consumer-credit prior. **We should not**, because consumer credit is a different
wedge, and 0.55 on a 2018 consumer competition is not a result worth optimising. The finding
worth keeping is the contrast between 0.78 corporate and 0.56 consumer.

## 24. The frontier labs are restricting their weights, and that is a market gap

**Date:** 2026-09-09. **Status:** MEASURED by reading the licences; the strategic reading is
inference and labelled as such.

Three independent data points, all from primary sources:

| model | code licence | **weights licence** |
| --- | --- | --- |
| Google TabFM | Apache-2.0 | **`tabfm-non-commercial-v1.0` — non-commercial, non-production** |
| Google TimesFM 3.0 | Apache-2.0 | **`timesfm-non-commercial-license-v1.0` — non-commercial** |
| Google TimesFM ≤ 2.5 | Apache-2.0 | Apache-2.0 |
| V4FinBench | MIT | data CC BY 4.0 |

TimesFM is the informative case: weights through 2.5 were Apache-2.0, and **3.0 — the
version that tops fev-bench, TIME and GIFT-Eval — is not**. The pattern is that as these
models become genuinely good, the weights stop being commercially usable.

**The inference, and it is judgment rather than measurement:** a commercial lender cannot
deploy the current best tabular or time-series foundation models at all. Not for want of
quality or money — the licence forbids it. That is a gap that no amount of accuracy work by
the frontier labs closes, because it is a deliberate business choice on their part.

**What follows for this project.** `docs/DECISIONS.md` D6 chose Apache-2.0 on the reasoning
that the moat is the weights and the prior rather than the code. This finding suggests the
sharper version: **commercially usable weights may themselves be the differentiator** in a
market where the best models are locked. That does not mean giving the weights away — it
means that whatever we do ship must be deployable in production by a regulated lender, which
the alternatives currently are not.

It also sets a trap to avoid: **never build on non-commercially-licensed weights**, however
convenient for a benchmark. Evaluating against published *numbers* is fine; running their
checkpoints inside anything commercial is not. Recorded in `CLAUDE.md`'s licensing boundary.

**Caveat:** licences change, in both directions. Re-read before relying on any of this, and
treat the table as of 2026-09-09.

## 25. Every gradient boosting comparison so far used the weakest member of the family

**Date:** 2026-09-09. **Status:** MEASURED by audit. **This weakens findings 12, 16 and 17
in our favour, so it is recorded prominently rather than quietly fixed.**

An audit of what `bench.py` actually runs found:

- **LightGBM has never executed once.** The import fails with `OSError` on a missing
  `libomp`, and §5's fix guarded the crash — which meant it was **silently skipped on every
  run**, exactly the failure that guard was added to prevent, one layer up.
- **CatBoost and XGBoost were never installed.** Not skipped: absent.

So every statement in this project of the form "gradient boosting beats us above N rows"
compared against **sklearn's `GradientBoostingClassifier`**, the weakest of the family. The
real gap against a tuned CatBoost or LightGBM is very likely **larger**, not smaller.

### And fixing it produced a harder problem

Installing Homebrew's `libomp` fixed the import and introduced a **segfault**. The crash
report shows two OpenMP runtimes in one process:

```
/Users/.../site-packages/torch/lib/libomp.dylib     (bundled with PyTorch)
/opt/homebrew/opt/libomp/lib/libomp.dylib           (Homebrew, loaded by LightGBM)
```

Confirmed by bisection: LightGBM alone works; `import torch` followed by LightGBM segfaults
(exit 139). **`KMP_DUPLICATE_LIB_OK=TRUE` does not fix it** — still 139.

The robust answer is process isolation: fit the boosting baselines in a subprocess that never
imports torch. Tracked in `openspec/changes/strong-baselines`.

### Two process notes worth keeping

**A guarded exception is not a passing test.** The guard added in §5 turned a crash into a
silent absence, and a silently absent baseline flatters us. Skips must be *announced*, which
`_boosting_family()` now does.

**I fell into the pipe trap again** while diagnosing this — `python ... | grep` reported
`exit=0` for a segfaulting process, because that is grep's status. It is written down in
`CLAUDE.md` and it still caught me, which is an argument for the harness enforcing it rather
than a human remembering.

### Measured, once the subprocess fix made it possible

Fitting the family out-of-process (`evaluation/boosting.py`) works with torch loaded. On the
UCI Polish panels, 30% held out, out-of-the-box on every side:

| panel | logreg | RF | **sklearn gboost** | lightgbm | **catboost** | xgboost |
| --- | --- | --- | --- | --- | --- | --- |
| 1-year | 0.6974 | 0.9400 | **0.9624** | 0.9799 | **0.9864** | 0.9761 |
| 3-year | 0.6990 | 0.8552 | **0.8953** | 0.9316 | **0.9353** | 0.9245 |
| 5-year | 0.7793 | 0.8824 | **0.9252** | 0.9426 | **0.9432** | 0.9390 |

**The weak baseline understated the field by +0.018 to +0.040 AUC.** CatBoost wins every
panel.

**The Brier-skill comparison is the one that hurts.** Against the feature-free reference
(§17), CatBoost scores **+59.9%, +41.7%, +46.4%** skill. This project's best model scores
**1-2%** (§17). That is not a narrow gap and no amount of framing closes it.

### Consequence, restated at the strength this evidence supports

Findings 12, 16 and 17 compared against a baseline 0.02-0.04 AUC weaker than the real field,
so **their margins are optimistic and their conclusions are directionally unchanged but
worse**. Specifically:

- §16's "above ~1,000 rows a calibrated gradient boosting is the better model" holds *more
  strongly*, and the crossover is probably lower than 1,000.
- §17's 1-2% Brier skill now has a reference point: a competent gradient booster achieves
  40-60% on the same panels. **Our headline calibration story does not survive contact with
  CatBoost's skill scores**, and the honest position is that the model is not yet competitive
  on discrimination *or* on proper-scoring-rule terms at these dataset sizes.

What survives untouched is coherence (§20, §21), which is a property no gradient booster has
at all, and provenance (§1). Those were already the thesis; this finding is why they must
remain it.

## 26. First out-of-time result on real corporate data: coherence holds, everything else fails — and the prior cannot generate the target regime

**Date:** 2026-09-09. **Status:** MEASURED on V4FinBench, out-of-time. The first evaluation
in this project that tests the actual thesis on real corporate panels with a date-based split.
Re-derivable:

```bash
uv run fintfm-v4oot --model runs/v4-hazard-136f.pt --max-rows 120000 \
    --train-until 2016 --test-from 2017
```

120,000 company-years: **72,622 train (≤2016), 47,378 test (≥2017)**, no year in both.
Train default rate 1.545%, test 0.998% — the split crosses a real change in conditions.

| arm | mean AUC | violations | fully monotone |
| --- | --- | --- | --- |
| **fintfm hazard head** | **0.5869** | **0.00%** | **100.0%** |
| per-horizon logistic regression | **0.8616** | 39.06% | 1.4% |

Per horizon:

| arm | h0 | h1 | h2 | h3 |
| --- | --- | --- | --- | --- |
| fintfm AUC | 0.6539 | 0.6020 | 0.5662 | 0.5257 |
| logreg AUC | **0.9717** | **0.8908** | **0.8310** | **0.7530** |
| fintfm ECE | 0.1085 | 0.1853 | 0.2519 | **0.3322** |
| logreg ECE | **0.0028** | **0.0014** | **0.0002** | **0.0016** |

### The one thing that held, and it held completely

**Coherence: 0.00% violations and 100% monotone curves, against logistic regression's 39.06%
and 1.4%.** Independent per-horizon models produced a self-contradicting term structure for
**98.6% of firms** on real out-of-time data. The structural guarantee (§20) transferred from
synthetic to real without qualification, and it is the only claim in this project that did.

### Everything else failed, and by a wide margin

Logistic regression — not CatBoost, not a tuned anything, **plain logistic regression** —
beat us by **0.27 mean AUC**, and its calibration is 40× to 200× better. Our ECE degrades
from 0.11 at the first horizon to **0.33 at the fourth**, meaning stated probabilities are off
by thirty-three percentage points against a base rate near 1%.

### The cause is diagnosed, specific, and embarrassing

`prior/financial.py` samples the base rate as `exp(U(log 0.01, log 0.30))`. Measured over
20,000 draws: **minimum 1.0003%, and exactly 0% of tasks fall below 1%**.

V4FinBench's cumulative default rates by horizon are **0.36%, 0.30%, 0.25%, 0.23%, 0.21%,
0.19%**.

**The model has never once seen a task as imbalanced as the target** — not in any of the
~96,000 synthetic tasks it trained on. Its entire calibration behaviour is anchored to a
1-30% regime and it is being asked about a 0.2-0.4% one. That explains the ECE rising with
horizon, since the cumulative rate *falls* with horizon here and the model pushes it up.

### The strategic problem this exposes

`docs/STRATEGY.md` targets **low-default portfolios** — a Basel category defined by having
very few defaults — and §9 named them as the wedge because that is where the incumbent's
remedies fail.

**Our prior cannot generate a low-default portfolio.** The floor is 1%; the regime we claim
starts an order of magnitude below it. The strategy and the prior have been pointing at
different problems since the prior was written, and nothing surfaced it until a real
low-default panel was scored.

This is the most actionable finding so far, and it is a one-line change to the sampling range
followed by a retrain — with §19's lesson attached: **widening the range will move task
difficulty, and difficulty must be re-measured, not assumed.**

### Honest reading

Do not read this as "the approach fails". Read it as: the thesis-critical property transferred
perfectly, and the model was asked a question its prior never posed. Whether fixing the base
rate closes the accuracy gap is **unknown and should not be assumed** — logistic regression at
0.97 AUC on h0 is a very strong baseline, and §25 suggests CatBoost would be stronger still.

## 27. Restated against CatBoost: the small-n win survives, and it is narrower than §16 claimed

**Date:** 2026-09-09. **Status:** MEASURED, single seed, `polish-bankruptcy-3y`, `mixed`
checkpoint. Resolves `strong-baselines` task 18.4. §25 established that every earlier
gradient-boosting comparison used sklearn's weakest implementation; this is the honest rerun.

| n (defaults) | model | AUC | ECE | Brier skill |
| --- | --- | --- | --- | --- |
| **100** (5) | **fintfm** | **0.6945** | **0.0070** | **+2.10%** |
| | catboost | 0.6940 | 0.0149 | +0.61% |
| | lightgbm | 0.6620 | 0.0500 | **−11.26%** |
| | xgboost | 0.6193 | 0.0355 | **−3.14%** |
| 250 (12) | fintfm | 0.6872 | **0.0031** | +1.92% |
| | **catboost** | **0.7791** | 0.0282 | **+4.21%** |
| 1,000 (47) | fintfm | 0.7154 | **0.0040** | +1.64% |
| | **catboost** | **0.8234** | 0.0251 | **+8.70%** |
| 4,000 (189) | fintfm | 0.6827 | **0.0013** | +1.05% |
| | **catboost** | **0.9066** | 0.0156 | **+35.17%** |

### At 100 rows we win on all three metrics, against the real competition

Best AUC (marginally, over CatBoost), best calibration by 2×, and best Brier skill by 3×.
This is the first time this project has beaten a competent gradient booster on a proper
scoring rule, and it happens exactly where the strategy says the market is.

**LightGBM and XGBoost post *negative* Brier skill at n = 100** — worse than a predictor that
ignores every feature and returns the base rate. That is §16's argument, measured against the
strong family rather than argued: thin books punish the incumbent, and here two of the three
best boosters are actively harmful on five defaults.

### But the window is narrower than §16 said

§16 put the Brier crossover between 250 and 1,000 rows. Against CatBoost it is **between 100
and 250** — CatBoost already leads on AUC and skill at 250. The usable window is therefore
roughly *under 200 obligors*, not under a thousand.

By 4,000 rows CatBoost reaches +35.17% skill against our +1.05%, so above the window this is
not a contest.

### What holds across every size

**Best ECE at every n tested**, by 2× to 20× (0.0013-0.0070 against 0.0149-0.0500). §12's
calibration claim survives contact with the strong family — unlike its 11.7× magnitude, which
§16 already retired.

### Consequence

§16 is **narrowed, not retracted**: the claim is now "below roughly 200 obligors we are the
best available model on a proper scoring rule, and the best calibrated at any size". That is a
smaller market than a thousand-obligor ceiling, and it is still precisely the low-default
segment §9 identified — and the one §26 showed the prior could not even generate until today.

**Single seed.** The n = 100 win is 0.0005 AUC over CatBoost, which is noise; the skill and
ECE margins are larger but still one draw. Three seeds before this is quoted anywhere.

---

## 28. §26's headline was our own bug: the term-structure path skipped the base-rate correction

**Date:** 2026-09-09. **MEASURED.** **Command:**

```bash
uv run fintfm-v4oot --model runs/v4-hazard-ldp.pt --max-rows 120000 \
    --train-until 2016 --test-from 2017 --out runs/v4-oot-corrected
```

§26 reported the hazard head stating a **12.8% mean default probability against a 0.47%
observed rate** out of time, with ECE degrading to 0.3322 by the fourth horizon. It
diagnosed the cause as the synthetic prior being unable to generate default rates below 1%,
and a 6,000-step retrain was spent widening the prior's floor to 0.195%.

**The diagnosis was wrong, and the retrain was not what fixed it.** The out-of-time harness
constructed its own forward pass — it fitted the classifier, reached into `_ctx_X`/`_ctx_y`,
and called `FinancialTFM.term_structure` directly. That call bypasses `predict_proba`, which
is the only place the base-rate correction of §6 and decision D5 was ever applied. So the
context was **resampled to 50% defaulters against a 1.545% population** and the model, doing
exactly what an in-context model is supposed to do, read the base rate out of its context and
reported it.

### The correction, applied on the same checkpoint and the same split

| horizon | observed | uncorrected | corrected | ECE uncorrected | ECE corrected |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.47% | 12.78% | **0.25%** | 0.1231 | **0.0023** |
| 1 | 0.34% | 21.49% | **0.48%** | 0.2115 | **0.0014** |
| 2 | 0.20% | 29.03% | **0.76%** | 0.2883 | **0.0056** |
| 3 | 0.08% | 36.85% | **1.22%** | 0.3677 | **0.0114** |

A **32× reduction in calibration error at the fourth horizon**, from one inference-time line.
At the first two horizons the corrected curve is now as well calibrated as per-horizon
logistic regression (0.0023 against 0.0028, and 0.0014 against 0.0014).

**Mean AUC did not move at all** — 0.5967 before and after, identical to four decimals. That
is not luck: the correction is a strictly increasing map applied elementwise, so it cannot
reorder rows at a fixed horizon, and it cannot break monotonicity either. Both properties are
now asserted rather than argued (`tests/test_hazard.py`).

### Why it went undetected

Three separate safeguards each failed to see a 27× level error:

- **AUC could not see it.** Every prediction inflates alike, which is the same blindness §5
  found and D5 was written to address.
- **The coherence check could not see it.** The curve was 100% monotone throughout. A
  perfectly coherent curve can state perfectly wrong levels.
- **The harness summary did not print the level.** `mean_predicted` was computed, stored in
  the JSON, and never rendered. It now prints beside the observed rate, because the one
  number that would have caught this was one column away from being visible.

### What changed, so it cannot recur

`FinancialTFMClassifier.predict_term_structure` is now the public path and applies the
correction itself; `base_rate_shift` and `shift_cumulative_pd` live in
`fintfm.modeling.hazard` with the monotonicity and rank-preservation properties tested. The
harness keeps an **uncorrected arm permanently**, so the distortion is measured beside the fix
rather than reasoned about.

### The rule this earns

**A correction that lives in one method will be bypassed by the next caller.** D5's base-rate
correction was implemented, tested, documented and reversed a decision — and then a second
entry point walked straight past it. The correction belongs on the object, not in one of its
methods, and an alternative forward path through a model is a defect even when it produces
numbers.

**And: a level error is a context problem before it is a prior problem.** The cheap check —
compare the context's base rate against the population's — takes one line and would have
saved 6,372 seconds of retraining. It is now printed by the harness on every run.

---

## 29. Balanced context sampling costs 10-12 AUC points here, reversing the default we adopted from the literature

**Date:** 2026-09-09. **MEASURED**, single seed per cell. **Command:**

```bash
uv run fintfm-ctxsweep --model runs/v4-hazard-ldp.pt --out runs/context-sweep
```


D5 adopted `context_strategy="balanced"` as the default on published evidence: Tanna et al.
(2026) benchmark seven context-construction strategies for credit-risk TFMs and report
balanced and hybrid sampling worth 3-4 AUC points over uniform. Once §28's correction made
the levels readable, the strategies could be compared on the real out-of-time split, and the
ordering is **the reverse of the published one, by three times the published margin**:

| max context | strategy | context rate | positives in context | mean AUC | AUC h0 | mean ECE |
| --- | --- | --- | --- | --- | --- | --- |
| 1,000 | balanced | 50.00% | 500 | 0.6255 | 0.7143 | 0.0041 |
| 1,000 | hybrid | 25.60% | 256 | 0.6794 | 0.8253 | 0.0045 |
| 1,000 | **uniform** | 1.20% | **12** | **0.6921** | 0.8026 | 0.0081 |
| 2,000 | balanced | 50.00% | 1,000 | 0.5967 | 0.6839 | 0.0052 |
| 2,000 | hybrid | 25.40% | 508 | 0.6468 | 0.7435 | 0.0051 |
| 2,000 | **uniform** | 1.55% | **31** | **0.7192** | **0.8398** | 0.0077 |
| 4,000 | balanced | 28.05% | 1,122 | 0.6013 | 0.7007 | 0.0051 |
| 4,000 | hybrid | 25.07% | 1,003 | 0.6118 | 0.7149 | 0.0052 |
| 4,000 | **uniform** | 1.73% | 69 | 0.6986 | 0.8168 | 0.0073 |

**Uniform wins at every context size, and the ordering uniform > hybrid > balanced holds in
all three.** Three context sizes give three replications of the ordering, which is why this is
reported at all from a single seed.

**It is not about how many defaults the context contains.** Uniform with **12** positives
(0.6921) beats balanced with **1,122** positives (0.6013). Mean AUC tracks the context's
*default rate* monotonically and ignores the positive count, which rules out the obvious
"balanced supplies more signal about the rare class" mechanism.

**The likely mechanism is the majority class, not the minority one.** A balanced 2,000-row
context spends half its budget on 1,000 of 71,500 non-defaulters — 1.4% of that class — so
the model's picture of a *healthy* firm is drawn from a thin and unrepresentative sample.
Uniform preserves the covariate distribution of both classes. In a low-default portfolio the
majority class is where nearly all the information about the decision boundary lives, and
balancing is precisely the operation that throws it away.

**Calibration goes the other way, mildly.** Balanced-plus-correction is slightly better
calibrated (mean ECE 0.0041-0.0052 against 0.0073-0.0081). So the choice is a real trade, not
a free win: roughly 10 AUC points for roughly 0.003 ECE. At that exchange rate, uniform.

**Scope, honestly.** One dataset, one checkpoint, one seed, and the *survival* path only. The
binary-classification evidence behind D5 has not been re-measured under uniform sampling, so
the class default is unchanged pending that; see
`openspec/changes/revisit-context-strategy`. Tanna et al. are not contradicted on their own
setting — theirs is single-horizon classification on different panels, and this is a
six-horizon hazard model evaluated out of time.

---

## 30. The low-default retrain did pay off — in calibration, and only once §28's bug was out of the way

**Date:** 2026-09-09. **MEASURED.** **Command:** `fintfm-ctxsweep` run against each
checkpoint in turn, identical split and protocol.

§26 concluded the prior's 1% base-rate floor was the problem and drove a 6,000-step retrain
that lowered it to 0.195%. §28 showed the floor was not what caused the reported failure. The
fair question is then whether the retrain bought anything at all, and it did:

| checkpoint | strategy | mean AUC | AUC h0 | mean ECE |
| --- | --- | --- | --- | --- |
| old, 1% floor | balanced | 0.5869 | 0.6539 | 0.0055 |
| old, 1% floor | uniform | 0.6965 | 0.8018 | 0.0209 |
| **LDP, 0.195% floor** | balanced | 0.5967 | 0.6839 | 0.0052 |
| **LDP, 0.195% floor** | **uniform** | **0.7192** | **0.8398** | **0.0077** |

Under the configuration that actually works, the retrain is worth **+0.023 mean AUC, +0.038
at the first horizon, and 2.7× better calibration** (ECE 0.0209 → 0.0077). The calibration
gain is the one that matches the mechanism: a prior that can generate a 0.2% default rate
produces a model that can state one.

**Note what this says about the previous configuration.** Under balanced-plus-uncorrected the
retrain looked worth +0.010 AUC and nothing else — a broken evaluation configuration hid a
2.7× improvement and made a correct change look like a null result. Both arms had to be right
before either effect was visible.

### Where this leaves the out-of-time standing

Best configuration to date on V4FinBench out of time: LDP checkpoint, uniform context,
corrected. **Mean AUC 0.7192, mean ECE 0.0077, 0% coherence violations.** Per-horizon logistic
regression on the same split: **mean AUC 0.8616, mean ECE ~0.0018, 39.06% violations** with
only 1.4% of firms fully monotone.

**We are still behind on both discrimination and calibration on this benchmark.** The AUC gap
narrowed from 0.27 to 0.14 and the calibration disaster is gone, but §26's honest conclusion
stands in weakened form: coherence is the only dimension where this model leads, and it leads
there by construction rather than by learning.

**The largest remaining defect is identified and unfixed.** The context is labelled with
binary `y` only — who defaulted, never *when* — so the hazard head must shape a six-horizon
curve with no timing evidence in context whatsoever. That is consistent with what the AUC
column does: 0.8398 at the first horizon, decaying to 0.5968 by the fourth. Supplying
per-horizon context labels is `openspec/changes/survival-context-labels`, and it is now the
top-priority change in the queue.

---

## 31. Correcting §30's own diagnosis: 84% of the out-of-time gap is horizon-independent, so it is a context-size problem, not a timing one

**Date:** 2026-09-09. **MEASURED**, from `runs/v4-oot-corrected/v4_out_of_time.json` and the
context sweep. **This finding corrects a claim made in §30 and in
`openspec/changes/survival-context-labels` earlier the same day.**

§30 read the hazard head's decay across horizons — 0.8398 at the first, 0.5968 at the fourth
— as the signature of a context that says *who* defaulted and never *when*, and the proposal
written from it called the widening gap "the signature of missing timing evidence". Decomposing
the gap shows that framing is wrong:

| horizon | ours (uniform) | per-horizon logreg | gap | our decay from h0 | its decay from h0 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.8398 | 0.9717 | 0.1319 | — | — |
| 1 | 0.7559 | 0.8908 | 0.1348 | −0.0838 | −0.0809 |
| 2 | 0.6842 | 0.8310 | 0.1468 | −0.1556 | −0.1407 |
| 3 | 0.5968 | 0.7530 | 0.1561 | −0.2429 | −0.2187 |

**The baseline decays almost exactly as fast as we do** — −0.2187 against −0.2429 across the
grid. Far horizons are simply harder for everyone, which is what a five-year default forecast
should look like. Our *excess* decay is 0.0242 over three steps.

So the gap decomposes into a **constant 0.132 deficit present already at the first horizon**
plus a 0.024 widening. **84% of the deficit is horizon-independent.** Fixing the term
structure's timing information addresses, at most, the 16%.

### Task 31.1's test was confounded, and the confound is instructive

The cheap premise test — restrict the context to firms observed across the whole grid — moved
mean AUC by +0.006, but the effect was **largest at the first horizon** (+0.0102) and smallest
at the fourth (+0.0018), the opposite of what the timing hypothesis predicts.

The reason is that `n_observed` is censored *by default itself*: a firm defaulting in year two
has two observed horizons. So filtering to full-grid observation removes the defaulters, and
it took the context's default rate from 1.54% to **0.19%**. That is not an observation-depth
manipulation, it is a base-rate manipulation, and the AUC it bought is what §29 already
predicts from a lower context rate. **No cheap unconfounded test of the timing hypothesis
exists**, because the only available proxy for observation depth is entangled with the label.

### What the constant deficit actually is

It is §16 and §27 restated on real data. Per-horizon logistic regression is *fitted on all
72,622 training rows*; the hazard head sees a **2,000-row context**. That is the known
crossover — above a few hundred rows a fitted model wins — appearing exactly where it should.

**And it cannot be closed by enlarging the context.** From the sweep, uniform context at
1,000/2,000/4,000 rows scores 0.6921 / **0.7192** / 0.6986 mean AUC. It peaks at 2,000 and
*falls* at 4,000, because the model was pretrained on tasks of 256-1,024 rows and a 4,000-row
context is out of distribution. Pretraining on larger tasks is the obvious response and
`docs/COMPUTE.md` prices it out: 2,048-row tasks cost 32× per step and 4,096-row tasks 500×.

### Consequence for the queue

**Choosing *which* 2,000 rows enter the context is the lever; supplying more rows is not.**
That promotes `retrieval-context` to the top of the queue and demotes
`survival-context-labels`, which now has an honest expected ceiling of roughly 0.024 AUC rather
than the 0.03+ its pre-registered prediction claimed. The pre-registration stands as written —
it was recorded before the run and it is now expected to fail, which is the point of writing
it down.

**The lesson, and it is the second one today.** §28's wrong diagnosis blamed the prior for what
was a context bug. §30's wrong diagnosis blamed the context labels for what is a context *size*
limit. Both times the story was built from a pattern in the numbers before the pattern was
decomposed. **Decompose before diagnosing**: a monotone trend and a constant offset look
identical in a summary table and imply completely different work.

---

## 32. Retrieval is the first change that moves accuracy: +0.066 to +0.095 AUC over the best blind strategy, Holm-significant

**Date:** 2026-09-09. **MEASURED**, single seed. **Command:**

```bash
uv run fintfm-ctxsweep --model runs/v4-hazard-ldp.pt --out runs/context-sweep-retrieval
```

§31 concluded that *which* rows enter the context was the only remaining lever on the
horizon-independent part of the out-of-time gap. It is, and it is a large one.

| max context | balanced | hybrid | uniform | **retrieval** | seconds (uniform → retrieval) |
| --- | --- | --- | --- | --- | --- |
| 1,000 | 0.6255 | 0.6794 | 0.6921 | **0.7507** | 24 → 44 |
| 2,000 | 0.5967 | 0.6468 | 0.7192 | **0.7872** | 33 → 73 |
| 4,000 | 0.6013 | 0.6118 | 0.6986 | **0.7934** | 61 → 155 |

Paired bootstrap against uniform at matched protocol, Holm-corrected across four horizons:

| horizon | retrieval − uniform | 95% CI | Holm p | verdict |
| --- | --- | --- | --- | --- |
| 0 | **+0.0665** | [+0.0539, +0.0801] | 0.0000 | significant |
| 1 | **+0.0632** | [+0.0408, +0.0867] | 0.0000 | significant |
| 2 | **+0.0952** | [+0.0501, +0.1408] | 0.0000 | significant |
| 3 | +0.0719 | [−0.0311, +0.1703] | 0.1510 | not significant |

The fourth horizon has 18 positives among 23,099 observed rows, so its interval spans zero
regardless of the effect. **This is the first accuracy improvement in the project that
survives a family-wise correction.**

### It also corrects §31: more rows do help, if they are the right rows

§31 stated that supplying more context does not work, from uniform peaking at 2,000 rows and
falling at 4,000. Retrieval **rises monotonically** across the same sizes — 0.7507, 0.7872,
0.7934 — and its calibration improves with size too (ECE 0.0257 → 0.0169 → 0.0111). So the
ceiling §31 identified was a property of *random* rows going out of distribution, not of
context size. The corrected statement: **beyond about 2,000 rows, additional context only
helps when it is selected for relevance.**

### The bug this produced first, and why it was inevitable

Retrieval initially scored **0.3679 mean AUC — far below chance**, which is the signature of
inverted ranking rather than of a weak method. The cause was applying the base-rate correction
**per query group**.

That correction is exact under label shift, and the docstring stating so is explicit about
why it holds: "resampling selects on ``y`` alone, so it holds by construction here."
**Retrieval selects on ``x``, so the assumption is violated by construction** — and the
failure is not subtle. A risky cluster retrieves risky neighbours, so its context rate is
high, so it receives the *largest downward* shift. The correction was systematically pushing
the riskiest firms down hardest, erasing exactly the between-group differences retrieval
exists to find.

Turning it off restored 0.7872. The fix in the code is a **single pooled shift** applied to
every query, which is constant and therefore cannot reorder anything, asserted in
`tests/test_retrieval.py`. Note the pooled correction costs a little calibration and buys no
ranking (ECE 0.0119 uncorrected against 0.0169 pooled at 2,000 rows), because there is not
much left to correct once the context is no longer resampled on the label.

**This is the third time today that a documented assumption was carried into a context where
it did not hold** — §28 bypassed the correction, §30 misattributed a level error, and here the
correction was applied where its own stated precondition fails. The pattern is not
carelessness about the assumption; it is that each new code path silently inherits it.

### Where it leaves the comparison

| arm | mean AUC | mean ECE | coherence violations |
| --- | --- | --- | --- |
| **fintfm, retrieval, 4,000** | **0.7934** | 0.0111 | **0.00%** |
| fintfm, uniform, 2,000 | 0.7192 | 0.0077 | 0.00% |
| per-horizon logistic regression | **0.8616** | **~0.0018** | 39.06% |

Against the incumbent, Holm-corrected: **−0.0654, −0.0716, −0.0516 at the first three
horizons, all significant**; the fourth is inconclusive. The mean AUC gap has closed from
0.142 to 0.068 and the first-horizon gap from 0.132 to 0.066 — **halved, not closed.** We are
still behind a logistic regression fitted on 72,622 rows, and saying otherwise would require
ignoring three significant negative differences.

**Cost:** roughly 2.5× the scoring time of blind sampling, and it gives up batch independence
— a query's prediction depends on its group-mates. Both are documented in
`fintfm.inference.retrieval`; neither is fatal for a batch scoring job, and both would matter
for a real-time API.

### A trap in our own API, recorded because it corrupted a write-up

`holm_bonferroni` returns **booleans**, and the first version of this finding printed them as
if they were adjusted p-values. `True` formatted as `1.0000` and read as "not significant",
inverting every verdict — so the first pass at these numbers concluded retrieval's wins were
insignificant and its one *insignificant* horizon was the significant one. Caught only because
the pattern was backwards on inspection: the horizon with the widest confidence interval was
the one being reported as significant. `holm_adjusted_p` now exists beside it, and both are
named for what they return.

---

## 33. Three seeds: retrieval replicates cleanly, and §32's "rises monotonically" was noise

**Date:** 2026-09-09. **MEASURED**, three seeds per cell. **Command:**

```bash
uv run fintfm-ctxsweep --model runs/v4-hazard-ldp.pt --seeds 0,1,2 --out runs/context-sweep-seeds
```

Every context-strategy result before this rested on one draw per cell. Replicated:

| max context | balanced | hybrid | uniform | **retrieval** |
| --- | --- | --- | --- | --- |
| 1,000 | 0.6206 ± 0.0236 | 0.6471 ± 0.0464 | 0.7052 ± 0.0114 | **0.7591 ± 0.0077** |
| 2,000 | 0.5938 ± 0.0094 | 0.6349 ± 0.0108 | 0.7071 ± 0.0111 | **0.7888 ± 0.0029** |
| 4,000 | 0.6106 ± 0.0110 | 0.6129 ± 0.0085 | 0.6903 ± 0.0086 | **0.7890 ± 0.0062** |

**Retrieval's worst seed beats uniform's best seed at every context size** — the distributions
do not overlap at all. §29's and §32's orderings both survive replication.

**Retrieval is also the most stable strategy**, at ±0.003 to ±0.008 against ±0.009 to ±0.046
for the blind ones. That follows from the mechanism rather than being a coincidence: a
retrieved context is determined by the data, so the random draw has much less left to
influence. Worth noting because hybrid at 1,000 rows has a ±0.046 spread — wide enough that
single-seed comparisons between blind strategies were never safe.

### The correction to §32

§32 stated that retrieval "rises monotonically" across context sizes — 0.7507, 0.7872, 0.7934
— and drew from that the conclusion that more rows help when they are retrieved. On three
seeds, **2,000 and 4,000 rows are tied** (0.7888 ± 0.0029 against 0.7890 ± 0.0062), so the
0.7934 was a favourable draw.

The corrected statement: retrieval improves on uniform *at every size*, and gains from 1,000
to 2,000 rows, then **plateaus**. §31's ceiling on context size was real; retrieval raises the
plateau's height without moving where it starts. Practically this makes **2,000 the operating
point**, since 4,000 costs 1.7× the time for nothing.

Calibration does keep improving with size (ECE 0.0223 → 0.0178 → 0.0145), so a
calibration-first configuration would still prefer 4,000. It remains 2× worse than uniform's
0.0072 either way.

---

## 34. A hazard checkpoint's classification head is untrained, and `predict_proba` served it anyway

**Date:** 2026-09-09. **MEASURED**, found while testing something else.

The training loop selects one objective per step:

```python
if model.hazard is not None and batch.period is not None:
    loss = model.survival_loss(...)
else:
    loss = model.loss(...)
```

It is an `if/else`, so **a hazard checkpoint never optimises the classification head at all**.
That head keeps its random initialisation, and `predict_proba` ran it without complaint. On
`polish-bankruptcy-3y`, using the hazard checkpoint through the classification path:

| metric | value | what it should be near |
| --- | --- | --- |
| AUC | **0.3745** | ≥ 0.5 |
| ECE | **0.6905** | ~0.01 |
| mean predicted | ~0.69 | 0.047 |

A model asserting a 69% default probability against a 4.7% base rate, ranking *worse than
chance*, with no error raised. This was found only because those numbers were too absurd to
belong to the hypothesis being tested — a subtler version would have been believed, and the
same checkpoint is the one every §28-§33 result uses through the survival path, where it is
correct.

**Fixed** by recording the objectives actually optimised into the checkpoint
(`trained_objectives`) and refusing to serve a head that is not among them.
`predict_proba` and `predict_term_structure` each assert their own. Checkpoints written before
this field carry no record and are allowed through, because refusing them would break every
in-memory model; that is a deliberate hole and the reason the field is written on save rather
than inferred on load.

**Not fixed:** the two objectives are still exclusive, so no single checkpoint can serve both
paths. Training them jointly is `openspec/changes/joint-objective-training`. Until then a
hazard model is a term-structure model and nothing else.

**The pattern, for the fourth time today.** An interface offered a capability the artefact
behind it did not have, and said nothing. §28 was the same shape: a path existed that skipped
a required step. The lesson recorded there — that a correction must be a property of the
object rather than of one method — generalises to heads: **a head that was never trained
should not be reachable.**

---

## 35. Rank-transforming features is worth +0.086 AUC, and it resolves D9

**Date:** 2026-09-09. **MEASURED**, three seeds, three panels.

`normalize_features` standardises each feature by its **context mean and standard deviation**,
then clips to ±10. That is the standard PFN treatment, and on this data it is close to
useless. Measured on the V4FinBench training rows:

| statistic | value |
| --- | --- |
| median ratio of standard deviation to interquartile range | **240** |
| 90th percentile | 4,687 |
| features with standard deviation above 10× their IQR | **110 of 136** |

A ratio is a quotient, and a firm heading for default is exactly where denominators go small,
so this is the normal case here rather than the tail case. One such firm inflates the standard
deviation enough to collapse every other firm toward zero; the ±10 clip bounds the outlier and
does nothing about the collapse.

Conditioning the features first — fitted on training rows only, NaN preserved — on the
V4FinBench out-of-time split, three seeds:

| strategy | transform | mean AUC | mean ECE |
| --- | --- | --- | --- |
| uniform | none | 0.7071 ± 0.0111 | 0.0075 |
| uniform | **rank** | **0.7930 ± 0.0038** | 0.0073 |
| retrieval | none | 0.7888 ± 0.0029 | 0.0178 |
| retrieval | **rank** | **0.8118 ± 0.0032** | **0.0126** |

**+0.086 for uniform and +0.023 for retrieval**, with calibration improving too. The
asymmetry is explained by the mechanism: retrieval's distance metric already normalised on
median and IQR, so retrieval was partly compensating for the tails before the model saw them.
Winsorising at the 1st and 99th percentile gets most of the way there (0.8067) and the rank
transform beats it, which says the problem is the *shape* of the distribution and not only its
extremes.

On the binary path, two independent panels, three seeds, the rank transform improves AUC in
**seven of eight** configurations (the exception is Polish hybrid, −0.005):

| dataset | balanced | hybrid | uniform | retrieval |
| --- | --- | --- | --- | --- |
| polish, none | 0.6943 | 0.7037 | 0.6719 | 0.7092 |
| polish, **rank** | 0.6989 | 0.6988 | 0.6969 | **0.7120** |
| taiwan, none | 0.8713 | 0.8723 | 0.8743 | 0.8745 |
| taiwan, **rank** | 0.8879 | 0.8887 | 0.8932 | **0.9062** |

Retrieval beats balanced on both panels with Holm-corrected significance: **+0.0138**
[+0.0026, +0.0261] on Polish and **+0.0154** [+0.0072, +0.0235] on Taiwan. So retrieval is now
the best strategy on **all three panels and both prediction paths**.

### D9 resolves, and §29's mechanism was too broad

D9 left `context_strategy` at `"balanced"` pending binary-path evidence. Here it is: on the
binary path the strategies are **nearly tied** — uniform − balanced is −0.0004 (not
significant) on Polish and +0.0077 on Taiwan. §29's 10-12 point margin does **not** generalise
to this path.

The reason is that "balanced" is not one operation. Polish 3-year has roughly 500 positives
against a 2,000-row budget, so balanced *cannot* reach 50/50 and lands near 25% — much closer
to hybrid than to the 50% it reaches on V4FinBench's 1,122 positives. **The effect scales with
how extreme the rebalancing actually is, not with the strategy's name**, which is a narrower
and more useful claim than §29's.

### Defaults changed, on this evidence

- `feature_transform` now defaults to **`"rank"`**.
- `context_strategy` now defaults to **`"uniform"`**, not `"balanced"`: never worse than
  balanced in any measurement here, and blind, so it preserves the batch independence that
  retrieval gives up.
- **Retrieval is the recommended accuracy setting** but stays opt-in, because it costs ~2.5×
  the scoring time and makes a prediction depend on its query group-mates. A default that
  silently breaks batch independence is the category of hidden behaviour that produced §28.

Every number recorded before 2026-09-09 was produced with `balanced` and no transform; pass
both explicitly to reproduce them.

### Where this leaves the gap

| arm | mean AUC | mean ECE | violations |
| --- | --- | --- | --- |
| **fintfm, retrieval + rank, 2,000** | **0.8118** | 0.0126 | **0.00%** |
| fintfm this morning (§26) | 0.5869 | ~0.19 | 0.00% |
| per-horizon logistic regression | **0.8616** | **~0.0018** | 39.06% |

The mean-AUC gap has gone **0.142 → 0.048** over the day, and at the first horizon 0.132 →
0.030 (0.942 against 0.9717). Still behind on both discrimination and calibration, and the
remaining gap is now small enough that closing it is a plausible target rather than a hope.

---

## 36. V4FinBench's published protocol is not out-of-time, and their best method pre-empts our §29 mechanism

**Date:** 2026-09-09. **Status:** MEASURED by others, read from the full PDF rather than an
abstract or a summary — see the method note at the end, which is the reason that distinction is
spelled out. **Source:** Kostrzewa, Tomczak, R. Furman, Poberezhna, Furgała, Farganus,
O. Furman, Zięba, *V4FinBench: Benchmarking Tabular Foundation Models, LLMs, and Standard
Methods on Corporate Bankruptcy Prediction*, arXiv:2605.10896v2, 13 May 2026.

`public-benchmark-claim` task 33.1. Every number this project has produced on V4FinBench uses
an out-of-time split of our own design, and the published results had never been read in. They
have now, and **our numbers are not comparable to theirs for four independent reasons.**

### Their protocol (§4 of the paper, quoted in substance)

**5-fold stratified cross-validation with company-level grouping within country.** All
observations from a company go to the same fold; country proportions are preserved across
folds. Per iteration: one fold test, one validation, three training — approximately 60/20/20.
**Fold indices are released and shared across all methods.** Missing values are imputed with
training-set medians and features standardised with training-set statistics, computed
separately within each fold. Metrics are accuracy, precision, recall, F₁ and ROC-AUC,
fold-averaged with standard deviations; given the 0.19-0.36% positive rate, **F₁ and ROC-AUC
are the primary metrics**. Decision thresholds are calibrated on the validation fold by
maximising F₁ on the precision-recall curve, then applied unchanged to the test fold.

### The four reasons our numbers cannot be placed against theirs

1. **It is cross-validation, not out-of-time.** Grouping is by *company*, not by date, so a
   fold can contain 2019 observations while predicting a 2008 one. Out-of-time is the harder
   split, so reporting our numbers against theirs would understate us — but it would also be
   the same category of error §26 made in the opposite direction, and "understates us" is not
   a licence.
2. **The horizon tasks are built on different rows.** For horizon *h*, a distressed company
   has its **final *h* years of data removed** and the resulting final observation is labelled
   positive. Our `_curve_truth` derives cumulative labels for a *fixed* row from its `period`.
   Their horizon 3 and our horizon 3 are not the same prediction.
3. **Their inference context is 10,000 rows**; ours is 2,000 (Table 3: `n_inference_context`
   = 10 000). §33 measured our own model *degrading* past 2,000, so this is not a knob we can
   simply match.
4. **Their TabPFN is fine-tuned on V4FinBench** — 10 epochs, learning rate 5×10⁻⁶, batch 1024,
   on an A100, taking 35:43 at horizon 0 (Tables 3 and 4). Ours never touches real data, which
   is decision D2's entire point. Their number answers "can a TFM be adapted to this data";
   ours answers "can a synthetic-only TFM transfer to it". Different questions.

Their Table 1 does corroborate our ingestion exactly: 1,000,087 rows at horizon 0 falling to
598,832 at horizon 5, distressed counts 3,587 → 1,154. §22's loader agrees with the paper.

### The published numbers we can quote

Table 2, QLoRA-finetuned Llama-3-8B against XGBoost on identical rows (a stratified
20,000-observation training subset; test is all held-out positives plus sampled negatives):

| horizon | Llama-3-8B ROC-AUC | XGBoost ROC-AUC | Llama F₁ | XGBoost F₁ |
| --- | --- | --- | --- | --- |
| 0 | 0.825 | **0.995** | 0.308 | 0.483 |
| 1 | 0.568 | 0.937 | 0.095 | 0.218 |
| 2 | 0.597 | 0.908 | 0.119 | 0.113 |
| 3 | 0.517 | 0.879 | 0.042 | 0.055 |
| 4 | 0.583 | 0.857 | 0.011 | 0.040 |
| 5 | 0.553 | 0.811 | 0.030 | 0.037 |

Figures 3 and 4 report the main comparison as plots rather than a table, so exact per-horizon
values for fine-tuned TabPFN against the six classical baselines live in their Appendix D,
which is not in the pages read. Read off the figures, ROC-AUC for prototype-undersampled
TabPFN runs from about 0.995 at horizon 0 to about 0.86 at horizon 5, with XGBoost close
beneath it and TabPFN without resampling at about 0.78 by horizon 5. **Those are figure
readings, not quoted numbers, and must not be tabulated as if they were.**

Their headline: prototype-undersampled TabPFN **matches or exceeds gradient boosting on
ROC-AUC at every horizon**, and on F₁ from horizon 2 onward. Llama-3-8B trails XGBoost on
ROC-AUC at every horizon.

### Their best method is our §29 mechanism, published four months earlier

This is the uncomfortable part and it goes at the top of any external claim.

Their TabPFN context-construction ablation compares no resampling, **random undersampling**
(minority-to-majority ratio 0.3), and **prototype undersampling** — the same class budget, but
the majority subset chosen by clustering majority-class samples with **MiniBatchKMeans** and
keeping, per cluster, the real observation closest to the centroid. Prototype undersampling
wins, and their stated conclusion is:

> The gap between prototype and random undersampling indicates that preserving majority-class
> structure matters beyond simply increasing minority exposure.

That is **exactly** §29's proposed mechanism — that balancing throws away the majority class,
which is where a low-default portfolio's information lives — and it is exactly what §32 and
§35's retrieval exploits. They published it in May 2026. Our sweep did not know that, which is
the cost of not having read the benchmark's own paper before scoring on its data.

**What survives as distinct.** Their context is built **once, globally**, by clustering the
majority class. Ours is built **per query group**, conditioned on the queries being scored, and
§32 measured grouping against blind sampling on the same data. Query-conditioned retrieval is
a real difference from global prototype selection, and it is a narrower claim than "retrieval
helps", which they had already shown in substance. It is also worth noting they used
MiniBatchKMeans for the same reason we did, which is mild evidence the design is the obvious
one rather than a contribution.

### Method note: do not read a paper through a summariser

The first pass at this finding used a fetched summary of the PDF rather than the PDF. That
summary produced **a results table that does not exist in the paper**, invented per-horizon
default rates of "4.2% at 1Y rising to 16.9% at 5Y" against the true 0.19-0.36%, and stated
that **TabPFN was used zero-shot** when the abstract on the same page says it was fine-tuned.
It was caught only because the fabricated default rates contradicted our own loader by tenfold
and the fine-tuning claim contradicted the abstract.

Had the protocol been slightly wrong rather than absurdly wrong, it would have been recorded.
**A secondhand summary of a source is not the source**, and every number in this section came
from reading the pages. This is the same failure family as §28 and §34: an interface that
answers confidently without the thing behind it being what it claims.

---

## 37. The retrieval grouping approximation costs about 0.012 AUC, and moves individual firms by up to 0.24

**Date:** 2026-09-09. **MEASURED**, single seed. **Command:**

```bash
uv run fintfm-retrgroup --model runs/v4-hazard-ldp.pt --n-positives 150 --n-negatives 250 \
    --group-sizes 2,8,32,128,400
```

§32 and §35 rest on an approximation that was documented as unmeasured. Per-query retrieval is
the correct operation; queries are clustered instead and each **group** shares one retrieved
context, so a firm's prediction depends on which other firms were scored beside it. This is
`retrieval-context` task 17.6.

Exact per-query retrieval as the reference, 400 test firms (150 defaulting, 250 not) against
a 72,622-row pool:

| queries per context | groups | seconds | mean abs deviation | max abs deviation | Spearman | mean AUC | vs exact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **1 (exact)** | 400 | **508** | — | — | — | **0.8449** | — |
| 2 | 200 | 194 | 0.0032 | **0.6425** | 0.9898 | 0.8332 | −0.0118 |
| 8 | 50 | 67 | 0.0021 | 0.1721 | 0.9757 | 0.8304 | −0.0146 |
| 32 | 13 | 24 | 0.0013 | 0.0399 | 0.9850 | 0.8492 | +0.0042 |
| 128 | 4 | 14 | 0.0028 | 0.0518 | 0.9605 | 0.8360 | −0.0089 |
| **400 (one shared context)** | 1 | 7 | **0.0327** | 0.3165 | **0.8935** | **0.7688** | **−0.0762** |

**Between 2 and 128 queries per context the portfolio-level cost is about 0.01 AUC**, and
non-monotone across that range, so the honest reading is "roughly 0.01 somewhere in here"
rather than a curve. Mean absolute deviation in cumulative PD is 0.0013-0.0032 and Spearman
correlation with the exact prediction is 0.96 to 0.99.

**A single shared context is a different animal: −0.076 AUC**, mean deviation 0.033 and
Spearman 0.894. With one group there is no clustering at all — the context is retrieved around
the mean of every query at once, which is closer to a global prototype than to retrieval. That
result is what prompted the group-count sweep in §38, because §32 and §35 used 64 groups
throughout and never varied it.

A confirmatory run on a second, positive-heavy subsample (473 defaulting, 60 not) gave the
same picture at 2/8/32 queries per context: deviations of 0.0004-0.0009, Spearman 0.982-0.997,
AUC deltas of +0.002 to −0.014, and an exact reference costing 561 s for 533 queries.

**Exact retrieval costs 1.05 s per query**, measured. Grouped scoring of the full 47,378-firm
panel takes about 110 s in total (§32), which is 0.0023 s per firm, so the exact operation is
roughly **450× more expensive** — confirming the "~500×" estimate the module docstring had
been asserting without measurement.

### The number that matters for the product is the maximum, not the mean

At 2 queries per context — the *tightest* grouping tested, where the mean deviation is a
harmless 0.0032 — **one firm's cumulative PD moved by 0.64**. The mean is reassuring and the
maximum is not, and a credit decision is made per obligor rather than per portfolio. A model
whose stated PD for a given firm can move by 64 percentage points depending on which other
firms were in the scoring batch is not something to put in front of a model-risk function
without saying so.

Note the maximum does **not** shrink as groups tighten: 0.64 at 2 queries per context against
0.05 at 128. Whatever produces the extreme cases is not simple group coarseness, and it is
unexplained.

That makes the batch-dependence trade a **product** constraint rather than an engineering
detail, and it argues for exact retrieval on the obligors that matter — 1.05 s for a single
firm is entirely affordable when scoring one firm, which is the actual decision context. Batch
scoring for portfolio analytics can use groups; an individual credit decision should not.

### Caveats, both material

**The subsample is 37.5% defaulters**, against a true rate near 1%. Positives were capped and
negatives sampled to keep the exact reference — at a measured **1.05 s per query** — inside ten
minutes. The *deviation* and *Spearman* columns do not depend on the label and are trustworthy;
the **AUC deltas are on an unrepresentative population and are not the cost on a real book.**
A run at a realistic base rate is `retrieval-context` task 17.7.

**Single seed, and non-monotone**: 32 queries per context beats exact by +0.004, which is
noise, not a real gain from coarser grouping.

---

## 38. Query-conditioned retrieval loses to the published global prototype context, on every axis

**Date:** 2026-09-09. **MEASURED**, three seeds, plus a Holm-corrected paired bootstrap.
**This finding retires §32's contribution claim.**

§36 established that our §29 context mechanism was published first, as prototype
undersampling. What was left as arguably ours is that retrieval is **query-conditioned** —
each group's context is retrieved around the firms being scored — where theirs is built once,
globally, by clustering the majority class. This measures whether that distinction pays.
Prototype undersampling was implemented from the paper's description (`prototype_context`),
with no code or data from that work.

| arm | mean AUC | mean ECE | seconds | blind? |
| --- | --- | --- | --- | --- |
| retrieval, 64 groups | 0.8118 ± 0.0032 | 0.0126 ± 0.0011 | 111 | no |
| retrieval, 256 groups | 0.8130 ± 0.0069 | 0.0119 ± 0.0022 | 221 | no |
| **prototype, minority ratio 0.3** | **0.8143 ± 0.0038** | **0.0072 ± 0.0005** | **56** | **yes** |
| uniform | 0.7930 ± 0.0038 | 0.0073 ± 0.0006 | 31 | yes |

Paired bootstrap, retrieval(256) − prototype, Holm-corrected across four horizons: **−0.0077
at horizon 0, [−0.0136, −0.0020], adjusted p = 0.036 — significant against us.** Horizons 1, 2
and 3 are inconclusive (+0.0053, +0.0229, −0.0071).

**Prototype wins or ties on every axis that matters:**

- **Accuracy**: equal within seed noise on the mean, significantly better at the first
  horizon, which is the horizon with the most defaults and the most commercial weight.
- **Calibration**: 0.0072 against 0.0119, non-overlapping across seeds — **1.6× better.**
- **Cost**: 56 s against 221 s, roughly **4×** cheaper, because it selects once instead of
  clustering queries and retrieving per group.
- **Stability**: ±0.0038 against ±0.0069 across seeds.
- **It is blind**, so it keeps the batch-independence property retrieval gives up — and §37
  measured a single firm's cumulative PD moving by **0.64** under retrieval depending on its
  scoring batch. That alone is disqualifying for per-obligor use.

### Consequence: retrieval is retired as a contribution and as a recommendation

§32 called retrieval "the first accuracy gain in this project to survive a family-wise
correction". Against *blind uniform sampling* that remains true and replicated (§33). Against
**the actual state of the art on this benchmark** it is not a gain at all, and it carries two
real costs the alternative does not.

So the honest position is: **context construction matters, Kostrzewa et al. showed why, and
their construction is better than ours.** `prototype` is the recommended strategy; retrieval
stays in the codebase as a measured arm and as the exact per-query reference of §37, not as
the recommendation.

### Two self-corrections inside this one finding

**The group count does not help.** A single-seed sweep showed 64 → 256 groups worth +0.008
(0.8126 → 0.8209) and I treated that as a real lever, including as the reason to re-run this
comparison. On three seeds, 256 groups gives **0.8130 ± 0.0069** — the 0.8209 was a favourable
draw, and the seed standard deviation is larger than the effect. Tightening groups from 740
firms per context to 185 buys nothing measurable and costs 2× the time. 1,024 groups also gave
0.8207 on a single seed, so that number is suspect for the same reason and was never
replicated.

**And the first version of this comparison was against an under-tuned arm.** Retrieval was run
at its default 64 groups, prototype at its paper value; the rematch at 256 groups was the fix,
and it changed nothing. Both arms were fine. The lesson is narrower than "tune before
comparing": a single-seed sweep is not tuning, it is noise with a direction.

**This is the sixth wrong diagnosis in two days** (`docs/POSTMORTEM.md`), and the first where
the error was over-crediting our own result rather than mis-attributing a failure.

---

## 39. The first comparable number: 0.9811 ROC-AUC on V4FinBench's own protocol, from a model that has never seen a real company

**Date:** 2026-09-09. **MEASURED**, 5 folds, full data. **Command:**

```bash
uv run fintfm-v4protocol --model runs/v4-clf-small.pt --horizon 0 --folds 0,1,2,3,4
```

§36 established that nothing this project had produced could be placed against V4FinBench's
published table. This is the first number that can be, on their protocol, their folds, their
horizon-0 task, all 1,000,087 rows.

**The reproduction lands on the right data.** Our loader reports 1,000,087 rows with 3,587
positives (0.359%) at horizon 0; their Table 1 reports 1,000,087 and 3,587. Fold sizes come out
at 199,392-200,822, which is the even split their company-grouped assignment should give.

| arm | ROC-AUC | F₁ |
| --- | --- | --- |
| **CatBoost** | **0.9959 ± 0.0002** | **0.4275 ± 0.0135** |
| logistic regression | 0.9839 ± 0.0013 | 0.2439 ± 0.0177 |
| **fintfm** (847K params, synthetic-only) | 0.9811 ± 0.0008 | 0.2202 ± 0.0129 |
| LightGBM | 0.9649 ± 0.0148 | 0.3727 ± 0.0080 |
| XGBoost | 0.8873 ± 0.0752 | 0.3379 ± 0.0131 |

**The baselines here are untuned**, and that is a defect rather than a footnote. Their
protocol grid-searches every baseline on the validation fold (their Table 5); these run at
library defaults. It shows: default XGBoost lands at 0.8873 with a ±0.0752 fold spread, and
default LightGBM at 0.9649, **both below logistic regression**, while the paper reports
gradient-boosted trees as its strongest classical cluster. So two of the three boosting arms
above are broken rather than beaten, and the field is understated — the same defect as §25,
pointed the other way. The harness now carries their grids behind `--tune` and prints a
warning when they are not used.

**CatBoost is the arm that matters, and it lands at 0.9959** — within 0.0009 of the ~0.995 the
paper reports for its strongest methods, which is the best evidence yet that this reproduction
is faithful. CatBoost happens to be well-configured out of the box; the other two boosters are
not.

Against it we are **0.0148 AUC and roughly half the F₁ behind**. Against logistic regression we
are 0.0028 behind on AUC, consistently across all five folds, with a fold spread of ±0.0008
that is smaller than the gap — so even that small deficit is real rather than noise.

That is a materially better showing than out-of-time, where the gap to per-horizon logistic
regression is 0.047 (§38). Two protocols, two answers, and the difference is the protocol: this
one is company-grouped cross-validation, so a fold may contain 2019 observations while
predicting a 2008 one. **Out-of-time is the harder split, and we are further behind on it** —
which is the honest direction for that difference to run, since out-of-time is also the one a
model-risk function cares about.

### The F₁ gap is the real problem, and it is not what it first looks like

Their published XGBoost reaches far higher F₁ than either arm here. Before treating that as a
2× deficit, note **their Table 2 F₁ of 0.483 is not comparable to our 0.2202.** That table uses
a different sub-protocol: a 20,000-observation training subset, and a test set built from *all*
held-out positives plus sampled negatives. F₁ depends directly on class balance, so enriching
the test set with positives inflates it. Our 0.2202 is on the full fold at a 0.359% base rate.

Running the boosters ourselves settles it without needing their figure: **CatBoost reaches
0.4275 F₁ on these folds against our 0.2202.** The gap is to gradient boosting, it is roughly
a factor of two, and it is not an artefact of their test-set construction.

**Separately, AUC 0.981 with F₁ 0.220 is itself informative.** Ranking is excellent and the
operating point is poor, which is a threshold and probability-shape problem rather than a
discrimination one — at a 0.359% base rate the F₁-maximising threshold sits at 0.032 for our
model against 0.102 for logistic regression, so our probabilities are compressed toward zero
relative to a fitted model's.

### What this does and does not license

**It licenses:** "on V4FinBench's published protocol at the immediate horizon, a
synthetic-only in-context model with 847K parameters reaches 0.9811 ROC-AUC, within 0.003 of a
logistic regression fitted on 600,000 rows, having never seen a real company."

**It does not license** any claim against their headline TabPFN result. Theirs is **fine-tuned
on this data** for 10 epochs on an A100; ours cannot be, by decision D2, because the
provenance argument is the point. Those answer different questions and a table putting them
side by side without that sentence attached would be misleading.

**Still missing before this is publishable:** the other five horizons, the five classical
baselines we do not run, and the boosters. Horizon 0 is also the easiest horizon — the label is
a deterministic rule on three features that are present in the feature set, so 0.98 here is
closer to rule recovery than to forecasting.

---

## 40. Context strategy does not transfer between protocols, for the fourth time

**Date:** 2026-09-09. **MEASURED**, 5 folds. **Command:**

```bash
uv run fintfm-v4protocol --model runs/v4-clf-small.pt --horizon 0 --folds 0,1,2,3,4 \
    --no-boosting --config configs/best.yaml
```

§39's protocol run used `uniform` context, the class default. That looked like an oversight:
at a 0.359% base rate a 2,000-row uniform context holds about **7 positive examples**, and the
horizon-0 label is a *conjunction* of three thresholds, so seven examples to infer an AND from
seemed obviously too few. `prototype` context supplies about **462**, and §38 had measured it
as the best strategy on the out-of-time survival path.

It made things **worse**, consistently:

| context | ROC-AUC | F₁ |
| --- | --- | --- |
| **uniform** (~7 in-context positives) | **0.9811 ± 0.0008** | **0.2202 ± 0.0129** |
| prototype (~462 in-context positives) | 0.9687 ± 0.0015 | 0.1967 ± 0.0168 |

**−0.0124 AUC and −0.024 F₁**, on all five folds, with fold spreads far smaller than the gap.
Sixty-six times the positive examples in context makes the model worse. Whatever limits it
here, it is not a shortage of positives.

### The pattern this completes

Four settings, four different answers about which context construction wins:

| setting | result |
| --- | --- |
| survival, out-of-time, V4FinBench | balanced loses to uniform by 10-12 AUC points (§29, §33) |
| binary, Polish and Taiwan panels | strategies nearly tied; the effect scales with how extreme the rebalancing actually is (§35) |
| survival, out-of-time, V4FinBench | prototype beats uniform and beats retrieval (§38) |
| **binary, their cross-validation protocol** | **prototype loses to uniform by 0.012 AUC** |

**Context construction is protocol-dependent and does not transfer.** Not between prediction
paths, not between split designs, and not from the literature — §29 already contradicted the
published ordering it was adopted from. This is now a standing result rather than a series of
surprises, and it has two consequences:

- **A context strategy must be re-measured in the setting it will be used in.** Carrying one
  over is how §39's headline nearly got run on a configuration chosen for a different protocol
  — in the direction that would have understated us by 0.012 AUC had the default gone the
  other way.
- **Decision D10's caution was right.** The class default was left at `uniform` pending
  per-path measurement rather than flipped to the strategy that won once. Had it been flipped,
  this benchmark would have silently used the worse option.

### What it rules out, and what it leaves

**Ruled out:** that the model's weak F₁ at horizon 0 is caused by too few positive examples in
context. That was the obvious hypothesis and it is wrong.

**Left standing:** that fintfm (0.9811 / 0.2202) sits almost exactly where logistic regression
does (0.9839 / 0.2439), while CatBoost reaches 0.9959 / 0.4275. The horizon-0 label is an AND
of three thresholds, which a tree path represents natively and an additive model cannot — a
weighted sum scores a firm extreme on one axis highly even when it fails the other two
conditions, and those false positives land exactly where F₁ is decided. An in-context
transformer *should* be able to represent a conjunction. At 847K parameters this one does not,
and whether capacity is the reason is what the scaling curve measures next.

---

## 41. The prior teaches smooth additive boundaries; the benchmark's label is a hard conjunction

> **FALSIFIED the same day by §42, before any of it was built.** The premise — that the model
> can represent a smooth boundary but not a conjunction — is false: it scores 0.68 on *both*,
> including a linear task logistic regression solves at 0.9997. Kept in full, unedited, because
> the pre-registration is the point: a tidy mechanism explaining four observations was wrong,
> and the five-minute premise check that killed it cost less than the 1.5-hour retrain it would
> otherwise have justified. Read §42 instead.

**Date:** 2026-09-10. **Status: HYPOTHESIS**, with a pre-registered prediction, recorded before the
experiment was run. Not a measurement — see the provenance note in `openspec/tools/validate.py`. Derived from reading `prior/financial.py` against §39 and §40.

### The mismatch

The financial prior builds its label from a latent distress score:

```python
distress = drivers @ w                    # linear in nine standardised drivers
distress += sector_hazard[sector] - cycle * sector_cyclicality[sector] * u   # additive
distress += rate * u * leverage           # one multiplicative term
if rng.random() < 0.7:                    # 70% of tasks
    hidden = np.tanh(drivers[:, idx] @ W + b)
    distress += hidden @ v                # smooth, MLP-shaped
p_default = sigmoid(distress + b)
```

Every one of those terms is **smooth**. The only nonlinearity is `tanh`, which is the function
class a neural network represents naturally.

V4FinBench's label, from their `docs/benchmark_protocol.md`, is:

```text
Equity/total_assets < 0  AND  EBITDA/total_assets < 0  AND  Current_assets/short_term_liabilities <= 0.6
```

A hard **conjunction of three axis-aligned inequalities** — the function class a decision tree
represents natively as a single root-to-leaf path, and that a smooth additive model
approximates badly: a weighted sum scores a firm that is extreme on one axis highly even when
it satisfies neither of the other two conditions, and those false positives land precisely
where F₁ is decided at a 0.359% base rate.

**The prior never generates a threshold conjunction.** Not once, in any task.

### It explains every result of the last two days

| observation | explanation under this hypothesis |
| --- | --- |
| fintfm 0.9811 / 0.2202 sits almost exactly on logistic regression's 0.9839 / 0.2439 (§39) | both are smooth and near-additive; the model was trained to be |
| CatBoost reaches 0.9959 / 0.4275 (§39) | axis-aligned thresholds are a tree's native representation |
| 847K → 14.5M parameters changes nothing (§42, pending) | more capacity executes the wrong inductive bias more precisely |
| 66× more in-context positives makes it *worse* (§40) | the bias is wrong, not the evidence; more examples of a rule it cannot represent do not help |

Compounded by a choice of mine: these checkpoints were trained with `--p-financial 1.0`, for
comparability with the hazard checkpoint, so the model saw **only** the financial prior — not
even the generic SCM prior's `relu`, the one piecewise-linear activation in the codebase.

### Pre-registered prediction, recorded before implementing anything

Adding threshold-conjunction labels to the prior — with some probability, generating `y` from a
shallow axis-aligned rule over the drivers instead of `sigmoid(linear + tanh)` — will:

1. **Narrow the horizon-0 ROC-AUC gap to CatBoost by at least half**, from 0.0148 to 0.0074 or
   better, at matched compute and matched architecture.
2. **Improve F₁ by more than it improves AUC in relative terms**, because the deficit is
   concentrated at the top of the ranking rather than spread through it.
3. **Not** materially change the out-of-time survival results (§38), whose label is a
   *timing* construction rather than a threshold rule.

If (1) fails, the limitation is architectural rather than in the prior — most likely the
mean+max pooling over features, which is structurally additive and may be unable to express a
conjunction whatever the training signal contains. That would be the more expensive finding and
it is worth knowing either way.

**Why this is written down first.** Over 2026-09-09 six diagnoses were wrong (`docs/POSTMORTEM.md`),
including two that over-credited a result. A hypothesis this tidy — one mechanism explaining
four separate observations — is exactly the kind that gets confirmed by a story rather than by
evidence. The numbers above are the test, and they were fixed before the experiment existed.

---

## 42. §41 is falsified, and the real defect is far worse: the model barely learns in context at all

**Date:** 2026-09-10. **MEASURED**, synthetic controls plus a feature audit.
**This finding retracts §41 and puts every accuracy number in this project in question.**

§41 predicted that a conjunction-capable prior would close the gap to CatBoost. Testing the
premise first — can the model represent a conjunction at all? — cost five minutes and destroyed
the hypothesis.

### The model is equally bad on a task with a perfect linear boundary

Synthetic, 20 features, 5% base rate, 1,000 context rows, three seeds:

| label shape | fintfm (847K / 4.9M / 14.5M) | logistic regression | gradient boosting |
| --- | --- | --- | --- |
| smooth linear | 0.685 / 0.685 / 0.688 | **0.9997** | 0.972 |
| threshold conjunction | 0.695 / 0.694 / 0.701 | 0.955 | 0.991 |

**It is not a conjunction problem.** The model scores 0.68 on a clean linear task that logistic
regression solves at 0.9997. Varying the input changed nothing that matters:

- **Feature count** 5 / 10 / 20 / 60 / 130 → 0.643 / 0.553 / 0.700 / 0.576 / **0.505**. At 130
  features it is at chance.
- **Feature transform** none / winsor / rank → 0.7010 / 0.7019 / 0.7001. Irrelevant here.
- **Context size** 200 / 500 / 1,000 / 2,000 → 0.687 / 0.696 / 0.700 / 0.701. **Four times the
  context buys 0.014.** A working in-context learner improves with context; this one does not.

### Pretraining bought almost nothing

Against an untrained model of the same architecture, on a clean linear task:

| | AUC |
| --- | --- |
| trained (6,000 steps) | 0.6334 |
| **untrained, random weights** | **0.6180** |
| logistic regression | 0.9999 |

**+0.015 for the entire pretraining run.** §15 measured +0.031 against an untrained control and
read it as "pretraining buys calibration more than ranking". The more economical reading, now
that the ceiling is visible, is that pretraining bought very little of anything.

### And V4FinBench horizon 0 is nearly solved by one column

Auditing every feature's individual ability to separate the label:

| single feature | AUC alone |
| --- | --- |
| `Working_capital/total_assets` | **0.9799** |
| `Equity/long_term_liabilities` | 0.9759 |
| `EBITDA/total_assets` | 0.9742 |
| `Total_liabilities/total_assets` | 0.9738 |

Eight individual columns exceed 0.972. Our headline 0.9811 (§39) is **0.0012 above the best
single feature in the table.** The label is a deterministic rule on three ratios that are
themselves present, so horizon 0 rewards reading one column, not modelling. That number was
never evidence the model works — and neither was logistic regression's 0.9839.

### The training loop's only quality signal was uninformative

`_eval_accuracy` reports **accuracy**. Measured over 20 prior tasks, the mean base rate is
0.047, so **always predicting the majority class scores 0.953**. The training logs reported
"held-out synthetic query accuracy: 0.935 … 0.945" — *below the constant predictor*, presented
as progress.

`CLAUDE.md` already says accuracy is not a proper scoring rule and that this is why training
uses cross-entropy. The evaluation inside the training loop did not follow its own rule, so
three GPU runs reported a number that a constant predictor beats, and nothing flagged it.

### The likely root cause, and why it was designed in

The prior is **deliberately tuned to be hard**. §19 records the sharpness multiplier being
narrowed to `rng.uniform(0.7, 2.7)` so that logistic regression scores ≈0.74 on synthetic tasks
against 0.769 on real ones — an explicit "difficulty match", recorded as a success.

The consequence, unnoticed until now: **the prior contains almost no learnable tasks.** Every
task the model has ever seen has a label that is mostly noise, with a ceiling near 0.75. On a
task from its own prior it scores 0.6607 against logistic regression's 0.6068 — respectable
against that ceiling. It has learned to do as well as anything can on irreducibly noisy tasks,
and has never been shown a task where a sharp boundary exists and can be extracted.

So it does not know how to exploit clean signal, which is exactly what the controls show.
TabPFN-style priors span difficulty from trivial to impossible; ours is clamped to hard. **§18
and §19 optimised the prior to look realistic rather than to teach**, and recorded that as a
finding in favour.

### What this retracts or weakens

- **§41** — retracted outright. The conjunction hypothesis was wrong; all three of its
  pre-registered predictions are void because the premise was false.
- **§39** — the 0.9811 stands as a number but not as evidence of capability. Horizon 0 is a
  one-column task.
- **§15** — "pretraining buys calibration" survives only as a statement about calibration;
  its +0.031 ranking gain is of the same order as the +0.015 seen here between trained and
  random weights.
- **§16, §17, §27** — that gradient boosting overtakes us above a few hundred rows, and that a
  constant predictor beats every model on ECE, are exactly what a barely-learning model
  produces. They were read as properties of the regime; they are at least partly properties of
  this checkpoint.
- **The scaling result** (§43) is now unsurprising rather than informative: 17× the parameters
  cannot help a model whose training signal is noise.

### What to do, in order

1. **Fix the training-loop metric.** Report AUC and a proper scoring rule against a constant
   baseline, not accuracy. Cheap, and it is the instrument everything else is read through.
2. **Widen the prior's difficulty range** to include learnable and near-deterministic tasks,
   rather than clamping it to a realistic-looking band. This inverts §19's change and needs its
   reasoning rewritten, not merely reversed — the difficulty match was aimed at a real property
   and solved it in a way that removed the training signal.
3. **Re-measure the untrained control on everything.** It is the only baseline that separates
   "the model learned this" from "the architecture and context did".
4. Only then revisit architecture and scale.
