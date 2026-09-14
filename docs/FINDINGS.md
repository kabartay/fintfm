# Findings

Numbered, dated, with the source or command that produced each. A finding that lives only in
a conversation is lost when the conversation compacts.

## Where we stand — the honest scorecard

Updated 2026-09-10 after forty-seven findings. **Read this before quoting any number below**,
because several findings temper, amend or outright retract earlier ones and the amendments
matter more than the originals. §28 retracts §26's headline as our own bug.

### Read this first

**§47 is the root cause and it invalidates every accuracy number produced before 2026-09-10.**
Shuffling the context labels — destroying any relationship between features and labels — left
the model's predictions unchanged (rank correlation 0.977) and its AUC *higher* than with true
labels. **It was not doing in-context learning at all.** The cause was one line of the prior:
the driver signs were fixed, so every task ever generated shared one feature-to-label mapping,
and a model could memorise it and never consult a labelled example. Fixed by randomising the
sign per task; verification runs in flight.

Read §47 first, then §42. Coherence (§11, §20, §26) is structural and unaffected — it is a
property of the output parameterisation, not of anything learned.

**§42 also puts every accuracy number in this repository in question.** Against an untrained model of
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

---

## 43. We have never trained this model properly: 48,000 tasks against a field norm of ~10⁷

**Date:** 2026-09-10. **MEASURED** (the count and the cost), **ESTIMATED** (the field norm).

Every checkpoint in this project comes from **6,000 steps × batch 8 = 48,000 synthetic tasks**.
Published prior-fitted networks train on the order of **10⁷** synthetic datasets. We are two to
three orders of magnitude short, and have been reasoning about architecture and scale on that
basis for two days.

**The remedy is compute, not research**, and it is cheap. Measured throughput on a T4 is
0.64 s/step at 847K parameters and batch 8 (`docs/HF_JOBS.md`), i.e. 12.5 tasks per second:

| tasks seen | GPU-hours (T4) | approx cost |
| --- | --- | --- |
| **48,000** — every checkpoint to date | 1.1 | $0.70 |
| 500,000 | 11 | ~$7 |
| **1,000,000** | **22** | **~$13** |

**A 20× increase in training data costs about $13.** Until that is run, "the architecture is
wrong" is an untested claim, and so is "scale does not help" — §42's scaling result compared
three model sizes that were each trained on 48,000 tasks, which measures capacity at a fixed,
tiny data budget rather than a scaling law.

This reorders the queue. `adopt-published-methods` task 36.3 (context scaling) and any
architectural work sit behind simply training the model once at a defensible volume, because
every measurement taken before that is a measurement of an undertrained model.

**The uncomfortable part:** this was always visible. The step count was in every command, and
`docs/COMPUTE.md` has priced runs since day one. Nobody multiplied 6,000 by 8 and compared it
to the literature the architecture was copied from.

---

## 44. The prior-difficulty fix changed nothing, and the real defect is generalisation across feature distributions

**Date:** 2026-09-10. **MEASURED**, capability probes, three seeds.
**This falsifies §42's second remedy as a sufficient fix.**

§42 diagnosed a prior clamped to a narrow difficulty band and predicted that widening it would
let the model learn. The prior was widened — learnable tasks went from 8% to 42%, the ceiling
from 0.945 to 1.000 — a checkpoint was retrained on it, and the probes say it made **no
difference at all**:

| arm | linear | conjunction | xor | noise |
| --- | --- | --- | --- | --- |
| **fixed prior** | **0.685 ± 0.119** | 0.692 ± 0.005 | 0.530 | 0.490 |
| old prior | 0.684 ± 0.119 | 0.695 ± 0.005 | 0.537 | 0.492 |
| untrained control | 0.367 ± 0.129 | 0.298 ± 0.004 | 0.517 | 0.496 |
| logistic regression | **1.000** | 0.955 | 0.522 | 0.502 |
| gradient boosting | 0.983 | 0.989 | **0.997** | 0.517 |

0.685 against 0.684. The two checkpoints are indistinguishable.

### But the same run shows the model learning perfectly well

The new training metric (§42's first remedy, which *did* work) reports held-out **AUC 0.83-0.88
with Brier skill +0.16 to +0.18** throughout the run — on tasks drawn from its own prior. So
the model does in-context learning competently on data that looks like its training
distribution, and collapses to 0.685 on iid Gaussian features.

**That is a generalisation failure across feature distributions, not a failure to learn.** It
also explains §42's confusing pattern — more context and more capacity cannot help a model
whose problem is that the test features are unlike anything it has seen.

### The likely cause is a flag I set myself

Every checkpoint in §39-§44 was trained with **`--p-financial 1.0`**, chosen for comparability
with the hazard checkpoint. That disables the generic SCM prior entirely, so the model has only
ever seen financial-statement features: heavy-tailed, accounting-linked, strongly correlated
ratios. The default is 0.7, and the 30% generic component is precisely where distributional
diversity comes from — it is why TabPFN-style priors use structural causal models rather than
one domain generator.

I removed the component that teaches the model to handle unfamiliar tabular distributions, and
then measured it failing on unfamiliar tabular distributions.

### Consequence for §43

§43 argued that 48,000 tasks against a field norm of 10⁷ is the biggest untapped lever, and
that remains true — but it is now **second** in order. Training 20× longer on a distribution
the model already fits, and still cannot generalise from, would buy a better fit to the same
narrow distribution. Prior *diversity* is cheaper to test and more likely to be the binding
constraint, and it is a $0.70 experiment against $13.

### Untrained control at 0.367 is itself a finding

The untrained control scores **below chance** on `linear` and `conjunction` (0.367, 0.298),
where it scored 0.618 on an earlier balanced version of the same task. Random weights produce
predictions anti-correlated with the label at a 5% base rate. So "trained beats untrained" is
satisfied here trivially and is *not* evidence of learning — the floor is unstable and must be
read alongside the fitted baselines, never alone.

---

## 45. The model's predictions depend on which class is called "1"

**Date:** 2026-09-10. **MEASURED**, `runs/v4-clf-small.pt`, synthetic linear task, 300-row
context.

Prompted by a question about whether TabPFN's ensembling axes apply here. Testing them found
one that does not, one that does, and one that is a defect rather than an opportunity.

Take a fitted model, **relabel the context** so 0 becomes 1 and 1 becomes 0, predict, and invert
the output. A model reading its context as *evidence* should return almost exactly the original
probabilities — the labelled examples carry the same information either way, only the names
changed.

| ensembling axis | max abs change | correlation with baseline |
| --- | --- | --- |
| feature permutation | 1.19e-07 | **1.000000** |
| **class-label swap** (then inverted) | 0.344 | **−0.62** |
| feature subset, 70% | 0.194 | 0.496 |

**Correlation −0.62 where +1.0 is expected.** Swapping the names of the classes does not
perturb the model's predictions, it substantially reverses them. Its output is driven by which
class occupies the "1" slot rather than by what the context demonstrates.

This is the same defect as §42 and §44 seen from a third angle. A model that scores 0.685 on a
linear task, does not improve with more context, and reverses under a relabelling is not doing
in-context inference — it is doing something else that occasionally correlates with the answer.

### Why the asymmetry is not simply expected

The label embedding is learned per class (`y_proj` over a one-hot), so *some* asymmetry is
unavoidable and a small correlation shortfall would be unremarkable. A negative correlation is
not: it means the two labellings induce systematically opposed predictions, which no amount of
learned-embedding asymmetry justifies. Label-swap invariance is a property a working
prior-fitted network approximately satisfies, and it is now a cheap standing check.

### Feature permutation is confirmed dead as an axis

TabPFN ensembles over feature permutations because its predictions depend on column position.
Ours do not: 1.19e-07 maximum change is float noise, confirming decision D4's column-order
invariance end to end on a real checkpoint rather than only in the unit test. **So that axis
buys nothing here and should not be copied across.** The two axes that remain are label
assignment and feature subsetting, both of which measurably move predictions.

### Consequence

Label-swap averaging was added on the reasoning that it **cancels a measured pathology**:
averaging `p` with `1 − p_swapped` removes the component of the prediction that depends on the
label naming.

### Amendment, measured immediately after: cancelling the pathology cancels the model

| probe | single | ensemble ×4 | + label swap | + 70% feature subset |
| --- | --- | --- | --- | --- |
| linear | 0.6841 | 0.6838 | **0.4531** | 0.4124 |
| conjunction | 0.6948 | 0.6951 | **0.4985** | 0.4896 |

Two results, and the second is the important one.

**Context-draw ensembling does nothing** — 0.6841 against 0.6838 across three seeds. Consistent
with §42, where quadrupling the context bought 0.014: if more context does not help, neither do
more draws of it.

**Label-swap averaging drops the model to chance.** Not "helps less than hoped" — 0.684 → 0.453
on `linear`, and 0.695 → 0.499 on `conjunction`, which is exactly chance.

The arithmetic is the diagnosis. Averaging two anti-correlated predictors cancels their shared
component; there was almost nothing shared to keep. **So the discrimination the model appeared
to have *was* the label-slot asymmetry.** It is not weakly learning the task — it is reading
which class occupies the "1" position, which happens to correlate with the answer often enough
to produce 0.685.

The alignment was checked rather than assumed: the swapped member's class-0 column is the
original class-1 probability, so the inversion is correct and this is not an indexing error.

**Consequence.** The workaround is not a workaround. Ensembling is not a lever on this model in
any of its three axes, and `adopt-published-methods` task 36.2 is answered: **no**, until the
model learns something that survives relabelling. The implementation stays because it is the
cheapest available test of whether a future checkpoint has real discrimination — a model whose
score is unchanged by label-swap averaging is one whose signal is in the evidence rather than
in the labelling.

---

## 46. Prior diversity helps, and is not sufficient: 0.685 → 0.712, with the real gain on interactions

**Date:** 2026-09-10. **MEASURED**, three seeds, `fintfm-capability`.
**Tests §44's diagnosis. Confirmed in direction, wrong in magnitude.**

§44 found the model reaching held-out AUC 0.83-0.88 on its own prior and collapsing to 0.685
on iid Gaussian features, and blamed a flag: every checkpoint had been trained with
`--p-financial 1.0`, disabling the generic SCM prior and leaving the model to see only
financial-statement features. A checkpoint at `p_financial=0.7` tests it.

| arm | linear | conjunction | **xor** | noise |
| --- | --- | --- | --- | --- |
| financial-only | 0.685 ± 0.119 | 0.692 | 0.530 | 0.490 |
| **mixed, p_financial 0.7** | **0.712 ± 0.074** | 0.693 | **0.614 ± 0.014** | 0.483 |
| logistic regression | **1.000** | 0.955 | 0.522 | 0.502 |
| gradient boosting | 0.983 | 0.989 | **0.997** | 0.517 |
| untrained control | 0.357 | 0.351 | 0.518 | 0.515 |

**The interaction probe carries the result: 0.530 → 0.614 on `xor`.** That probe is
parity-structured, so logistic regression is pinned at chance by construction (0.522) and
anything above it is genuine interaction learning rather than a linear shortcut. The generic
SCM prior taught the model something the financial prior could not — which is the mechanism
§44 proposed, showing up on exactly the probe that isolates it.

`linear` improves from 0.685 to 0.712 and its seed variance nearly halves (±0.119 to ±0.074),
so the model is also more stable. `conjunction` does not move at all.

**And it is nowhere near enough.** 0.712 against a ceiling of 1.000, on a task a linear model
solves exactly. Prior diversity is *a* cause of §42's failure, not *the* cause.

### During training the same run looked far better than it is

Held-out AUC on its own prior ran 0.905-0.912 with Brier skill +0.39 to +0.42, against
0.83-0.88 and +0.16 to +0.18 for the financial-only prior — a doubling of skill against the
base-rate predictor. Read alone, that looks like a solved problem. The probes say the transfer
gain is 0.027.

**Held-out-on-your-own-prior is not a capability measurement**, and this is the cleanest
demonstration of it in the project: the same checkpoint that doubled its held-out skill moved
0.027 on the task that matters. It belongs beside §42's warning about the training metric.

### What this leaves

§43 is now the next test and is properly motivated for the first time: with a defensible prior
in place, **48,000 training tasks against a field norm near 10⁷** is the remaining known
deficiency, at roughly $13 for a million. The architecture critique — the mean+max pooling over
features, which is structurally additive — stays behind it, because diagnosing an architecture
on a model trained at 0.5% of the field's data volume is not a diagnosis.

---

## 47. The mechanism: the prior had a fixed feature-to-label direction, so the model never learned to read its context

**Date:** 2026-09-10. **MEASURED**, three checkpoints, three seeds.
**This is the root cause behind §42, §44, §45 and §46.**

Four hypotheses had been tested and found wrong or marginal — conjunction representation
(§41, falsified), prior difficulty (§44, no effect), prior diversity (§46, +0.027), training
volume (§43, no effect at 10×). The pattern was guessing at causes. This is the diagnostic
that isolates one.

### The test: shuffle the context labels

Randomise the context labels, destroying any feature-label relationship, and predict. A model
doing in-context learning should collapse to chance. A model ignoring its labels should not
move at all.

| checkpoint | AUC, true labels | AUC, **shuffled** labels | rank correlation | verdict |
| --- | --- | --- | --- | --- |
| 48k, financial-only | 0.6841 | **0.6895** | **0.977** | **ignores labels entirely** |
| 48k, mixed prior | 0.7116 | 0.6006 | 0.404 | uses labels |
| 480k, mixed prior | 0.7158 | 0.6793 | 0.804 | partly ignores again |

**The financial-only model scores higher with randomised labels than with true ones.** Its
0.684 was never in-context learning. It was unsupervised structure in the features plus the
label-slot bias §45 measured, and that is why label-swap averaging collapsed it to chance:
there was no label-derived signal to preserve.

### The cause, in one line of the prior

```python
drivers = [z(leverage), -z(coverage), -z(margin), -z(cash_ratio), ...]   # signs hardcoded
w = np.abs(rng.normal(1.0, 0.5, size=...)) * rng.uniform(0.3, 1.0, ...)  # all positive
```

`np.abs` over drivers with hardcoded orientation means that **in every task the prior has ever
generated, higher leverage is riskier and higher margin is safer.** The feature-to-label
mapping is universal, so a model can memorise one global distress score and apply it to every
task without ever consulting a labelled example.

It is economically correct, and it removed the only reason to learn in-context inference.

This also explains §46 cleanly. The generic SCM prior randomises its structure per task, so
mixing it in was the first thing that ever *required* label-reading — which is why it moved
`xor` (0.530 → 0.614) and why the mixed model is the only one that uses labels (correlation
0.404 against 0.977).

And it explains §43. More data on a prior that rewards ignoring labels buys a better
label-ignorer: the 480k checkpoint regressed to 0.804, further from label-reading than the
48k mixed model, matching its external-task regression.

### The fix

A random sign per driver per task. Verified: the first feature's direction now points each way
in 55%/45% of tasks, so no global rule can work, and knowing whether high leverage means risky
*in this task* requires reading the context.

Half the tasks become economically nonsensical. **That is correct.** A prior's job is to teach
in-context inference, not to resemble the deployment distribution — §42's lesson, restated and
now with a mechanism. The features keep their accounting identities and realistic correlations;
only the direction of the label relationship varies.

`tests/test_prior.py` pins it: if the first feature points the same way in more than 75% of
tasks, a global rule would still work and the test fails.

### Why this took three days to find

Every earlier measurement compared a model against baselines or against itself. **None asked
whether the context mattered at all** — the one question that separates in-context learning
from a fixed function of the features. It costs five minutes and it should be the first
diagnostic run against any prior-fitted network, before accuracy is discussed.

---

## 48. The fix works: the model reads its context, and improves with more of it

**Date:** 2026-09-10. **MEASURED**, three seeds. **Confirms §47's mechanism.**

Two checkpoints trained with the randomised label direction, one financial-only and one mixed.

### It reads the context

| checkpoint | AUC true labels | AUC shuffled | gap | rank correlation |
| --- | --- | --- | --- | --- |
| old, financial-only | 0.6841 | 0.6895 | **−0.005** | 0.977 |
| old, mixed | 0.7116 | 0.6006 | +0.111 | 0.404 |
| **new, signfix financial-only** | 0.7067 | 0.5746 | +0.132 | **0.295** |
| **new, signfix mixed** | 0.7089 | 0.5573 | **+0.152** | **0.226** |

**The sign fix alone is sufficient.** `signfix-financial` contains no generic SCM prior at all
and still moves from 0.977 to 0.295. That isolates the mechanism: §46's diversity gain was a
side effect of the SCM prior randomising structure per task, not of domain variety. The
production config (mixed) is best at 0.226, but the causal ingredient is the sign.

### And it improves with more context, which it never did before

Mean AUC on the `linear` probe against context size, three seeds:

| checkpoint | 100 | 250 | 500 | 1,000 | 2,000 | slope |
| --- | --- | --- | --- | --- | --- | --- |
| old, financial-only | 0.5564 | 0.5663 | 0.5246 | 0.5448 | 0.5386 | **−0.018** |
| **new, signfix mixed** | 0.5899 | 0.6401 | 0.6503 | **0.6620** | 0.6614 | **+0.072** |

The old model got *worse* with more evidence. The new one improves monotonically and saturates
around 1,000 rows. **A model that improves with more context is doing in-context inference**;
this is the first checkpoint in the project that does.

### And accuracy barely moved: 0.709 against 0.712

That has to be said plainly. The model now genuinely reads its context and scores the same as
when it ignored it. Label-reading was **necessary and is not sufficient** — the previous 0.71
was reached by a different route (unsupervised feature structure plus label-slot bias, §45),
and the ceiling is set by something else.

`xor` is the exception and moves in the right direction: 0.614 → 0.650, the probe that
requires genuine interaction learning.

### What this changes about everything measured before

**Every null result in §42-§46 was obtained on a model that ignored the input the lever acts
through**, and must be re-run:

- **§43's volume null** (10× data, no change) — measured on a label-ignorer, where more data
  buys a better label-ignorer. The 480k checkpoint's *regression* to correlation 0.804 is
  direct evidence of exactly that.
- **§42's flat scaling curve** (847K / 4.9M / 14.5M within 0.001) — same objection. Capacity
  cannot help a model not using the mechanism capacity would serve.
- **§45's ensembling null** — the axes were averaging a prediction whose signal was artefact.

None of those are now known to be false. They are unmeasured. The saturation at 1,000 context
rows is the new open question, and capacity is the first suspect worth re-testing.

---

## 49. The model degrades with feature count, and that is where the architecture is suspect

**Date:** 2026-09-10. **MEASURED**, two seeds, synthetic linear task with a perfect boundary.

With §48's checkpoint reading its context, the question becomes what limits it at 0.71. Sweeping
the number of features, holding everything else fixed:

| checkpoint | F=5 | F=10 | F=20 | F=40 | F=80 | F=130 |
| --- | --- | --- | --- | --- | --- | --- |
| old, financial-only | 0.695 | 0.630 | 0.571 | 0.534 | 0.546 | 0.570 |
| **new, signfix mixed** | **0.846** | 0.762 | 0.716 | 0.568 | 0.551 | 0.612 |
| logistic regression | 0.9998 | 0.9996 | 0.9996 | 0.9985 | 0.9964 | 0.9945 |

**On narrow tasks the fixed model is much better than the probes suggested: 0.846 at five
features**, against the old model's 0.695. Every probe in §42-§48 used twenty features and so
understated the improvement.

**And it collapses as features multiply**, 0.846 → 0.716 → 0.568. Logistic regression is flat
at ~0.999 across the entire range, so the difficulty is not in the task: every feature carries
signal, and a linear model combines 130 of them as easily as 5.

### Why this points at the pooling

The column stage produces one token per feature and `encode_rows` reduces them to a single
row vector by **masked mean and max pooling**. Combining five tokens into a summary is easy;
combining 130 into one that preserves each feature's weighted contribution is what a mean
cannot do — the informative directions are averaged against 129 others.

This is the first evidence that names a specific component rather than a hyperparameter, and
it makes a **falsifiable prediction for the capacity re-test now running**: if pooling
*capacity* is the constraint, the 4.9M and 14.5M models should degrade less steeply with F.
If they degrade identically, capacity is not the issue and the pooling *design* is — which
would be the first properly motivated case for architecture work in this project.

### The distribution-shift caveat was wrong, and removing it strengthens this

This finding originally carried a caveat: that the prior exposes "roughly 9-59 columns per
task" (citing §18), so F=80 and F=130 were partly extrapolation. **That was stale and it is
wrong.** Measured directly over 120 sampled tasks:

| prior | median width | ≥80 features | ≥130 features |
| --- | --- | --- | --- |
| financial | **77** | **49%** | 9% |
| mixture, p_financial 0.7 | 76 | 49% | 8% |

The prior was widened since §18 was written, and now produces a median of 77 columns with
**half of all tasks at 80 or more**. So F=80 and F=130 are squarely inside the training
distribution, and the collapse from 0.846 at five features to 0.551 at eighty is **not**
extrapolation. The model has been trained on plenty of tasks that wide and still cannot
aggregate them.

That removes the confound the caveat introduced and leaves the aggregation failure clean —
which makes the pooling a stronger suspect, not a weaker one. The lesson is narrower and worth
keeping: **a width quoted from an old finding is not a measurement.** It took one command to
check and it reversed the conclusion of a paragraph written twenty minutes earlier.

---

## 50. Capacity is ruled out: 17× the parameters gives identical curves. The pooling is the constraint.

**Date:** 2026-09-10. **MEASURED**, three seeds, three model sizes, prediction registered in

> **RETRACTED READING (2026-09-11).** The measurements here stand; the conclusion drawn
> from them does not. Every variant compared was equally constrained by the encoder bug
> of §54 — the row representation could not represent "column j matters" at all — so
> this finding's *reading* is unsafe to carry forward and the comparison must be re-run.
> §56 records the fix and what it invalidates.
§49 **before** the runs finished.

§49 predicted: if pooling *capacity* is the limit, larger models degrade less steeply with
feature count; if they degrade identically, the pooling *design* is at fault. Three checkpoints
on the fixed prior, identical in everything but size:

| arm | params | F=5 | F=10 | F=20 | F=40 | F=80 | F=130 | drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small | 846,818 | 0.8265 | 0.7341 | 0.7089 | 0.6492 | 0.5617 | 0.5510 | **−0.2755** |
| medium | 4,878,146 | 0.8264 | 0.7342 | 0.7083 | 0.6461 | 0.5631 | 0.5531 | **−0.2732** |
| large | 14,506,466 | 0.8243 | 0.7355 | 0.7099 | 0.6466 | 0.5618 | 0.5541 | **−0.2701** |
| logistic regression | — | 0.9999 | 1.0000 | 0.9997 | 0.9994 | 0.9983 | 0.9965 | −0.0034 |

**The three curves are identical to three decimal places at every width.** Not merely the same
slope — the same values, from independently trained models spanning 17× in parameters. Three
runs converging that precisely are converging to the same function, which is the signature of
a structural bottleneck rather than a capacity one.

Meanwhile a linear model loses 0.003 across the same range. The task does not get harder with
width; our model gets worse.

### The component

`FinancialTFM.encode_rows` produces one token per feature through the column stage, then
reduces them to a single row vector by **masked mean and max pooling**:

```python
pooled_mean = (cells * keep).sum(dim=2) / denom
pooled_max  = cells.masked_fill(keep == 0, -inf).max(dim=2).values
return self.row_proj(torch.cat([pooled_mean, pooled_max], dim=-1))
```

A mean is permutation-invariant and **weight-blind**: it cannot preserve that feature 7 matters
three times as much as feature 40, because every token contributes 1/F regardless. At five
features the surviving signal is enough; at 130 the informative directions are averaged against
129 others. Max recovers a little — the strongest single activation — which is why the curve
flattens rather than reaching chance.

That is exactly the shape observed, and it explains why capacity cannot help: a wider or deeper
model still has to pass everything through the same lossy reduction.

### What this licenses, and what it does not

**Licensed:** replacing the pooling with a mechanism that can weight features — attention
pooling with a learned query, in the manner of a set transformer's pooling-by-multihead-
attention. This is now the first architecture change in the project supported by a registered
prediction and a controlled test, rather than by a hunch.

**Not licensed:** claiming the pooling is the *only* remaining constraint. Even at F=5, where
aggregation is easiest, the model reaches 0.826 against logistic regression's 1.000. Something
else costs 0.17 at the narrow end, and fixing the pooling will not address it.

### Sequence worth recording

Five hypotheses, each narrowing the last, each measured: conjunction representation (§41,
falsified) → prior difficulty (§44, no effect) → prior diversity (§46, +0.027) → **fixed
label direction (§47, the root cause)** → training volume and capacity (§43, §50, both ruled
out) → **pooling design (§50, indicted)**. The first four were about whether the model learned
anything at all; only now that it demonstrably reads its context (§48) is an architectural
question even well-posed.

---

## 51. The model detects extremes rather than ordering: performance falls as the base rate rises

**Date:** 2026-09-10. **MEASURED**, three seeds, five-feature linear task, `signfix-mixed`.

§50 indicted the pooling for the collapse with feature count, but noted that even at F=5 —
where aggregation is trivial — the model reaches 0.826 against logistic regression's 1.000.
Something else costs 0.17 at the narrow end. Two probes locate it.

**More evidence does not help at F=5.** Context 250 / 1,000 / 2,000 / 4,000 gives 0.849 /
0.836 / 0.850 / 0.850. Sixteen times the labelled examples, flat. Whatever limits it here is
not a shortage of evidence — which is notable given §48 showed the model *does* improve with
context at F=20.

**And performance falls monotonically as the task becomes balanced:**

| base rate | fintfm | logistic regression |
| --- | --- | --- |
| 5% | **0.850** | 1.0000 |
| 15% | 0.754 | 1.0000 |
| 30% | 0.685 | 1.0000 |
| 50% | **0.658** | 1.0000 |

That is backwards on its face. More positives means more information about the positive class,
and the linear baseline is unmoved at 1.0000 throughout.

### The obvious explanation is wrong

The first hypothesis was prior coverage — that the prior never generates balanced tasks, as
its rate ceiling is 0.30. **Measured, it does:** over 150 sampled tasks the base rate has a
median of 0.096 and **24% of tasks are at 50% or above**, because the generic SCM component
produces them. Refuted in one command.

### What fits instead, and it points back at the pooling

At a 5% base rate, ranking well largely means **detecting extremes** — and max pooling is
precisely an extremeness detector. At 50%, ranking requires *ordering the whole distribution*,
which a mean-and-max reduction cannot do: the mean is weight-blind (§50) and the max reports
only the strongest activation. The monotone decline as the task shifts from tail-detection to
full ordering is what that predicts.

**Registered prediction for the attention-pooling checkpoint now training:** the gain should
be **largest at high base rates**, where ordering matters and extremeness does not suffice.
If attention pooling lifts the 50% case substantially more than the 5% case, this reading is
confirmed; if it lifts both equally, the base-rate effect has some other cause and this
explanation should be discarded rather than adjusted.

---

## 52. Attention pooling changes nothing, and neither does label noise. Two more falsifications.

**Date:** 2026-09-10. **MEASURED**, three seeds. **Falsifies §50's and §51's readings, and the

> **RETRACTED READING (2026-09-11).** The measurements here stand; the conclusion drawn
> from them does not. Every variant compared was equally constrained by the encoder bug
> of §54 — the row representation could not represent "column j matters" at all — so
> this finding's *reading* is unsafe to carry forward and the comparison must be re-run.
> §56 records the fix and what it invalidates.
probe-artefact hypothesis.**

Both predictions registered in §50 and §51 failed.

**Feature-width curve, mean+max against attention pooling:**

| arm | F=5 | F=10 | F=20 | F=40 | F=80 | F=130 | drop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| meanmax | 0.8265 | 0.7341 | 0.7089 | 0.6492 | 0.5617 | 0.5510 | −0.2755 |
| attention | 0.8281 | 0.7346 | 0.7081 | 0.6479 | 0.5593 | 0.5539 | −0.2741 |

**Base-rate sweep:**

| arm | 5% | 15% | 30% | 50% | drop |
| --- | --- | --- | --- | --- | --- |
| meanmax | 0.8426 | 0.7476 | 0.6791 | 0.6563 | −0.1862 |
| attention | 0.8473 | 0.7494 | 0.6796 | 0.6582 | −0.1891 |

Indistinguishable on both. **The pooling is exonerated.** §50 inferred from three model sizes
producing identical curves that the reduction must be at fault; the correct inference was
narrower — that *nothing about the model's capacity or reduction* matters, which includes the
replacement.

### And the probes were not the problem either

The prior samples labels stochastically — `y = rng.random(n) < sigmoid(distress + b)`, plus an
explicit 2% flip — while every probe used a hard threshold, `y = score >= quantile`, which is
perfectly separable. So the model had never seen a noiseless task, and a model trained only on
stochastic labels might reasonably hedge. Testing it:

| probe label type | fintfm | logistic regression | **gap** |
| --- | --- | --- | --- |
| threshold (separable) | 0.8516 | 1.0000 | **0.1484** |
| Bernoulli (matches the prior) | 0.6071 | 0.7523 | **0.1451** |

The gap is the same to three decimals. **Our probes were representative**; the deficit is not
an artefact of noiseless evaluation.

### What is now ruled out

The ~0.15 AUC deficit against logistic regression on linear tasks is invariant to:

| varied | result |
| --- | --- |
| model capacity, 847K → 14.5M | identical curves (§50) |
| pooling design, mean+max → attention | identical curves (this finding) |
| context size, 250 → 4,000 rows | flat (§51) |
| base rate, 5% → 50% | level moves, gap does not |
| label noise, deterministic → Bernoulli | gap identical |

**A deficit that survives every one of those is not an architecture problem.** Five
architectural and evaluative explanations have now been tested and eliminated, which is worth
more than it feels: the remaining space is much smaller.

### The one lever not yet re-tested

**Training volume, with the fixed prior.** §43 measured 10× data and found nothing — but that
was on a checkpoint from the *broken* prior, which §47 showed was ignoring its context
entirely, and §48 argued makes every pre-fix null uninformative. Capacity was re-tested after
the fix; volume was not.

Every sign-fixed checkpoint in §48-§52 is 6,000 steps — **48,000 tasks**, against a field norm
near 10⁷. That is the last untested lever, and the fact that three model sizes and two pooling
designs all converge to precisely the same function is what one expects when the *training
signal*, not the model, is the binding constraint.

---

## 53. The model scores ~0.67 on everything: it is a capped predictor, not a failing transfer

**Date:** 2026-09-11. **MEASURED**, twelve tasks per source, `v4-vol4x`.

> **RETRACTED READING (2026-09-11).** The measurements here stand; the conclusion drawn
> from them does not. Every variant compared was equally constrained by the encoder bug
> of §54 — the row representation could not represent "column j matters" at all — so
> this finding's *reading* is unsafe to carry forward and the comparison must be re-run.
> §56 records the fix and what it invalidates.

§52 left one lever untested — training volume with the fixed prior. Tested at 4× (48,000 →
192,000 tasks): the feature-width curve is unchanged, drop −0.2702 against −0.2755.

**But the training log says the model learned more.** Held-out AUC on its own prior reached
**0.916 with Brier skill +0.494**, the best of any run. More data teaches it more, and none of
it reaches the probes. Scoring the same checkpoint on three task sources explains why:

| task source | **fintfm** | logistic regression | gap |
| --- | --- | --- | --- |
| financial prior — 70% of its training | **0.6708** | 0.7270 | 0.056 |
| SCM prior — 30% of its training | **0.6894** | 0.8887 | 0.199 |
| iid Gaussian — never seen | **0.6261** | 0.9947 | 0.369 |

**The model produces 0.63-0.69 on all three.** Its own training distribution, a partly-seen
one, and one it has never encountered. The gap widens only because logistic regression gets
*better* on easier tasks while our model does not move at all.

**This is not a transfer failure.** It is a model that has converged to a capped predictor,
returning roughly the same quality whatever it is given — including tasks whose achievable
ceiling is 0.995. It is not tracking task difficulty.

### Why that explains six null results at once

§50 (capacity, 17×), §52 (pooling design, context size, base rate, label noise) and this
finding (volume, 4×) all produced *identical* curves. Six independent interventions changing
nothing looked suspicious and now makes sense: **none of them addresses a model that is not
extracting the signal available to it.** Varying the capacity of something that has stopped
short, or the width of its reduction, or how much data it sees, cannot move a ceiling that is
not set by any of those.

### The honest caveat on volume

4× is a small factor against a field norm near 10⁷ — roughly 200× our 48,000. If the scaling
is logarithmic, 4× returning +0.002 is entirely consistent with 200× returning something real.
**This is weak evidence against volume, not strong**, and saying "volume is eliminated" would
overstate it.

### The experiment this forces

Everything above is consistent with two very different worlds: our prior is too hard
*everywhere*, or the architecture cannot learn sharp in-context inference *at all*. The
control that separates them is a deliberately trivial prior — few clean features, a
deterministic linear rule, no noise, no missingness, where logistic regression scores **0.9998**
(`prior/trivial.py`, verified in the tests).

- **A model trained there reaches ~0.95** → the architecture works, our prior is too hard
  everywhere, and the fix is difficulty coverage at the easy end.
- **It cannot** → something in the training loop or architecture prevents sharp in-context
  inference regardless of data, and no prior work will fix it.

Running now. The trivial prior is a **diagnostic, not a candidate**: a model trained only on
trivial tasks would learn nothing about abstention, which is exactly the failure §42's
difficulty span exists to prevent.

## 54. The mechanism: the row encoder is permutation-invariant *within* a row, so the model cannot represent "column j matters"

**Date:** 2026-09-11. **MEASURED** — the trivial-prior control is a completed 6,000-step run
on a T4 (`fintfm-trivial`); the invariance table, the per-task distribution (192 tasks) and
the symmetric/asymmetric sweep (six rules, ten seeds each) were all run locally against saved
checkpoints. The architectural claim is additionally *proved*, not only measured: the
composition of a column-shared cell embedding, a position-free column encoder and a reduction
over the feature axis is symmetric by construction.

§53 set up a control — train on a prior so trivial that logistic regression scores 0.9998 —
and named the two worlds it would separate. The answer is the second one, and this finding
gives the mechanism, a proof, and a confirming prediction. **This is the root cause of §47,
§50, §51, §52 and §53 alike.**

### What the control returned

A model trained *exclusively* on trivial tasks plateaus at held-out **AUC 0.73** with training
loss flat at 0.52 from step 1850 onward. Scored per task against logistic regression fitted on
the identical context:

| | mean | median | sd |
| --- | --- | --- | --- |
| model | 0.684 | 0.683 | 0.130 |
| logistic regression, same context | **0.995** | **0.997** | — |

The per-task distribution is **unimodal at 0.68 and not one of 192 tasks was solved** (none
above 0.99). So this is not a model that identifies the rule sometimes and mistakes the sign
otherwise — which would be bimodal. It is uniform mediocrity, and that points at
representation rather than inference.

### The mechanism, read off the code

In `modeling/model.py::encode_rows`, three things compose:

- `cell_embed` is a `Linear(2 -> d_cell)` over `[z, missing]`, **shared across every column**,
  with no feature-identity term — a cell's embedding depends only on its value;
- `column_encoder` is a `TransformerEncoder` over the F feature tokens with **no positional
  encoding**, hence permutation-equivariant;
- pooling (`mean`/`max`, or the attention variant) reduces over the feature axis, hence
  permutation-invariant.

Composed, **the row representation is a symmetric function of the multiset of that row's
feature values.** The model has no representation of "column 3" anywhere. This was designed in
deliberately, for column-order invariance (D4) — but D4 asks for invariance to permuting
columns *consistently across rows*, and what is implemented is invariance to permuting each
row's values *independently*, which is far stronger and is fatal.

### Verified, not asserted

Permuting each row's values independently and measuring the change in the row embedding,
relative to its own scale:

| context rows | identically-distributed columns | different-scale columns |
| --- | --- | --- |
| 32 | 0.1915 | 0.4639 |
| 128 | 0.1119 | 0.4179 |
| 512 | 0.0376 | 0.3455 |
| 2048 | **0.0154** | 0.3412 |

With identically-distributed columns the change converges to **zero** as the context grows.
The residual at small context is estimation noise in `normalize_features`' per-column
mean/std — the only thing in the model that distinguishes columns at all, and it is an
artefact of finite samples, not an identity. (Columns of genuinely different scale stay
distinguishable, which is why the right-hand column does not converge; after normalisation
that signal is a scale tag, not a stable identity, and no prior here relies on it.)

### The confirming prediction

If the representation is symmetric, the model should be near the ceiling when the rule is a
symmetric function of the row and capped when it is not. Measured on constructed tasks, 6 iid
normal features, 400 context / 200 query rows, 10 seeds:

| rule | symmetric? | signfix mixed | vol4x |
| --- | --- | --- | --- |
| `sign(sum x)` | yes | 0.9892 | **0.9964** |
| `count(x > 0)` | yes | 0.9377 | 0.9568 |
| `sign(max x)` | yes | 0.8624 | 0.9834 |
| `sign(x_0)` | no | 0.7185 | 0.7206 |
| random linear `w·x` | no | 0.6559 | 0.6685 |
| **`x_0 - x_1`** | **antisymmetric** | **0.4930** | **0.5097** |

The decisive row is the last. For `x_0 - x_1` the label is independent of the multiset of
values, so the best achievable AUC from *any* symmetric function is exactly 0.5 — and the
model scores 0.5097. It is at its ceiling, not below it.

The model exceeds a `sum`-only oracle on the other two asymmetric rules (0.72 vs 0.685, 0.67
vs 0.486). That is not a contradiction: `sum` is a lower bound on the symmetric ceiling, since
order statistics (max, min, the sorted values) are symmetric too and carry more information.

### Why this explains everything that came before

- **§47.** The pre-fix financial prior took `np.abs()` of every driver weight, making the label
  monotone in the *sum* of drivers — a symmetric function. Shuffling context labels changed
  nothing because a global symmetric rule was learnable and sufficed. The sign fix made the
  prior's rules asymmetric, which is why §48 saw context sensitivity return *and* why
  performance did not improve: the tasks became representable-in-principle-only.
- **§50 (capacity, 17×), §52 (attention pooling), §53 (volume, 4×).** None of these touches
  permutation invariance. Parameters, pooling design and data volume cannot buy a
  representation the architecture excludes by construction. Their identical curves were the
  signature of a structural ceiling, correctly measured and wrongly attributed.
- **§51 (extremes over ordering), §53 (0.63-0.69 on every source).** `max` and `count(x > 0)`
  are exactly the symmetric functions available. The model was not failing to transfer — it
  was reporting the best symmetric summary of each table.

### The fix, and what it must preserve

The requirement is that rows **agree** on which column is which, without the model learning a
*fixed* meaning for column index j — the latter would break D4 and is what column-order
invariance exists to prevent. The standard resolution is a **per-task random column
embedding**: draw a random vector per column slot, resample it for every task, add it to the
cell embeddings. Columns become mutually distinguishable and consistently so across rows,
while the distribution over tasks stays exactly column-order invariant.

The alternative is to let the column stage attend **across rows** within each feature slot, so
a cell is contextualised by its whole column. **This is TabPFN's central architectural claim**,
and their description of it is a direct statement of the distinction this finding measures
(Nature 2025):

> our architecture ... uses a two-way attention mechanism, with each cell attending to the
> other features in its row (that is, its sample) and then attending to the same feature
> across its column (that is, all other samples). This design enables the architecture to be
> invariant to the order of both samples and features

The decisive detail is that TabPFN **assigns a separate representation to each cell and never
pools features into a row vector**. Feature identity is carried by the tensor structure: cell
*(i, j)* attends to cell *(k, j)* down its own column, so rows agree on which column is which
without any column ever acquiring a fixed learned meaning. Invariance there is a property of
*equivariant operations*; invariance here was bought by *destroying the information*, which is
the strictly stronger and fatal version. We adopted the two-way attention idea and then
undercut it with a pooling step the reference design does not have.

**These two are not alternatives — TabPFN uses both**, and their stated reason for the second
is verbatim our failure mode:

> To allow our model to differentiate features more easily that have the same statistics, for
> example, two features that have the same entries just in different orders, we use random
> feature embeddings that we add to all embeddings before the first layer. We generate one
> embedding per feature by projecting a random vector of one-fourth the size of our embeddings
> through a learned linear layer and add this to all embeddings representing an instance of
> that feature.

This matters: equivariant column attention **alone** still cannot separate two
identically-distributed features, which is precisely the regime the table above shows
collapsing to invariance as the context grows. The random embedding is what breaks that tie,
and the projection-through-a-learned-linear-layer detail is worth copying exactly — it lets
the model shape the embedding distribution rather than being handed raw noise.

The per-task random column embedding remains the cheaper test of the diagnosis, and is now
also the higher-confidence one.

Other implementation details from the same section worth adopting, several of which we already
match: z-normalisation per feature across the training rows (we do); an extra per-cell missing
indicator with the value set to 0 (we pass ``[z, missing]``, same idea); test rows attending to
training rows but **not to each other** (our ``_row_mask`` already does this, for the same
stated reason). Ones we do not have: encoding **two features per position** as an efficiency
win, and the multi-query attention variant that makes the cached train state cheap. The cell-level two-way design is the one with published evidence behind it, and
is what a serious version of this model should end up with — with the caveat that it changes
the memory profile substantially, since the state becomes ``(B, N, F, d)`` through the whole
stack rather than ``(B, N, d_model)`` after the first stage (`docs/COMPUTE.md` already records
that this model is memory-bound in ``batch x rows x features``).

A second thing worth taking from the same source: TabPFN separates inference on training and
test rows, running ICL over the training set once and caching the state for reuse, reporting
**~300x CPU speedup** at 10,000 training rows. Our `fit()`/`predict()` split currently re-runs
the context on every call, and the cell-level rewrite is the right moment to fix that.
**Everything measured before this finding was measured through this ceiling**, so the
architecture comparisons in §50 and §52 must be re-run once it is lifted; they compared
variants that were all equally constrained.

## 55. What TabPFN's published protocol does not measure — and why that is where the gap is

**Date:** 2026-09-11. **HYPOTHESIS**, and deliberately labelled as one: this finding contains
no numbers of ours. The quotations are checkable fact; the conclusion drawn from them — that
the gap is where their protocol is silent — is a conjecture until each half of it is measured,
and §55's whole value evaporates if it is quoted as though it were a result. It is
a reading of the TabPFN Nature 2025 paper's methods sections, quoted directly so the claims
can be checked against the source rather than against a summary of it (a summariser
previously fabricated a results table for this project, `docs/POSTMORTEM.md`). Every
competitive inference drawn here is about **what their protocol reports**, which is
verifiable, and not about how their model would score on an unreported metric, which is not.

§54 settled the architecture question by adopting their design. This finding is the opposite
exercise: reading their **evaluation protocol** for what it leaves unmeasured. These are
durable competitive facts, sourced from the Nature 2025 paper's methods sections, and they
matter more to this project's positioning than any accuracy number in it.

### 1. They never measure calibration

> To obtain scores for classification tasks, we use two widely adopted evaluation metrics:
> ROC AUC (One-vs-Rest) and accuracy.

**Neither is a proper scoring rule.** `CLAUDE.md` already records why that matters and §42
records this project shipping three runs that logged accuracy below a constant predictor
without noticing. So TabPFN's published evidence establishes that it *ranks* well and says
essentially nothing about whether a stated probability is correct.

That is precisely and only what a credit supervisor grades. §12 measured this project's
calibration advantage over gradient boosting at 2.3x to 11.7x across every dataset size. A
head-to-head on **Brier and ECE with reliability bins**, rather than AUC, is a comparison
nobody in this literature has published — and one where the incumbent has no stated result
to defend.

Caution before claiming it: not measured is not the same as not good. TabPFN's loss is
cross-entropy, which *is* proper, so it may well be well calibrated and simply unreported.
The claim available today is about the evidence, not about the model.

### 2. There is no out-of-time evaluation

> 10 repetitions, each with a different random seed and train-test split (90% train and 10%
> test samples)

Random splits, throughout. **No temporal holdout anywhere in the protocol.** For credit risk
this is not a minor omission: out-of-time performance is a mandatory model-validation test,
because a model that interpolates within a period can fail completely across one, and a random
split cannot see that. Our V4FinBench out-of-time protocol already exists and already tests it.

### 3. Their benchmarks are not selected for imbalance, and neither is their prior

The test suites are the AutoML Benchmark and OpenML-CTR23, curated for "sufficient complexity,
real-world relevance, absence of free-form text features and diversity of problem domains" —
class imbalance is not a selection criterion, and nothing in the SCM prior forces extreme skew.
Combined with §9 (the credit field's own benchmark averages a **22%** default rate), the
low-default regime at **0.19%** is unrepresented in both literatures at once.

### 4. The prior generator is closed

> The code to generate synthetic pre-training data has not been released with our models.

The weights and inference code are open; **the prior is not**. That is a deliberate choice and
it identifies where they believe the value sits — which is the same conclusion §54 forces from
the other direction. It also means a domain-specific prior cannot be dismissed as trivially
replicable by pointing at their repository, and that our prior work is genuinely our own rather
than a reimplementation.

### Two protocol cautions when comparing numbers

- **Their headline figures are min-max normalised per dataset**, scaled so the best method
  scores 1.0 and the worst 0.0 across the compared methods. "0.939 versus 0.752" is therefore
  not an AUC difference and must never be quoted as one. Absolute numbers are in their
  supplementary tables and are the only ones comparable to ours.
- **"TabPFN (default)" is already a four-way ensemble** (eight-way for regression) over
  different pre- and post-processors. A fair comparison either ensembles ours the same way or
  states plainly that it does not.

Worth copying regardless of the competitive angle: they keep **development datasets strictly
disjoint from test datasets** for choosing hyperparameters and search spaces. This repository
does not currently have that separation formalised, and every tuning decision made on an
evaluation set is a quiet leak.

### The resulting position

We should not contest general small-to-medium tabular classification; the evidence there is
theirs and it is strong. The defensible gap is the intersection of four things their protocol
never tests: **low default rates, out-of-time splits, proper scoring rules, and a term
structure over horizons** — the last having no counterpart in their model at all, since the
architecture has no time axis and their own future work names time-series priors as open.

None of it is measurable until §54's encoder fix lands, because every number this project has
was measured through that ceiling.

## 56. The fix works: 0.713 -> 0.997 on the trivial prior, and the unlearnable rule is solved

**Date:** 2026-09-11. **MEASURED** — two completed 6,000-step T4 runs (`fintfm-trivial`,
`fintfm-trivial-colid`) differing in exactly one config field, plus a six-rule sweep at ten
seeds each, eight column-identity draws averaged per prediction. The prediction being tested
was registered in §54 *before* either number was known.

### The control, rerun with column identities

Identical configuration, identical prior, identical step count. The only change is
`column_id_dim`: `0` before, `d_cell // 4 = 12` after.

| | held-out AUC | loss | Brier skill | ceiling |
| --- | --- | --- | --- | --- |
| no column identities | 0.713 | 0.515 | +0.125 | 0.9998 |
| **with column identities** | **0.997** | **0.058** | **+0.906** | 0.9998 |

The before-arm was fully converged — flat from step 1,850 through 6,000 with the learning rate
annealed to zero — so this is not a training-length artefact. The after-arm passed AUC 0.99 by
step 1,500, a quarter of the way in.

### §54's prediction, tested

§54 predicted that the model would be at ceiling on symmetric rules and capped on asymmetric
ones, and that the fix should lift specifically the asymmetric ones. Measured on the two
trivial-prior checkpoints:

| rule | symmetric? | no col ids | with col ids |
| --- | --- | --- | --- |
| `sign(sum x)` | yes | 0.9911 | 0.9996 |
| `count(x > 0)` | yes | 0.9034 | 0.9232 |
| `sign(max x)` | yes | 0.8426 | 0.8867 |
| `sign(x_0)` | no | 0.6946 | **0.9994** |
| random linear `w·x` | no | 0.6618 | **0.9991** |
| **`x_0 - x_1`** | **antisymmetric** | **0.5007** | **0.9995** |

The asymmetric rules move from 0.50-0.69 to 0.999; the symmetric ones, which were never the
constrained case, improve slightly. **`x_0 - x_1` is the one that settles it**: its symmetric
ceiling is provably exactly 0.5, it measured 0.5007 before, and it measures 0.9995 after. No
amount of capacity, data or pooling redesign could have moved that number, and §50, §52 and
§53 each spent a run confirming as much.

Fifteen lines in `encode_rows`.

### What this invalidates

**Every quantitative result in this repository before today was measured through the §54
ceiling.** Specifically:

- **§50 (capacity), §52 (pooling, label noise), §53 (volume, capped predictor)** compared
  variants that were all equally constrained. Their measurements were correct and their
  conclusions — "capacity is not the constraint", "pooling changes nothing" — are unsafe to
  carry forward. They must be re-run, and their *readings* are now retracted rather than
  merely qualified.
- **§26-§40, all survival and hazard work.** Already invalid for a separate reason (untrained
  classification heads, §34); now invalid for this one too.
- **§33's public-benchmark claim and §32's retrieval gains.** Re-measure before quoting.
- **§9, §12, §23, §35, §38** and anything comparing us to boosting baselines. The baselines
  are unaffected; our side of every comparison is not.

What survives untouched: the prior-side findings (§42's difficulty span, §47's sign bug), the
protocol and honesty findings, `docs/COMPUTE.md`'s timings, and §55, which contains no numbers
of ours.

### The weakest remaining cell

`count(x > 0)` at 0.9232 and `sign(max x)` at 0.8867 are now the low scores, and both are
symmetric rules requiring a *nonlinear* aggregation over features rather than a linear one.
They improved only slightly, so the column identities were not their constraint.

**Check the reference before calling this a failure.** Logistic regression scores only 0.8919
on `count(x > 0)`, so the model is *ahead* of the linear baseline there — LR is simply the
wrong yardstick for a nonlinear rule. The right one is the Bayes ceiling, obtained by scoring
with the rule's own statistic, which is **1.0000** for all three symmetry probes. Against that
the model is 0.077 short on `count(x > 0)`, which is a real gap and not an artefact of an
unreachable target.

So it stays the first candidate for a second, independent bottleneck — nonlinear aggregation
across features rather than column identity — and is worth a look once the financial prior is
re-run. Stated carefully because the temptation was to read 0.9232-against-0.8919 as a
deficiency when it is an improvement on the baseline and a shortfall against the ceiling at
the same time.

## 57. The training loop's held-out AUC was pooled across tasks, overstating discrimination by 0.26

**Date:** 2026-09-11. **MEASURED**, 80 tasks per prior, two checkpoints, scored both ways from
the same forward passes.

While setting up the before/after comparison for §56, the before-arms' logged held-out AUC —
0.943 for the mixed prior, 0.834 for the pure financial one — flatly contradicted §53, which
found 0.63-0.69 on every task source. Both were right. The metric was reading the wrong thing.

`_eval_quality` concatenated the query rows of 5 batches × 16 tasks and computed **one AUC over
the pool**. Scored per task instead, from the identical forward passes:

| checkpoint | pooled AUC | per-task mean | per-task median | base-rate spread |
| --- | --- | --- | --- | --- |
| `signfix-mixed` | 0.9080 | **0.6426** | 0.6566 | 0.003 - 0.986 |
| `signfix-fin` | 0.8629 | **0.6488** | 0.6494 | 0.002 - 0.312 |

**The mechanism.** AUC asks whether a positive outranks a negative. Pooled across tasks, most
such pairs come from *different* tasks, so a model that merely predicts each task's base rate
ranks a positive from a 90%-default task above a negative from a 0.3%-default task and is paid
for it — without discriminating between two firms inside either task, which is the only thing
a credit model is for. The financial prior's rates span three orders of magnitude, so there is
a great deal to be paid for.

**Why this hid for so long.** §42 caught the previous version of the same mistake — the loop
reported *accuracy*, which at a 4.7% base rate sat below a constant predictor — and replaced it
with AUC and Brier skill. The replacement fixed the metric and kept the pooling, and pooling is
invisible unless base rates vary. The trivial prior of §54 and §56 has a fixed 0.30 rate for
every task, which is exactly why its 0.713 and 0.997 were trustworthy and comparable.

**Fixed** by reporting `auc_per_task` alongside the pooled number, printing both, and
documenting in the docstring which one to read. The pooled figure is kept rather than removed,
so earlier runs remain comparable to later ones — deleting it would silently break every
number in this file that came from a training log.

**Consequence for §56.** The trivial-prior comparison is unaffected (uniform base rate). The
financial arms now training were launched against the old wheel and will log the pooled
number; their per-task figures are computed offline from the checkpoints, which is where the
comparison to 0.6426 and 0.6488 belongs.

**The wider lesson, and it is the third time.** §42 was accuracy versus a base rate. §34 was a
head that was never trained. This is pooling across heterogeneous tasks. Each time the
instrument was wrong in the flattering direction, and each time it was caught by a *disagreement
between two measurements* rather than by inspection. Two numbers that should agree and do not
are the most productive thing in this repository; the rule worth keeping is to chase the
disagreement before chasing the result.

## 58. The architecture was necessary, not sufficient: the financial prior does not teach column-specific inference

**Date:** 2026-09-11. **MEASURED** — three completed 6,000-step T4 runs differing only in
`p_financial`, each against its own before-arm, scored per task (§57) and on the symmetry
probes. The isolation run that tests this finding's proposed mechanism was launched **before**
its result was known; the prediction below is registered, not fitted.

§56 showed the column-identity fix takes the trivial prior from 0.713 to 0.997. Rerunning the
*financial* prior with the same fix gives a much less comfortable answer.

### Dose-response in the generic-SCM share

Probe AUCs are comparable across arms (identical probes, eight column-identity draws averaged,
eight seeds):

| SCM share | arm | antisymmetric | orientation | linear |
| --- | --- | --- | --- | --- |
| 0% | `p_financial=1.0` before | 0.4960 | 0.6916 | 0.6387 |
| 0% | `p_financial=1.0` **after** | **0.5352** | 0.6881 | 0.6503 |
| 30% | `p_financial=0.7` before | 0.4887 | 0.7054 | 0.6437 |
| 30% | `p_financial=0.7` **after** | **0.9116** | 0.9464 | 0.8943 |
| 100% | `p_financial=0.0` **after** | **0.9932** | 0.9918 | 0.9897 |

Monotone in the SCM share on all three probes. **Trained on the financial prior alone, the fix
buys almost nothing** — 0.4960 to 0.5352 against a chance floor of 0.5 — while the same
architecture and the same step count on a 70/30 mixture reaches 0.9116.

So §54's encoder bug was a real bottleneck and removing it was necessary. It was not
sufficient: the capability is now *available*, and the financial prior does not *teach* it.

### Two explanations eliminated by measurement

- **Task width.** Excluded. Financial tasks have a median of 70 real columns, generic SCM
  tasks 68. Further, probing the antisymmetric rule at widths 2 through 64 on a model that
  *did* learn the capability shows graceful decay (1.0000 at two features to 0.7773 at 64),
  whereas the financial-prior model sits at 0.6339 even at **two** features, where there is
  nothing to aggregate. The learned strategy, not the probe, is what fails.
- **Dilution by uninformative columns.** Excluded. The fraction of columns that individually
  separate the classes is 77.7% for the financial prior and 80.0% for the SCM prior.

### The registered prediction

What remains is stark:

| prior | median base rate | missing cells | best single-column AUC |
| --- | --- | --- | --- |
| financial | **0.021** | 8.9% | 0.812 |
| generic SCM | **0.500** | 0.0% | 0.911 |

A 24x difference in positive density. At 512 rows the financial prior yields roughly **eleven
positives per task**, a fraction of which fall in the context — too little to learn a
column-specific rule from, while a symmetric summary remains cheaply available.

**Prediction:** opening the financial prior's rate envelope, changing nothing else, will
recover the capability. `configs/balanced-financial.yaml` does exactly that — same accounting
identities, same drivers, same missingness, median base rate 0.401 instead of 0.021 — and is
training now. If it reaches the mixture's ~0.91 on the antisymmetric probe, base rate is the
cause. If it stays near 0.54, base rate is **excluded** and the cause is missingness or task
separability, which is worth the run either way.

### The uncomfortable part

**The target segment is what starves the pretraining signal.** The low-default portfolio is
what this project exists for (`docs/GLOSSARY.md`), and a prior faithful to it supplies too few
positives for the model to learn the discrimination that regime demands. If the prediction
holds, the resolution is a **curriculum** spanning both densities rather than a move to
balanced tasks — a model trained only on balanced data would be trained away from its purpose,
and §42's difficulty span exists for the same reason.

### What is *not* claimed

Per-task AUC is **not** comparable across these arms: each is scored on its own training prior,
and the SCM prior is balanced and cleaner, so `p_financial=0.0`'s 0.8029 against
`p_financial=1.0`'s 0.6693 mostly reflects an easier evaluation, not a better model. Only the
probe columns support cross-arm comparison.

And on the financial prior itself, the fix moved per-task AUC from 0.6491 to **0.6863** — a
real gain, measured, and far too small to quote as a result. The capability is present; turning
it into discrimination on realistic low-default tasks is the next problem, not a solved one.

## 59. On V4FinBench, column identity is worth 0.0009 AUC — the §54 fix barely moves real data

**Date:** 2026-09-11. **MEASURED**, and it is a corrective to §54, §56 and §58 rather than an
extension of them. Two protocol runs differing only in checkpoint, plus a direct bound on how
much column identity can possibly be worth on this dataset.

### The real-data result

V4FinBench published protocol, horizon 0, fold 0, **0.380% default rate** (402 positives in
105,900 rows). Baselines were byte-identical across both runs, confirming only our model
changed:

| arm | ROC-AUC | F1 |
| --- | --- | --- |
| catboost | 0.9944 | 0.3706 |
| **fintfm, after the §54 fix** | **0.9827** | 0.2500 |
| logistic_regression | 0.9819 | 0.2724 |
| **fintfm, before the fix** | **0.9756** | 0.2364 |
| xgboost | 0.9577 | 0.3417 |
| lightgbm | 0.9229 | 0.2807 |

**+0.0071 AUC**, against **+0.42** on the synthetic antisymmetric probe. With 79 positives in
the test fold that gain is very likely inside sampling error, and these baselines are untuned.

### Why: the task does not need column identity

Fit gradient boosting on the **order statistics** of each row's standardised features — the
sorted vector, which is a *complete* symmetric summary and discards column identity entirely —
and compare against the same model on the raw columns:

| representation | AUC |
| --- | --- |
| order statistics (purely symmetric) | **0.9985** |
| raw columns (identity preserved) | **0.9994** |

**Column identity is worth 0.0009 AUC on this dataset.** A permutation-invariant model is
essentially unhandicapped, which is exactly why the pre-fix checkpoint reached 0.9756 and why
the fix could not add much. §47's bug hid for three days for the same underlying reason.

*Caveat on those two numbers:* they use a random 90k/30k split, not the protocol's country
folds, so both are inflated by firm-level leakage and neither is comparable to the protocol
table above. The **comparison between them** is valid, since both sides share the split, and
that comparison is the whole point.

### Two of my own inferences that this killed

- **"Crude symmetric predictors cap at 0.78, so the model cannot be using one."** Wrong. Sum,
  max, min, `max|z|` and missing-count all score 0.54-0.78, but those are a tiny subset of the
  symmetric functions available. The complete symmetric summary reaches 0.9985. A learned
  permutation-invariant network is not limited to the statistics one happens to think of.
- **"Real columns have different scales, so per-column normalisation leaks identity."**
  Wrong, and the measurement that suggested it was confounded. Per-row permutation sensitivity
  reads 0.4334 on real columns against 0.0053 on iid-normal ones — but permuting raw values
  between columns of different scale changes the *normalised* values drastically, so the
  embedding moves for a reason that has nothing to do with the model knowing which column is
  which. The valid test is predictive: the antisymmetric probe scores **0.5044 on iid-normal
  columns and 0.5044 on heterogeneous ones**, identical to four decimals. No usable leak.

### What this does and does not change

**Unchanged.** §54's mechanism is proved, not merely measured. §56's trivial-prior result
(0.713 to 0.997) and §58's dose-response in the SCM share are real, reproducible, and correct.
The architecture could not represent a rule as simple as `x_0 - x_1`, and now it can.

**Changed.** The claim that this was *the* thing holding the project back is retracted. It was
the thing holding back performance on tasks that **require** column identity, and V4FinBench
horizon 0 is not one. The day's headline number does not transfer to the product metric, and
saying otherwise would be the §50 mistake again — a correct measurement carrying a conclusion
it does not support.

**The honest summary of the fix's value:** it removes a provable representational ceiling whose
cost on our current benchmark is under 0.01 AUC. Worth keeping — a model that cannot represent
`x_0 - x_1` will fail on some future table, and we cannot know which in advance — but it is
insurance, not the unlock.

### What to look at instead

The F1 column is the more interesting anomaly. Ours is the **worst of the five arms** (0.2500)
while our AUC is second-best. Strong ranking with weak thresholded classification is a
calibration or threshold-transfer failure, and calibration is the one dimension where this
project claims a measured advantage (§12) and where TabPFN publishes nothing (§55). That gap
is both larger and more aligned with the thesis than anything §54 was about.

## 60. ROC-AUC flattered us by a factor of two: on average precision we are third, not second

**Date:** 2026-09-11. **MEASURED**, V4FinBench protocol horizon 0 fold 0, 0.380% base rate,
79 positives in the test fold, baselines at library defaults.

§59 reported ROC-AUC and F1 because those are their protocol's metrics, and noted our F1 was
the worst of five arms while our ROC-AUC was second-best. Adding **average precision** and an
**oracle F1** to the harness explains that, and the explanation is not flattering:

| arm | ROC-AUC | **AP** | F1 | F1-oracle | transfer loss |
| --- | --- | --- | --- | --- | --- |
| catboost | 0.9944 | **0.3310** | 0.3706 | 0.3865 | 0.016 |
| xgboost | 0.9577 | **0.2685** | 0.3417 | 0.3957 | 0.054 |
| **fintfm** | 0.9827 | **0.1853** | 0.2500 | 0.2740 | 0.024 |
| logistic_regression | 0.9819 | 0.1786 | 0.2724 | 0.3163 | 0.044 |
| lightgbm | 0.9229 | 0.1638 | 0.2807 | 0.3093 | 0.029 |

**ROC-AUC 0.9827 against CatBoost's 0.9944 reads as a 0.012 gap. Average precision says 0.1853
against 0.3310 — CatBoost is 1.8x better.** Ordering by ROC-AUC puts us second of five;
ordering by AP puts us third, behind both CatBoost and XGBoost.

**Why ROC-AUC misleads here.** At a 0.380% base rate the test fold holds 79 positives and
20,838 negatives. ROC-AUC averages over all sensitivity-specificity trade-offs, the vast
majority of which live in a region no credit decision is ever taken in; separating the bulk of
obviously-safe firms is nearly free and dominates the score. AP integrates precision against
recall, so it is sensitive to exactly the high-precision region where a decision is made. This
is why AP was added, and it should be read first at low base rates.

**It is not calibration, and it is not the threshold.** The oracle column settles a question
§59 left open. Our transfer loss -- best-achievable F1 on test minus F1 at the threshold chosen
on validation -- is **0.024, the second smallest in the table**, better than logistic regression
(0.044) and XGBoost (0.054). The threshold transfers fine. But our **oracle** F1 of 0.2740 is
the **lowest of all five arms**: with a perfect threshold handed to us we would still finish
last. The deficit is in the ranking near the decision boundary, not in turning a ranking into a
decision.

That retracts the reading proposed at the end of §59 — that the F1 gap was a calibration or
threshold-transfer failure and therefore aligned with this project's stated advantage. It is
neither. It is a plain discrimination deficit in the region that matters, and the calibration
thesis is untouched by this measurement rather than supported by it.

**The uncomfortable symmetry with §55.** That finding criticises TabPFN for reporting only
ROC-AUC and accuracy, neither a proper scoring rule, and argues the gap is where their protocol
is silent. Our own harness reported ROC-AUC and F1 and would have carried "second of five" into
a document. The criticism was correct and we were committing a version of it at the same time.
`docs/DECISIONS.md` should record AP as the metric read first on any low-base-rate split.

### The ordering, tested rather than asserted

"Third of five" was over-precise, and a paired bootstrap over the 20,917 test rows says so.
Pairing matters because every arm scores the same rows, so their errors are correlated and an
unpaired interval would overstate the uncertainty of a *difference*. 2,000 resamples,
Holm-adjusted across the family of four:

| fintfm vs | dAP | 95% CI | Holm p | verdict |
| --- | --- | --- | --- | --- |
| logistic_regression | +0.0067 | [-0.0588, +0.0826] | 1.000 | indistinguishable |
| lightgbm | +0.0215 | [-0.0488, +0.1099] | 1.000 | indistinguishable |
| xgboost | -0.0832 | [-0.1912, +0.0255] | 0.423 | indistinguishable |
| **catboost** | **-0.1457** | **[-0.2269, -0.0623]** | **<0.001** | **DIFFERENT** |

**Exactly one comparison of four is established: CatBoost is genuinely ahead.** fintfm,
logistic regression, LightGBM and XGBoost form a four-way tie at this sample size. The
intervals run +/-0.07 to +/-0.09, so with 79 positives only a gap of roughly 0.15 was ever
detectable — which is precisely Baesens et al.'s finding of significance in 22 of 406
comparisons, reproduced on our own data.

So the league table in this finding should be read as: CatBoost first, everything else
unresolved. Ranking arms by point estimate and reporting a position is the error this project
already warned itself about (`evaluation/metrics.py`: "a win count is not a result"), and the
first draft of this finding committed it anyway.

### Is 0.1853 signal at all?

Yes, and by a wide margin. A random ranker's average precision **equals the prevalence**, which
is 0.00378 here, so 0.1853 is **49x the chance floor** (CatBoost is 88x, LightGBM 43x). The
absolute number looks small only because AP is scaled to the base rate; that is the property
that makes it readable here, where ROC-AUC compresses everything from useless to excellent
into its top 2%.

**Caveats that cut both ways.** One fold and 79 positives, which the intervals above quantify.
The baselines are **untuned**, and their protocol grid-searches each one — so the established
CatBoost gap is, if anything, **understated**, while the three ties might resolve in either
direction. Five folds and `--tune` before any of this is quotable.

## 61. The financial prior doubles average precision on credit — it specialises rather than helps generally

**Date:** 2026-09-11. **MEASURED**, V4FinBench protocol horizon 0 fold 0, two checkpoints
identical in architecture, steps and seed, differing only in `p_financial`.

§58 found that the *generic SCM* share is what teaches column-specific in-context inference,
and that a pure financial prior teaches almost none of it. Read alone, that invites the
conclusion that the financial prior is dead weight. On the benchmark it is the opposite:

| arm | ROC-AUC | **AP** |
| --- | --- | --- |
| catboost | 0.9944 | 0.3310 |
| xgboost | 0.9577 | 0.2685 |
| **fintfm, `p_financial=0.7`** | 0.9827 | **0.1853** |
| logistic_regression | 0.9819 | 0.1786 |
| lightgbm | 0.9229 | 0.1638 |
| **fintfm, `p_financial=0.0`** | 0.9611 | **0.0918** |

**The financial prior doubles average precision on the credit task**, 0.1853 against 0.0918,
and the ordering is the same on ROC-AUC (0.9827 against 0.9611) so it is not an artefact of
metric choice. Dropping the domain prior would have cost half the precision on the task this
project exists for.

### Why this nearly went the other way

§58's probes measure a *capability* — can the model condition on which column is which — and
the SCM-heavy arms win those decisively. Capability is not the same as fit to a target
distribution, and the pure-SCM checkpoint demonstrates the gap: best-in-class on every
synthetic probe, worst-in-class here. A finding about probes does not transfer to a finding
about benchmarks without being measured on the benchmark, which is the §50 lesson in a new
costume.

### The plausible mechanism, and how to test it

The financial prior's median base rate is **2.1%** and V4FinBench horizon 0 is **0.380%** —
the same regime. The generic SCM prior's median is **50%**. So this may be **base-rate regime
match** rather than domain match in any richer sense, and that is the sharper claim because it
is testable: a prior can be moved along the base-rate axis while holding its generator fixed,
which `configs/balanced-financial.yaml` already does in the other direction.

The two readings make opposite predictions for a balanced-rate *financial* prior. If domain
content is what matters, it should keep most of the benefit here; if base-rate regime is what
matters, it should lose it. That run is in flight for a different reason (§58) and will
discriminate these as a side effect.

### What this does not license

**No single checkpoint here is good at both regimes.** The model does not generalise across
base rates: the arm that wins at 0.38% loses badly where positives are common, and vice versa.
A foundation model that must be told its deployment base rate in advance to pick a checkpoint
is not yet a foundation model, and this is the strongest argument so far for §58's curriculum
rather than a choice between the two priors.

Caveats unchanged from §60: one fold, 79 positives, untuned baselines, so the gap to CatBoost
and XGBoost is if anything understated, and adjacent arms are not separated.

## 62. §58's registered prediction is falsified: base rate is not why the financial prior fails to teach

**Date:** 2026-09-11. **MEASURED**, one completed 6,000-step T4 run against three existing
arms, using the prediction §58 registered **before** this run's result was known.

§58 eliminated task width and dilution by uninformative columns as explanations for the
financial prior's failure to teach column-specific in-context inference, and named the
remaining candidate: base rate. The financial prior's median is 2.1%, the generic SCM prior's
is 50%, and at 512 rows that is roughly eleven positives per task against 256. It predicted
that opening the rate envelope, changing nothing else, would recover the capability.

**It does not.** `configs/balanced-financial.yaml` keeps the financial generator identical —
same accounting identities, same drivers, same missingness — and raises the median base rate
from 0.021 to 0.401:

| arm | base rate | antisymmetric probe |
| --- | --- | --- |
| `p_financial=1.0`, default envelope | 0.021 | 0.5352 |
| **`p_financial=1.0`, balanced envelope** | **0.401** | **0.5669** |
| `p_financial=0.7` (30% generic SCM) | mixed | **0.9116** |
| `p_financial=0.0` (pure generic SCM) | 0.500 | **0.9932** |

A 19x increase in positive density bought **+0.03** on a probe where the mixture gains +0.38.
Base rate is not the mechanism. Nor is it a partial one: 0.5669 against a chance floor of 0.5
is the same "barely above chance" regime as before.

### What is left

Three of the four differences between the two priors are now eliminated — width (70 columns
against 68), informative fraction (77.7% against 80.0%), and base rate. What remains
unmeasured:

- **Missingness.** 8.9% of financial cells against 0.0% for the SCM prior. Cheap to test the
  same way: strip missingness from the financial generator and re-run.
- **Per-column separability.** Best single-column AUC 0.812 against 0.911 — the SCM prior's
  tasks are simply cleaner, so the "which column matters" signal is sharper per example.
- **Correlation structure.** The financial prior's features are tied by accounting identities,
  so its columns are far from independent. A rule over correlated drivers may be learnable by
  many different weightings, which would weaken the pressure to identify any particular column.

The third is the most interesting and the hardest to test, because removing the identities
would stop it being a financial prior at all.

### Why this finding exists in this form

The prediction was written into §58 and the run launched before its result was known,
specifically so it could fail visibly. It failed. Recording that is the point: three of this
session's readings (§59, §60, §61) had to be retracted after further measurement, and each
retraction came from a check made *after* the conclusion was drafted. A prediction registered
in advance is the cheap version of the same discipline.

### Consequence for `base-rate-curriculum`

Task 38.1 gated that proposal on this result, and the answer changes its shape. The sampler is
still worth building — §61's regime effect is real and no single checkpoint yet works at both
ends — but it can no longer be justified as *the* fix for the financial prior's teaching
failure, because density is not what is broken. The proposal's "Open question" section should
be read as answered in the negative, and the remaining candidates above are the live ones.

## 63. Two-factor control: the financial prior's benefit is its structure, and its density is irrelevant

**Date:** 2026-09-11. **MEASURED**, three checkpoints on the identical V4FinBench fold — same
20,917 rows, same 79 positives — compared with a paired bootstrap over 2,000 resamples,
Holm-adjusted.

§61 found the financial prior doubles average precision on credit and left an open question:
is that **domain content** or merely **base-rate regime match**? The financial prior's median
rate is 2.1% and V4FinBench's is 0.380%, so the two explanations were confounded.

§62's balanced-financial checkpoint breaks the confound, because it has the financial
generator at a base rate close to the SCM prior's:

| arm | financial content | prior base rate | AP |
| --- | --- | --- | --- |
| `colid-fin00` | none | 0.500 | 0.0918 |
| **`colid-finbal`** | **full** | **0.401** | **0.1900** |
| `colid-fin07` | 70% | 0.021 | 0.1853 |

| contrast | dAP | 95% CI | Holm p | verdict |
| --- | --- | --- | --- | --- |
| **domain content**, density matched (`finbal` - `fin00`) | **+0.0982** | [+0.0333, +0.1747] | **0.002** | **DIFFERENT** |
| **density**, domain matched (`finbal` - `fin07`) | +0.0047 | [-0.0675, +0.0750] | 0.895 | indistinguishable |

**The financial generator's structure is worth a significant +0.098 AP. A twenty-fold change
in its base rate is worth nothing measurable.** This is the second comparison in this session
to survive significance testing, the first being CatBoost's lead (§60).

That is a real result for this project's founding premise, and the first evidence for it that
is controlled rather than suggestive. The domain prior earns its keep through the accounting
identities and driver structure — not through matching the imbalance of the target, which was
the more deflationary reading and is now excluded.

### It also sharpens §62 rather than resolving it

The financial prior **transfers well** to credit and **teaches poorly**: 0.5669 on the
antisymmetric probe against the SCM mixture's 0.9116. Those are now cleanly separable
properties. What a prior teaches the model about conditioning on which column is which, and
what it contributes to performance on a matched domain, are different things carried by
different features of the generator. A prior can be good at one and bad at the other, and ours
is exactly that.

This means the two open problems should stop being discussed as one:

- **Teaching** — why the financial generator does not produce column-specific in-context
  inference. Candidates remaining after §62: missingness, per-column separability, and the
  correlation imposed by the accounting identities.
- **Coverage** — why no single checkpoint works across target domains. `p_financial=0.7`
  already mixes the two priors and does *not* deliver best-of-both; it matches the
  financial-only arm on credit and is beaten by the SCM-only arm elsewhere.

### Caveats

One fold, 79 positives. The significant contrast has a lower bound of +0.033, so its
*direction* is established while its magnitude is loose. Both fintfm arms remain
indistinguishable from logistic regression, LightGBM and XGBoost, and both lose to CatBoost
(§60) — this finding is about which prior to build, not about being competitive yet.

## 64. What a prior teaches tracks its signal-to-noise, not how much column identity it demands

**Date:** 2026-09-11. **MEASURED**, 22-40 sampled tasks per prior, gradient boosting fitted
twice per task on a two-thirds/one-third split. **Registered hypothesis, falsified.**

§62 eliminated width, dilution and base rate as reasons the financial prior fails to teach
column-specific in-context inference, leaving missingness, per-column separability and
correlation structure. The hypothesis stated before this measurement was that the financial
prior's tasks simply **do not need** column identity — which would make its failure to teach
correct behaviour rather than a defect, and would explain §59's finding that V4FinBench itself
has almost no use for it.

**Identity advantage** measures this directly: fit the same model on raw columns, and on the
**order statistics** of each row's standardised values — a *complete* symmetric summary that
discards which column is which. The gap is exactly what column identity is worth.

| prior | raw AUC | order-statistic AUC | identity advantage |
| --- | --- | --- | --- |
| financial (default envelope) | 0.7276 | 0.5796 | **+0.1480** |
| financial (balanced envelope) | 0.7434 | 0.5983 | **+0.1451** |
| generic SCM | 0.8306 | 0.6881 | **+0.1425** |
| trivial | 0.9912 | 0.6902 | +0.3010 |
| *V4FinBench h0 (§59)* | *0.9994* | *0.9985* | *+0.0009* |

**The hypothesis is false.** The financial and generic-SCM priors demand column identity
equally — 0.1480 against 0.1425 — while one teaches it (antisymmetric probe 0.9932) and the
other does not (0.5669). The prior's tasks require the capability; the model fails to acquire
it regardless.

### What does track teaching

The **raw** column, which is simply how learnable each prior's tasks are by any means:

| prior | raw AUC | antisymmetric probe after 6,000 steps |
| --- | --- | --- |
| trivial | 0.9912 | 0.9995 |
| generic SCM | 0.8306 | 0.9932 |
| financial | 0.7276 | 0.5669 |

Monotone, and the mixture sits between its components on both. **Signal-to-noise, not identity
demand.** Gradient boosting tops out at 0.73 on financial tasks — they are intrinsically hard,
so the gradient carrying "which column matters" arrives buried in noise. The SCM prior's tasks
are cleanly solvable, so the same signal arrives intact.

This generalises §62's per-column-separability candidate from a property of single columns to a
property of the whole task, and it explains §58's dose-response without further assumption:
raising the SCM share raises the fraction of high-SNR tasks. It is also §42 again — that
finding fixed a prior clamped to a narrow *difficulty* band, and this is the same variable
seen from the learning side rather than the sampling side.

**Three points is not a curve.** The ordering is consistent and the mechanism is plausible, but
this is a correlation over three priors and should be treated as a hypothesis with one
falsification already behind it, not a law. The causal test is to raise the financial prior's
SNR while holding its structure fixed, and measure whether teaching follows.

### The separate observation, and it may matter more

**V4FinBench's identity advantage is +0.0009. Every prior we train on is between 150x and 330x
higher.** Our priors demand a capability the target barely uses, and conversely offer little
practice at whatever credit data actually rewards — which, since order statistics reach 0.9985
there, is extracting a great deal from a symmetric summary of a firm's standardised ratios.

That is a prior-target mismatch on a measurable axis, it was invisible until this metric
existed, and it reframes §59: the §54 encoder fix bought 0.0071 AUC on V4FinBench not because
the fix was small but because **the benchmark is a task where column identity is nearly
worthless**. Whether that generalises to credit data at large, or is specific to this panel's
131 heavily-engineered ratios, is unknown and worth knowing before the next prior is designed.

## 65. Signal-to-noise is falsified too: five explanations eliminated, the financial prior still will not teach

**Date:** 2026-09-11. **MEASURED.** **Second registered prediction of the day, also falsified.**

§64 proposed that what a prior teaches tracks how learnable its tasks are, and named the causal
test: raise the financial prior's signal-to-noise while holding its structure fixed. That test
was specified, `sharpness_min` was made configurable for it, the manipulation was verified to
work *before* the run, and the prediction — antisymmetric probe ≳0.93 — was written down first.

| arm | prior raw AUC | antisymmetric probe |
| --- | --- | --- |
| financial, default envelope | 0.7276 | 0.5352 |
| financial, balanced base rate (§62) | 0.7434 | 0.5669 |
| **financial, sharpened (`configs/sharp-financial.yaml`)** | **0.8885** | **0.5861** |
| generic SCM | 0.8306 | **0.9932** |
| trivial | 0.9912 | **0.9995** |

**The sharpened financial prior is now more learnable than the generic SCM prior — 0.8885
against 0.8306 — and still teaches at 0.586 against 0.993.** The correlation §64 reported over
three points does not survive a fourth placed deliberately to break it.

All three financial arms sit at 0.535-0.586 regardless of base rate, sharpness, or density. The
constant is the generator.

### The elimination table

| explanation | measured | verdict |
| --- | --- | --- |
| task width | 70 columns against 68 | eliminated (§58) |
| dilution by uninformative columns | 77.7% informative against 80.0% | eliminated (§58) |
| base rate | 0.021 raised to 0.401, probe +0.03 | eliminated (§62) |
| how much column identity tasks demand | +0.1480 against +0.1425 | eliminated (§64) |
| learnability / signal-to-noise | raised to 0.8885, probe +0.05 | **eliminated here** |
| near-duplicate columns | 11.7% against 4.0% | weak, wrong direction for rank |
| overall correlation | effective rank 25.8 against 18.3 — financial is *less* correlated | weak |
| missingness | 6.8% against 4.4% | too small |

The last three are measured but none is a plausible cause of a 0.41 gap on the probe. Note the
missingness figure **corrects §58 and §62**, which recorded 8.9% against 0.0%: measured properly
over real columns rather than the padded matrix, the SCM prior has missing cells too, so the
"missingness" candidate those findings named was overstated.

### What remains

The one structural difference not yet excluded is **how concentrated the label's dependence
is**. An SCM generates its target from a few specific parent nodes, with other columns merely
correlated downstream; the financial generator builds a distress score as a weighted sum over
many drivers. If any reasonable weighting of a diffuse driver set predicts nearly as well as
the true one, there is no pressure to identify any particular column — which would produce
exactly this: tasks that *demand* identity by the order-statistic measure (§64) while never
*rewarding* the effort of acquiring it.

That is measurable as the AUC lost by deleting the single best column, and it is the next test.

### Why this finding is worth its space

Two predictions registered in advance, both falsified, on the same afternoon. That is the
process working: §59, §60 and §61 each had to be retracted *after* being drafted as
conclusions, and the difference here is that the failure cost one 70-minute run instead of a
retraction. The elimination table is a genuine asset — five named, measured, closed
explanations — even though it does not yet contain the answer.

The honest status is: **we do not know why the financial prior will not teach
column-specific in-context inference**, and the space of cheap explanations is nearly exhausted.

## 66. Seven explanations eliminated, and the pattern is a discontinuity — so stop testing task statistics

**Date:** 2026-09-11. **MEASURED**, and the useful content is a closed list plus a redirection.

§65 left one structural candidate: how concentrated the label's dependence is. Two further
measurements killed it and killed its successor.

**Concentration of the label's dependence** — AUC lost by deleting the single best column,
stratified splits:

| prior | AUC lost | teaches |
| --- | --- | --- |
| financial | +0.0297 | 0.5352 |
| generic SCM | **+0.0099** | **0.9932** |
| trivial | +0.2419 | 0.9995 |

Backwards: the SCM prior's label is the *least* concentrated and it teaches best.

**How much in-context inference is worth** — a model trained across 40 tasks scored on held-out
tasks, against a model fitted per task:

| prior | global | per-task | gap | teaches |
| --- | --- | --- | --- | --- |
| financial, default | 0.4819 | 0.7200 | +0.2381 | 0.5352 |
| **financial, sharpened** | 0.4965 | 0.8571 | **+0.3607** | **0.5861** |
| generic SCM | 0.4485 | 0.9168 | +0.4683 | 0.9932 |
| trivial | 0.4983 | 0.9874 | +0.4891 | 0.9995 |

The first three rows look like a clean monotone story. The sharpened arm was measured
specifically to break it and does: its gap is nearer the SCM prior's than the default financial
prior's, and it teaches like the financial prior.

Worth keeping from this table anyway: **a global predictor is at chance (0.45-0.50) on every
prior**, so no prior here is solvable without conditioning on the context. The §47 failure mode
is gone from all of them.

### The closed list

| explanation | measured | verdict |
| --- | --- | --- |
| task width | 70 against 68 columns | eliminated (§58) |
| dilution by uninformative columns | 77.7% against 80.0% | eliminated (§58) |
| base rate | 0.021 raised to 0.401 | eliminated (§62) |
| column-identity demand | +0.1480 against +0.1425 | eliminated (§64) |
| learnability / signal-to-noise | raised to 0.8885, above SCM's 0.8306 | eliminated (§65) |
| concentration of label dependence | +0.0297 against +0.0099, wrong direction | **eliminated** |
| value of in-context inference | +0.3607 against +0.4683, sharp arm breaks it | **eliminated** |
| near-duplicate columns, effective rank, missingness | all small or wrong-signed | implausible (§65) |

### Why the next test should not be another task statistic

Three financial arms — default, balanced, sharpened — span base rates 0.021 to 0.401,
learnability 0.7276 to 0.8885, and in-context value 0.238 to 0.361. **All three teach between
0.535 and 0.586.** The generic SCM prior teaches 0.9932. That is a *discontinuity*, and seven
attempts to find a continuous task property that crosses it have failed.

What differs categorically is the **functional form of the generator**. Every financial task is
a monotone function of a linear combination of latent drivers, then pushed through accounting
identities; every SCM task is a freshly sampled computational graph with nonlinear activations,
discretisation and tree-structured rules. One family, versus a distribution over families.

### The experiment that isolates it

A **crossed design**, which no measurement so far has run: take the two generators' *features*
and their *label functions* and combine them.

| | financial label | SCM label |
| --- | --- | --- |
| **financial features** | current financial prior (teaches 0.535) | ? |
| **SCM features** | ? | current SCM prior (teaches 0.993) |

If the off-diagonal cells follow the **label function**, the financial generator's single
functional family is the cause and the fix is functional diversity over financial-shaped data.
If they follow the **features**, the accounting identities are the cause and the trade is
sharper, because those identities are what makes it a financial prior at all.

Either answer is actionable and the design is the cheapest way to get one, because it changes
one factor at a time across a boundary that seven continuous properties failed to cross.

**Status, stated plainly: unknown, and now well-bounded.** Seven named explanations are closed
with measurements behind each. That is worth more than an eighth guess.

## 67. The crossed design implicates the financial generator's features, not its label function

**Date:** 2026-09-13. **MEASURED**, one completed 6,000-step T4 run
(`fintfm-colid-crossed`, `--p-crossed 1.0`) against the four existing symmetry-probe arms.

§66 closed seven candidate task statistics and named the remaining variable: the generator's
functional form. The financial generator produces every task from one family — a monotone
function of a signed linear combination of accounting drivers — while the generic SCM
generator samples a fresh nonlinear computational graph per task. §66 proposed a crossed
design to separate whether teaching tracks the **label function** or the **features**.

`prior/crossed.py`'s `sample_scm_features_financial_label` keeps the SCM generator's feature
pool and replaces its label with the financial generator's own mechanism — signed random
weights over a driver subset, an optional two-way nonlinear interaction, log-uniform
sharpness, sigmoid, Bernoulli — with base rate held fixed by the identical calibration policy
used everywhere else (`_finish_binary`), so it cannot confound the result as it did not in §62.

| arm | antisymmetric probe | orientation | linear |
| --- | --- | --- | --- |
| financial, default envelope | 0.5352 | 0.6881 | 0.6503 |
| financial, balanced base rate | 0.5669 | 0.6983 | 0.6602 |
| financial, sharpened | 0.5861 | 0.6938 | 0.6555 |
| **SCM features x financial label (crossed)** | **0.5647** | 0.7127 | 0.6637 |
| SCM features x SCM label (on-diagonal) | **0.9932** | 0.9918 | 0.9897 |

**Swapping the financial generator's label mechanism onto SCM's own features collapses
teaching from 0.9932 to 0.5647 -- landing inside the financial arms' 0.535-0.586 range, not
anywhere near the SCM baseline.** The label function transplants cleanly onto a feature source
that otherwise teaches perfectly well, and the result tracks the financial side.

**This implicates the financial generator's features, not its label function.** The
accounting-identity structure, the driver correlations, or something else about how the
exposed columns are built is what suppresses column-specific in-context inference -- not the
fact that the label is a monotone function of a linear combination rather than a fresh
nonlinear graph.

### The pre-registered sanity check, and why the other cell was not run to completion

A quick gradient-boosting probe of both off-diagonal cells (100-1000 rows, not a pretraining
run) found `financial-feat x SCM-label` close to unlearnable at this scale -- raw AUC ~0.54-0.56
across two independent samples, below both on-diagonal baselines rather than between them.
That is a task the probe cannot cleanly read regardless of mechanism, since a near-chance task
teaches nothing about anything. `scm-feat x financial-label` instead sat between the two
on-diagonal baselines and closer to SCM's -- the informative cell, and the only one worth the
GPU spend. The full pretraining run above confirms what the cheap probe suggested.

### What this does not yet tell us

**Not which specific feature property.** §65 already eliminated near-duplicate columns,
overall correlation (financial is *less* correlated, wrong direction) and label-dependence
concentration (also wrong direction) as continuous statistics. This finding says the cause is
somewhere in the feature-generating process, not that any of the previously-measured
statistics is it — those remain eliminated. The candidate not yet isolated is the **accounting
identities themselves**: ratios built as quotients and sums of a small set of underlying
account balances, so many exposed columns are deterministic functions of few latent
quantities in a way no correlation coefficient on standardised values necessarily captures.

**Not yet a trade with a stated cost.** If the accounting identities are confirmed as the
cause, weakening them to test that hypothesis makes the generator progressively less a
financial prior — which is why `docs/DECISIONS.md` and this project's stated non-goals treat
that trade as one requiring an explicit argument, not a default fix.

### Status

Eight explanations now addressed: seven eliminated (§58, §62, §64, §65) and one confirmed
(this finding) as the side of the boundary the cause sits on, without yet naming the specific
mechanism. The next test is the one this finding could not avoid deferring: hold the exposed
feature *statistics* fixed while removing the accounting-identity dependency structure between
them, and see whether teaching returns.

## 68. The identity-weakening screen is inconclusive — additive noise is not a clean test of §67's hypothesis

**Date:** 2026-09-13. **MEASURED**, scratch-only screen (not committed as a prior variant),
five noise scales, 12-16 tasks each.

§67 implicated the financial generator's features and named the candidate: accounting-identity
structure between exposed ratio columns. The cheap test tried was adding independent Gaussian
noise to each exposed column, scaled to that column's own standard deviation, before scoring —
intended to break exact algebraic ties (e.g. `leverage + equity_ratio = 1`) while leaving each
column's individual relationship to the label diluted rather than destroyed.

| perturb scale | raw AUC | sorted AUC | identity advantage |
| --- | --- | --- | --- |
| 0.0 (baseline) | 0.7508 | 0.6577 | +0.0931 |
| 0.3 | 0.6750 | 0.5523 | +0.1227 |
| 0.6 | 0.6775 | 0.5445 | +0.1331 |
| 1.0 | 0.6158 | 0.5368 | +0.0791 |
| 2.0 | 0.5710 | 0.5200 | +0.0510 |

**Hump-shaped, not monotone.** Read naively at three points this looked like the predicted
dose-response; the fourth and fifth points show both raw and sorted AUC converging toward
chance (0.5) as noise grows, and "identity advantage" is the gap between two quantities
collapsing to the same floor — which produces exactly this shape whether or not accounting
identities are the true cause. **This screen cannot distinguish "breaking identities helps
column-awareness" from "adding enough noise degrades everything, briefly favouring whichever
representation degrades slower."**

### Why the design was wrong, stated plainly

Additive per-column noise degrades **both** the cross-column algebraic structure **and** each
column's own marginal usefulness simultaneously, with no way to separate the two effects from
the output alone. A clean test needs to hold each column's *individual* relationship to the
label fixed while varying *only* the cross-column dependency structure — which post-hoc
additive noise cannot do, because the noise itself is what erodes the individual signal.

### What a valid design requires

Perturb the **shared latent accounts** before deriving ratios, not the derived ratios
themselves. Concretely: give each ratio's numerator and denominator independent copies of the
underlying account (e.g. two decorrelated draws of `total_assets` for two different ratios that
currently share the literal same array), calibrated to preserve each account's own marginal
distribution and its own contribution to the label-generating drivers, while breaking only the
literal shared-array identity between ratios. This requires modifying `_accounts` and
`_ratio_family` directly rather than perturbing the assembled `Task.X`, and is accordingly a
larger change than the screen attempted.

### Status

§67 stands: the crossed design cleanly implicated features over the label function. This
finding narrows what a valid follow-up test must look like and rules out the cheap version of
it, rather than ruling out the hypothesis itself. Task 38.11 is revised to specify the
account-level perturbation rather than the column-level one.

## 69. Five folds, properly tuned: fintfm ties logistic regression and loses clearly to all three boosters

**Date:** 2026-09-13. **MEASURED**, V4FinBench published protocol, all 5 folds, `--tune`
(baselines grid-searched on the validation fold per their Table 5). Closes task 38.8, which
every quoted number since §60 was waiting on.

Pooled across all 105,900 rows and 402 positives (prevalence 0.380%, identical to §60's fold),
with a paired bootstrap over 2,000 resamples, Holm-adjusted across the family of four:

| arm | AP | fintfm vs it: dAP | 95% CI | Holm p | verdict |
| --- | --- | --- | --- | --- | --- |
| catboost | 0.3425 | -0.2059 | [-0.2455, -0.1697] | <0.001 | **DIFFERENT** |
| xgboost | 0.3406 | -0.2041 | [-0.2458, -0.1647] | <0.001 | **DIFFERENT** |
| lightgbm | 0.3076 | -0.1711 | [-0.2116, -0.1355] | <0.001 | **DIFFERENT** |
| logistic_regression | 0.1493 | -0.0127 | [-0.0363, +0.0120] | 0.309 | indistinguishable |
| **fintfm** | **0.1365** | -- | -- | -- | -- |

**The tie with logistic regression survives properly tuned baselines and five folds.** §60
found the same tie on one untuned fold; pooling five tuned folds narrows the interval and it
still crosses zero.

**What changed is the boosters.** Untuned (§60), only CatBoost's lead was significant --
LightGBM and XGBoost were statistically indistinguishable from fintfm on one fold. Tuned, all
three separate from fintfm with Holm p < 0.001, and CatBoost's gap **roughly doubled**, from
-0.1457 untuned to -0.2059 tuned. §60's caveat -- "the gap to CatBoost and XGBoost is, if
anything, understated" by using untuned baselines -- is now a measured fact rather than a
caveat.

### Why tuning mattered for the boosters and not for us

Grid search finds each booster's operating point on *this exact panel*. fintfm has no
comparable step: it is scored zero-shot, with a fixed context strategy (`uniform`),
`max_context=2000`, and `n_ensemble=1` -- none of the machinery already in this codebase for
improving in-context scoring (retrieval-based context, §32's measured +0.066 to +0.095 AUC
gain; ensembling over context draws and column-identity seeds, D12's stated mitigation for
exactly the invariance §54 traded away) was engaged for this comparison. Tuning the baselines
and not tuning our own inference configuration is not a fair fight, and it is the most likely
reason for the doubled gap.

### What this settles and what it does not

**Settles:** the honest ranking on V4FinBench horizon 0 is a two-way tie for last (fintfm,
logistic regression) against a tuned three-way cluster (LightGBM, XGBoost, CatBoost) roughly
2-2.5x higher on average precision. This is now a five-fold, tuned, paired-bootstrap result --
the first number in this run of findings that clears every caveat stated against it.

**Does not settle:** whether fintfm's own inference configuration is doing it justice. Every
context-quality lever this codebase has built and measured -- retrieval, ensembling, feature
transform choice -- was left at its default for this comparison. The next experiment is not
another architecture or prior change; it is re-scoring this exact fold with those levers turned
on, before concluding the gap to the boosters is architectural rather than configurational.

## 70. Retrieval sabotages fintfm on V4FinBench; ensembling and wider context each help, and stack to the best result yet

**Date:** 2026-09-13. **MEASURED**, one fold (identical to §60/§69's fold 0: 63,588 train,
20,917 test, 79 positives), six variants of an isolated fintfm-only comparison (no baseline
fits, so the effect is attributable to the inference configuration alone), reproduced twice
independently with identical results to four decimals.

§69 scored fintfm at the classifier's library defaults — `context_strategy="uniform"`,
`n_ensemble=1`, `max_context=2000` — and lost to all three tuned boosters. None of this
project's own measured context-quality levers were engaged. Testing each in isolation and
combination on the identical fold:

| variant | AP | delta vs baseline | wall time |
| --- | --- | --- | --- |
| baseline (uniform, n_ens=1, ctx=2000) | 0.1853 | -- | 19s |
| retrieval only | **0.0527** | **-0.133** | 68s |
| n_ensemble=8 only | 0.2107 | +0.025 | 162s |
| max_context=4000 only | 0.1982 | +0.013 | 35s |
| retrieval + ctx=4000 | 0.1184 | -0.067 | 152s |
| retrieval + n_ensemble=8 + ctx=4000 | 0.1225 | -0.063 | 1092s |
| **n_ensemble=8 + ctx=4000, no retrieval** | **0.2116** | **+0.026** | 254s |

**Retrieval alone drops AP by 0.133 — the opposite direction from `docs/FINDINGS.md` §32's
+0.066 to +0.095 AUC gain on the same protocol.** Widening the context partially offsets the
damage (0.0527 to 0.1184) but no combination that includes retrieval recovers the baseline,
let alone improves on it. **Dropping retrieval and keeping the other two levers reaches 0.2116
-- the best AP measured for this checkpoint on this benchmark, and combining almost exactly
additively** (0.1853 + 0.025 + 0.013 = 0.223 predicted, 0.2116 measured).

### Why retrieval likely hurts here and did not in §32

Not yet isolated, but the two measurements differ on the axis most likely to matter:
prevalence. §32 measured retrieval's gain on the term-structure/hazard evaluation at a
different, unrecorded base rate; this fold is 0.380%, with 79 positives spread across a
63,588-row pool. Retrieval groups queries and retrieves per-group (`retrieval_groups=64`,
`retrieval_min_positive=8` by default) — at this few positives, a retrieval group may draw a
near-degenerate context, or the per-group logit correction may behave differently than at a
denser base rate. This is a mechanism to chase, not yet a finding.

### What is safe to conclude now

**§69's comparison against the boosters used an inference configuration this project's own
measurements say is actively harmful.** The honest re-run uses `context_strategy="uniform"`
(not the codebase default's mention of retrieval as an improvement — that claim does not
transfer to this regime), `n_ensemble=8`, `max_context=4000`. `configs/v4finbench-full-
inference.yaml` is updated accordingly, with the retrieval reversal documented in its own
header so a future reader does not repeat the same untested assumption.

### What is not yet known

Whether 0.2116 closes a meaningful fraction of the gap to the tuned boosters (0.31-0.34 AP,
§69) once measured properly across all five folds with a paired bootstrap. One fold and one
seed is exactly the standard this project has repeatedly found insufficient (§60, §65); the
next step is the five-fold run, not a conclusion from this table.

## 71. The lever fix is real: +0.031 AP, significant, and still short of the boosters

**Date:** 2026-09-13. **MEASURED**, all five V4FinBench folds, fintfm re-scored with §70's
corrected configuration (`context_strategy="uniform"`, `n_ensemble=8`, `max_context=4000`)
against the identical rows and the same tuned baseline predictions saved from §69 (row
equality asserted per fold before pooling, so this is a same-data comparison).

§70's single-fold result (0.1853 to 0.2116) was optimistic. Per fold, the corrected config
won on 3 of 5 and lost narrowly on 2:

| fold | S70 config | S69 config |
| --- | --- | --- |
| 0 | 0.2116 | 0.1853 |
| 1 | 0.2020 | 0.2083 |
| 2 | 0.1184 | 0.1208 |
| 3 | 0.1930 | 0.1794 |
| 4 | 0.1527 | 0.1450 |

Pooled across all five (n=105,900, 402 positives), paired bootstrap, Holm-adjusted:

| comparison | dAP | 95% CI | Holm p | verdict |
| --- | --- | --- | --- | --- |
| S70 config vs S69 config (fintfm, same checkpoint) | **+0.0311** | [+0.0142, +0.0501] | <0.001 | **DIFFERENT** |
| S70 config vs logistic_regression | +0.0183 | [-0.0105, +0.0488] | 0.204 | indistinguishable |
| S70 config vs lightgbm | -0.1400 | [-0.1812, -0.1007] | <0.001 | **DIFFERENT** |
| S70 config vs catboost | -0.1749 | [-0.2120, -0.1378] | <0.001 | **DIFFERENT** |
| S70 config vs xgboost | -0.1730 | [-0.2134, -0.1329] | <0.001 | **DIFFERENT** |

**The fold-to-fold variance was real, and the pooled effect is real too.** Dropping retrieval
and turning on `n_ensemble=8` plus a wider `max_context` improves fintfm's AP from 0.1365 to
0.1676, and this is the first configuration-only change all session to clear a five-fold
paired bootstrap at conventional significance.

**It changes the standing versus logistic regression** from a tie (§69: dAP -0.0127, p=0.309)
to a numerical lead that does not yet clear significance (dAP +0.0183, p=0.204). Worth another
data point before calling it resolved either way.

**It does not touch the booster gap.** -0.14 to -0.17 AP against all three, all p<0.001. §69's
central finding is unchanged: on a properly tuned comparison, fintfm is clearly behind
LightGBM, CatBoost and XGBoost on this benchmark.

### Status of task 38.12

Closed as "partial win, quantified." The gap was never purely architectural — a meaningful
slice of it (§69's untested-defaults gap) was inference configuration, and that slice is now
measured at +0.031 AP. What remains after fixing it is +0.14 to +0.17 AP against the boosters,
which is the honest size of whatever gap is left to close by other means (fine-tuning,
architecture, or prior work).

### Open thread

Why retrieval hurts here and helped in §32 is still unexplained (§70). Worth chasing if
retrieval's mechanism can be fixed rather than disabled, since disabling it forgoes whatever
its intended benefit was.

## 72. Identity-shuffle moves the antisymmetric probe, but induces a reproducible sub-chance inversion on symmetric ones

**Date:** 2026-09-14. **MEASURED**, one completed 6,000-step T4 run (`fintfm-colid-idshuffle`,
`--identity-shuffle`), symmetry probes at 16 seeds each (up from the usual 8, specifically to
verify this result before reporting it).

§68's revised design — permute each account independently so cross-account identities break
exactly while every account's own marginal distribution and the label (computed from the true,
unpermuted accounts) are completely unaffected — was verified correct before spending the GPU
run: the identity `equity = assets - liabilities` holds to machine precision in the true
accounts and is violated at the scale of the accounts themselves after permutation; every
account's sorted values are unchanged; `y` is byte-identical between `identity_shuffle=True`
and `False` at the same seed; `identity_shuffle=False` reproduces every existing checkpoint's
behaviour exactly (regression-tested).

### The result is not a clean confirmation

| probe | financial arms (default/balanced/sharp) | **identity-shuffle** | generic SCM |
| --- | --- | --- | --- |
| antisymmetric | 0.535 - 0.586 | **0.7488** (sd 0.025, n=16 seeds) | 0.9932 |
| symmetric_sum | 0.986 - 0.992 | **0.2988** (sd 0.027, n=16) | 0.9964 |
| symmetric_count | 0.920 - 0.930 | **0.2597** (sd 0.026, n=16) | 0.9336 |

**Antisymmetric moved further than any other single manipulation tried against the pure
financial prior** — 0.749 against the next-best (sharpened) prior's 0.586. That is consistent
with §67's implication of the features, and stronger evidence for the accounting-identity
hypothesis specifically than anything measured so far.

**But `symmetric_sum` and `symmetric_count` collapsed to 0.30 and 0.26 — below the 0.5 chance
floor, and every other checkpoint in this project scores at or above 0.92 on these two probes,
without exception.** An AUC below 0.5 means the ranking is *systematically inverted*: the model
is confidently wrong, not merely uninformative. Verified stable across 16 independent seeds
(sd ~0.026), so this is not a fluke of one measurement.

**And the checkpoint is worse than every other financial-only arm on its own in-distribution
metric**, not only on the exotic probes:

| arm | per-task AUC (financial prior, in-distribution) |
| --- | --- |
| default (colid, no shuffle) | 0.6693 |
| balanced (base rate) | 0.6433 |
| sharpened (learnability) | 0.6601 |
| **identity-shuffle** | **0.5772** |

Identity-shuffle is the worst of four financial-only checkpoints on the metric that matters
most, and it developed a severe, reproducible inversion the others do not have.

### What this does and does not support

**Some support for the accounting-identity hypothesis**, because the antisymmetric probe moved
further than any prior manipulation. **Not clean confirmation**, because the same intervention
produced a serious new failure mode rather than a strict improvement, and made the model worse
where it is actually deployed. Identity-shuffle as implemented is not a candidate fix; it is
evidence the mechanism is real but that severing the accounting identities has side effects on
what the model learns that are not yet understood.

### A candidate mechanism for the inversion, untested

Breaking cross-account identities can produce far more extreme derived ratios than the true
accounts ever would (e.g. `debt_to_ebitda` when `debt` and `ebitda` are independently
permuted, rather than tied through the same firm's true financials). If the model learned
"extreme values across many columns" as a spurious cue for this prior's label — plausible,
since real defaulting firms in the *true* accounts tend toward extreme ratios, and
identity-shuffle may have decoupled "extreme" from "defaulting" while leaving the association
partially learned — that heuristic would misfire in the opposite direction on a probe where
the positive class is defined by a large sum or count of positive values, which is exactly
where the inversion appears. Untested; the next step if this thread continues.

### Status

Task 38.11 is not closed by this result. It substantially narrows the mechanism (features, and
now specifically something about the identity structure moves the needle) while opening a new,
unexplained problem that must be understood before any identity-breaking variant is a candidate
for anything beyond diagnosis.

## 73. Generalization check: the calibration edge replicates everywhere, the discrimination gap varies wildly and does not track base rate

**Date:** 2026-09-14. **MEASURED**, `evaluation/bench.py`'s existing `run_credit` harness (Polish
bankruptcy at 3 horizons, Taiwan bankruptcy — two independent economies, already wired into
this repo), single 70/30 split per dataset (seed 0), `colid-fin07` with §71's lever fix
(`n_ensemble=8`, no retrieval). `CreditMetrics` now reports average precision alongside ROC-AUC
(D13), added as part of this measurement since the harness predated that decision.

| dataset | base rate | fintfm AP | best booster AP | ratio | fintfm vs LR |
| --- | --- | --- | --- | --- | --- |
| V4FinBench h0 (§71, 5-fold) | 0.38% | 0.1676 | 0.3425 (catboost) | 2.0x | ahead, not significant |
| Polish 1y | 3.86% | 0.1185 | **0.8588** (catboost) | **7.2x** | ahead |
| Polish 3y | 4.71% | 0.1071 | 0.6651 (catboost) | 6.2x | ~tied |
| Polish 5y | 6.94% | 0.2763 | 0.7159 (xgboost) | 2.6x | ~tied |
| Taiwan | 3.23% | 0.2569 | **0.4444** (catboost) | **1.7x** | slightly behind |

### The gap does not track base rate

Polish 1y (3.86%) and Taiwan (3.23%) are at nearly identical base rates and have the *most*
and *least* favourable gaps measured, respectively. Whatever drives the difference is specific
to each dataset's structure, not its class balance — consistent with §63/§67's reading that
domain match, not density match, is the thing that carries transfer.

### The calibration edge is the one thing that held up everywhere

| dataset | fintfm ECE | best baseline ECE |
| --- | --- | --- |
| Polish 1y | **0.0016** | 0.0090 (gboost) |
| Polish 3y | **0.0018** | 0.0088 (gboost) |
| Polish 5y | 0.0102 | 0.0082 (random_forest) |
| Taiwan | **0.0053** | 0.0064 (random_forest) |

Best or near-best on all four independent panels. This is the first time this project's
calibration thesis (§12, measured 2.3x-11.7x better than boosting) has been tested outside
V4FinBench, and it replicated without exception. §60's finding that the discrimination deficit
is real and the calibration edge is real *simultaneously* now has cross-dataset support rather
than resting on one benchmark.

### Caveats

**Single split, not five folds.** Every number here is one 70/30 split at one seed — directional,
not the paired-bootstrap standard §69/§71 met on V4FinBench. Sampling error at these dataset
sizes (5,910-10,503 rows, 66-190 positives in the test fold) is comparable to what turned "third
of five" into "one significant difference of four" in §60.

**Untuned baselines**, same caveat as §60/§69: the gap to boosters, if anything, understates
what tuning would show.

**One checkpoint.** `colid-fin07` was selected for V4FinBench; whether a different arm (pure
SCM, or the balanced/sharpened financial variants) generalises differently to these panels is
untested and is the natural next question, since §61/§63 predict the ordering should depend on
domain match rather than being fixed across datasets.

## 74. The Bayes-ceiling test resolves it: pure financial training induces a severe capacity-like cap the architecture does not have

**Date:** 2026-09-14. **MEASURED**, three existing checkpoints, a task with an *exactly known*
Bayes-optimal AUC rather than an estimated one -- proposed externally (relayed by the user) as
the decisive test of whether §53's "capped predictor" finding is an architecture/capacity
bottleneck or a prior-content effect.

### The construction

`X in R^6`; class 0 drawn `N(0, I)`, class 1 drawn `N(mu * e_1, I)` -- a mean shift in one
informative dimension, five pure-noise dimensions, no cross-column structure whatsoever. For a
1-D equal-variance mean shift, the Bayes-optimal classifier thresholds the informative
dimension and its AUC has the closed form `Phi(mu / sqrt(2))`, verified numerically before use
(empirical AUC of the true statistic matched the formula to 3 decimals at every target tested).
Solving for `mu` given a target lets the task's true difficulty be dialled exactly, from chance
to near-certainty, with nothing left to estimate.

### The result

| Bayes AUC (exact) | fin07 (70% financial) | **fin00 (pure SCM)** | **fin10 (pure financial)** |
| --- | --- | --- | --- |
| 0.500 | 0.499 | 0.503 | 0.506 |
| 0.700 | 0.669 | 0.692 | 0.584 |
| 0.900 | 0.849 | 0.891 | 0.666 |
| 0.990 | 0.953 | **0.984** | **0.713** |
| 0.999 | 0.978 | **0.997** | **0.728** |

10 seeds x 8 column-identity draws per point; sd ranged 0.002-0.032, tightest near the extremes.

**Pure SCM tracks the true Bayes curve almost exactly across the whole range** -- 0.997 achieved
against 0.999 true, no meaningful gap anywhere measured.

**Pure financial hits a hard ceiling around 0.73 regardless of true difficulty.** At Bayes AUC
0.999 -- a task an optimal classifier separates almost perfectly -- it still achieves only
0.728. This is not a column-identity failure: the task has one informative dimension and no
identity structure to resolve, ruling out §54-§67's mechanism as the cause here. It is a more
basic failure to use strong, cleanly-available signal at all.

### What this settles

**Architecture capacity is ruled out as the cause.** Identical architecture, identical
capacity, and one training-prior choice produces near-perfect tracking while the other produces
a severe, reproducible cap. A capacity bottleneck would show up regardless of what the model
was trained on; this does not.

**Training on the financial prior, especially as the dominant source, induces the cap.**
`fin07` (70% financial, 30% SCM) sits between the two extremes and degrades gracefully with
financial share rather than falling off a cliff -- a third independent dose-response in this
direction, joining §58's antisymmetric-probe result and §72's identity-shuffle result. The
mechanism is still not identified (this finding does not explain *why* the financial prior
teaches under-extraction of strong signal), but the *locus* is now established beyond doubt:
it is something about what the financial generator teaches, not a limit of what the encoder can
represent.

### Why this probe is more damning than every prior one

§54-§67 all probed column-specific reasoning specifically -- tasks that require identifying
which column matters. This probe requires none of that: the signal sits in a single named
dimension with five inert companions. A model that has merely failed to learn column identity
should still find this signal, since no column-identity reasoning is needed to read dimension
0. That it does not, and caps at almost exactly the same ~0.73 region §51 and §53 measured
across a whole family of *different* probes days ago, suggests the financial prior teaches a
general under-confidence or under-extraction strategy that is not specific to any one probe
family.

### Connection to §73

Read alongside this finding, the Polish/Taiwan reversal (pure SCM beating 70%-financial on all
four external panels, opposite of V4FinBench) stops looking like a domain-match curiosity and
starts looking like a symptom of the same underlying defect: the financial prior's only
demonstrated benefit anywhere is the +0.098 AP on V4FinBench specifically (§63), and training
on it measurably damages the model's basic signal-extraction capability everywhere else,
including on tasks with no relationship to finance, columns, or identity at all.

### What this does not yet establish

*Why* the financial prior induces this. Candidates worth testing, none yet measured: whether
the effective SNR of financial tasks (even under §42's widened sharpness range) is lower in
practice than the sharpness parameter alone suggests, once accounting-identity correlations and
missingness are accounted for; whether the objective (cross-entropy under heavy class
imbalance) interacts badly with the prior's base-rate distribution to teach systematic
under-confidence; or something in the missingness/MNAR mechanism training the model to hedge.
This is now the highest-priority open question the whole "what should the production prior look
like" thread reduces to.

## 75. The financial prior's benefit does not generalize past V4FinBench: pure SCM wins on Polish and Taiwan

**Date:** 2026-09-14. **MEASURED**, same harness as §73 (`run_credit`, single 70/30 split per
dataset), `colid-fin00` (pure SCM) against `colid-fin07` (70% financial, §73's checkpoint).

| dataset | fin00 (pure SCM) AP | fin07 (70% financial) AP | winner |
| --- | --- | --- | --- |
| Polish 1y | **0.1363** | 0.1185 | SCM |
| Polish 3y | **0.1191** | 0.1071 | SCM |
| Polish 5y | **0.3021** | 0.2763 | SCM |
| Taiwan | **0.3223** | 0.2569 | SCM, by +0.065 |
| *V4FinBench h0 (§63)* | *0.0918* | *0.1900* | *financial, +0.098, Holm p=0.002* |

**Pure SCM beats the financial-content checkpoint on all four external panels** -- the exact
opposite ordering from V4FinBench, where §63 measured the financial prior's advantage as
significant and controlled for base rate. AUC shows the same pattern: fin00 reaches 0.9220 on
Taiwan against fin07's 0.8858.

### The reading this forces

§63 was correct about what it measured -- the financial prior's structure genuinely helps on
V4FinBench, and that comparison was a controlled two-factor design, not a fluke. What this
finding adds is that the benefit **does not generalize to "credit risk" as a domain**. Polish
and Taiwan are both real corporate bankruptcy panels, built from different accounting
conventions and a different feature-engineering process than V4FinBench's. The financial
prior's advantage evidently comes from matching something specific to how V4FinBench's 130
features were constructed -- plausibly the numerator/denominator ratio pairs `_ratio_family`
samples resembling V4FinBench's own engineered ratios by construction -- rather than from
encoding anything general about corporate default.

**§63's title should be read narrowly from here on**: "the financial prior helps on
V4FinBench" is established; "the financial prior helps on credit risk" is now contradicted by
two independent panels.

### Consistent with §74

This is the same story §74 tells from the architecture side. Training on the financial prior
caps basic signal extraction (§74) and its one measured real-data benefit does not travel past
the one benchmark it was measured on (this finding). Both point the same direction: the
current financial generator, particularly at high or exclusive share, is a narrow fit to one
dataset's engineering choices rather than a genuine domain prior, and the SCM prior is the
stronger default until the mechanism behind §74's cap is understood and fixed.

### Caveat

Single split per dataset, as in §73 -- directional, not the five-fold standard.

## 76. Four content axes bisected, none fixes §74's cap: the bottleneck does not track any prior-content variable tested

**Date:** 2026-09-14. **MEASURED**, six existing checkpoints, zero new GPU spend, the exact
§74 Bayes-ceiling probe.

§74 found pure financial training caps basic signal extraction at ~0.73 AUC regardless of true
difficulty, while pure SCM tracks the true curve almost exactly. This finding bisects the
candidate causes using checkpoints already trained this session, each isolating one axis
against the two established endpoints.

| variant | isolates | 0.900 | 0.990 | 0.999 |
| --- | --- | --- | --- | --- |
| fin10 (financial, default) | baseline, capped | 0.666 | 0.713 | 0.728 |
| finbal (financial, balanced ~40% rate) | base rate | 0.665 | 0.714 | 0.730 |
| finsharp (financial, raised raw SNR) | signal-to-noise | 0.667 | 0.716 | 0.733 |
| crossed (clean SCM features, financial label) | features vs label mechanism | 0.680 | 0.736 | 0.755 |
| idshuffle (financial, broken accounting identities) | identity structure | 0.592 | 0.611 | 0.616 |
| **fin00 (pure SCM)** | -- | **0.891** | **0.984** | **0.997** |

**None of the four content-axis interventions comes close to closing the gap.** Balancing the
base rate: no measurable effect. Raising raw signal-to-noise (§65's sharpening, which itself
raised the financial prior's own learnability above the SCM prior's): no measurable effect.
Swapping to clean, uncorrelated SCM features while keeping the financial label mechanism:
a small improvement (0.728 to 0.755 at the top), nowhere near closing the 0.997 gap. Breaking
accounting identities: **makes it worse** (0.616), consistent with §72's finding that
identity-shuffle training damages general capability alongside its narrow antisymmetric gain.

**What is common to every capped variant and absent from the one that is not**: the financial
label's functional form. `fin10`, `finbal`, `finsharp` and `crossed` all construct their label
via the same mechanism -- a signed linear combination of a driver subset, an optional single
two-way interaction, then a sigmoid -- while `fin00`'s SCM label is read from an arbitrary node
of a random 1-4-layer computational graph with five candidate nonlinearities and sparse random
connectivity. This is the one variable that was never isolated in this bisection (it would
require training a new checkpoint with the label functional form varied while everything else
is held fixed), and it is the natural next test if prior-content work continues.

### Why this result changes the priority

An externally-proposed review (relayed by the user, 2026-09-14) argued that this project should
prioritize architecture -- specifically completing the two-way cell attention D12 already named
as the unresolved answer, and ensuring labels participate throughout context encoding, not only
after pooling -- ahead of further prior-content experiments. This finding is direct evidence for
that read: four different, real, measured content interventions collectively moved the ceiling
by at most 0.03 AUC (and one moved it backward), while the only thing that has ever closed the
gap is discarding the financial generator's structure entirely for the SCM prior's. That pattern
is more consistent with a representational limitation specific to what the current architecture
can extract from financial-shaped tasks than with a content property of the prior that the next
experiment might happen to fix.

**Status:** task 38.15 is not closed. The label-functional-form candidate remains untested, and
architecture (two-way cell attention, `openspec/changes/cell-attention-and-task-inference`) is
now the priority track per the user's explicit direction, with the label-form test as the
cheaper prior-side alternative if architecture does not resolve it.

## 77. Even sharpened, the financial prior almost never reaches near-deterministic realized difficulty

**Date:** 2026-09-14. **MEASURED**, ~45-57 sampled tasks per prior, gradient boosting on a
clean in-task split, run in parallel with §76's bisection.

§76 found sharpening the financial prior (`finsharp`) barely moved the Bayes-ceiling cap
(0.666 to 0.667 at target 0.900). This finding gives the mechanism: sharpening raises the
*median* realized difficulty substantially (0.702 to 0.906) without ever reaching the extreme
tail.

| prior | median | tasks >0.90 | tasks >0.95 | **tasks >0.99** | n |
| --- | --- | --- | --- | --- | --- |
| financial, default | 0.702 | 22.2% | 8.9% | **0.0%** | 45 |
| financial, sharpened | 0.906 | 53.2% | 21.3% | **0.0%** | 47 |
| generic SCM | 0.954 | 70.2% | 52.6% | **17.5%** | 57 |

**Zero of 92 sampled financial tasks, at either sharpness setting, reached realized AUC above
0.99. 17.5% of SCM tasks did.** The §74 Bayes-ceiling probe specifically tests targets of 0.99
and 0.999 -- exactly the region the financial label's construction structurally does not visit
during training, regardless of the nominal sharpness parameter.

### Why this is structural, not a tuning miss

Financial's label is `drivers @ w` (a bounded-magnitude linear combination over ~9 driver
dimensions, §47) plus an optional single two-way interaction, then a sigmoid. Averaging several
bounded terms is exactly the setting where extreme realizations become rare by construction --
a central-limit-like smoothing that no amount of rescaling (which is what "sharpness" does: a
uniform multiplier on the whole score) can undo, because it scales the *typical* case and the
*extreme* case together. SCM's label instead reads one node off a random 1-4-layer graph with
five candidate nonlinearities; a narrow, monotonic path through such a graph can concentrate
variance far more sharply than a bounded linear sum ever can.

### Relationship to §76

This does not contradict §76's bisection -- it explains one leg of it. `finsharp` raising raw
learnability (median realized AUC) without closing the Bayes-ceiling cap is exactly what this
distribution predicts: more of the mass moves into the 0.80-0.95 band, essentially none reaches
the 0.99+ band the probe specifically tests. The other three bisected variants (`finbal`,
`crossed`, `idshuffle`) are not directly explained by this finding, since none of them touch
the label's linear-combination structure -- `crossed` keeps the identical label mechanism on
different features and is capped for the same structural reason this finding identifies.

### What this adds to `mechanism-diverse-prior`

Concrete evidence for that proposal's premise, sharper than the qualitative "financial's label
is always the same shallow shape" argument: **the shape matters because it caps how extreme a
realized task can be, independent of how the sharpness parameter is set.** A financial-style
prior with genuinely near-deterministic tasks would need either an unbounded-magnitude driver
weighting (changing what the weights represent) or a qualitatively different label mechanism,
not merely a wider sharpness range on the current one.

## 78. Two-way cell attention closes §74's capacity cap: regret 0.27 to 0.003, on the exact prior that produced it

**Date:** 2026-09-14. **MEASURED**, two completed 6,000-step T4 runs
(`fintfm-cellattn-fin10`, `p_financial=1.0`; `fintfm-cellattn-fin00`, `p_financial=0.0`),
`n_cell_blocks=1` with `cell_labels=True` (task 39.1/39.2), otherwise matching §74's
`fin10`/`fin00` configuration. This is `cell-attention-and-task-inference` task 39.4 — the
experiment the whole architecture-first thread (§74, §76, §77, and an externally-relayed
review converging with D12) was built to run.

### The result

| target Bayes AUC | financial, OLD architecture (§74) | **financial, two-way cell attention** | SCM, OLD (§74) | **SCM, two-way cell attention** |
| --- | --- | --- | --- | --- |
| 0.900 | 0.666 (regret 0.234) | **0.897 (regret 0.003)** | 0.891 (regret 0.009) | 0.899 (regret 0.001) |
| 0.990 | 0.713 (regret 0.277) | **0.986 (regret 0.004)** | 0.984 (regret 0.006) | 0.988 (regret 0.002) |
| 0.999 | 0.728 (regret 0.271) | **0.998 (regret 0.001)** | 0.997 (regret 0.002) | 0.999 (regret 0.0005) |

**A checkpoint trained exclusively on the financial prior -- the exact prior that produced a
hard ~0.73 ceiling regardless of true task difficulty -- now tracks the true Bayes-optimal
curve almost perfectly.** Regret in the 0.90-0.999 range falls from 0.234-0.277 to 0.001-0.005,
roughly a 60-90x reduction. The SCM arm, already near-optimal under the old architecture, shows
**no regression** and if anything a small further improvement at every target.

### Confirmed by two further, independent instruments before trusting one metric

Per this project's own standing discipline (§50, §52's retracted readings), a single metric
moving is not sufficient. The symmetry probes (§54, §56), run on both new checkpoints:

| arm | antisymmetric | symmetric_sum | symmetric_count | orientation | linear |
| --- | --- | --- | --- | --- | --- |
| financial, OLD architecture | 0.5352 | 0.9921 | 0.9236 | 0.6881 | 0.6503 |
| **financial, cell attention** | **0.9890** | 0.9849 | 0.8745 | **0.9961** | **0.9757** |
| SCM, OLD architecture | 0.9932 | 0.9964 | 0.9336 | 0.9918 | 0.9897 |
| SCM, cell attention | 0.9962 | 0.9775 | 0.8872 | 1.0000 | 0.9739 |

**The financial-only checkpoint's antisymmetric score -- 0.5352 to 0.9890 -- essentially closes
to the SCM baseline (0.9932), on a probe with a provable 0.5 ceiling for any architecture that
cannot represent column identity.** Nothing tried against the financial prior's *content*
across §58-§77 came close: the best any content intervention achieved was the 70/30 mixture's
0.9116 (§58), and every pure-financial content variant (default, balanced, sharpened, clean
features, broken identities) stayed at 0.53-0.59. The architecture change does in one run what
seven content interventions could not.

**One honest exception**: `symmetric_count` is lower on both new checkpoints (0.87-0.89) than
their old-architecture counterparts (0.92-0.93). Both new arms land in a narrow band regardless
of prior, which suggests a property of the architecture change generally rather than noise, but
this is one data point and not yet understood. Recorded rather than glossed over.

### What this settles

**§74's cap was architectural, not a property of the financial prior's content.** Every
content-side hypothesis this project spent §58-§77 testing -- base rate, signal-to-noise,
feature cleanliness, accounting-identity structure, label functional form -- was chasing an
effect whose actual cause was that the row encoder had no way to develop column semantics from
the context, only column *identity* (D12, §54). Two-way cell attention gives it exactly that:
a cell can now attend to the rest of its own column across every context row, not only to the
other features of its own row.

### What this does not yet settle

**No real-data claim.** Every number in this finding is synthetic. `cell-attention-and-task-
inference` task 39.5 -- re-running V4FinBench and the credit panels before any claim that this
is a strict improvement -- is next, and it is not optional: §69-§77 all warn that a synthetic
result and a real-benchmark result have dissociated before in this project.

**A deliberate configuration deviation.** Both runs used `--batch-size 4 --n-rows-choices
256,512` rather than fin10/fin00's `--batch-size 8 --n-rows-choices 256,512,1024`, forced by a
CUDA OOM the first launch attempt hit (`_eval_quality`'s held-out evaluation batch size was
hardcoded to 16, independent of the training batch size, and row-attention-within-feature's
`(B*F, heads, N, N)` attention score matrix made that difference decisive -- 34 GB attempted
on a 14.74 GB card). The eval-batch-size bug is fixed for every future run; the training
`n_rows` cap of 512 for *this* pair of runs is a real, documented difference from §74's
baseline and should be closed before the comparison is called final.

### Consequence for the rest of this project's architecture-first thread

Per `cell-attention-and-task-inference` task 39.6: since 39.4 *did* close the gap, task 39.5 is
now the immediate next step (re-run real-data protocols) and `mechanism-diverse-prior`'s
premise (task 40.1) is substantially weakened -- the cap this project spent three days chasing
through prior-content experiments was not a prior-coverage gap. The label-functional-form
candidate §77 developed remains scientifically interesting on its own terms, but it is no
longer the leading explanation for §74's finding.
