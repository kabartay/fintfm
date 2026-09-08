# Findings

Numbered, dated, with the source or command that produced each. A finding that lives only in
a conversation is lost when the conversation compacts.

## Where we stand — the honest scorecard

Updated 2026-09-08 after nineteen findings in one day. **Read this before quoting any number
below**, because several findings temper or amend earlier ones and the amendments matter more
than the originals.

### What is measured and large

| claim | evidence | § |
| --- | --- | --- |
| **PD term structures are incoherent** — 39% of firms get a curve where a longer horizon carries *lower* default probability, while the portfolio aggregate looks monotone and hides it | measured, real data | §11 |
| **Pretraining buys calibration** — 7× on Brier, up to 130× on ECE against an untrained model that predicts a 39% default rate against a 4.7% base | measured, untrained control | §15 |
| **Firm-level financial data is licence-locked**, so synthetic pretraining is the only clean route into this domain, which is why the space is empty while energy is crowded | dataset census | §4, §8 |
| **The prior matches real task difficulty** — logistic-regression AUC 0.743 synthetic against 0.769 real | measured, held-out seed | §18, §19 |

### What is measured and small

| claim | size | § |
| --- | --- | --- |
| Domain prior beats a generic one | **+0.049 AUC**, 3 of 6 cells significant | §14 |
| Model beats logistic regression | **+0.033 AUC** | §17 |
| Model beats an untrained model of the same architecture | **+0.031 AUC** | §15 |
| Calibration advantage over a *calibrated* gradient boosting | **~2×**, and only below ~250 rows | §16 |
| Brier skill over a feature-free constant predictor | **1-2%** | §17 |

### What has been ruled out

- **"More accurate than gradient boosting."** False above a few hundred rows; above ~1,000 a
  calibrated gradient boosting wins on both AUC and Brier (§16).
- **"Best calibrated" as a standalone claim.** A constant base-rate predictor beats every
  model here on ECE, so calibration numbers cannot carry an argument alone (§17).
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

### What this does not establish

- **Single seed, no significance test.** The AUC differences (±0.004) are almost certainly
  within noise, which supports "no difference" but does not establish it. Three seeds and the
  paired test before this is quoted.
- **Synthetic evaluation only.** The term structure cannot be scored on the UCI panels at all
  (no firm identifiers, §7). Real validation needs V4FinBench, whose loader now exists but
  whose data needs Kaggle credentials.
- **Small models near a ceiling.** Giving the baseline 5× compute bought +0.003 AUC, which
  suggests every arm is close to what this size can do. The comparison may look different at
  scale, in either direction.
- **Nothing here is about calibration of the curve levels.** §17's warning applies: coherence
  and calibration are independent, and a monotone curve can still state the wrong numbers.
