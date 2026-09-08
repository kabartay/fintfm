# Findings

Numbered, dated, with the source or command that produced each. A finding that lives only in
a conversation is lost when the conversation compacts.

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

**Date:** 2026-09-08. **Status:** positioning input; see `docs/LANDSCAPE.md`.

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

**Date:** 2026-09-08. **Status:** analysis of a published table; the explanation is inference,
the absence is fact. **Source:** Table 5 of Meyer et al. (arXiv:2510.13654), which catalogues
pretraining (P), train/test (T/T) and zero-shot (ZS) dataset use across 15 TSFMs.

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
