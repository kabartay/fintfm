# Competitive landscape

Recorded 2026-09-08 from public company material (websites, blog posts, LinkedIn
announcements) supplied during the founding session. **Everything here is a claim made by
the company about itself**, not something measured or verified independently. Treat it as
positioning intelligence, not as fact about capability. Re-check before citing: this page
will decay.

## Who is in this space

| company | what they build | shape of the data | posture |
| --- | --- | --- | --- |
| **Fundamental** (NEXUS) | general tabular foundation model, "pretrained on billions of tables"; **now verticalising** — sells NEXUS as purpose-built for oil & gas, in production | single tables | proprietary; $255M Series A per their site |
| **Prior Labs** | tabular foundation models for spreadsheets/databases; the TabPFN lineage | single tables | open weights/code history, commercial company |
| **Neuralk** (Seldon) | tabular foundation model, in-context, synthetic-prior pretraining | single tables | proprietary, API + self-hosted |
| **Kumo** (KumoRFM-2) | relational foundation model, **in-context prediction at query time**, trained on real + synthetic relational data; PQL query language, NL coding agent, optional fine-tuning; Online Serving distills to sub-100 ms | **multi-table relational** | commercial platform; heavy press (AP, Forbes); **documented on docs.nvidia.com** |
| **Feedzai** (RiskFM) | "universal tabular foundation model" for financial risk | transactional | direct competitor on **fraud** |
| **Google** (TabFM) | general TFM, hundreds of millions of synthetic datasets | single tables | research release |
| **The Forecasting Company** | time-series foundation model (t0-alpha), temporal engine, Retrocast UI | **temporal** | argues explicitly that time series are not tables |
| **Aionic Labs** | time-series *language* models (OpenTSLM lineage), reasoning in NL | temporal + text | SPRIND-funded, up to €26.5M |

## What this means for this project

**The "no feature engineering" pitch is spent, and so is "in-context, no training".**
Kumo has the first in AP and Forbes. The second is worse: NVIDIA's own docs describe KumoRFM
as "performs in-context predictions... learns from your existing data at query time without
training" — the same sentence this project would use about itself. Neither claim
differentiates any more; both are table stakes messaging. **The repo description currently
leads with "no training on your data" and should not survive into a pitch deck in that
form.**

**Distribution is the asymmetry, not the model.** KumoRFM appearing in NVIDIA's
documentation puts it inside a stack enterprise buyers already run. That channel cannot be
matched by a solo founder and it decides more outcomes than model quality does. Plan around
it rather than against it: pick a buyer and a problem where a general platform's default
presence is not sufficient.

**Fraud is claimed; corporate credit is not.** Kumo (real-time serving, relational) and
Feedzai (RiskFM) both target fraud with infrastructure that would have to be rebuilt to
compete. Corporate default on a company panel remains comparatively uncontested, which is
why this repository targets it.

**Kumo's relational framing is the most serious threat to the credit use case**, more than
the single-table TFM companies are. Bank risk data is natively relational (customers,
accounts, transactions, filings over time). Expect a sophisticated buyer to ask why they
should not point a relational platform at the warehouse instead. Have an answer.

**Their latency moat does not transfer.** Sub-100 ms serving is decisive for fraud, where
the model sits inside transaction authorization. Corporate credit underwriting takes days.
Kumo's hardest engineering is on an axis this market does not price.

**The financial prior is thinner protection than it first appears.** Kumo trains on real
*and synthetic* relational data; synthetic-prior pretraining is not this project's private
idea. The domain specificity is real but is a difference of degree, not of kind, and it
should be argued with measured benchmark evidence on credit panels rather than asserted.

**The gap nobody is filling: regulated model risk management.** A credit model at a bank
does not win on accuracy; it wins on surviving validation — calibrated PD, stability across
economic regimes, documented time-based backtesting, explainability a regulator accepts,
evidence for Basel / IFRS 9 review. "Upload data, get predictions" platforms structurally
cannot ship that, because there the certificate *is* the product. This is also the one place
where `finkele-axiom` (validation protocol, conformal coverage, honest refusals) is directly
transferable. See `docs/POSITIONING.md` if that thesis is ever written up properly.

**Tabular is the right frame for this problem, and that is contested ground worth knowing
about.** The Forecasting Company argues tabular priors do not transfer to temporal data:
rows are exchangeable, time is not, and encoders leak future information unless masked by
hand. That argument is correct and it cuts *for* this project rather than against it —
corporate default on an annual company panel is genuinely a tabular problem. It becomes a
problem the moment this repository drifts toward forecasting cash flows or market series,
which would need a causal decoder, not this architecture.

## The competitive bar in enterprise deals is the incumbent, not the SOTA

Fundamental's oil & gas post (2026-09) reports NEXUS at **75% better MAE and 43% better
RMSE across 13 regional markets, beating the prior approach in 70% of regions and moving to
production** — and the baseline it beats is **linear regression**. Not gradient boosting,
not another foundation model. That is the number a funded competitor chose to publish, so
take it as the shape of what enterprise buyers actually compare against: whatever legacy
model they run today.

**This is unusually favourable for credit.** Bank credit scorecards genuinely are logistic
regression, in large part *because* regulators demand interpretability. So the accuracy bar
in this market is low and the real barrier is regulatory acceptance — the third independent
signal pointing at validation evidence rather than prediction quality as the product.

Consequence for `bench.py`: **keep both baselines and read them differently.** Logistic
regression is the commercially realistic comparison (what a customer would actually
replace); gradient boosting is the scientific one (what tells you whether the model is any
good). Never report only the flattering one — see the claims section of `CLAUDE.md`.

**Verticalisation is the confirmed playbook, and it is also the threat.** Fundamental going
vertical validates specialising rather than chasing a general TFM. It also means financial
services is an obvious next vertical for a company that already has the general model, the
capital and the team, for whom entering it is a GTM motion rather than a research project.
Assume the wedge is defended by regulatory depth and evidence, not by being first.

## Technical notes worth keeping

**In-context serving cost is a known problem with a known answer.** A PFN-style model carries
its context table on every request and attention is quadratic in context length — hence the
`max_context` cap in `classifier.py`. Kumo's published answer is two-stage: train the heavy
relational model offline, distill into a light serving model over precomputed embeddings.
If latency ever becomes a requirement here, distillation is the escape hatch to reach for
rather than a novel problem to solve.

**Nothing in this file may be used as a source of code, weights, or data.** See the licensing
boundary in `CLAUDE.md`. These are competitors read from public material, and that is the
only relationship this project has with them.

## Four camps, and the object each one predicts

Added 2026-09-08. The clearest way to see where this project can win is to ask what *object*
each camp's model actually predicts. They are not competing for the same target.

| camp | who | object predicted | rows exchangeable? |
| --- | --- | --- | --- |
| **time series** | The Forecasting Company (t0-alpha), Google TimesFM-3, Chronos-2, Toto, Moirai, NXAI TiRex, Aionic (TSLMs) | future values of a sequence | no — order *is* the signal |
| **tabular** | Neuralk (Seldon), Fundamental (NEXUS), Prior Labs (TabPFN), Google TabFM, Feedzai (RiskFM) | a label for one row | yes, assumed IID |
| **relational / graph** | Kumo (KumoRFM), GraphPFN | a label for a node in a linked structure | no — edges carry information |
| **panel hazard** | **nobody** | **the probability path of an event, per entity, over time, given time-varying covariates** | neither |

Credit risk is the fourth row. Its object is a hazard path, and **IFRS 9 makes that path
mandatory** by requiring lifetime expected credit loss rather than a single-horizon
probability. See `openspec/changes/pd-term-structure`.

**Why no camp takes it for free:**

- **Time-series models forecast the covariates, not the event.** They have no notion of an
  absorbing state, competing risks, or a cumulative probability that must not decrease. The
  Forecasting Company's own argument that time series are not tables cuts both ways: a
  hazard over a panel of firms is not a series to continue either.
- **Tabular models predict one label at one horizon.** Multiple horizons become unrelated
  tasks with no coherence constraint, which is exactly the defect
  `pd-term-structure` task 11.1 is designed to expose in our own current output.
- **Graph models add cross-entity structure but still emit a label**, not a path. Kumo's
  relational framing is the strongest threat to the credit use case generally
  (bank data *is* relational), but its object is still a prediction per node.

**GraphPFN** is worth watching for a different reason: it extends the prior-data-fitted
paradigm to attributed graphs, encoding tabular node features and graph structure in one
transformer, and there is related work turning tabular foundation models into graph ones. If
that line matures, the tabular and relational camps converge and Kumo's structural advantage
becomes reproducible from open weights. Encountered via a DeFi credit-risk paper
(arXiv:2602.03981) that fine-tunes open GraphPFN weights; **that paper has not been read
beyond its related-work section**, and network contagion is systemic-risk modelling rather
than single-obligor PD, so it is adjacent rather than a competitor.

## Neuralk as a product and pricing reference

Observed 2026-09-08 from their public site and a trial account.

**Positioning:** "THE PREDICTIVE AI COMPANY" / **"Predict anything."** / "The Foundation
Model for Prediction. No training. No pipelines."

**Product surface:** a natural-language prompt box ("Predict customer…"), a CSV upload, and
a Predict button, plus "try API for free". Model served as `nicl-small`.

**Commercials:** credit-based. Roughly **2 credits per inference request of ~2,000 test
rows**, observed latency ~250ms. A dashboard covers API keys, team members with
owner/admin/member/viewer roles, usage charts and a voucher redemption box. Notably they
**auto-refund** requests whose delivery is unconfirmed ("no SDK telemetry after 10m"), which
implies they treat unverified delivery as a billing risk worth automating away.

**Two things to take from it, and one not to.**

Take the **pricing shape** as a reference point: per-request credits scaled by rows scored,
not a seat licence. And take the **auto-refund** detail as a signal that inference-billing
disputes are real enough to engineer around.

Do **not** take the product framing. "Predict anything" from a CSV is the exact generic
positioning `docs/STRATEGY.md` rules out: it competes with Neuralk, Kumo and Google BigQuery
simultaneously on the one axis where this project has no advantage. An API is eventually the
delivery mechanism for the certificate, but the thing delivered has to be narrow — a PD term
structure with its interval, its calibration and its attributions — or it is a worse copy of
something already shipping.

## Neuralk's Financial Services page names the constraint and does not solve it

Read 2026-09-08 from [neuralk.ai/solutions](https://www.neuralk.ai/solutions). This is the
single most useful competitive document encountered, because it is a direct competitor
describing our exact market in their own words.

**Their structure is horizontal with vertical marketing.** Six verticals — financial
services, telecom, industry and manufacturing, energy and utilities, commerce and retail,
healthcare and life sciences — each with four use cases, all served by the same model. The
messaging is explicit: *"one model, every industry"*, no retraining. The financial services
page is generic capability with domain copy on top, not a purpose-built financial product.

**They name the binding constraint themselves:**

> "Financial institutions face two hard constraints: extreme accuracy requirements and
> **tight regulatory scrutiny**."

**And then address none of it.** The page has no mention of calibration, model validation,
probability-of-default term structure, IFRS 9, Basel, low-default portfolios, uncertainty
quantification, or AML. It names regulatory scrutiny as one of the two hard constraints and
ships nothing for it.

**Their own demo mockup is the argument.** The financial services illustration shows a
personal loan applicant with a credit score of 721 and:

> Default risk **6%** — Recommendation: **Approve at 4.9% APR**

A bare point estimate, with a pricing decision attached, and no interval, no horizon, no
calibration statement, no refusal path for an applicant unlike anything in the context. That
is precisely the object `docs/STRATEGY.md` argues is insufficient: a single 6% carries no
information about whether 6% means anything, and a model-risk function cannot validate it or
provision against it.

**Be fair about where the gap actually bites.** For a fast consumer-credit decision, a point
PD at 250ms may be exactly what the buyer wants, and their retail traction is real (a
reported 1.3M products at 89.8% zero-shot accuracy). The gap matters for **regulated PD used
in provisioning and capital** — IFRS 9 lifetime ECL, Basel PD, low-default portfolios — where
the number must be defensible rather than merely fast. That is a narrower market than
"financial services" and it is the one this project targets.

**Three further reads:**

- **Their credit angle is consumer, not corporate.** The example is a €14,000 personal loan
  with a credit-bureau-style score. SME and specialty corporate lending, where the
  small-data advantage is strongest (`FINDINGS` §9), is not what this page is selling.
- **They claim time series too** ("alpha generation", "time-series forecasting to power and
  commodity price data"). Given Beyond IID on non-IID data and The Forecasting Company's
  argument that time series are not tables, that is a broad claim for a tabular model and
  not a fight worth joining on either side.
- **Breadth across six verticals means depth in none** — the structural opening for a
  specialist. But it cuts both ways: nothing stops them adding depth once a customer pays
  for it, so the defence has to be regulatory substance and an accumulated track record, not
  a head start.

## How the open TFM projects present themselves, and what fintfm should copy

Recorded 2026-09-22 after reading ten peer projects from primary sources. **Licensing posture
and README structure both reveal commercial intent more reliably than any stated positioning
does**, and the pattern is consistent enough to plan against.

### Three postures, distinguishable at a glance

| posture | who | code / weights | what the README leads with |
| --- | --- | --- | --- |
| **Research-adoption** | TabICL (Soda-Inria), LimiX (Stable AI + Tsinghua), Nori (Synthefy) | permissive **both** — BSD-3, Apache-2.0 | method, ablations, full pretraining code, the synthetic generator |
| **Commercial hedge** | EXAONE (LG AI Research), Causilo (Nums AI), Google TabFM/TimesFM | permissive **code**, non-commercial **weights** | benchmark rank, usually self-reported |
| **Productised** | Prior Labs (TabPFN) | permissive, plus a paid tier | a Nature paper, an extensions ecosystem, and a **distillation engine** for deployment |

**The commercial-hedge pattern is the one to recognise.** Permissive code costs them nothing —
the architecture is reproducible from the literature in days, which is exactly why this project
chose Apache-2.0 (decision D6). The weights carry the restriction because the weights are the
asset. EXAONE's licence goes further than most and bars commercial use of the **Output**, not
just the model. Four of the projects read this month split their licences this way, and in
every case the split is invisible from the repository's headline licence badge.

### What they lead with, and why fintfm must not copy it

**Every commercial entrant leads with a leaderboard rank**, and Causilo leads with one that does
not match this project's own measurement of it (§109's neighbour: their README claims Elo
position 1 at 1792.9; our live run puts them 6th at 1751). Leading with rank is rational when
you have one.

**We place 93rd of 95.** So rank-first presentation is unavailable, and imitating it would mean
either burying the number or reporting a favourable slice of it — which is exactly the
absolute-AUC-without-rank error §107 already retracted once.

### What fintfm should copy instead

- **From the research-adoption camp: publish the generator.** TabICL, LimiX and Nori all ship
  their synthetic data generation. This project's prior is its distinctive asset *and* its
  auditability argument, and those pull in the same direction: a provenance claim nobody can
  check is a slogan. `CLAUDE.md` currently treats the mature prior as the private moat, which
  is defensible commercially and in tension with the audit story — a decision to make
  deliberately rather than by inertia.
- **From TabPFN: the deployment path.** Their distillation engine converts a fitted model into
  a dataset-specific MLP or tree ensemble for "orders-of-magnitude lower latency", aimed at
  pipelines "constrained by latency, interpretability, or regulatory requirements". That is a
  precise description of this project's target buyer, and this project's predict time is
  **8.6 s/1K against a field norm near 0.1**.
- **From nobody: lead with the claims ledger.** No peer publishes a document that tags its own
  claims SURVIVES / SUPERSEDED / RETRACTED and records the retractions as prominently as the
  wins. That is a genuine differentiator in a regulated domain where a model must arrive with
  its own validation evidence, and it is the one axis on which this project is ahead of the
  field rather than behind it.

**The honest positioning that follows:** not "a competitive tabular foundation model" — the
measurement says otherwise — but *a credit model that arrives with an auditable record of what
has and has not been shown about it*. That is `docs/STRATEGY.md`'s existing thesis, and the peer
sweep confirms it is the only claim here that no better-funded competitor is also making.

