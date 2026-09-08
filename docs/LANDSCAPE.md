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
