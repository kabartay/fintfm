# Competitive landscape

Recorded 2026-09-08 from public company material (websites, blog posts, LinkedIn
announcements) supplied during the founding session. **Everything here is a claim made by
the company about itself**, not something measured or verified independently. Treat it as
positioning intelligence, not as fact about capability. Re-check before citing: this page
will decay.

## Who is in this space

| company | what they build | shape of the data | posture |
| --- | --- | --- | --- |
| **Fundamental** (NEXUS) | general tabular foundation model, "pretrained on billions of tables" | single tables | proprietary; $255M Series A per their site |
| **Prior Labs** | tabular foundation models for spreadsheets/databases; the TabPFN lineage | single tables | open weights/code history, commercial company |
| **Neuralk** (Seldon) | tabular foundation model, in-context, synthetic-prior pretraining | single tables | proprietary, API + self-hosted |
| **Kumo** (KumoRFM-2) | relational foundation model over warehouse schemas, graph transformers, NL query interface; Online Serving distills to sub-100 ms | **multi-table relational** | commercial platform, heavy press (AP, Forbes) |
| **Feedzai** (RiskFM) | "universal tabular foundation model" for financial risk | transactional | direct competitor on **fraud** |
| **Google** (TabFM) | general TFM, hundreds of millions of synthetic datasets | single tables | research release |
| **The Forecasting Company** | time-series foundation model (t0-alpha), temporal engine, Retrocast UI | **temporal** | argues explicitly that time series are not tables |
| **Aionic Labs** | time-series *language* models (OpenTSLM lineage), reasoning in NL | temporal + text | SPRIND-funded, up to €26.5M |

## What this means for this project

**The "no feature engineering" pitch is spent.** Kumo has it in AP and Forbes. Leading with
it now reads as derivative. It is table stakes messaging, not a differentiator.

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
