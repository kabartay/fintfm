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
