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
