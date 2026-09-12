# Decisions

Choices made, the alternatives that were actually considered, and **what would reverse
them**. A decision recorded without its alternatives is indistinguishable from an accident,
and a decision recorded without a reversal condition cannot be revisited honestly when the
evidence changes.

Newest last. Where a decision rests on measurement, the measurement is in
`docs/FINDINGS.md`.

---

## D1 — Corporate credit, not fraud, markets, or general tables

**Date:** 2026-09-08. **Status:** active.

Finance splits into problems with little in common. The candidates were corporate credit and
default, fraud and financial crime, treasury, and market or investment prediction.

**Chose corporate credit.** Fraud is claimed by well-funded incumbents with real-time serving
infrastructure — Kumo's Online Serving distils to sub-100 ms, and Feedzai's RiskFM targets
fraud explicitly — and matching that means rebuilding an infrastructure stack, not a model.
Market prediction is harder science with severe temporal-validation traps and sophisticated
proprietary competition. Corporate credit is comparatively uncontested, is genuinely a
*tabular* problem rather than a temporal one, and its buyers are under supervisory obligation,
which is the wedge (D3).

**Reversed if:** a credible party ships a validated corporate-credit foundation model with
regulatory evidence, or a credit panel with usable licensing proves impossible to obtain,
making the domain undemonstrable.

---

## D2 — Synthetic-only pretraining, and the prior stays parametric

**Date:** 2026-09-08. **Status:** active, load-bearing invariant.

The alternative was pretraining on real company panels, which is what Fundamental does with
"billions of tables" and Kumo does with "real and synthetic relational data".

**Chose synthetic only**, for two reasons that turned out to be stronger than the original
cost argument. First, firm-level financial data sits behind Bloomberg, S&P and Moody's, so
synthetic generation is not a budget substitute in this domain — it is the only licence-clean
route, which is why the space is empty while energy is crowded (`FINDINGS` §4). Second, a
model that never saw real data **cannot** have memorised a benchmark, and the leakage
literature quantifies that contamination at up to 32 points of MAPE (`FINDINGS` §1). That is
an auditable property, not a marketing line.

The stronger half of the decision: **the prior samples a macro regime as a parameter rather
than learning real crisis history.** Fitting the generator to real panels to make it more
realistic would improve benchmarks while silently destroying the auditability claim, and
nothing would fail to warn us.

**Reversed if:** Phase 1 shows the synthetic prior cannot transfer at all, at which point the
choice is not "add real data" but "abandon the model and keep the validation layer" (D3).
Partially revisited if a genuinely open, commercially licensed firm-level panel appears.

---

## D3 — The product is validation evidence, not prediction

**Date:** 2026-09-08. **Status:** active; the central strategic bet.

The obvious product is "upload a table, get predictions, no training required". Rejected
because every mechanism claim in it is already owned — Kumo's is in NVIDIA's documentation
and AP and Forbes, Google's TabFM is in BigQuery, Neuralk and Prior Labs and Feedzai each
hold a piece. Competing there means competing on capital and distribution against a $255M
Series A. That is an unwinnable fight and picking it would be the main way this project
fails.

**Chose the validation layer.** Credit scorecards are logistic regression *because*
regulators demand interpretability, which is why Fundamental's published oil-and-gas result
beats linear regression rather than gradient boosting (`FINDINGS` §2). The accuracy bar is
low; the barrier is model risk management. Calibrated PD, stability across regimes,
documented out-of-time backtesting, evidence a validation committee accepts. Vendors whose
product is the prediction cannot ship that as an afterthought.

**Reversed if:** conversations with actual credit risk functions show they will not pay for
validation evidence separately from a model, in which case this becomes a feature of someone
else's platform rather than a company.

---

## D4 — Column attention over a flat row projection

**Date:** 2026-09-08. **Status:** active. **Supersedes** the original architecture.

The first architecture fed a flat row vector through `Linear(2·max_features → d_model)`.
Simple, fast, and wrong: it gave every feature a fixed weight slot, so permuting columns
changed predictions, padding position mattered, and column 7 of one dataset shared weights
with column 7 of an unrelated one.

**Chose per-cell embedding plus attention across columns, then pooling into rows.** Costs
more compute — the column stage runs on `B × N` sequences — and buys column-order invariance
and padding-width invariance, both now asserted in tests. Deliberately **no feature-index
embedding**, so identity comes from the data distribution rather than position.

**Alternative not taken:** adding learned per-column embeddings, which would be cheaper and
might help within a fixed schema, but reintroduces exactly the positional identity this
removes.

**Reversed if:** the column stage proves to dominate cost at scale without a measurable
accuracy return, in which case the fallback is row compression with a cheaper set encoder,
not a return to the flat projection.

---

## D5 — Calibration is a first-class metric, and balanced context is corrected

**Date:** 2026-09-08. **Status:** active. See `FINDINGS` §5 and §6.

Balanced context sampling was adopted on published AUC evidence, then adding calibration
metrics showed it inflated predicted default rates roughly threefold, because an in-context
model reads the base rate out of its context. AUC could not see this, since every prediction
inflates alike.

**Chose to keep balanced sampling and correct the base rate analytically**, shifting logits
by `log P_true(y) − log P_context(y)`. Exact under label shift, which holds by construction
because context selection looks only at `y`, and provably ranking-preserving.

**Alternatives considered:** abandoning balanced sampling, which forfeits the ranking gain;
or fitting a Platt scaler on a validation split, which absorbs whatever the model actually
does rather than assuming a mechanism.

**Reversed if:** at real pretraining scale the analytic correction over- or under-shoots
systematically. There is already a warning sign — at the 1-year horizon it *worsens* ECE
slightly — so the fitted-scaler alternative is the likely successor rather than a
hypothetical.

---

## D6 — Apache-2.0, and no LICENSE while private

**Date:** 2026-09-08. **Status:** active.

Considered AGPLv3 with a commercial dual licence, mirroring `finkele-axiom`.

**Chose Apache-2.0.** Dual licensing has no revenue model here: customers would consume
predictions through an API or licensed weights and never deploy the training code, so there
is no copyleft obligation for them to pay to escape. Meanwhile AGPL is blanket-banned at many
banks and insurers, which are precisely the target buyers, and Apache's patent grant is what
their counsel looks for. The moat is the weights and the mature prior, not the training code,
which is reproducible from the public PFN literature in days.

**Reversed if:** the code itself turns out to be the differentiator, which would be evidence
the architecture is far more novel than currently believed.

---

## D7 — Metal for local training

**Date:** 2026-09-08. **Status:** active. See `docs/COMPUTE.md`.

**Chose `--device mps`.** Measured 3.5x faster than CPU at identical loss, but the deciding
factor is that it runs on the GPU, which the genomics pipeline sharing this machine does not
touch. That converts "wait for the pipeline" into "run now".

**Reversed if:** an MPS numerical discrepancy appears at scale, or Phase 2's multiplied run
count makes rented NVIDIA the cheaper path in wall-clock terms.

---

## D8 — The base-rate correction belongs on the object, not in one method

**Date:** 2026-09-09. **Status:** active. **Amends D5.** See `FINDINGS` §28.

D5 chose to keep balanced context sampling and correct the base rate analytically. That
correction was implemented in `predict_proba` and only there. The term-structure path was
added later, built its own forward pass through `FinancialTFM.term_structure`, and walked
straight past it — producing a 12.8% default rate against a 0.47% truth for a full evaluation
cycle, and a wrong root-cause diagnosis that cost a 6,000-step retrain.

**Chose to make the corrected path the only public path.**
`FinancialTFMClassifier.predict_term_structure` applies the correction itself, the shift
functions live beside the hazard head with their monotonicity and rank-preservation properties
tested, and the out-of-time harness retains a permanently **uncorrected arm** so the
distortion is measured rather than assumed absent.

**Alternative considered:** asserting the correction in a test on the harness. Rejected — the
harness is one caller of several, and the next one would have the same freedom to bypass it.
The property has to be unavailable to get wrong.

**Reversed if:** a use case needs raw uncorrected curves as a primary output, in which case
the correction moves to a required explicit argument rather than a default, so that skipping
it is a visible choice at the call site instead of an omission.

---

## D9 — Uniform context on the survival path; balanced stays the class default pending re-measurement

**Date:** 2026-09-09. **Status:** active. **Partially reverses D5.** See `FINDINGS` §29.

D5 adopted `balanced` on published evidence (Tanna et al. 2026: balanced worth 3-4 AUC points
over uniform on credit-risk TFMs). Measured on the V4FinBench out-of-time split, the ordering
is reversed and the margin is roughly three times larger: uniform beats balanced by 10-12 mean
AUC points at every context size tested, and 12 in-context defaults outrank 1,122.

**Chose to report uniform as the headline survival configuration and keep all three strategies
as measured arms**, rather than silently flipping a default. Two reasons. The binary
single-horizon evidence that motivated D5 has not been re-measured under uniform sampling, so
flipping the class default would change a path this experiment says nothing about. And the
trade is real rather than free: uniform costs roughly 0.003 ECE for roughly 10 AUC points, so
a calibration-first use case could legitimately want the other side of it.

**Alternatives considered:** flipping the default globally, which over-generalises from one
dataset and one seed; and removing balanced sampling, which discards the strategy that is
still better calibrated and is still supported by the published benchmark on its own setting.

**Reversed if:** `openspec/changes/revisit-context-strategy` finds uniform also wins on the
binary path across datasets and seeds, at which point the class default changes and D5's
context half is fully retired. Reversed the other way if the effect fails to replicate across
seeds, since §29 rests on a single draw per cell.

---

## D10 — Rank-transform features by default; uniform context by default; retrieval opt-in

**Date:** 2026-09-09. **Status:** active. **Resolves D9.** See `FINDINGS` §33 and §35.

Three changes to what the classifier does when nobody passes an argument, each on measured
evidence rather than on the literature this project took its first defaults from.

**`feature_transform="rank"`.** 110 of 136 features on V4FinBench have a standard deviation
more than ten times their interquartile range, with a median ratio of 240, so the model's
mean/standard-deviation normalisation was being destroyed on almost every column. Conditioning
first is worth **+0.086 mean AUC** to uniform context and +0.023 to retrieval out of time, and
improves AUC in seven of eight configurations across two further panels.

**`context_strategy="uniform"`,** replacing `"balanced"`. D5 took balanced from Tanna et al.;
§29 measured it losing by 10-12 AUC points on the survival path; §35 shows the strategies are
nearly tied on the binary path because those panels lack the positives for balanced to reach
50/50. **The effect scales with how extreme the rebalancing is, not with the strategy's name.**
Uniform is never worse than balanced in any measurement here, and it is blind, so it keeps the
batch independence retrieval gives up.

**Retrieval stays opt-in**, despite being the best strategy on all three panels and both
paths, and significantly better than balanced on both binary panels. It costs about 2.5× the
scoring time and makes a prediction depend on its query group-mates. **A default that silently
breaks batch independence is the category of hidden behaviour that produced §28**, so this one
is chosen at the call site.

**Alternatives considered:** making retrieval the default, rejected for the reason above;
keeping `"balanced"` for continuity, rejected because no measurement supports it; and
special-casing the default per prediction path, rejected because a default that changes
depending on which method you call is worse than either default.

**Reversed if:** the rank transform loses on a panel with well-behaved features, which is
plausible — it discards magnitude information and the gain here comes from tails that a
cleaner dataset may not have. That would make it a data-dependent choice rather than a
default, and `winsor` is the intermediate already implemented.

---

## D11 — A head that was never trained is not reachable

**Date:** 2026-09-09. **Status:** active. See `FINDINGS` §34.

The training loop optimises one objective per step, so a hazard checkpoint's classification
head stays at random initialisation. `predict_proba` served it — AUC 0.3745, a stated 69%
default rate against a 4.7% base — silently.

**Chose to record the objectives actually optimised into the checkpoint and refuse to serve
any other head.** The alternative was training both objectives jointly so the situation cannot
arise, which is the better long-term answer and is now
`openspec/changes/joint-objective-training`; it needs a pretraining run, and the guard is
worth having regardless of how that lands, because a checkpoint should be able to say what it
is.

Checkpoints written before the field carry no record and are allowed through. That is a
deliberate hole — refusing them would break every in-memory model — and it is why the field is
written on `save` rather than guessed on `load`.

**Reversed if:** joint training lands and makes every checkpoint serve both paths, at which
point the guard becomes a formality. Keep it anyway.

## D12 — Random per-task column identities, and exact D4 invariance traded for expressiveness

**Date:** 2026-09-11. **Status:** active. **Amends D4.** See `FINDINGS` §54 and §56.

D4 chose per-cell embedding plus column attention, deliberately with **no feature-index
embedding**, so that "identity comes from the data distribution rather than position". Its
*Alternative not taken* names learned per-column embeddings and rejects them because they
"reintroduce exactly the positional identity this removes".

**That reasoning was right about learned fixed embeddings and overshot.** It rejected every
form of column identity, and the result was an encoder whose row representation is a symmetric
function of the row's values — verified converging to exactly permutation-invariant as the
context grows (§54). Such a model cannot represent "column j matters" at all: on `x_0 - x_1`
it scored 0.5007 against a provable ceiling of 0.5, and on the trivial prior it converged at
AUC 0.713 where logistic regression reaches 0.9998. D4's stated hope that identity would come
from the data distribution is precisely what the column stage cannot deliver, because it runs
on one row at a time and so never sees a column.

**Chose random per-task column identities.** A random vector per column, projected through a
learned linear layer, added to every cell of that column, resampled for every task. Rows agree
on which column is which; no column index acquires a fixed meaning, because the tag is
redrawn. This is the distinction D4 missed: *fixed* identity is positional and bad, *random
per-task* identity is neither. Result: 0.713 to 0.997, and `x_0 - x_1` to 0.9995.

**What this costs, stated plainly.** Exact pointwise column-order invariance — D4's headline
property — **is gone.** It is now distributional, recovered by averaging over identity draws
at roughly 1/sqrt(K): 0.018 absolute probability at K=1, 0.0056 at K=16, 0.0023 at K=64. Both
halves are asserted in `tests/test_model.py`, which is parametrized to require *exact*
invariance at `column_id_dim=0` and *distributional* invariance with identities on, so the
test states the guarantee the code actually has. Padding-width invariance is weakened the same
way, being a special case of the same permutation.

This trade is accepted because the property D4 protected is worth nothing on its own: an
encoder symmetric enough to guarantee it pointwise is provably unable to learn a
column-specific rule. TabPFN makes the same trade and for the same reason, citing the need to
"differentiate features ... that have the same statistics".

**Predictions are deterministic anyway.** Identities are random while training — the
resampling is the mechanism — and **seeded while evaluating**, because a credit model whose
score changes between two identical calls fails model validation before anyone examines its
accuracy. Callers ensemble by passing explicit seeds.

**Reversed if:** the cell-level two-way attention of `FINDINGS` §54's rung 2 lands. Attending
across rows within each column gives each column a *data-derived* identity, which is both
deterministic and exactly column-order equivariant, and would restore D4's property in full
rather than in distribution. That is the better answer; this one is fifteen lines and shipped
today. Losing exactness is a real regression against D4 and should not be left standing
permanently.

## D13 — Average precision is read first on any low-base-rate split

**Date:** 2026-09-11. **Status:** active. See `FINDINGS` §60.

On V4FinBench horizon 0 (0.380% base rate, 79 positives against 20,838 negatives), ROC-AUC
ranked this model **second of five** at 0.9827 against CatBoost's 0.9944 — a gap of 0.012.
Average precision on the identical predictions ranked it **third**, at 0.1853 against 0.3310,
a factor of 1.8. Both numbers are correct; they measure different things, and only one of them
is about the region where a credit decision is taken.

**Chose to report average precision alongside ROC-AUC everywhere, and to read AP first
whenever the base rate is low.** ROC-AUC stays, because V4FinBench's published table uses it
and dropping it would make us incomparable to the field. It is no longer the number quoted
first.

Also added: **oracle F1**, the best F1 any threshold reaches on test. Alone it means nothing —
it tunes on the test set. The quantity of interest is `F1_oracle - F1`, which separates a
ranking that is weak near the decision boundary from a threshold that failed to transfer.
Measured, those two diagnoses pointed opposite ways: our transfer loss was the second smallest
in the table while our oracle F1 was the lowest, so the deficit is discrimination, not
calibration. Without the oracle column the obvious reading was the wrong one.

**Why this is a decision and not a preference.** `FINDINGS` §55 criticises TabPFN for
reporting only ROC-AUC and accuracy. Our own harness reported ROC-AUC and F1, and would have
carried "second of five" into a document while the field was nearly twice as good at the thing
being sold. The criticism was right and we were committing a version of it simultaneously.

**Reversed if:** never for low base rates. On balanced data ROC-AUC and AP largely agree and
the ordering is a matter of taste.
