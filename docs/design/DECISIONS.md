# Decisions

Choices made, the alternatives that were actually considered, and **what would reverse
them**. A decision recorded without its alternatives is indistinguishable from an accident,
and a decision recorded without a reversal condition cannot be revisited honestly when the
evidence changes.

Newest last. Where a decision rests on measurement, the measurement is in
`docs/results/FINDINGS.md`.

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

**Date:** 2026-09-08. **Status:** active. See `docs/infra/COMPUTE.md`.

**Chose `--device mps`.** Measured 3.5x faster than CPU at identical loss, but the deciding
factor is that it runs on the GPU, which the genomics pipeline sharing this machine does not
touch. That converts "wait for the pipeline" into "run now".

**Reversed if:** an MPS numerical discrepancy appears at scale, or Phase 2's multiplied run
count makes rented NVIDIA the cheaper path in wall-clock terms.

**Partially reversed, 2026-10-03: not safe at the current reference recipe without
investigation first.** Launching the exact §91 reference recipe (`--n-rows-choices
256,512,1024 --batch-size 8 --max-features 136 --cell-labels --device mps`) grew to **78 GB
resident memory on a 64 GB machine** within 17 minutes, never reaching step 50, forcing heavy
memory compression and the exact bwa-plus-real-training freeze risk this file already warns
about. Killed before any damage; no checkpoint had been written yet. The same recipe needs
15-30 GB on HF Jobs' CUDA path, so this is specific to MPS at this row/feature count, not a
property of the model or recipe in general — the original 3.5x-faster measurement predates the
current cell-attention architecture's scale and never tested this many rows/features. Do not
launch this recipe on `--device mps` again until the memory growth is root-caused; a smaller
smoke config (fewer `--n-rows-choices`, smaller `--feature-chunk`) is untested but may be safe.

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

## D14 — Synthetic-only training, against the field's counter-thesis (task 48.7)

**Date:** 2026-10-02. **Status:** active; restates D2 against a counter-argument that has
since published.

ConTextTab (arXiv:2506.10707) argues explicitly that "exclusive training on synthetic data
limits their ability to fully leverage the rich semantics and world knowledge contained in
real-world tabular data", and trains on large-scale real tables instead; iLTM and TabSTAR make
the same choice. This is not a different tactic aimed at the same goal D2 already weighed — it
is a different goal. Those projects want semantic transfer from real column names, real units,
real world-knowledge priors a table's text carries. Synthetic-only training cannot supply that
by construction, and no amount of prior engineering closes the gap, because the information
genuinely is not there.

**Cost, stated plainly.** This project is giving up whatever real column semantics would buy
it — plausibly real, unmeasured here — in exchange for D2's auditability claim: a model that
never saw real data cannot have memorised a benchmark, and the leakage literature quantifies
that risk at up to 32 points of MAPE (`FINDINGS` §1). The three peers made the opposite trade.
Neither side is wrong; they are optimising different things, and this project's target buyer
(a bank or insurer whose compliance review is the actual gate, per `docs/roadmap/STRATEGY.md`) is
one for whom the auditability side of that trade is likely to matter more than for an open
leaderboard entrant.

**Not reversible for a licence reason, unlike most trade-offs in this file.** D2 already
states firm-level data sits behind Bloomberg/S&P/Moody's, so "train on real tables" is not a
budget question here — there is no large, licence-clean panel of the kind ConTextTab-style
training needs in this domain. The three counter-thesis projects evidently found one in their
domain; this project has not and the search already informed D2.

**Reversed if:** a genuinely open, commercially licensed firm-level panel at the scale these
peers use appears (restates D2's own reversal condition — this is the same fork, now with
named opposition), **or** if this project's own measured deficit (§101-§102's uniform ~0.035
ROC-AUC gap, worse on small tables per D15) turns out to be semantic-knowledge-shaped rather
than architecture- or scale-shaped, which would mean the synthetic prior is the actual
bottleneck and no further architecture or prior-diversity work (Phase C) can close it.

## D15 — The small-data thesis and §102's measured direction (task 48.8)

**Date:** 2026-10-02. **Status:** active; names a tension rather than resolving it.

TabPFN v2 (*Nature*, 2025) claims dominance "for datasets with up to 10,000 samples and 500
features". `docs/roadmap/STRATEGY.md`'s Phase 1 exit condition B bet on the same segment: "the model
beats gradient boosting **somewhere on the size sweep** — most plausibly below 1,000 rows."
Both treat small-data as this architecture family's natural strength.

**§102 measured the opposite shape on real data.** Re-reading §101's TabArena results against
dataset size: the ROC-AUC deficit against tuned baselines is **−0.0465 on the ten datasets
under 5,000 rows, and −0.0276 on the seventeen above** — almost double the gap on the smaller
half. The single clearest contrast: −0.0074 on 150,000-row `GiveMeSomeCredit`, −0.0863 on
1,000-row `credit-g`. Where TabPFN v2 and this project's own stated bet both expect the
largest edge, this architecture currently shows its largest deficit.

**Why this is not simply "the small-data thesis is dead."** §102 also found minority-class
fraction, not size, is the variable that actually tracks the residual (balanced classes lose
most, rare-event classes least) — V4FinBench itself is a rare-event panel (0.3-4% positive
rates depending on horizon), which is closer to the regime where §102's data says this
architecture does comparatively better, not worse. TabArena's small datasets skew more
balanced than V4FinBench's credit panels. So the contradiction may be between "small" as
TabPFN v2 and `docs/roadmap/STRATEGY.md` use it (row count) and "small" as the actual lever (something
correlated with, but not identical to, row count) -- which §102 could not fully separate from
size given only 27 datasets and one confound already found (§100's categorical-preprocessing
effect).

**The honest state: a contradiction recorded, not reconciled.** This project's stated small-
row-count bet is not supported by its own TabArena measurement; its credit-panel results (where
the actual product claim lives) have not been sliced by size or minority fraction the way §102
sliced TabArena, so whether the same shape holds on V4FinBench is untested rather than refuted.

**Reversed if:** a size-and-minority-fraction slice of V4FinBench's own five folds either
confirms §102's pattern transfers (minority fraction is the real lever, `docs/roadmap/STRATEGY.md`'s exit
condition B should be restated in those terms) or shows V4FinBench's gap also widens on its
smallest folds (the small-data thesis loses its strongest piece of indirect support and
`docs/roadmap/STRATEGY.md` should say so rather than continue to assume it).

## D16 — The Bitter-Lesson critique against this project's own effort allocation (task 48.11)

**Date:** 2026-10-05. **Status:** active; names the trade rather than resolving it.

TabDPT's appendix states the field's version of Sutton's Bitter Lesson plainly: compute and
high-quality data matter more than architectural manipulation, and a lot of published
tabular-foundation-model work amounts to the latter. This project's own record now makes the
critique concrete rather than abstract, and the concrete version is harder to wave away.

**The score, stated once, in one place.** Every lever this project has pulled on the
architecture/prior-mechanism side, with its measured return:

| lever | result |
| --- | --- |
| 5x training-task volume (§93) | −0.0012 AP, null |
| 5.7x parameters at matched tasks (§108) | −0.0077 AP, a loss |
| learnability filter on the prior, full strength (§145) | −0.0106 AP, a loss |
| cheap realism augmentations, bundled (§144) | −0.0035 AP, noise with a negative lean |
| schedule-free optimisation, untuned (§146) | −0.0180 AP, a loss (confound found, §146's addendum) |
| dedicated mask embedding (§147) | −0.0312 AP, a clean loss |
| pinball quantile head vs. binned head (§148) | small negative lean, premise not confirmed |

Seven measured changes, seven nulls or losses, zero wins. This is not a curated selection —
it is the complete record of every architecture/prior lever this project has pretrained a
checkpoint to test. **What this project has spent its effort on is exactly the category
TabDPT's appendix names as the lower-return one**, and the data confirms the appendix's claim
about this architecture specifically, not just in general.

**What this project believes it gets in exchange, stated so it can be checked against
evidence rather than assumed.** Not raw accuracy — nothing above claims that. The bet (D2,
D14) is that synthetic-only, architecturally-legible pretraining buys **auditability**: a
model that provably never saw real data cannot have memorised a benchmark, which is the
property this project's target buyer (a bank's compliance review, per `docs/roadmap/STRATEGY.md`)
is claimed to value enough to accept whatever the architecture-tinkering nulls above cost in
raw rank. That is the actual trade this project is making, and it is not the same trade as
"architecture work will eventually find a win" — this file's own position is closer to "we
accept architecture work is low-expected-value and are paying for something else."

**What would show the trade is not worth it — the falsification this critique needs to not be
a slogan.** Three observations, any one sufficient:

1. **A real customer engagement is lost or blocked specifically on an accuracy gap this
   architecture family's own compute/data levers (more real data, bigger pretraining budget)
   would close, and auditability is not raised as a mitigating factor by the buyer.** This
   would mean the auditability premium this project is pricing in does not exist at the price
   this project is paying for it.
2. **A size/minority-fraction slice of V4FinBench (D15's own open reversal condition) shows
   this architecture's gap against gradient boosting is driven by something data volume would
   fix** — not architecture, not prior mechanism — which would mean the lever this project is
   locked out of (real data, D2/D14) is the one that actually matters, and no amount of
   further architecture tinkering of the kind §93-§148 tried can substitute for it.
3. **An eighth architecture or prior-mechanism lever, chosen in advance rather than
   post-selected, also returns a null or loss.** Seven failures could still be seven unlucky
   draws from a distribution with real wins in it; an eighth failure, pre-registered rather
   than cherry-picked, is the point at which "we haven't found the right lever yet" stops
   being an available excuse and "this category of lever has a low ceiling here" becomes the
   better-supported reading.

**What this does not settle.** D2's reversal condition (a licence-clean real panel at scale)
remains the only route to the lever this project is locked out of, and nothing here changes
that it has not appeared. This decision does not recommend abandoning architecture work
either — Phase A'/Phase C's remaining items (§110, §114's own open half, task 48.19) are
still worth their now-small marginal cost, since the infrastructure to test them already
exists and the tests themselves are cheap. It only names, plainly, that the project's actual
track record on this axis is seven losses and no wins, so the next one should be evaluated
against that base rate rather than against hope.

## D17 — Two model families: FinTFM stays synthetic-only, FinTFM-R trains on real data

**Date:** 2026-10-06. **Status:** active; direction agreed, nothing implemented yet.

D14 recorded synthetic-only training as a deliberate trade and named its reversal condition:
"a genuinely open, commercially licensed firm-level panel at the scale these peers use". D16
then scored the alternative honestly — seven architecture/prior levers, seven nulls or losses
— which makes the lever this project is locked out of the only untested one with a credible
ceiling. This decision opens that lever **without spending the thing D14 bought.**

**Two families, two names, one repository.**

| family | training data | name |
| --- | --- | --- |
| the existing model, unchanged | synthetic prior only | **FinTFM** |
| the new line | real data (plus, optionally, the synthetic prior) | **FinTFM-R** |

**The existing model is not renamed.** `FinTFM` keeps its name, its PyPI package, its Zenodo
DOI, its TabArena entry (`TA-FINTFM`) and every citation already made. Appending "-synthetic"
retroactively would rewrite something already shipped, the same category of mistake
`CLAUDE.md`'s release section forbids for tags. Where the contrast needs naming in prose,
write "FinTFM (synthetic-only)" as a parenthetical, never as a renamed identifier. "-R" is
documented as *real-data*, and that expansion belongs in `README.md` wherever the family is
first mentioned.

**What this costs, stated before any code is written.** The claim "fintfm has never seen real
data" stops being a property of the repository and becomes a property of a *checkpoint*. Every
future mention — release note, paper draft, leaderboard row, model card — must say which
family, permanently. A careless future sentence that says "fintfm never sees real data" without
the qualifier becomes false the moment FinTFM-R's first checkpoint exists. This is a
documentation tax with no end date, accepted knowingly; `docs/paper/CLAIMS.md` carries the
per-family form of the claim and is the ledger that has to hold the line.

**The benchmark cost, which is permanent and worth more attention than the naming.** Pretraining
FinTFM-R on real US SEC filer data **permanently forfeits the ability to evaluate any FinTFM-R
checkpoint cleanly on a US-filer benchmark.** The two benchmarks this project actually uses stay
safe, and that is not luck — it was checked before the source was chosen:

- **V4FinBench** is Visegrád-group firms (CZ/HU/PL/SK, 2006-2021). Different jurisdiction,
  different firm population, different source. Disjoint from SEC EDGAR.
- **TabArena**'s two corporate-bankruptcy datasets are `polish_companies_bankruptcy` and
  `taiwanese_bankruptcy_prediction`; its credit datasets (`GiveMeSomeCredit`, `credit-g`,
  `credit_card_clients_default`, `heloc`) are all consumer credit. None are SEC filers.

So the forfeit is of a benchmark this project does not currently use and has never published
against. It is still a one-way door: a future US corporate-distress benchmark could not be used
to evaluate FinTFM-R, only FinTFM. **Any new benchmark adopted from here must be checked for
SEC-filer overlap before a FinTFM-R number is computed on it**, the same way licences are
checked separately from code (`CLAUDE.md`).

**The source, and why it was chosen (verified, not assumed).** SEC EDGAR's **Financial
Statement Data Sets** — XBRL numeric facts from the face financials of every SEC filer, January
2009 to June 2026, quarterly, with SIC industry codes. A US federal government work: public
domain, no stated copyright restriction, commercially usable, with no registration, agreement
or vendor relationship required. Labels come from the same archive: **8-K Item 1.03**, the
SEC-mandated bankruptcy-or-receivership disclosure, roughly 400 filings a quarter, joinable to
the financials by CIK. Features and labels from one public-domain source, in this project's own
domain.

**Two real sources are intended, and they start at different times for a non-technical
reason.** Freddie Mac's Single-Family Loan-Level Dataset (≈56M mortgages, 1999-2026, with
observed defaults and losses) is the second source, adopted deliberately rather than as an
afterthought: one EDGAR panel is a *single* task family, and what real-table pretraining buys
the peers is diversity across heterogeneous real tables, which a second genuinely different
credit domain (consumer mortgage against corporate) supplies. It is also the only one of the
two with observed, loan-level default outcomes rather than disclosure-derived events.

**But it cannot be touched until a commercial licensing agreement is executed.** Freddie Mac
requires registration on its Clarity Data Intelligence platform and a licensing agreement for
commercial use; free access is academic/non-commercial, which this project is not. That is a
human and legal action, not an engineering one, and **no Freddie Mac data may enter this
repository, any training run, or any derived artefact before it is signed** — the same rule
`CLAUDE.md`'s licensing boundary applies to every other third-party source. EDGAR therefore
starts immediately because it is public domain and needs no permission; Freddie Mac starts
when the agreement does.

The prior mixture carries **separate weights per real source** (`p_real_edgar`,
`p_real_mortgage`) from the first commit, so the second source drops in without rework and so
any published checkpoint can state exactly which real corpora it saw — which the per-family
claim above makes mandatory rather than merely tidy.

**Implementation shape, so the scope is not mistaken for larger than it is.** FinTFM-R is
expected to need **no parallel codebase**. `PriorConfig` already mixes task generators by
weight (`p_financial`, `p_scm`, `p_tree`, `p_task_family`, `p_regression`), and a real panel
enters as one more weighted source that *samples* in-context tasks rather than generating them.
`collate()`, the training loop, the checkpointing and the whole evaluation harness are reused
unchanged. That also means the synthetic-versus-real question is measurable by the same paired
-bootstrap V4FinBench comparison every other lever in §144-§148 was judged by — a fair fight on
instruments this project already trusts.

**Reversed if:** the EDGAR panel proves unusable in a way that is about the data rather than the
modelling — label censoring too severe to work around (a firm that stops filing entirely never
files an Item 1.03, so it is silently absent rather than labelled distressed), or a coverage
check showing the eligible firm-quarters are too few or too clean to constitute a rare-event
panel comparable to the regime this project targets. Either would send the search back to a
licensed vendor panel, which is D14's original reversal condition and its original blocker.
