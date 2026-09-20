# Architecture

How the system is put together and why each piece is shaped the way it is. For *what* is
being built and in what order, see `docs/STRATEGY.md`; for decisions and the alternatives
that were priced against them, `docs/DECISIONS.md`; for measured throughput,
`docs/COMPUTE.md`.

## The core idea in one paragraph

A conventional model is fitted to your data. This one is fitted to a *prior* — a generative
story about what company financial tables look like and how they relate to default — and is
then handed your table as **context** at inference time. It answers by pattern-matching the
new table against the millions of synthetic problems it saw in pretraining. There is no
gradient step on customer data, ever. That is what "in-context" means here, and it is why
`fit()` is nearly free while pretraining is expensive.

## Package layout, which follows the pipeline

```
src/fintfm/
  prior/          synthetic task generation — the ONLY source of pretraining data
    base.py         Task container, padding and batching
    financial.py    the financial prior: balance sheets, P&L, macro regime, default labels
    scm.py          a generic structural-causal-model prior (random MLP graphs)
    mixture.py      draws from both; p_financial is the variable Phase 1 tests
  modeling/       architecture and pretraining
    model.py        FinancialTFM, the three-stage encoder
    train.py        one gradient step per freshly sampled batch; no epochs, no dataset
  inference/      in-context prediction
    classifier.py   sklearn-compatible API, context construction, calibration correction
  evaluation/     real data and honest scoring
    datasets.py     real credit panels — EVALUATION ONLY, never pretraining
    metrics.py      calibration as a first-class citizen, not just AUC
    bench.py        harnesses comparing against classical baselines
  experiments/    designed experiments with pre-stated exit conditions
    prior_ablation.py   Phase 1: does the financial prior beat a generic one?
```

The dependency direction is one-way down that list. `prior/` knows nothing about models,
`modeling/` knows nothing about real datasets, and `evaluation/` is the only place a real
dataset is ever loaded. That last boundary is not stylistic — it is what makes the
provenance claim in `docs/FINDINGS.md` §1 checkable by grep:

```bash
grep -rn "fetch_openml\|read_csv\|urlopen" src/fintfm/   # evaluation/ only, never prior/
```

## Model: three stages

Input is a single tensor `(B, N, F)` — batch of tasks, rows per task, features — with `NaN`
marking both genuinely missing cells and padding out to `max_features`. The first `n_ctx`
rows are labelled context; the rest are queries.

### Stage 0 — normalisation using context statistics only

Every feature is z-scored using the mean and standard deviation of the **context rows
alone**. Using all rows would let a query row influence its own normalisation, which is a
quiet form of test-set leakage. Missing cells are set to zero *and* flagged, so "missing"
and "average" are distinguishable — in credit they mean very different things, and
`financial.py` deliberately generates missingness that correlates with distress.

### Stage 1 — cell embedding and column attention

Each cell becomes a token from its normalised value and its missingness flag. Cells within
a row then attend to each other, so a leverage ratio is interpreted in the light of the
other ratios of the same company.

**No feature-index embedding is added, deliberately.** A column's identity has to come from
its data distribution, not from its position, because tables have no canonical column order.
The consequence is a property asserted in the test suite:

- **Column-order invariance.** Permuting features cannot change any prediction.
- **Padding-width invariance.** A 5-feature table scores identically whether padded to 8
  columns or 64, and wherever the padding sits.

The previous design — one `Linear(2·max_features → d_model)` over a flat row vector — failed
both by construction, giving every feature a fixed weight slot and making column 7 of one
dataset share weights with column 7 of an unrelated one. That was the single biggest limit
on transfer and the reason for the rewrite.

### Stage 2 — row compression, then row attention

Cells are pooled into one vector per row: a masked mean, which is order-*invariant*,
concatenated with a masked max, which preserves the one extreme ratio in an otherwise
ordinary firm that a mean washes out. Context rows then receive an embedding of their label;
query rows receive a learned "unknown" token instead.

Rows attend under a mask with two rules:

- Every row may attend to **context rows** — this is where the in-context learning happens.
- Every row may attend to **itself**, so a query's own features reach the output directly
  rather than only through the residual path. This leaks nothing, since a query carries no
  label.
- No row may attend to **another query**. A prediction therefore never depends on which
  other rows happen to be in the same batch, which is also asserted in the tests.

A linear head produces logits over `max_classes`, with classes a task does not have masked
to `-inf`.

## Why cross-entropy is the training loss

Cross-entropy is a *proper scoring rule*: it is minimised only by honest probabilities, not
merely by correct rankings. That matters more here than in most classification work, because
the deliverable is a probability of default that a lender prices and provisions against. A
loss that rewarded ranking alone would optimise for the one number the buyer cannot use.

## Inference: three corrections that are easy to get wrong

`FinancialTFMClassifier.fit()` stores the table rather than training on it. Three things then
happen that are not obvious:

**Context construction.** Attention is quadratic in context length, so tables above
`max_context` must be subsampled. Doing that *uniformly* on a 4% default rate throws away
almost every defaulter before the model sees one, and context strategy turns out to explain
more variance in AUC than the choice of model family (`docs/FINDINGS.md` §5). Balanced and
hybrid strategies keep the minority class.

**Base-rate correction.** But rebalancing the context *lies to the model about how common
default is*, because an in-context learner reads the base rate out of its context — measured
at a 14.9% predicted mean against a 4.7% actual rate. The fix shifts the logits by
`log P_true(y) − log P_context(y)`, which is exact under label shift (guaranteed here,
because context selection looks only at `y`) and provably leaves ranking untouched. See
`docs/FINDINGS.md` §6.

**Categorical encoding** (`inference/categorical.py`). Every stage above reads a cell as an
*ordered numeric scalar*, so a categorical column has to arrive as a number and the choice of
number is a modelling decision, not plumbing. Label encoding asserts that resource code 4127
lies between 4126 and 4128; `docs/FINDINGS.md` §100 measured what that costs — a −0.0894
ROC-AUC deficit on mostly-categorical datasets against −0.0320 on numeric ones, correlating
−0.668 with log cardinality. The replacement gives each level the smoothed target rate among
context rows carrying it, which is ordered on the axis the model actually reads.

**The out-of-fold part is the whole difficulty, and it is an invariant.** If a row's own label
enters its own encoding, then on a high-cardinality column — where most levels appear once —
the encoded value is very nearly the label. The column becomes an almost perfect predictor
*inside the context* and carries nothing at query time, so the model learns to trust a feature
that will not be there. **This leak makes the model worse rather than flattering the score,
which is exactly why it survives careless validation.** Context rows are therefore encoded
from a K-fold partition excluding their own fold; query rows use the full context, having no
labels to leak.

These three interact, and none is visible in an AUC-only evaluation. That is the whole
argument for `evaluation/metrics.py`.

## Pretraining loop

There is no dataset and there are no epochs. Each step samples a fresh batch of synthetic
tasks, takes one gradient step, and discards them. Data is effectively infinite, so dropout
defaults to zero — there is nothing to overfit. The context/query split point is re-drawn
per batch so the model learns to work at many context sizes rather than one.

## Invariants that must not be broken

1. **No real data in pretraining.** It is the basis of the auditability claim and it can be
   destroyed silently, by a well-meaning change that improves the benchmarks.
2. **The financial prior stays parametric.** It samples a macro regime rather than learning
   real crisis history. Conditioning the generator on a real panel would reintroduce exactly
   the global-pattern memorisation the leakage literature warns about.
3. **Trained weights never enter git.** `.gitignore` excludes `*.pt`.
4. **Every reported number carries how it was produced** — SMOKE-TEST, MEASURED, or
   SIMULATED (`CLAUDE.md`). In an ML repo a wrong number does not crash; it just looks like
   a result.
5. **Target encoding is out of fold for context rows.** Pinned by
   `tests/test_categorical.py`, which asserts the contrast rather than the property alone:
   the naive statistic correlates above 0.9 with the row's own label on a column of unique
   levels while the out-of-fold statistic is constant at the prior. Without that contrast the
   test would also pass for an encoder that encodes nothing, so a second test requires it to
   recover a real level effect.
