# Architecture

How the system is put together and why each piece is shaped the way it is. For *what* is
being built and in what order, see `docs/roadmap/STRATEGY.md`; for decisions and the alternatives
that were priced against them, `docs/design/DECISIONS.md`; for measured throughput,
`docs/infra/COMPUTE.md`.

## The core idea in one paragraph

A conventional model is fitted to your data. This one is fitted to a *prior* — a generative
story about what company financial tables look like and how they relate to default — and is
then handed your table as **context** at inference time. It answers by pattern-matching the
new table against the millions of synthetic problems it saw in pretraining. By default there
is no gradient step on customer data. That is what "in-context" means here, and it is why
`fit()` is nearly free while pretraining is expensive. The exception is opt-in:
`finetune_steps > 0` (default 0, §163) takes AdamW steps on a copy of the checkpoint inside
`fit()`, and a model fine-tuned that way is no longer the zero-shot FinTFM.

## Package layout, which follows the pipeline

```
src/fintfm/
  config.py       layered configuration; packaged defaults in configs/default.yaml
  provenance.py   what produced a checkpoint, dump or run, recorded beside it (DR1.2)
  prior/          task generation for pretraining
    base.py         Task container, padding and batching
    financial.py    the financial prior: balance sheets, P&L, macro regime, default labels
    scm.py          a generic structural-causal-model prior (random MLP graphs)
    tree.py, task_families.py, crossed.py, trivial.py, learnability.py
                    further synthetic generators, probes and the learnability filter
    gated.py, knobs.py, levers.py, dump.py
                    gated-label prior (DR2.2), opt-in knob distributions for the general priors
                    (DR2.3), command-line wiring for the levers, pre-generated batches on disk
    real_edgar.py, real_mortgage.py, real_panel.py
                    FinTFM-R ONLY: samplers over real panels (D17); weight 0.0 by default
    real_lendingclub.py
                    retired (D19): the sampler raises RetiredSourceError
    mixture.py      PriorConfig and sample_task: draws from all of the above by weight
  modeling/       architecture and pretraining
    model.py        FinancialTFM, the three-stage encoder
    hazard.py       the discrete-time hazard head for PD term structures
    train.py        one gradient step per freshly sampled batch; no epochs, no dataset
    loader.py       streaming pretraining batches, batch i from its own generator (DR5)
    muon.py, masked_column.py
                    opt-in Muon optimiser (AU5.17) and the masked-column objective (48.15)
  inference/      in-context prediction
    classifier.py   sklearn-compatible API, context construction, calibration correction
    retrieval.py    query-conditioned context selection (opt-in)
    categorical.py  out-of-fold target encoding for categorical columns
    preprocess.py   feature transforms (rank by default; none, winsor, power, squash)
    large_n.py      PartitionedTFMClassifier: a shallow tree routes rows to leaves, each with
                    its own context (§169); also a boosting wrapper
    regressor.py, binning.py, quantile_regressor.py
                    regression through the binned head, or a pinball-loss quantile head
  evaluation/     real data and honest scoring
    datasets.py, fetch.py   real credit panels for evaluation
    metrics.py      calibration as a first-class citizen, not just AUC
    bench.py, boosting.py   harnesses against classical baselines (boosters out of process)
    splits.py       time-based splitting that refuses a shared identifier
    overlap.py, proxy_suite.py, credit_suite.py
                    overlap check against TabArena and BeyondArena, the proxy Elo suite
                    (DR4.2) and the credit suite outside V4FinBench (DR4.3)
  experiments/    designed experiments with pre-stated exit conditions
```

The dependency direction is one-way down that list: `prior/` knows nothing about models, and
`modeling/` knows nothing about real datasets.

**Which family a checkpoint belongs to decides the provenance claim** (D17). FinTFM is trained
with every `p_real_*` weight at 0.0 and `real_panels` empty, so no real panel is opened during
its pretraining; that is the claim in `docs/results/FINDINGS.md` §1. FinTFM-R checkpoints set
those weights and do see real data, through the `prior/real_*.py` samplers only. Real datasets
are otherwise loaded in `evaluation/` and in `experiments/openml_breadth.py`, for scoring:

```bash
grep -rn "fetch_openml\|read_csv\|read_parquet\|urlopen" src/fintfm/
# expected: evaluation/, experiments/openml_breadth.py, and prior/real_*.py (FinTFM-R only)
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
`financial.py` deliberately generates missingness that correlates with distress. The z-score is
`normalize_features` in `modeling/model.py`, with the result clipped to ±10; a conditioning step
in `inference/preprocess.py` (`feature_transform`, below) runs before it at inference.

### Stage 1 — cell embedding and column attention

Each cell becomes a token from its normalised value and its missingness flag. Cells within
a row then attend to each other, so a leverage ratio is interpreted in the light of the
other ratios of the same company.

**No fixed feature-index embedding is added, deliberately**, because tables have no canonical
column order. Instead each column gets a **random per-task identity** (D12, §54): a random
vector of width `column_id_dim` (default `d_cell // 4`), projected by a learned linear layer
and added to every cell of that column. It is redrawn for every training task, so no column
index acquires a fixed meaning, and seeded at inference, so predictions are deterministic.
Without it the encoder cannot represent "column j matters" at all (§54).

The cost is that the invariances are **distributional, not exact**. Both are asserted in
`tests/test_model.py`, exactly at `column_id_dim=0` and in distribution otherwise:

- **Column-order invariance.** Permuting features changes a single prediction slightly; the
  average over `n_ensemble` identity draws converges at about 1/sqrt(K), measured at 0.018
  absolute probability for K=1 and 0.0023 for K=64.
- **Padding-width invariance.** A special case of the same permutation, with the same
  guarantee.

**Optional two-way cell attention** (`n_cell_blocks > 0`, §78). Before pooling, each block
first lets a cell attend to the same feature's cells in other rows (under the row mask below),
then runs the column attention above. With `cell_labels`, context cells also receive their
row's label and query cells a learned mask token. The default `ModelConfig` has
`n_cell_blocks=0`; the published reference recipe uses 1 block with cell labels (§91 found
both are needed together).

The first design, one `Linear(2·max_features → d_model)` over a flat row vector, gave every
feature a fixed weight slot and made column 7 of one dataset share weights with column 7 of
an unrelated one. That was the reason for the rewrite (D4).

### Stage 2 — row compression, then row attention

Cells are pooled into one vector per row: a masked mean, which is order-*invariant*,
concatenated with a masked max, which preserves the one extreme ratio in an otherwise
ordinary firm that a mean washes out (`pooling="meanmax"`, the default). `pooling="attention"`
adds a learned query attending over the feature tokens as a third component (§50; §52 found
it changed nothing). Context rows then receive an embedding of their label; query rows
receive a learned "unknown" token instead.

Rows attend under a mask with three rules:

- Every row may attend to **context rows** — this is where the in-context learning happens.
- Every row may attend to **itself**, so a query's own features reach the output directly
  rather than only through the residual path. This leaks nothing, since a query carries no
  label.
- No row may attend to **another query**. A prediction therefore never depends on which
  other rows happen to be in the same batch, which is also asserted in the tests.

The head depends on `head_type`:

- `"binned"` (default): a linear head producing logits over `max_classes` (default 10), with
  classes a task does not have masked to `-inf`. Used for classification and, through
  quantile bins, for regression (below).
- `"quantile"`: a linear head over `n_quantiles` (default 99) evenly spaced quantile levels,
  trained by pinball loss (§148 found it does not beat the binned head).
- A separate **hazard head** (`modeling/hazard.py`, built when `n_horizons` is set) emits a
  per-horizon hazard path for PD term structures. A checkpoint records which objectives were
  trained and refuses to serve an untrained head (D11).

`ModelConfig` defaults: `max_features=24`, `max_classes=10`, `d_cell=64`, `d_model=192`,
`n_heads=4`, `n_col_layers=2`, `n_layers=6`, `d_ff=512`, `dropout=0.0`. The published binary
checkpoint uses a different recipe (`d_cell=48`, `d_model=128`, `n_layers=4`,
`max_features=136`, one cell-attention block with `cell_labels`, `column_id_dim=16`, 885,650
parameters, built and counted in `docs/design/attention_cost_model.md`); read a checkpoint's own
stored config rather than these defaults.

**Structural options added since this document was written**, all off by default so that
every earlier checkpoint reproduces (`tests/test_golden_outputs.py` pins the published one bit
for bit). None has been promoted. Screening results in `docs/results/FINDINGS.md`: `cell_row_attn="induced"` and
`readout="cls"` harmful at one seed (§178), the RBF value embedding harmful (§213), the mask
embedding harmful (§147, §160). No result is recorded there for the other fields.

| `ModelConfig` field | default | what it does |
| --- | --- | --- |
| `n_summary_tokens` | 1 | K learned queries cross-attend over a row's feature tokens, replacing mean and max when above 1 (AU5.3) |
| `readout`, `n_readout` | `"pool"`, 4 | `"cls"`: learned readout tokens join column attention and their outputs are the row vector (DR6.3, §178) |
| `head_mlp` | `False` | an MLP head in place of the linear one (DR6.3) |
| `cell_row_attn`, `n_inducing` | `"full"`, 64 | `"induced"`: inducing points in the row-within-feature stage, linear in rows (DR6.2, §178) |
| `cell_mixing` | `"attention"` | `"latent"`: row and column latents replace both cell stages (48.27; no result yet) |
| `value_embedding`, `value_basis_size`, `value_basis_learnable`, `value_ecdf` | `"mlp"`, 16, `False`, `False` | `"rbf"` or `"fourier"` value embeddings, and a channel with each cell's rank within its column's context rows |
| `attn_length_scale`, `attn_length_ref`, `attn_impl` | `False`, 512, `"masked"` | row-attention logits scaled by log(n_ctx)/log(attn_length_ref) (AU5.17); `"split"` computes the same attention without a boolean mask so fused kernels apply (AU5.2) |
| `mask_embedding` | `False` | a dedicated missingness embedding |
| `hazard_period_context` | `False` | period labels as hazard-head context; needs `n_horizons` |

`ModelConfig.__post_init__` refuses incompatible combinations (for example `cell_mixing="latent"`
without cell blocks, or a quantile head with a hazard head).

## Why cross-entropy is the training loss

Cross-entropy is a *proper scoring rule*: it is minimised only by honest probabilities, not
merely by correct rankings. That matters more here than in most classification work, because
the deliverable is a probability of default that a lender prices and provisions against. A
loss that rewarded ranking alone would optimise for the one number the buyer cannot use.

## Inference: three corrections that are easy to get wrong

`FinancialTFMClassifier.fit()` stores the table rather than training on it (unless
`finetune_steps > 0`). Three things then happen that are not obvious:

**Context construction.** Attention is quadratic in context length, so tables above
`max_context` (default 2,000 rows) must be subsampled. The strategies are `uniform` (the
default, D10), `balanced`, `hybrid`, `prototype` (the published global prototype context,
§38) and `retrieval` (query-conditioned, opt-in because it makes a prediction depend on its
query group-mates). The project first adopted `balanced` from the literature (§5); measured on
V4FinBench out of time, uniform beat it by 10-12 AUC points (§29; the three-seed margins in §33
are 8-11 points), and on the binary path the two are nearly tied (§35). §84 measured the cell-attention gain at 6-7 times any context
effect on the full panel, so context is a second-order lever there (the architecture arms may
also differ in prior mix; audit task AU2.8). (2026-11-13, §85: §84's reading that cell attention
prefers less context did not reproduce on a probe with a known optimum.) Resolved 2026-10-18 (§172): at a matched prior the
cell-attention gain is +0.0520 mean AP at the default context, 5 of 5 folds.

**Context and inference options added since** (all off by default; defaults read from
`FinancialTFMClassifier.__init__` and `configs/default.yaml`, and held equal by
`tests/test_config.py`):

- `context_positive_floor` and `context_positive_cap` (defaults `None`) fix the positive count or
  bound the positive share of a uniform context. DR3.3 measured the floor as null (§200).
- `feature_transform` takes `none`, `winsor`, `rank` (default), `power` or `squash`. `squash`
  (robust scaling then a soft clip) was the one large move on TabArena-Lite (§210); the default
  stays `rank`.
- `n_ensemble` (default 1), `ensemble_label_swap`, `ensemble_feature_frac`,
  `ensemble_transforms`, `ensemble_svd_components`, `ensemble_class_shift`. Eight members against
  one was null on V4FinBench (§196); the class-shift arm stopped for futility (§217).
- `max_features_strategy` is `raise` (default), `subsample` or `select`, with `select_method`
  `extra_trees` (default) or `mutual_info`, for tables wider than the checkpoint (DR8.3).
- `retrieval_per_class` (default `None`) was harmful (§213).
- `finetune_*` options include inner-validation early stopping (`finetune_val_frac`, default 0.0).
- `calibration_val_frac` (default 0.0) holds out a slice, fits one temperature by log loss and
  applies it in `predict_proba` (openspec `fitted-calibration` 4.1, 4.2).
- `fit(n_observed=...)` carries each row's observation count; `context_period_` and
  `context_n_observed_` return copies aligned with `context_indices_` (0.10.0).
- `PartitionedTFMClassifier` (`inference/large_n.py`; `max_depth=2`, `n_bags=1`) gives each leaf
  of a shallow tree its own context. Zero-shot on V4FinBench's five full-panel folds it lifted
  mean AP from 0.1986 to 0.2362, 5 of 5 folds positive (§169, MEASURED, no training).

**Base-rate correction.** Any strategy that rebalances the context *lies to the model about
how common default is*, because an in-context learner reads the base rate out of its
context — measured at a 14.9% predicted mean against a 4.7% actual rate. The fix shifts the
logits by `log P_true(y) − log P_context(y)`, which is exact under label shift (guaranteed
for `balanced` and `hybrid`, whose selection looks only at `y`; `prototype` and `retrieval`
also select on features, so there it is an approximation) and provably leaves ranking
untouched. See
`docs/results/FINDINGS.md` §6.

**Categorical encoding** (`inference/categorical.py`). Every stage above reads a cell as an
*ordered numeric scalar*, so a categorical column has to arrive as a number and the choice of
number is a modelling decision, not plumbing. Label encoding asserts that resource code 4127
lies between 4126 and 4128; `docs/results/FINDINGS.md` §100 measured what that costs — a −0.0894
ROC-AUC deficit against tuned logistic regression on mostly-categorical TabArena datasets,
against −0.0320 on numeric ones, correlating
−0.668 with log cardinality. The replacement gives each level the smoothed target rate among
context rows carrying it, which is ordered on the axis the model actually reads. Applied to
TabArena-Lite's 27 datasets it raised mean ROC-AUC by 0.0181 and took the correlation
with log maximum cardinality from −0.668 to −0.025 (§101).
`CategoricalTargetEncoder` is a standalone class (defaults `smoothing=10.0`, `n_folds=5`): the
classifier does not call it, so the caller encodes the columns first, as the README says.
`unseen_level_fraction` reports the share of query cells whose level the context never saw.

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

**What the encoder does with a non-binary target** is the same decision one level up. A rate
is a mean of an indicator, so a *continuous* target takes the identical out-of-fold treatment
with the level's smoothed mean in place of its smoothed rate. An *integer class code* does
not: "class 3" is a name, so a mean taken against those codes is §100's mistake moved to the
target side of the problem, and multiclass falls back to frequency encoding instead. The line
is quantity versus name, not binary versus not.

## Regression is the classification head, read differently

`inference/regressor.py` adds no architecture. `FinancialTFM.head` is already
`nn.Linear(d_model, max_classes)` trained by cross-entropy on an integer index, and a
continuous target cut into `K` quantile bins **is** an integer index over `K` outcomes — so
the head, the loss and the per-cell label injection all work unchanged.
`FinancialTFMRegressor` (`n_bins=10` by default, classifier settings through
`classifier_params`) bins on fit and takes the predicted distribution's mean on predict.

Bin edges come from the **context** targets' quantiles, per task, which buys three things at
once: scale invariance (the model never sees the target's units), balanced occupancy (no bin
starved of signal), and no query leakage (edges fitted where the labels are allowed to be
seen). The cost is that bin widths vary, so the point estimate weights *within-bin means*
rather than evenly spaced centres — the estimator that stays correct when the outer bins are
unbounded, which for financial targets is the usual case.

**The point estimate is the least interesting output.** The head emits a distribution, so
`predict_quantile` and `predict_interval` come free, and the distribution can be multimodal —
which is precisely what loss given default is, piling up at both ends of `[0, 1]`, and
precisely what a Gaussian head cannot represent. The limits are equally structural:
resolution is bounded by `K` no matter how confident the model is, an interval can never be
narrower than one bin, and a prediction is a convex combination of representatives so it
cannot leave the training target's range. `binning.interval_coverage` exists to measure the
first of those rather than assume it.

## Pretraining loop

There is no dataset and there are no epochs. Each step samples a fresh batch of synthetic
tasks, takes one gradient step, and discards them. (Since DR5, `modeling/loader.py` can stream batches from per-batch seeds
in worker processes, and `prior/dump.py` can pre-generate them; `fintfm-train` also offers an
opt-in Muon optimiser, harmful in §211.) Data is effectively infinite, so dropout
defaults to zero — there is nothing to overfit. The context/query split point is re-drawn
per batch so the model learns to work at many context sizes rather than one.

## Invariants that must not be broken

1. **No real data in FinTFM pretraining.** It is the basis of the auditability claim and it
   can be destroyed silently, by a well-meaning change that improves the benchmarks. FinTFM-R
   (D17) trains on real panels by design; every checkpoint, result and claim must say which
   family it belongs to.
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
