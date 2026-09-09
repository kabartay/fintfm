# Changelog

Hard-wrapped, because it is read in an editor and a diff. Release bodies on GitHub are
**not** wrapped — they are read in a browser at full width. Same words, different shape; do
not paste one into the other. See `CLAUDE.md`.

## [Unreleased]

### Fixed

- **The term-structure path skipped the base-rate correction**, so out-of-time evaluation
  stated a 12.8% default probability against a 0.47% observed rate. The out-of-time harness
  built its own forward pass through `FinancialTFM.term_structure` and reached into the
  fitted estimator's private context, bypassing `predict_proba` — the only place decision
  D5's correction was ever applied. Fourth-horizon calibration error falls **32×** (0.3677 →
  0.0114) on the same checkpoint and the same split, and mean AUC is unchanged to four
  decimals, as the rank-preservation property requires. `docs/FINDINGS.md` §28.
- `pyarrow` was declared only in the `kaggle` extra, so `uv sync --extra bench` broke the
  V4FinBench loader that every out-of-time finding depends on. CI never caught it because CI
  has no data and the test skips.

### Added

- **A feature conditioner** (`rank` and `winsor`), applied before the model's normalisation.
  Worth **+0.086 mean AUC** to uniform context and +0.023 to retrieval on the V4FinBench
  out-of-time split, and improving AUC in seven of eight configurations across two further
  panels. 110 of 136 features here have a standard deviation more than ten times their
  interquartile range, which is what the model's mean/standard-deviation normalisation could
  not survive. `docs/FINDINGS.md` §35.
- **Checkpoints record which objectives they were trained on**, and a head that was never
  trained can no longer be served. The training loop optimises one objective per step, so a
  hazard checkpoint's classification head sat at random initialisation and `predict_proba`
  reported AUC 0.3745 with a stated 69% default rate against a 4.7% base — no error raised.
  §34, decision D11.
- **A `retrieval` context strategy**, building each query group's context from its nearest
  training rows instead of a blind sample. **+0.066 to +0.095 AUC over the best blind
  strategy** on the V4FinBench out-of-time split, Holm-significant at three of four horizons
  — the first accuracy gain in this project to survive a family-wise correction. Costs about
  2.5× the scoring time and gives up batch independence: a query's prediction depends on its
  group-mates. `docs/FINDINGS.md` §32.
- `holm_adjusted_p`, beside `holm_bonferroni`, which returns booleans. Reading those booleans
  as p-values inverted every verdict in the first write-up of §32.
- `FinancialTFMClassifier.predict_term_structure`, the corrected public path for a PD term
  structure, chunked exactly and coherence-preserving. Calling
  `FinancialTFM.term_structure` directly is now a documented defect.
- `fintfm.modeling.hazard.base_rate_shift` and `shift_cumulative_pd`, with monotonicity and
  cross-row rank preservation asserted rather than argued.
- `fintfm-ctxsweep`, sweeping context strategy against context size out of time. It exists
  because §29 reverses a decision, and a decision reversal has to be re-runnable.
- The out-of-time harness scores three hazard arms — corrected, **uncorrected**, and
  uniform-context — and prints mean predicted PD beside the observed rate. The uncorrected
  arm is permanent: the distortion is measured beside the fix rather than assumed absent.

### Changed

- **Defaults now follow measurement rather than the literature** (decision D10):
  `feature_transform="rank"` and `context_strategy="uniform"`, replacing `"balanced"`.
  Retrieval is the best strategy on all three panels and both prediction paths but stays
  opt-in, because it costs ~2.5× the scoring time and makes a prediction depend on its query
  group-mates. Every number recorded before 2026-09-09 used `balanced` and no transform.
- **Best out-of-time result: mean AUC 0.5869 → 0.8118** over the day, closing the gap to
  per-horizon logistic regression from 0.142 to 0.048, from three inference-time changes and
  **no retraining**. Still behind on discrimination and calibration.
- **Uniform context sampling beats balanced by 10-12 mean AUC points** on the V4FinBench
  out-of-time survival split, at every context size tested, reversing the ordering this
  project adopted from the literature. Twelve in-context defaults outrank 1,122. The class
  default is unchanged pending re-measurement on the binary path (decision D9).
- The base-rate correction is **not applied per query group** under retrieval. It is exact
  under label shift, which holds by construction only when context selection looks at `y`
  alone; retrieval selects on features. Applied per group it pushed the riskiest clusters
  down hardest and drove mean AUC to **0.3679, below chance**. A single pooled shift replaces
  it, which cannot reorder anything.
- §31's "more context does not help" is **corrected twice**: retrieved rows do help, but on
  three seeds retrieval is tied at 2,000 and 4,000 rows, so it raises the plateau's height
  rather than moving where it starts. 2,000 is the operating point; §32's monotone rise was a
  single favourable draw (§33).
- §26's headline is **retracted**: the synthetic prior's 1% base-rate floor was real and is
  now 0.195%, but it was never what caused the out-of-time failure. The retrain it prompted
  is worth +0.023 mean AUC and 2.7× better calibration, which the broken evaluation
  configuration had hidden. `docs/FINDINGS.md` §30.

## [0.1.0] — 2026-09-08

First release. A working prior-fitted tabular model for corporate credit risk, evaluated on
real corporate defaults, plus the strategic and evidentiary scaffolding around it.

### Model

- Three-stage architecture: cells are embedded individually, attend across columns within a
  row, then pool into one vector per row before rows attend to labelled context rows.
- **Column-order invariance and padding-width invariance**, both asserted in tests. The
  first architecture used a flat `Linear(2·max_features → d_model)`, which gave every
  feature a fixed weight slot and broke under permutation. That was the main limit on
  transfer and the reason for the rewrite.
- Query rows attend to context rows and to themselves, never to each other, so a prediction
  never depends on which rows share its batch.
- Normalisation uses context-row statistics only, so a query cannot influence its own
  normalisation.

### Priors

- A financial prior generating company balance sheets, P&L, a sampled macro regime, sector
  effects, distress-correlated missingness and default labels at sampled base rates from 1%
  to 30%.
- A generic structural-causal-model prior for nonlinear and multiclass structure.
- **Pretraining is entirely synthetic.** No real data can reach it, which is what makes a
  benchmark number auditable rather than merely asserted.

### Evaluation

- Real corporate-default evaluation on the UCI Polish companies bankruptcy panels, at 1, 3
  and 5 year horizons, CC-BY-4.0, evaluation only.
- **Calibration is a first-class metric.** AUC is rank-only and cannot see a stated 2%
  probability of default that occurs 15% of the time. Every result also carries Brier score,
  expected calibration error, reliability bins and recall at a base-rate operating point.
- Benchmark harness against logistic regression, random forest, gradient boosting and
  LightGBM.
- Phase 1 ablation harness that enforces matched compute across prior variants and refuses
  to report if parameter counts diverge.

### Findings

Six numbered findings in `docs/FINDINGS.md`, of which two changed the code:

- Balanced context sampling improves ranking but **inflated predicted default rates roughly
  threefold** (14.9% against a 4.7% actual rate), because an in-context model reads the base
  rate out of its context. AUC cannot see this, since every prediction inflates alike.
  Corrected by an exact label-shift adjustment to the logits, which leaves ranking untouched
  and reduced expected calibration error from 0.102 to 0.007 at the 3-year horizon. The
  1-year horizon, where it slightly overcorrects, is recorded rather than omitted.
- Firm-level financial data is absent from every major foundation model's pretraining corpus,
  because it sits behind commercial licences. Energy got foundation models because its data
  was free. That makes synthetic generation the only licence-clean route into this domain
  rather than a budget substitute.

### Documentation

- `docs/STRATEGY.md`, the plan of record, with a falsifiable exit condition per phase.
- `docs/ARCHITECTURE.md`, `docs/DECISIONS.md` with seven decisions and what would reverse
  each, `docs/COMPUTE.md` with measured throughput, `docs/LANDSCAPE.md`, and
  `docs/REFERENCES.md` where every entry was verified against its source.
- `openspec/` with nine proposals and 42 tasks, and `docs/NEXT.md` as the tiered queue.

### Known gaps

- **No CI.** Tests pass locally, on a tree that carries a cached download the repository does
  not. Covered by `openspec/changes/ci-and-release-gates`, including a retrospective check of
  this release.
- **All splits are random, not time-based.** Optimistic for credit data in exactly the way a
  validation committee looks for. No headline accuracy number should be published before
  `time-based-evaluation` lands.
- **One dataset.** Every real-data number rests on Polish firms in 2000-2013.
- **No result yet on the central question.** Whether the financial prior beats a generic one
  at matched compute is unmeasured; the run was in flight when this was tagged. Every number
  in the repository so far is smoke-test scale.
