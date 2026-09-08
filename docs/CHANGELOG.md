# Changelog

Hard-wrapped, because it is read in an editor and a diff. Release bodies on GitHub are
**not** wrapped — they are read in a browser at full width. Same words, different shape; do
not paste one into the other. See `CLAUDE.md`.

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
