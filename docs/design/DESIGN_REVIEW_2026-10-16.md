# Design review, 2026-10-16

Six reviews of FinTFM's design (architecture, synthetic prior, training recipe and compute,
inference and data processing, evaluation strategy, code architecture), run on 2026-10-16 after
the audit of 2026-10-15 had fixed the bugs it found. Each review read the code, the findings log
and the peers' primary sources; the claims below that carry weight were checked against the
code or the data before being written here. The tasks they lead to are in
[`openspec/changes/design-review-2026-10-16/`](../../openspec/changes/design-review-2026-10-16/).

Every figure is labelled. **MEASURED** means produced here from code or saved runs;
**ESTIMATED** means arithmetic or a reading of a peer's paper, not a run of ours.

## The result in one paragraph

FinTFM's macro-design is the one the board's leaders use, and its biggest problems are not
there. On V4FinBench the label sits almost entirely behind one gate that a uniform context
wastes; on TabArena the model has seen about a hundredth of a peer's training volume, compresses
each row too early and pads half of every task; and the evaluation loop that should catch this
steers on a benchmark subset that overlaps TabArena, with a resubmission bar set below three
simple baselines. One past verdict that shaped the prior is confounded: §116's "the tree prior
hurts credit" compared arms with different financial shares. Several integrity gaps in the code
would let the next GPU runs repeat that kind of error silently, so they come first.

## 1. Findings, ranked by what they cost

### 1.1 V4FinBench's label is gated, and the context ignores the gate (MEASURED)

3,586 of V4FinBench's 3,587 positives have `Insolvency_flag = 1` (negative equity). The default
rate is 3.39% inside the flagged 10.6% of rows and 1 in 894,400 outside them
(`company_years_h1.parquet`, horizon 0). A uniform 2,000-row context therefore spends about 89% of
its rows on near-certain negatives and holds about 7 positives; a tree isolates the gate with one
split. This is a plausible large share of CatBoost's lead (0.4005 against 0.2062 mean AP, §163).

Consequences: context composition, not size, is the binding constraint (§162 measured size);
`ensemble` and `max_context` changes cannot fix it; partitioned contexts (AU5.8,
`PartitionedTFMClassifier`, built 2026-10-16) can, raising the flagged leaf's context to about 68
positives (ESTIMATED). Every V4FinBench report should add AP within the flagged rows beside the
headline AP, since the headline is in effect a ranking inside that 10.6%.

Related data facts (MEASURED): every positive is its company's final row; the base rate drifts
from 0.05% (2006) to 0.66% (2019). The review's further figure of 8,171 rows sharing an
identical financial vector with another company's row (6 positives) did not reproduce: on the
harness's 130 features it is 1,081 rows and 0 positives, and 1,081 to 4,933 rows with 0 or 1
positive across eleven column sets (§168). Do not quote 8,171.

### 1.2 §116's tree-prior loss is confounded with the financial share (MEASURED)

`prior/mixture.py` draws a task's source in a fixed order: tree, regression, trivial, crossed,
task families and real panels first, then financial, then SCM as the remainder. So
`p_tree 0.3, p_financial 0.7` yields financial 0.49, SCM 0.21 and tree 0.30, while the control
(`p_financial 0.7`) yields financial 0.70 and SCM 0.30. §116 described the two checkpoints as
"identical except for the prior mixture" and read the difference as the tree prior's effect.

Against `scm50` (financial 0.50, SCM 0.50), the arm with the matching financial share, the tree
arm is **+0.0118 mean AP** (fold differences −0.0028, +0.0191, +0.0133, +0.0207, +0.0089),
recomputed from the saved `runs/v4proto-{treeclean,scm50}` predictions (§166). §116, §118 and
§141 share one shape: moving 0.2-0.3 of the financial share elsewhere costs 0.022-0.034 AP,
whatever replaces it. The README's "the tree prior ships off because it hurts credit" rests on
the confounded comparison. The weights in every mixed recipe since need reading as conditional,
and future weights should be normalised and recorded in the checkpoint.

### 1.3 Training volume is about a hundredth of the field's (MEASURED for us, ESTIMATED for peers)

| model | tasks | parameters | tasks per parameter |
| --- | --- | --- | --- |
| FinTFM | 48,000 | 885,650 | 0.054 |
| LightPFN | 7.68M | 4.60M | 1.67 |
| TabICLv2 | about 35M | about 28M | about 1.25 |
| Mitra-v2 | 28.7M | 75.7M | 0.38 |

Peer figures are from their reports (`docs/research/REFERENCES.md`). LightPFN, one author, about
36 GPU-hours on two RTX 5090s, reaches 1322 Elo (1441 on classification only) and is the right
calibration point for this project's compute.

The cost is higher than `CLAUDE.md` states. Its "$13 per million tasks on a T4" predates cell
attention; the current recipe runs 1.90 s/step on a T4 (§86) and about 1.02 s/step on an L4
(§127), which is about 66 T4-hours or about $28 of L4 time per million tasks (ESTIMATED from
those MEASURED step times). Pre-generated task dumps do not scale (2M tasks is about 588 GB at
2.35 MB per batch); generation has to stream on the GPU host from per-batch seeds.

Every closed architecture verdict was measured at 48,000 tasks: scale (§114), depth (§124,
§126), volume at fixed size (§93), the `column_id_dim` peak (§104), pooling (§52, also before
§54's fix), context preference (§84, §85) and the prior verdicts (§141, §144, §145). They are
verdicts about this budget, not about the design.

### 1.4 Architecture: right pipeline, wrong compression (ESTIMATED effects)

The pipeline (cell embedding, alternating column and row attention with labels inside the
cells, compression, row-level in-context learning) is the one Kumo-Tabular, TabFM and TabICLv2
use. Labels in every context cell is TabICLv2's "early target inclusion" (about 100 Elo in its
ablation) and is already here (`cell_labels`, task 39.2). What differs:

- **Parameters sit where the compute is not.** 793,088 of 885,650 parameters (90%) are in the
  four-layer row encoder; the cell stage has about 80k at width 48, and every measured win came
  from the cell stage (§78, §80, §97, §104). Depth 4 to 16 costs 1.01x step time (§123).
- **Mean and max pooling to one 128-d row vector.** Peers read out with 4 to 8 learned tokens
  (TabICLv2 4 to 512-d, TabFM 8 to 1,024-d, LightPFN 4 to 256-d). `n_summary_tokens` (AU5.3)
  pools after the cell stage instead of taking part in feature attention, so it is not the
  peer design yet.
- **Full N x N attention across rows within each feature.** Peers gather from context cells
  through inducing points (TabICLv2 128, TabFM 256, LightPFN 64), so cost is linear in rows;
  ours puts 4,000-row contexts out of reach (§79: about 16 GB at 2,024 rows).
- **About half of every task is padding.** Every task pads to 136 columns while financial tasks
  draw 4 to 136, and within-feature attention runs over the padded columns too.

A sketched v2 (three grouped value slots with Fourier, ECDF and missingness channels; three
inducing-point cell blocks at width 64 with 4 readout tokens; a 6-layer row encoder at width 256
with a 2-layer head) is about 3.6M parameters and about 0.7-0.9 s/step on a T4 in fp16, against
1.90 s today (all ESTIMATED). It is to be tested as a bundle against v1 at matched tasks before
any part is credited.

### 1.5 The prior lacks the structures the benchmarks have

- **No gates.** The financial label is a sigmoid of a weighted driver sum
  (`prior/financial.py:475-535`); V4FinBench's is a gate then a ranking (1.1). Only the tree
  prior produces gates, which fits 1.2.
- **Feature counts are uniform on [2, 136] at every row count**, so the feature-to-row ratio
  reaches 0.53 at 256 rows; peers skew narrow.
- **SCM tasks are 89% multiclass**, and imbalance comes only from random cut points: in 5 of 20
  SCM and 7 of 20 tree tasks a class has two rows or fewer (MEASURED, 20 tasks per source).
- **Categoricals are ordered quantile codes with 2-7 levels**; peers permute codes and draw
  cardinalities up to 64-256 with Zipf frequencies.
- **No ties, point masses or zero inflation** in the general priors, which is what survives the
  rank transform; heavy tails do not.
- **Missingness** is MCAR in the general priors; the financial prior's is label-informative
  (11.7% of tasks, MEASURED, AU5.11).

The proposed prior v2 (a general trunk with normalised weights, plus a credit source with gated
labels) and its checks are in task DR2.

### 1.6 Inference and data processing

- **The V4 harness imputes and standardises before FinTFM sees the data**
  (`v4_protocol.py:466-468`), so V4FinBench never exercises the missingness mask. §147's reading
  that the joint embedding learns a value-and-missingness interaction has no support on this
  benchmark (MEASURED: no scored cell had missing = 1).
- **Every TabArena number to date was produced with NaN filled with 0** (the adapter's
  `X.fillna(0)`, removed on 2026-10-16 in the fork's `fintfm-audit-fixes` branch, unscored):
  §98, §101, §122, §132 and §150.
- **V4 scores with one ensemble member.** Eight members at 2,000 rows have never been scored
  five-fold on the full panel for the published checkpoint.
- **§130-§136's "the representation beats its own head" is confounded with label count.** The
  linear probe trains on about 2,150 positives; the head conditions on about 7.
- **Nominal codes enter as numbers** in V4 (NAICS codes up to 445,299; legal form, country), and
  `select_dtypes` silently drops the boolean `has_multiple_industries`, so the model sees 130
  features, not the 136 §35 states.

### 1.7 Evaluation steers on the wrong signals

- **The resubmission bar is too low.** ROADMAP Phase E resubmits on clearing Linear (default) at
  936 Elo; in the same pool Linear (tuned) is 966, Linear (tuned + ensembled) 984 and KNN
  (tuned + ensembled) 1008. The defensible bar is about 1,010, parity with RF (default), which is
  also the value TabArena imputes for a missing split.
- **The decision loop overlaps TabArena.** Six of the 15 datasets in
  `experiments/openml_breadth.py` are TabArena datasets (bank-marketing, churn,
  blood-transfusion, credit-g, qsar-biodeg, diabetes), and the Polish and Taiwanese panels are
  too (§165 used them). This is evaluation, not training, so D2 is intact, but choosing arms on
  them biases the eventual TabArena number.
- **Lite Elo is used to decide.** Its interval is +125/−175, wider than any milestone step.
  Decide on paired per-dataset differences.
- **The noise yardstick is one replicate pair** (AU2.9): σ about 0.0047, so t·σ is about 0.013 AP
  at 4 degrees of freedom (ESTIMATED); the 0.0073 bar is about 1.55σ for pretraining arms.
  Fine-tuning noise is far smaller: the S1 seed gap has a per-fold SD of 0.0008 (§164).
- **Coverage:** LightPFN was admitted classification-only with its regression results imputed,
  after our decline. Whether that precedent applies to us is a question for the maintainers.

### 1.8 Code integrity

- **Five inference paths.** `_representation_chunk` rebuilt the label embedding and called the
  encoder directly (now routed through `_row_attention` by AU5.2's change);
  `_term_structure_chunk` never passes `column_id_seed`, so it uses seed 0 while
  `predict_proba` uses `random_state` (v4 sets it to the fold); `quantile_regressor` has its own
  path; `capability.py` calls the model on raw features; three scripts read private
  `_ctx_y`/`_ctx_rate`.
- **No provenance.** Checkpoints save only the model config: no prior config, training config,
  git SHA or task count. That is why AU2.8 could not be settled from the checkpoint, and why 1.2
  went unnoticed. Run JSONs store the model path but no checksum or commit.
- **Config keys accepted and ignored.** `fintfm-v4protocol` never passes `correct_prior`,
  `retrieval_groups`, `retrieval_min_positive` or `prototype_minority_ratio`;
  `ensemble_label_swap`, `ensemble_feature_frac` and `finetune_clip` have no config key;
  `winsor_quantile` and `transform_subsample` have no consumer. This is the misspelled-key
  failure `CLAUDE.md` forbids, in a new form.
- **Config layering is broken for training.** 35 `fintfm-train` flags carry literal defaults,
  so the published recipe cannot be written as YAML.
- **Untested lever combinations that misbehave**: fine-tuning a hazard or quantile checkpoint
  (silently damages the encoder); `ensemble_class_shift` with `ensemble_label_swap` (cancel on
  binary); `ensemble_svd_components` with `max_features_strategy="subsample"` (SVD silently
  dropped); task dumps without a code fingerprint (a dump from older prior code is accepted).
- **Muon's settings are about 10x peers' per-element update** (lr 0.02 with Keller scaling,
  against 0.2·√max(n, m) scaling at lr 8e-4 to 1e-3 in TabICLv2, LightPFN and Mitra-v2;
  ESTIMATED by arithmetic). It is off by default and must not enter a long run as it stands.

## 2. The order of work

1. **Integrity first, CPU only, about five days** (DR1): a golden-output fingerprint test on the
   published checkpoint; provenance in every checkpoint, dump and run JSON; config from one
   source with signature-driven drift tests; one inference path; a lever-combination validator.
   Nothing here changes a weight or an output, and the fingerprint test enforces that. Until
   provenance exists, every new GPU checkpoint repeats AU2.8.
2. **Cheap measurements that need no pretraining** (DR3):
   - V4: partitioned contexts against uniform; 8 members against 1; a fixed positive count in
     the uniform context; the §130 probe fitted on the 2,000 context rows only; AP within
     flagged rows reported beside headline AP.
   - TabArena-Lite on the fork: NaN against zero fill (same day, same source), then
     `ensemble_class_shift`, then `ensemble_transforms`, then context 500/750/1,000.
   - A licence-clean proxy suite outside TabArena and BeyondArena, validated against existing
     Lite results before it is trusted (DR4).
3. **Throughput, then volume** (DR5): E0 measures tasks per GPU-hour with and without mixed
   precision and split attention on a Kaggle T4 and an HF L4; a streaming loader replaces dumps;
   E1 tests 48,000 against 480,000 tasks at two seeds under a pre-registered rule before any
   2M-task run.
4. **Prior v2 and architecture v2** (DR2, DR6), each against v1 at matched tasks, two seeds,
   V4FinBench first and TabArena second, as `CLAUDE.md` requires.

The stop rule stands: if the proxy is still below about 950 Elo after the 2M-task step, the
honest conclusion is that this line of work does not produce a listable general model.
