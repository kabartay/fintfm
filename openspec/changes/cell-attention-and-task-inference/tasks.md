# Tasks

## This round: architecture

- [x] 39.1 **Done.** `ModelConfig.n_cell_blocks` implemented as specified. Verify: passed
      — `n_cell_blocks=0` confirmed byte-identical to a model built without the field (a first
      version of this check was itself buggy, comparing two stochastic forwards without
      controlling shared RNG state; caught before trusting it). Local cost measurement before
      any GPU spend: `n_cell_blocks=1` roughly doubles step time, `=2` roughly quadruples it,
      at this project's production scale (n_rows up to 1024, max_features=136) — steeper than
      baseline's own scaling but tractable. T4 pretraining launched at `n_cell_blocks=1`.
- [x] 39.2 **Done.** `ModelConfig.cell_labels` implemented as specified. Verify: passed —
      `cell_labels=True` with `n_cell_blocks=0` builds no unused parameters (asserted directly,
      not just "should"). **Ablation from 39.1 not yet run**: the first T4 comparison combines
      both (cell attention + early labels together) as the strongest test of the combined
      hypothesis; isolating them is only worth the extra compute if the combination shows
      something worth attributing to one or the other.
- [x] 39.3 **Done.** `bayes_ceiling_probe()` in `experiments/capability.py`, matching
      §74's original measurement exactly (direct model calls, column-identity ensembling) so
      results are comparable to every recorded value. Also returns per-target regret (task
      39.18), free once both numbers exist. Verify: passed — closed form checked against an
      empirical Bayes-optimal-statistic AUC on 200,000 rows, kept as a permanent test.
- [x] 39.4 **Done — the gap closed.** §78: `n_cell_blocks=1`, `cell_labels=True`, both
      `p_financial` arms. Financial-only regret at Bayes AUC 0.90-0.999 fell from 0.234-0.277
      to 0.001-0.005 (~60-90x). SCM arm unregressed. Confirmed by the antisymmetric probe
      independently: 0.5352 to 0.9890, closing to the SCM baseline's 0.9932 — something no
      content intervention across §58-§77 came within 0.4 of. **Deviation from the planned
      protocol**: forced to `--batch-size 4 --n-rows-choices 256,512` (dropping 1024) after a
      CUDA OOM on the first launch attempt (`_eval_quality`'s hardcoded eval batch size of 16,
      independent of training's batch size, now fixed for every future run) — task 39.5 should
      close this gap before the comparison is called final.
- [x] 39.5 **(b) DONE — the fix transfers (§80). (a) still open.** Two complete five-fold
      V4FinBench runs on the full 1.0M-row horizon-0 panel under one shared config
      (`configs/cellattn-v4-validation.yaml`), differing only in `n_cell_blocks`/`cell_labels`:
      cell attention gains **+0.0486 mean AP, 5/5 folds, every 95% CI excluding zero, every
      Holm-corrected p < 0.001**. Against logistic regression, §69's tie becomes a significant
      win (+0.0307); against every tuned booster the deficit survives unchanged in character
      (-0.204 to -0.236). Symmetry probes were already run in §78. Verify: paired-bootstrap AP
      on V4FinBench and per-task symmetry-probe scores are both reported against the
      pre-change checkpoint. Done — §80 and §78 respectively.
      **Still open, split out as 39.5a and 39.24**: (a) re-run at §74's exact protocol
      (`--batch-size 8 --n-rows-choices 256,512,1024`), which §79 shows needs more GPU memory
      than a T4 has; and the caveat §80 raises — the new architecture's best *reachable* score
      (0.1941 at context 1000) is below the old one's best *recorded* score (0.2116 at context
      4000), so no claim of improved real-data *standing* may be made yet.

- [x] 39.24 **DONE (§81) — and it cost nothing.** Implemented as
      `FinancialTFM.feature_chunk` (runtime-only, never enters a checkpoint), default 16 via
      `InferenceConfig.feature_chunk`. Measured 21.0 GB / 10.6 s unchunked against 3.4 GB /
      6.8 s chunked at N=2024 on 136 features — **6.2x less memory and 1.6x faster**, not the
      memory-for-time trade this task predicted; that wrong prediction is corrected in place
      in §79, §80, Claim 10 and `LIMITATIONS.md`. `max_context=4000` now runs at 22.5 GB
      against §79's ~63 GB estimate. Identity asserted byte-for-byte by
      `tests/test_model.py::test_feature_chunking_is_an_identity` at chunk sizes 1/2/5/16/17/64.
      Verify: a test asserts chunked and unchunked outputs are equal on a fixed seed. Done —
      `test_feature_chunking_is_an_identity` asserts `torch.equal`, difference 0.000e+00.
      **Follow-on, task 39.25**: the re-scoring this unblocks has not been run.

- [x] 39.25 **DONE (§84) — the architecture wins best-vs-best.** Five arms, five folds, full
      1.0M-row panel. Old architecture peaks at ctx 2000 (0.1523); cell attention peaks at ctx
      1000 (0.1941). Each at its own best context, identical rows: **+0.0417 AP, 5/5 folds**.
      Architecture is worth +0.039 to +0.049 at matched context against a context span of
      0.0069 — a 6-7x ratio. Verify: paired-bootstrap AP reported per context, and §71's
      single-fold 0.2116 re-measured at five folds so the comparison is like-for-like. Done —
      §84's table, and §82 established §71 was a 10x subsample so the re-measurement is §83's
      full-panel curve rather than a like-for-like reproduction of an invalid number.
      **Open, split to 39.26**: cell attention prefers *less* context, contradicting §83's
      stated prediction; unexplained.

- [ ] 39.26 **Why does cell attention prefer less context?** §84 measured -0.0032 AP from
      ctx 1000 to 2000 (2/5 folds positive) where the old architecture gained +0.0069 (5/5).
      §83 predicted the opposite on the grounds that row-attention-within-feature should turn
      extra rows into a better-estimated column distribution. One untested candidate: that
      stage attends over all context rows per feature, so a larger context may dilute
      attention mass across near-duplicate rows rather than sharpening the estimate. Verify: a
      context sweep on the synthetic Bayes-ceiling probe, where the optimum is known in closed
      form, reports whether the effect reproduces off real data — which separates an
      architectural property from a V4FinBench idiosyncrasy.

- [x] 39.6 **Resolved — 39.4 closed the gap, so this branch does not trigger.** The
      label-functional-form candidate (§77) remains scientifically interesting but is no
      longer the leading explanation for §74; see the update to `mechanism-diverse-prior`.

## Backlog: sequenced after architecture, per the external review and the user's ordering

Recorded so none of this is lost. Each is a `docs/FINDINGS.md`-worthy experiment on its own;
none is scoped or scheduled yet. Verify: the label-functional-form experiment is scoped as its own task before being started, not assumed.

- [ ] 39.7 **DGP/task-representation probe.** Twenty-plus canonical task families (linear,
      threshold, XOR, interaction, sparse, dense, latent-factor, SCM); train a small classifier
      on the model's context/task representation to predict which family generated the context.
      If representations do not separate by family, the model is not forming a task posterior
      regardless of downstream AUC — this is a Bayesian-regret-adjacent diagnostic and the
      reviewer's strongest complement to the Bayes-ceiling test. Verify: probe classification accuracy on held-out DGP families is reported against a chance baseline.

- [ ] 39.8 **Context-sufficiency curve against information content, not row count.** AUC vs
      context size for tasks of controlled *information* requirement (e.g. `x0+x1>0` needs few
      examples; a 4-way interaction needs many), to separate genuine ICL from a strong global
      prior that happens to score well on average. Verify: the sufficiency curve is plotted against an information-content measure, not row count alone.

- [ ] 39.9 **Feature-selection recovery.** Extremely simple tasks (`y=x_3`, `y=-x_3`, `y=x_7`,
      `y=1[x_2>0]`, `y=1[x_2+x_5>0]`); test directly whether the model identifies the relevant
      feature subset, not just whether it gets the final AUC right. More fundamental than
      overall accuracy — this is "column identity vs column semantics," precisely. Verify: per-feature attention weight or ablation-based importance is compared against the true relevant feature for each task.
- [ ] 39.10 **Sign/direction inference.** Two tasks with identical feature distributions and
      opposite label direction (`y=1[x0>0]` vs `y=1[x0<0]`); measure whether the model infers
      `sign(beta_j)` from context alone, extending to a full `sigma(beta^T x)` with `beta`
      varying per task. A more precise diagnosis than aggregate AUC when it fails. Verify: recovered sign is compared against the true sign per task, not inferred from AUC alone.
- [ ] 39.11 **Compositional generalisation.** Train on `f1(x0)`, `f2(x1)`, `f3(x2,x3)` as
      separate task families; test on `f1(x0)+f2(x1)+f3(x2,x3)`, a combination never seen
      exactly. One of the stronger tests of whether the prior is foundational rather than a
      lookup over seen task shapes. Verify: AUC on the unseen composition is compared against AUC on each component task seen individually.
- [ ] 39.12 **Schema variation at fixed task simplicity.** Vary column count (5/10/50/200),
      useful-vs-useless ratio, categorical/numerical mix, duplicated and constant columns,
      while keeping the underlying task the same; the model should learn "3 of these 200
      columns matter," not have signal diluted by width. Verify: recovered feature-importance ranking is compared against the true relevant set as width grows.
- [ ] 39.13 **Irrelevant-feature robustness curve.** Fixed task `y=f(x0,x1)`, progressively add
      pure-noise dimensions up to d=100; plot the AUC-vs-d curve. Directly comparable to how
      tree ensembles and other architectures are known to behave here. Verify: the AUC-vs-d curve is plotted and its slope reported.
- [ ] 39.14 **Correlated-feature and confounding tests.** `x1=z+eps1`, `x2=z+eps2`, `y=f(z)`,
      then redundant predictors, proxies, confounders, collider structures — the SCM machinery
      already in this repo is the natural generator for this. Verify: the model's reliance on the confound vs the true cause z is measured directly, e.g. via an intervention that breaks the confound only.
- [ ] 39.15 **Causal/interventional comparison.** For a simple SCM (`Z->X1`, `Z->X2`,
      `X1->Y`), compare `P(Y|X1)` against `P(Y|do(X1))`. Not a product goal, but scientifically
      informative about whether the learned prior encodes structure or only association. Verify: the two conditional distributions are compared directly on held-out interventional data.
- [ ] 39.16 **Sample-size x task-difficulty 2D grid.** Rows in {16,64,256,1k,4k} crossed with
      difficulty in {easy,medium,hard}; look for `dAUC/dn` conditional on difficulty. A model
      doing genuine Bayesian ICL should show different learning curves per difficulty class. Verify: curves for each difficulty tier are plotted and compared for shape, not just endpoint.
- [ ] 39.17 **Prior-vs-evidence override test.** Heavily pretrain exposure to `y~f(x0)`-shaped
      tasks, then present a context where `y~f(x7)` instead; measure whether the model
      overrides its learned prior when the in-context evidence contradicts it. Distinguishes
      genuine posterior updating from a memorised "x0 is usually predictive" shortcut. Verify:
      the override is measured directly as achieved AUC on the contradicting context, not
      inferred from aggregate behaviour.
- [ ] 39.18 **Bayesian regret as the standard reported metric.** `R = AUC_Bayes - AUC_model`,
      and normalised `R_norm = (AUC_Bayes - AUC_model) / (AUC_Bayes - 0.5)`, alongside every
      Bayes-ceiling measurement from 39.3 onward — makes the §74 pattern (AUC 0.70 excellent at
      ceiling 0.71, terrible at ceiling 0.995) visible in the headline number instead of
      requiring the full curve to notice. Verify: 39.3's probe reports both raw AUC and
      regret by construction once this lands.
- [ ] 39.19 **Posterior task uncertainty, not just predictive uncertainty.** Construct
      deliberately ambiguous contexts consistent with two plausible DGPs; a model with a real
      task posterior should become less confident rather than arbitrarily committing to one —
      connects directly to this project's calibration thesis (§12, §73). Verify: predicted entropy/confidence is measured on ambiguous contexts against unambiguous ones of matched difficulty.

## Added 2026-09-14: deltas from a second external review round, not already covered above

- [ ] 39.20 **Full label-condition matrix**, extending the existing shuffle test (§47/§48).
      Compare achieved AUC under: correct labels, shuffled labels (have this), inverted
      labels, no labels at all (context features only), and labels drawn from an unrelated
      task. Verify: all five conditions reported side by side on the same context features,
      for at least the trivial and Bayes-ceiling probes.
- [ ] 39.21 **Separate marginal realism from mechanism realism.** Two cells not yet tested:
      real feature structure (e.g. V4FinBench's own columns) with destroyed/shuffled labels,
      and synthetic financial labels applied to real feature marginals. Verify: both cells
      scored on the Bayes-ceiling-style probes where possible, and the finding states plainly
      whether the gap tracks feature realism or label-mechanism realism.
- [ ] 39.22 **Semantic column information, three modes.** (A) anonymous numeric columns
      (current default), (B) derived statistical descriptors per column
      (mean/std/skew/missing-rate/quantiles) made available to the model, (C) column
      name/metadata embeddings. Verify: A vs B vs C compared on the same checkpoint family and
      the same real panels as §73/§75, since this is specifically about transfer to real
      tables with meaningful column names.
- [ ] 39.23 **Baseline expansion — gated on licensing.** TabICLv2, TabDPT, TabPFN,
      FT-Transformer, TabM, RealMLP alongside the existing tuned CatBoost/LightGBM/XGBoost.
      **Constraint, not optional**: this project's standing rule is that no code or weights
      from TabPFN/TabICL/TabDPT may enter the repository, and any evaluation against them is
      gated on checking each weight's licence separately from its code licence, before use,
      every time — not assumed from a prior check. Verify: a licence check is recorded in the
      commit or finding that adds each baseline, not deferred to "later."
