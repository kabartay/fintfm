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
- [ ] 39.4 **Pretrain and compare.** Same protocol as every other T4 comparison this session
      (6,000 steps, `p_financial=1.0` and `p_financial=0.0` arms, matching §74's `fin10`/`fin00`
      exactly except for `n_cell_blocks`). Verify: the §74 Bayes-ceiling probe run on both new
      checkpoints, reported against the existing `fin10`/`fin00` curve. This is the actual test
      of the proposal's hypothesis — report the result whichever way it comes out.
- [ ] 39.5 **If 39.4 closes the gap**, re-run §54/§56's symmetry probes and §69's V4FinBench
      protocol on the new architecture before any claim that it is a strict improvement —
      §50/§52's retracted readings are the standing warning against declaring victory from one
      metric. Verify: paired-bootstrap AP on V4FinBench and per-task symmetry-probe scores are both reported against the pre-change checkpoint.

- [ ] 39.6 **If 39.4 does not close the gap**, the label-functional-form candidate §76 left
      untested (financial's shallow linear-plus-one-interaction label versus SCM's deeper
      random-graph label) becomes the next hypothesis, and this proposal's premise (the cap is
      architectural) is itself falsified — record that plainly rather than moving the goalposts.

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
