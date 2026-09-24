# Tasks

- [ ] 48.1 **Test depth against width at matched parameters.** §108 scaled *width* — `d_model`
      128→256, `d_ff` 512→1024, depth only 4+2→6+3 — and got −0.0077 on TabArena binary.
      Nori-6M is **16 transformer layers at embed_dim 128**: deep and narrow, the opposite
      corner of the design space, and a corner this project has never visited. A
      `d_model=128, d_ff=512, n_layers=12, n_col_layers=6` arm is **2,623,506** parameters —
      between the 885K baseline and §108's 4.98M — so it is cheaper than the run that already
      failed. Verify: trained at matched *tasks* (not matched steps — §108's confound) and
      scored on the same 27 TabArena binary datasets with paired per-dataset deltas against
      `cat_full`. A null is publishable; the point is to learn whether the aspect ratio, not
      the parameter count, is what §108 got wrong.
- [ ] 48.2 **Recalibrate the scaling expectation in writing, before spending more GPU on it.**
      Verify: `docs/roadmap/STRATEGY.md`'s Phase 1 target of 10–50M parameters is restated with Nori's
      published curve beside it, and either defended with a reason the return should be larger
      here or revised. A target inherited from a plan written before any scaling evidence
      existed should not survive contact with evidence merely because it is in the document.
- [ ] 48.3 **Replace the binned head's fixed resolution with a quantile head.** §106 states the
      limitation plainly: resolution is bounded by `K`, an interval can never be narrower than
      one bin, and `n_bins` is coupled to `max_classes`. Nori emits a **999-quantile pinball**
      distribution, which has none of those properties and needs no bins. Verify: a pinball-loss
      head is measured against the binned head on `fintfm-capability --regression-sweep`, with
      the `binning_oracle` floor reported for both — the binned arm's floor is a real constraint
      and the quantile arm should not have one, which is the whole claim.
- [ ] 48.4 **Widen the prior's target mechanisms to a published target list.** `prior/scm.py`
      has several nonlinearities; Nori documents **9 target families** (dense/sparse linear,
      GAM, interactions, random MLP, random tree, radial/RBF, Fourier features, chained
      trigonometric) over **8 SCM edge-function types** (MLP, decision tree, piecewise-linear,
      polynomial, periodic, RBF, log/exp, conv1d). This is `mechanism-diverse-prior`'s open
      question with someone else's answer sheet. Verify: the families this prior already covers
      and those it does not are listed explicitly before anything is added, so the work is
      scoped by a gap rather than by the length of their list.
- [ ] 48.5 **Add a learnability filter to the prior, and measure whether it matters.** Nori
      rejects unlearnable synthetic datasets with an ExtraTrees signal-quality filter. §42
      established that a prior of only-easy or only-noise targets teaches the wrong thing, and
      this project's answer was to *span* difficulty; filtering is the complementary move and
      neither has been measured against the other. Verify: the fraction of tasks rejected is
      reported per prior, and a checkpoint trained with filtering is compared against one
      without on the same seeds — if the rejection rate is near zero the filter is inert here
      and that is the finding.
- [ ] 48.6 **Adopt the cheap realism augmentations.** Nori lists discretized features, noise
      features, correlated blocks, structural missingness and label noise. `prior/financial.py`
      already has missingness; the rest are cheap. Verify: each augmentation is added behind its
      own flag and the capability suite is run with each off, so a gain is attributable to one
      augmentation rather than to the bundle.
- [ ] 48.7 **Record the counter-thesis, because it is aimed at this project's foundation.**
      ConTextTab (arXiv:2506.10707) argues explicitly that "exclusive training on synthetic
      data limits their ability to fully leverage the rich semantics and world knowledge
      contained in real-world tabular data", and trains on large-scale real tables instead;
      iLTM and TabSTAR do likewise. This project cannot follow, and not for licence reasons —
      `docs/roadmap/STRATEGY.md`'s differentiator is auditable freedom from benchmark contamination,
      which real-table pretraining destroys. Verify: `docs/design/DECISIONS.md` carries this as a
      decision with its cost stated and **what would reverse it** named, rather than as an
      assumption nobody has revisited since the field moved.
- [ ] 48.8 **Re-examine the small-data premise against §102.** TabPFN v2 (Nature 2025) claims
      dominance "for datasets with up to 10,000 samples and 500 features", and this project's
      stated bet is the small-data regime. §102 measured the opposite shape: the TabArena gap
      *narrows* with dataset size. Verify: the two are reconciled or the contradiction is
      recorded — if our deficit is worst on small tables, the segment this project targets is
      the one where it is currently weakest, and that is a strategy finding, not a modelling one.
- [ ] 48.9 **Try schedule-free optimisation, which would decouple the run length from the
      schedule.** Defazio, Mehta, Mishchenko, Khaled & Cutkosky, *The Road Less Scheduled*
      (NeurIPS 2024), cited by TabDPT. This project's cosine schedule is a real operational
      constraint, not just a hyperparameter: §108's matched-task rerun had to start **fresh**
      rather than resume, because a schedule spanning 6,000 steps cannot be extended to 12,000
      without training the second half at an annealed-to-zero rate. Schedule-free removes the
      horizon from the optimiser entirely. Verify: a schedule-free run is compared against the
      cosine baseline at matched tasks on the same five V4FinBench folds, **and** the claim that
      a run can be extended without restarting is demonstrated rather than assumed — the second
      property is worth more here than any accuracy delta.
- [ ] 48.10 **Test the is-missing encoding, where two peers disagree.** TabDPT reports NaN
      tokens and binary is-missing features as no better than mean imputation; Nori uses learned
      mask embeddings. Our adapter does `fillna(0)`, which is the mean after normalisation.
      Verify: a learned mask embedding is measured against the current default on V4FinBench,
      whose missingness is structural rather than random — and the result is reported as
      settling *our* case only, since two published projects already disagree and a third data
      point does not resolve a disagreement it was not designed to arbitrate.
- [ ] 48.11 **Record the Bitter-Lesson critique against this project's own effort allocation.**
      TabDPT's appendix concludes that compute and high-quality data matter more than
      architectural manipulation. This project has spent its effort on architecture (§54, §104,
      §44) while §93 (5× data) and §108 (5.7× parameters) both returned nulls or losses, and it
      is locked out of "high-quality data" by decision D2. Verify: `docs/design/DECISIONS.md` states
      what this project believes it gets in exchange, and names the observation that would show
      the trade is not worth it — an unfalsifiable differentiator is a slogan.
- [x] 48.12 **Multiply tasks per prior draw by re-targeting, the way TabDPT's SSL does.** Their
      procedure generates **both** classification and regression targets from *every* training
      table, so 123 datasets yield far more than 123 tasks. The synthetic analogue is free and
      this project does not do it: `prior/scm.py` samples a random DAG and then uses **one**
      node as the target, discarding a graph whose every other node is an equally valid target
      with a different, genuinely non-redundant dependency structure. Sampling `k` targets per
      draw multiplies task count by `k` at roughly `1/k` the generation cost per task.
      **This is the one idea here that attacks the 48,000-task shortfall without real data**
      (`CLAUDE.md`, "Count the tasks, not the steps"). Verify: task diversity is measured, not
      assumed — re-targeting the same graph could produce correlated tasks that inflate the
      count without adding signal, which would look identical to a win in the step counter and
      nowhere else. Compare against §93's 5×-volume null, which is the result this has to beat.
      **Implemented and measured (§113), downstream still open.** `scm_reuse_graph`, default 1.
      2.53× tasks per second at `n_targets=4`, and the named failure mode is absent: a model
      fitted on one sibling scores **0.4775 — chance — on another** from the same graph over
      the same feature columns, against 0.7023 on its own. The remaining half of this task is a
      checkpoint trained at `scm_reuse_graph > 1` against one at 1, matched on **steps** so the
      task count genuinely differs. **That sentence is wrong and §113 corrects it**: a batch
      returns `batch_size` tasks at any `scm_reuse_graph`, so at matched steps the volume is
      identical. What changes is generation cost (~20% of step time, two-thirds of it saved,
      worth ~13% more steps per dollar) and **batch-level graph diversity, which falls to a
      quarter**. The deciding run is therefore the cheaper and less interesting question of
      whether that diversity loss hurts — not whether extra volume helps.
- [ ] 48.13 **Add a tree-based prior, the family MITRA singles out and this project does not
      have.** `PriorConfig` is financial (0.7) plus SCM, with no prior that generates threshold
      structure — while every baseline fintfm loses to is a tree ensemble. MITRA
      (arXiv:2510.21204) selects tree-based priors (gradient boosting, random forest, decision
      tree, extra tree) specifically on **distinctiveness**: TFMs pretrained on SCMs "do not
      always generalize well to all types of data generated from TBPs". Their priors are
      reported model-agnostic, so this needs no architecture change and is cheaper than anything
      in `factorized-attention`. Verify: a checkpoint trained with the tree prior mixed in is
      compared against the current mixture on the same five V4FinBench folds **and** the 27
      TabArena binary datasets, with the mixture weight stated — this is the first prior change
      since the regression prior and must not be confounded with one.
- [x] 48.14 **Score our own prior mixture against MITRA's three criteria.** They propose
      **performance** (does a TFM pretrained on this prior alone do well on real data),
      **diversity** (does it cover a wide region of task space) and **distinctiveness** (does it
      generate structure the other priors do not). This project has never evaluated its prior on
      any of the three, despite the prior being its one genuinely distinctive asset. Verify:
      each of `financial`, `scm` and any new prior is trained alone at matched budget and scored
      on real data (performance), and distinctiveness is measured by cross-evaluation — a model
      trained on prior A scored on tasks from prior B — rather than asserted from the generator's
      source code. A prior whose tasks another prior's model already solves is adding step count,
      not signal, which is the §93 null's most likely explanation.
      **Done (§111, corrected by §112): `fintfm-priorscore`.** All three criteria computed
      from *fitted* baselines rather than our own model, so the statistic cannot confuse "the
      prior lacks this structure" with "our model cannot learn it". Result: `tree` is the only
      prior with positive tree-versus-linear distinctiveness (+0.0225, against −0.0292
      financial and −0.0021 SCM); the SCM prior has by far the widest difficulty spread
      (diversity 0.2046), which argues against reducing its weight; the financial prior is the
      hardest (0.6880). **The first version of this instrument was wrong in a plausible way** —
      an unconditioned linear baseline on heavy-tailed features inflated distinctiveness and
      reported the financial prior as the *most* tree-favourable, which would have argued
      against building the tree prior at all. §112 records the correction and the rule: a
      fitted baseline used as an instrument gets the same preprocessing the model under study
      gets. The cross-evaluation half of this task (train on prior A, score on prior B) still
      needs checkpoints and is not done.
      **Extended by §117, after §116 showed the criteria were incomplete.** `--base-rate`
      subsamples every task to a given positive rate and scores by average precision, so the
      instrument asks this project's question rather than a general-tabular paper's. It
      **inverts the performance ordering** — the tree prior is the best of the three at natural
      balance (0.7555) and the worst at 0.4% (0.0225 AP) — and would have flagged the tree
      prior before four checkpoints were spent on it. Whether low-rate learnability *predicts*
      credit performance has been measured on **one prior, retrospectively**: the instrument
      now asks a question it previously could not, which is not the same as its answers being
      established.
- [ ] 48.15 **Evaluate moving the objective from `p(y | x, D)` to `p(x, y | D)`.** LimiX-2 —
      **rank 0 on TabArena, Elo 1872, synthetic-only, SCM-pretrained, Apache-2.0** — replaces
      target-centric prediction with "a context-dependent representation of the joint structure
      underlying data generation", trained by context-conditional masked modelling. This project
      amortises `p(y | x, D_context)` explicitly, in `README.md` and `docs/design/ARCHITECTURE.md`.
      Three things follow from the change and the first is why it matters here: **every column
      becomes a training signal rather than only the designated target**, which is the general
      form of 48.12 and attacks the 48,000-task shortfall without real data; one model then
      serves classification, regression, imputation and generation as conditional queries,
      where this project needs a checkpoint per problem type plus a binning wrapper (§106); and
      LimiX-2 reports feature attention recovering the causal skeleton, which
      `cell-attention-and-task-inference` 39.7 proposed probing and never did.
      Verify: scope this as a **measurement before a rewrite** — a masked-column objective is
      added behind a flag and compared against the current target-only objective at matched
      tasks on the same five V4FinBench folds, *before* anything in the inference API changes.
      The failure mode to name in advance: masked modelling spends capacity on reconstructing
      features nobody will ever query, and at 885K parameters that trade may be strictly bad —
      LimiX-2 is 16M and its 2M variant is the relevant comparison, not its headline.
- [ ] 48.16 **Use a trained checkpoint as a surrogate for the next sweep, instead of brute
      force.** TabPFN-2.5's report: ~100 real hyperparameter runs, scored by an in-house
      validation suite, then TabPFNv2 itself used as a regression surrogate to rank 10,000
      candidate configurations before spending compute on the winners. Every sweep in this
      project's history (`column_id_dim`, §104; volume, §93; scale, §108) has paid full price
      per arm. Verify: scoped as a **method test on cheap data first** — fit a surrogate (even
      a plain gradient-boosted regressor, no TFM needed to start) on the ~15 architecture/prior
      runs already in `runs/`, predict the untried points on the `column_id_dim` curve absent
      from §104's five, and check the surrogate's prediction against a real run before trusting
      it on anything expensive.
- [ ] 48.17 **Randomise the SCM prior's graph connectivity, not just its weights.**
      `sample_scm_task` draws a fixed-depth layered graph with independent sparse edges;
      TabICLv2 (arXiv:2602.11139, Appendix E.4) samples edge probability as
      `sigmoid(A + B_i + C_j)` with `A, B_i, C_j` drawn i.i.d. standard Cauchy — heavy tails
      giving "higher probabilities of exceptions to the rule". Five-line change, highest
      expected-gain-per-line-of-code in the whole peer sweep. Verify: task-difficulty spread
      (the statistic §42 already tracks) is measured before and after, since the claim is
      about diversity, not about any single accuracy number moving.
- [ ] 48.18 **Implement the tree-based node function from TabICLv2's Appendix E.8 for task
      48.13**, rather than inventing one. Oblivious (CatBoost-style) trees, split dimension
      chosen proportional to feature standard deviation, leaf values standard normal, ensemble
      of `LogInt(1, 128)` trees averaged. Verify: read their stated reason for ensembles over
      single trees (computational, not accuracy) before choosing between single-tree and
      ensemble for the first implementation — a wrong guess here duplicates MITRA's own
      SCM-vs-TBP distinctiveness result without adding anything.
- [ ] 48.19 **Widen the SCM prior's activation set with order-statistic functions.** Ours is
      5 fixed activations (`tanh`, `sin`, ReLU, identity, signed-sqrt); TabICLv2 lists 21 fixed
      plus 4 parametric, including `rank`, `softmax`, `one-hot argmax`, `argsort`. §35 measures
      that this project's *inference-time* rank conditioning is worth +0.086 AUC on real
      financial ratios — but the *prior's label mechanism* never generates rank-like structure
      for the model to learn from. Verify: `rank` and at least one other order-statistic
      activation are added to `_ACTS` and a checkpoint trained with them is compared against
      the current one on V4FinBench, since this is a plausible mechanism for part of §35's
      result rather than a proven one.
- [x] 48.20 **Cache the context's key/value projections across query chunks.** **Closed without implementing (§119).** Inference cost
      became a blocking liability on 2026-09-22: the deep-narrow arm (48.1) could not finish
      TabArena, raising `TimeLimitExceeded` after 8 of 27 datasets, and this project's median
      predict time is **8.6 s/1K against a field norm near 0.1**. TabICL reports KV caching
      plus `O(n² + nm²)` giving "10× faster than TabPFN-2.5" and 50,000×100 in under 10 s.
      The structure here suits it exactly: `fit()` fixes the context, and `predict_proba`
      re-projects that same context for every query chunk. Verify: predictions are **identical**
      before and after to within float tolerance — this is an optimisation and any accuracy
      change means it is wrong — and the speedup is reported at V4FinBench's shape (136
      features, `max_context=1000`, 48,000 queries) rather than a synthetic best case. Note it
      does **not** apply to `context_strategy="retrieval"`, where the context is chosen per
      query group, the same exception that already breaks exact query chunking.
      **Measured before writing it, and the measurement said no.** The context costs **1.022 s
      fixed per chunk** against **2.474 ms per query** marginal, so at V4FinBench's 48,000
      queries perfect caching saves **17%** (143.3 s → 118.8 s). Against a median predict time
      of 8.6 s/1K versus a field norm near 0.1 — an 86× gap — 17% leaves 71×. TabICL's 10×
      comes from `O(n² + nm²)` attention, not caching; attributing their speed to the cache was
      the error. Raising `query_chunk` recovers two-thirds of the same saving as a one-line
      default change. The marginal cost is 83% of the total and only `factorized-attention`
      (44.x) touches it.
