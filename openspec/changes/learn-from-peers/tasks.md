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
      Verify: `docs/STRATEGY.md`'s Phase 1 target of 10–50M parameters is restated with Nori's
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
      `docs/STRATEGY.md`'s differentiator is auditable freedom from benchmark contamination,
      which real-table pretraining destroys. Verify: `docs/DECISIONS.md` carries this as a
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
      is locked out of "high-quality data" by decision D2. Verify: `docs/DECISIONS.md` states
      what this project believes it gets in exchange, and names the observation that would show
      the trade is not worth it — an unfalsifiable differentiator is a slogan.
- [ ] 48.12 **Multiply tasks per prior draw by re-targeting, the way TabDPT's SSL does.** Their
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
- [ ] 48.14 **Score our own prior mixture against MITRA's three criteria.** They propose
      **performance** (does a TFM pretrained on this prior alone do well on real data),
      **diversity** (does it cover a wide region of task space) and **distinctiveness** (does it
      generate structure the other priors do not). This project has never evaluated its prior on
      any of the three, despite the prior being its one genuinely distinctive asset. Verify:
      each of `financial`, `scm` and any new prior is trained alone at matched budget and scored
      on real data (performance), and distinctiveness is measured by cross-evaluation — a model
      trained on prior A scored on tasks from prior B — rather than asserted from the generator's
      source code. A prior whose tasks another prior's model already solves is adding step count,
      not signal, which is the §93 null's most likely explanation.
- [ ] 48.15 **Evaluate moving the objective from `p(y | x, D)` to `p(x, y | D)`.** LimiX-2 —
      **rank 0 on TabArena, Elo 1872, synthetic-only, SCM-pretrained, Apache-2.0** — replaces
      target-centric prediction with "a context-dependent representation of the joint structure
      underlying data generation", trained by context-conditional masked modelling. This project
      amortises `p(y | x, D_context)` explicitly, in `README.md` and `docs/ARCHITECTURE.md`.
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
