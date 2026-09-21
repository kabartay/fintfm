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
