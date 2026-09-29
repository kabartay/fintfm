# Tasks

- [x] 40.1 **Done.** `cell-attention-and-task-inference` 39.4 closed the gap it names
      (`n_cell_blocks=1, cell_labels=True`) and 39.6 confirmed the branch it would have gated
      does not trigger, so nothing here is blocked. No pretraining run is scoped by this task
      itself; that stays gated on 40.7.
- [x] 40.2 **Done.** `src/fintfm/prior/task_families.py` implements all nine families (linear,
      threshold, XOR, interaction, max/min, piecewise, sparse, dense, latent-factor), each
      tagged `Task.source = "family:<name>"`. Five reduce exactly to §74's closed-form
      construction (`linear`/`sparse`/`dense` vary only which features carry a random unit-norm
      weight; rotation invariance of an isotropic Gaussian keeps `Phi(mu / sqrt(2))` exact
      regardless; `threshold` reuses it under a rank-preserving reshape, since AUC depends only
      on rank; `latent_factor` derives its own closed form by averaging noisy proxies of a
      hidden Gaussian driver, which is itself a mean-shifted Gaussian with an inflated
      variance). The remaining four (`xor`, `interaction`, `max_min`, `piecewise`) have no known
      closed form and are calibrated by bisection against a fixed Monte-Carlo sample (`tests/
      test_task_families.py`'s `_calibrate_sharpness`, monotone in sharpness by construction, so
      plain bisection converges). Verify: `tests/test_task_families.py` measures realised AUC on
      a large, freshly-drawn sample per family per target -- via `LogisticRegression` for the
      five linearly-separable families, and via each family's own true, by-construction
      statistic (exposed through a `_score_out` hook mirroring `scm.py`'s `_latent_out`
      convention) for the four that a generic learner cannot recover by design -- all nine
      within a stated tolerance across four target-AUC points. Not yet wired into
      `prior/mixture.py`'s `PriorConfig`; that mixing step is 40.7, gated on 40.2-40.6
      individually validating first.
- [ ] 40.3 **Interaction-order curriculum, measured before touched.** Sample the same family at
      `k=1..5` and report the achieved-AUC-vs-k curve on an existing checkpoint first (no
      training), matching the discipline of §76's cheap bisection before committing GPU spend.
      Verify: the curve is reported for at least two existing checkpoints (one capped, one not,
      per §74) before any new pretraining is scoped.
- [ ] 40.4 **Compositional generalisation test**, on existing checkpoints first, matching 40.3's
      discipline. Verify: AUC on a held-out composition is compared against AUC on each
      component seen individually, on checkpoints that already exist.
- [ ] 40.5 **Correlation/confounding/collider task family**, built on the existing SCM
      machinery. Verify: a constructed collider-structure task is scored and the model's
      reliance on the confound vs the true cause is measured via an intervention that breaks
      only the confound.
- [ ] 40.6 **Missingness/shift/support-extrapolation axes, sampled independently.** Verify: a
      test asserts these can be varied without changing which task family or difficulty a task
      belongs to, so the factorisation is genuine rather than bundled.
- [ ] 40.7 **Only after 40.2-40.6 are individually validated on existing checkpoints**, scope a
      pretraining run mixing them. Verify: matched-compute discipline per
      `phase1-prior-ablation`'s original design, and the §74 Bayes-ceiling probe run on the
      result before any other claim is made about it.
- [ ] 40.8 **Random monotonic marginal augmentation, motivated by a measured train/inference
      shift (§87).** `train.py` applies no feature transform, while `FinancialTFMClassifier`
      defaults to `feature_transform="rank"` and every real-data number in this project was
      produced with it on. Measured consequence: the model is fitted on marginals of kurtosis
      **40.70** and served marginals of kurtosis **1.81**. Proposed independently by an
      external review and by a researcher in conversation, which is why it jumps the 40.2-40.6
      queue: it is motivated by a measurement rather than by a design argument.
      Apply a random monotonic map per column per task at training time, so the model learns
      that marginal *shape* carries no signal while rank information and all causal structure
      survive by construction. Prefer this over simply rank-transforming during training: the
      augmentation makes the model invariant to *any* inference-time conditioning choice,
      whereas matching one transform ties the weights to it and turns
      `feature_transform="none"` into a mismatch in the other direction.
      Verify: a test asserts the augmentation preserves each column's rank ordering exactly, so
      it cannot change a task's label-relevant information; and the §74 Bayes-ceiling probe is
      run on the resulting checkpoint before any other claim, since a prior change that
      degrades basic signal extraction must be caught by the instrument built for exactly that.
- [ ] 40.9 **Document the SCM prior as a method, not only as a comparison arm.** `docs/paper/`
      currently mentions it only as a baseline, and `docs/paper/OUTLINE.md` §3.1 describes "the prior" as
      the financial generator alone — while §73/§75 measured the generic SCM prior *beating*
      the financial one on two of three real panels. Verify: `docs/paper/OUTLINE.md`'s method section
      describes both priors and states which one the real-data evidence currently favours on
      which panel.
