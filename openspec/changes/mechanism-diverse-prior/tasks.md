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
- [x] 40.3 **Done (§137).** `fintfm.prior.task_families.sample_interaction_order_task`
      generalises `xor` to an explicit order `k` (parity of `k` feature signs; no `k-1`-way
      marginal carries label information by construction). `fintfm.experiments.capability.
      interaction_order_probe` reports achieved-AUC-vs-`k` on an existing checkpoint (no
      training), matching `bayes_ceiling_probe`'s convention exactly. Verify: the curve is
      reported for `runs/dl/v4-cellattn-fin10.pt` (capped on the linear task, §74) and
      `runs/dl/v4-cellattn-fin00.pt` (uncapped, §74) -- both collapse to chance at `k=2` and
      stay there through `k=5`, so the linear-task capped/uncapped split does not predict
      interaction capability at all. No new pretraining is scoped by this task; §137 sets the
      floor tasks 40.4 and 40.7 measure against.
- [x] 40.4 **Done (§138).** `fintfm.prior.task_families.sample_composition_task` draws two
      disjoint single-feature rules sharing one `X` draw, exposing either component's own label
      or their AND; `fintfm.experiments.capability.composition_probe` evaluates all three on a
      loaded checkpoint, matching 40.3's convention. Verify: on both §74 checkpoints, `component_a`/
      `component_b` land at ~0.90 (their calibrated target) and `composed` at ~0.86 -- a real
      but modest drop, not the collapse to chance 40.3 found for interaction order. No new
      pretraining scoped.
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
