# Tasks

- [ ] 40.1 **Wait on `cell-attention-and-task-inference` task 39.4/39.6.** Do not scope
      pretraining runs here until that result is in. Verify: this task is only closed by
      linking the finding that triggered starting the rest of this proposal.
- [ ] 40.2 **Implement the labelled task-family generators** (linear, threshold, XOR,
      interaction, max/min, piecewise, sparse, dense, latent-factor), each independently
      difficulty-controlled via §74's closed-form construction where the family admits one, and
      each tagged with its own family identifier for later use by
      `cell-attention-and-task-inference` task 39.7's DGP-classification probe. Verify: a test
      asserts each family's realised difficulty (measured, not assumed) matches its requested
      Bayes-AUC target within a stated tolerance.
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
