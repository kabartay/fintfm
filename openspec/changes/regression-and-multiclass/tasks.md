# Tasks

- [x] 46.1 **Train and verify a multiclass-capable checkpoint** at `--max-classes 10` with the
      SCM prior in the mixture, since `prior/financial.py` hardcodes two classes. Verify: the
      capability suite reports per-class-count accuracy above an untrained control of the same
      architecture at 3, 5 and 10 classes — `fintfm-capability` builds that control
      automatically, and without it "the model does multiclass" is unreadable.
      **Done (§99):** clears the control at every K — +0.261/+0.267/+0.157 macro OvR AUC at
      K=3/5/10 — while the control sits at chance. The deficit to multinomial logistic
      regression is large and widens with K, which is §96's shape, not a new problem.
- [ ] 46.2 **Measure what multiclass training costs binary credit accuracy.** At 885K
      parameters, capacity spent on 10-class structure is capacity not spent on the binary
      task, and §73/§75 suggest broader training may help rather than hurt. Verify: V4FinBench
      five folds with paired bootstrap against a `--max-classes 2` checkpoint matched on prior
      mixture, so the comparison isolates class count rather than confounding it with the
      prior change multiclass requires.
- [x] 46.3 **Implement the binned regression head.** Bin edges from the context targets'
      quantiles per task, logits over bins, cross-entropy on the bin index. Verify: a test
      asserts the predicted distribution integrates to one and that the induced point estimate
      recovers a known linear target to a stated tolerance on a synthetic task — a head that
      trains but cannot recover `y = 2x + noise` is not a regression head.
      **Done, and cheaper than this task assumed:** no architecture change is needed.
      `FinancialTFM.head` is already `Linear(d_model, max_classes)` trained by cross-entropy
      on an integer index, and a target binned into K quantile bins *is* such an index. The
      work is `inference/binning.py` plus a prior; the head, loss and per-cell label
      injection are untouched.
- [x] 46.4 **Extend the prior to emit continuous targets.** `prior/scm.py` already computes a
      continuous latent before thresholding it into classes; regression tasks use that latent
      directly. Verify: a test asserts the regression prior spans difficulty in the same sense
      `tests/test_prior.py` already pins for classification — a prior of only easy or only
      noise targets teaches the wrong thing (§42).
      **Done, after the first attempt failed this exact check.** Projecting the exposed
      features linearly gave ridge-Spearman 0.74–0.98 on every draw — uniformly easy. Using
      the SCM's own pre-threshold latent (a node of the same random graph, so nonlinear in
      the features) with a log-uniform noise multiplier spans 0.01–0.87.
- [x] 46.4a **The sklearn-facing regressor.** 46.3 built the transform and 46.4 the prior;
      neither is reachable from a benchmark without an estimator that closes the loop.
      Verify: `FinancialTFMRegressor` bins on `fit`, de-bins on `predict`, and a test asserts
      the bin edges are identical whatever the query rows contain — the grid the answer is
      read off must not depend on the answer.
      **Done.** `inference/regressor.py`, 8 tests. Two boundary conditions are the whole
      risk: the classifier only emits columns for labels it saw, so the distribution is
      widened back to the full bin grid before it meets the representatives (misalignment
      there produces a plausible-looking number paired with the wrong bin); and `n_bins`
      above the head's width is refused rather than truncated.
- [ ] 46.5 **Report calibration, not only error.** Verify: prediction-interval coverage is
      reported beside RMSE on held-out synthetic tasks — an 80% interval should contain the
      truth 80% of the time, and a head that is accurate but badly calibrated would pass an
      RMSE-only check while being useless for the risk quantities this exists to serve.
- [ ] 46.6 **Bounded and bimodal targets specifically.** LGD lives in [0, 1] and piles up at
      both ends. Verify: a synthetic bimodal-bounded task is scored and the predicted
      distribution is shown to place mass at both modes rather than averaging to the middle —
      the specific failure a point estimate or a Gaussian head would exhibit.
- [ ] 46.7 **Only then, coverage.** Re-run the TabArena eligibility count and report the new
      fraction. Verify: the coverage number is recomputed from TabArena's task metadata rather
      than assumed from this proposal's estimate.
      **Counted, not closed.** `eligible_datasets()` derives **46 of 51 (90%)** from the task
      metadata — 27 binary, 7 multiclass, 12 regression, with `max_features=136` the only
      remaining exclusion. The estimate this proposal carried (27/19/5) was wrong; the suite
      is 30/13/8. This stays open deliberately: "only then" means after 46.5 and 46.6, and a
      coverage fraction reported before the arm behind it has been scored is a capability
      claim with no measurement under it.
