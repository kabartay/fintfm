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
- [ ] 46.3 **Implement the binned regression head.** Bin edges from the context targets'
      quantiles per task, logits over bins, cross-entropy on the bin index. Verify: a test
      asserts the predicted distribution integrates to one and that the induced point estimate
      recovers a known linear target to a stated tolerance on a synthetic task — a head that
      trains but cannot recover `y = 2x + noise` is not a regression head.
- [ ] 46.4 **Extend the prior to emit continuous targets.** `prior/scm.py` already computes a
      continuous latent before thresholding it into classes; regression tasks use that latent
      directly. Verify: a test asserts the regression prior spans difficulty in the same sense
      `tests/test_prior.py` already pins for classification — a prior of only easy or only
      noise targets teaches the wrong thing (§42).
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
