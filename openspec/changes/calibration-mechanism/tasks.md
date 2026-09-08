# Tasks

- [ ] 14.1 Add Platt- and isotonic-calibrated gradient boosting arms to the probe, fitted on a
      validation split carved from the training data. Verify: the probe reports four arms and
      `docs/FINDINGS.md` §12 is updated with the ratio against the calibrated one, re-derived
      from the run.
- [ ] 14.2 Compare financial versus generic prior on ECE at matched compute, using the Phase 1
      checkpoints. Verify: numbers into `docs/FINDINGS.md` §13's discriminating-test section,
      stating plainly whether calibration is domain-specific or generic to PFNs.
- [ ] 14.3 Report prediction spread (sd, p99/p50) beside every calibration number, so
      conservatism cannot be mistaken for skill. Verify: `CreditMetrics` carries a spread
      field and a test asserts a near-constant predictor is flagged despite a near-zero ECE.
- [ ] 14.4 Add a degenerate-predictor guard: a model predicting the base rate for everyone has
      ECE near zero and AUC 0.5, and must never read as a success. Verify: a test constructing
      that predictor and asserting the scorecard marks it useless.
- [ ] 14.5 Record the outcome in `docs/DECISIONS.md` as a new decision on what the defensible
      claim is, with its reversal condition. Verify:
      `uv run python openspec/tools/validate.py` passes.
