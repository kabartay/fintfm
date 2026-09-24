# Tasks

- [ ] 14.1 Add Platt- and isotonic-calibrated gradient boosting arms to the probe, fitted on a
      validation split carved from the training data. Verify: the probe reports four arms and
      `docs/results/FINDINGS.md` §12 is updated with the ratio against the calibrated one, re-derived
      from the run.
- [ ] 14.2 Compare financial versus generic prior on ECE at matched compute, using the Phase 1
      checkpoints. Verify: numbers into `docs/results/FINDINGS.md` §13's discriminating-test section,
      stating plainly whether calibration is domain-specific or generic to PFNs.
- [ ] 14.3 Report prediction spread (sd, p99/p50) beside every calibration number, so
      conservatism cannot be mistaken for skill. Verify: `CreditMetrics` carries a spread
      field and a test asserts a near-constant predictor is flagged despite a near-zero ECE.
      **Partly done 2026-09-08 via 14.4's skill score and degenerate flag; the explicit
      spread fields (sd, p99/p50) are still open.**
- [x] 14.4 Add a degenerate-predictor guard: a model predicting the base rate for everyone has
      ECE near zero and AUC 0.5, and must never read as a success. Verify: a test constructing
      that predictor and asserting the scorecard marks it useless.
      **Done 2026-09-08. `CreditMetrics.brier_skill` and `.is_degenerate`, spec E9,
      `FINDINGS` §17. The constant predictor turned out to beat every trained model here on
      ECE, which is why four earlier findings needed amendment.**
- [x] 14.1 Add calibrated gradient boosting arms. **Done 2026-09-08, `FINDINGS` §16.**
- [x] 14.2 Compare financial versus generic on calibration. **Done via `FINDINGS` §14/§15:
      calibration tracks prior breadth, not the financial prior, so it is a method property.**
- [ ] 14.5 Record the outcome in `docs/design/DECISIONS.md` as a new decision on what the defensible
      claim is, with its reversal condition. Verify:
      `uv run python openspec/tools/validate.py` passes.
