# Tasks

- [x] 14.1 **Done (2026-11-15, evidence): already closed under the same ID further down this file (`FINDINGS` §16, 2026-09-08). §16 reports four arms (fintfm, gradient boosting raw, Platt, isotonic) and §12 carries the pointer that tempers its 11.7x ratio to about 2x. Caveat: §16 calibrated by `CalibratedClassifierCV(cv=3)` cross-fitting on the training data, not a single held-out slice; the held-out-slice arms exist in code but are unrun (`sample-efficiency-regime` 13.7).** Add Platt- and isotonic-calibrated gradient boosting arms to the probe, fitted on a
      validation split carved from the training data. Verify: the probe reports four arms and
      `docs/results/FINDINGS.md` §12 is updated with the ratio against the calibrated one, re-derived
      from the run.
- [x] 14.2 **Done (2026-11-15, evidence): already closed under the same ID further down this file; `FINDINGS` §13 poses the discriminating test and §14/§15 report generic against financial prior ECE and conclude calibration tracks prior breadth, not the financial prior.** Compare financial versus generic prior on ECE at matched compute, using the Phase 1
      checkpoints. Verify: numbers into `docs/results/FINDINGS.md` §13's discriminating-test section,
      stating plainly whether calibration is domain-specific or generic to PFNs.
- [x] 14.3 **Done (2026-11-15): the 2026-11-08 note below claimed a test that was never committed; it now exists. `spread_fields()` in `src/fintfm/evaluation/metrics.py` is the shared helper (wraps `prediction_spread()`); it now also feeds `hybrid_catboost._metrics` and `credit_suite.score_row` (new `COLUMNS` appended, printed lines too). Tests in `tests/test_metrics.py`: `test_spread_fields_constant_prediction_is_flagged_despite_near_zero_ece`, `test_spread_fields_zero_median_guard_and_varied_predictions`, `test_every_reporting_path_carries_spread_fields`. The frozen Kaggle script `scripts/kaggle/s6e11_fintfm/s6e11_fintfm.py` is left unchanged on purpose.** Report prediction spread (sd, p99/p50) beside every calibration number, so
      conservatism cannot be mistaken for skill. Verify: `CreditMetrics` carries a spread
      field and a test asserts a near-constant predictor is flagged despite a near-zero ECE.
      **Partly done 2026-09-08 via 14.4's skill score and degenerate flag; the explicit
      spread fields (sd, p99/p50) are still open.**
      **Done (2026-11-08, `CreditMetrics.pred_sd/pred_p50/pred_p99/pred_p99_over_p50/is_near_constant`,
      `prediction_spread()`, test `test_near_constant_predictor_is_flagged_despite_near_zero_ece`).
      Spread now reaches every `evaluate_binary` caller and the printed lines of bench, the prior
      probe, the context sweep and the V4 out-of-time table. No emitter writes a calibration
      slope; none was added.**
- [x] 14.4 Add a degenerate-predictor guard: a model predicting the base rate for everyone has
      ECE near zero and AUC 0.5, and must never read as a success. Verify: a test constructing
      that predictor and asserting the scorecard marks it useless.
      **Done 2026-09-08. `CreditMetrics.brier_skill` and `.is_degenerate`, spec E9,
      `FINDINGS` §17. The constant predictor turned out to beat every trained model here on
      ECE, which is why four earlier findings needed amendment.**
- [x] 14.1 Add calibrated gradient boosting arms. **Done 2026-09-08, `FINDINGS` §16.**
- [x] 14.2 Compare financial versus generic on calibration. **Done via `FINDINGS` §14/§15:
      calibration tracks prior breadth, not the financial prior, so it is a method property.**
- [x] 14.5 **Done (2026-11-16): D20 in docs/design/DECISIONS.md adopted (owner delegated, recommended option taken): the calibration claim is stated narrowly, as shrinkage, always beside Brier skill, AUC and prediction spread; reversal condition (b) waits on 4.3's measured comparison.** **Progress (2026-11-15): 14.1 and 14.2 are closed (`FINDINGS` §16, §14/§15), so the evidence for D20's main claim exists and D20 stands drafted. Still missing for closure: the owner's adoption of D20, and the multi-horizon, >= 3-seed fitted-calibration comparison (`fitted-calibration` 4.3, still SMOKE-TEST only), which D20's reversal condition (b) depends on. Not decided here.** Record the outcome in `docs/design/DECISIONS.md` as a new decision on what the defensible
      claim is, with its reversal condition. Verify:
      `uv run python openspec/tools/validate.py` passes.
      **Progress (2026-11-08): drafted as a PROPOSED decision, `docs/design/DECISIONS.md` D20, from FINDINGS
      §13, §15, §16, §17. Awaiting owner adoption and the fitted-calibration results.**
