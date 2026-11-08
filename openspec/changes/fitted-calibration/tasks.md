# Tasks

- [x] 4.1 **Done (2026-11-08, `calibration_val_frac` on `FinancialTFMClassifier`, default 0.0 byte-identical; `tests/test_classifier.py::test_calibration_slice_never_reaches_the_stored_context`).** Hold out a validation slice inside `fit()` without leaking it into the context.
      Verify: a test asserting validation rows are absent from the stored context.
- [x] 4.2 **Done (2026-11-08, one temperature fitted by log loss on the slice, applied to the final probabilities in `predict_proba`; `test_fitted_temperature_preserves_ranking_and_is_valid`; not applied to `predict_term_structure`).** Fit temperature scaling on that slice; apply in `predict_proba`. Verify: a test
      that ranking is preserved (temperature scaling is monotone) as for the analytic shift.
- [ ] 4.3 **Progress (2026-11-08): SMOKE-TEST only, one panel (Polish bankruptcy 3y, 7,352 train / 3,151 test rows), one seed, `v4-clf-medium.pt`, `max_context=1000`. Balanced context, ECE: uncorrected 0.2487, analytic 0.0076, fitted alone 0.0523 (T=0.403), analytic+fitted 0.0012 (T=1.019). Uniform context, ECE: uncorrected 0.0057, analytic 0.0063, fitted alone 0.0017, analytic+fitted 0.0005. The fitted arms train on 75% of the rows and so draw a different context; AUC 0.706-0.709 throughout. Says nothing about the real comparison: no 1-year horizon, one seed, ECE differences at the 0.001 level are noise. Still to do: all horizons, >= 3 seeds.** Compare analytic, fitted and uncorrected across all horizons and >= 3 seeds.
      Verify: numbers into `docs/results/FINDINGS.md` §6, re-derived from the run, including the
      1-year case that motivated this change.
- [ ] 4.4 Set the default to whichever wins and record the decision in `docs/design/DECISIONS.md`
      D5, updating its reversal condition rather than appending a new decision. Verify:
      `uv run python openspec/tools/validate.py` passes and D5 names the measurement that
      settled it.
