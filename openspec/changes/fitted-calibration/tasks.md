# Tasks

- [ ] 4.1 Hold out a validation slice inside `fit()` without leaking it into the context.
      Verify: a test asserting validation rows are absent from the stored context.
- [ ] 4.2 Fit temperature scaling on that slice; apply in `predict_proba`. Verify: a test
      that ranking is preserved (temperature scaling is monotone) as for the analytic shift.
- [ ] 4.3 Compare analytic, fitted and uncorrected across all horizons and >= 3 seeds.
      Verify: numbers into `docs/FINDINGS.md` §6, re-derived from the run, including the
      1-year case that motivated this change.
- [ ] 4.4 Set the default to whichever wins and record the decision in `docs/DECISIONS.md`
      D5, updating its reversal condition rather than appending a new decision. Verify:
      `uv run python openspec/tools/validate.py` passes and D5 names the measurement that
      settled it.
