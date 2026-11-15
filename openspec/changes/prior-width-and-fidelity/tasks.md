# Tasks

- [ ] 15.1 Add a ratio-family generator over the existing latent accounts, with sampled
      numerator/denominator pairs and named ratio classes. Verify: `uv run pytest
      tests/test_prior.py -q`, plus a test that a task can reach `max_features` columns.
- [x] 15.2 **Done (2026-11-15): the cap was already gone in fact (the ratio-family generator, `_ratio_family`, fills width up to `max_features`; the project trains at 136 and 512). The Verify test is `tests/test_prior_width_fidelity.py::test_max_features_64_gives_substantially_more_than_24_columns`.** Remove the effective ~24-column cap so `max_features` is honoured. Verify: a test
      asserting a sampled task with `max_features=64` has substantially more than 24 columns.
- [ ] 15.3 Re-measure §18's four properties (width, logistic-regression difficulty,
      correlation, missingness) against the real panels. Verify: numbers into
      `docs/results/FINDINGS.md` §18 as a follow-up table, re-derived from the probe script.
      **A lost difficulty match is a regression, not a side effect.**
- [ ] 15.4 **Progress (2026-11-15): the option is implemented, the measurement is pending.** `sample_financial_task(max_lr_auc=...)` / `PriorConfig.financial_max_lr_auc` / `prior.financial_max_lr_auc` redraw a task whose held-out logistic-regression AUC exceeds the ceiling (up to 20 draws); default `None` is byte-identical (tested). The before/after difficulty distribution and the real-panel coverage comparison need a probe run and are not yet measured; the task stays open. Consider rejecting near-separable draws (logistic-regression AUC above ~0.95).
      Verify: report the difficulty distribution before and after, and state whether the
      real-panel range is better covered.
- [ ] 15.5 Retrain at matched compute against the current prior and compare on real panels
      with the paired test. Verify: `results.json` for both, and a finding stating whether
      width was the binding constraint — including if it was not.
      **Partially done 2026-09-09 (`docs/results/FINDINGS.md` §30):** the low-default prior was
      compared against the previous one on a real panel out of time, and is worth +0.023 mean
      AUC and 2.7x better calibration under the configuration that works. **Not matched
      compute** — the two checkpoints differ in step count as well as prior — so the
      attribution to the prior alone is not yet clean, and this task stays open for the
      matched-compute run.
