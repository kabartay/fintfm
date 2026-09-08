# Tasks

- [ ] 15.1 Add a ratio-family generator over the existing latent accounts, with sampled
      numerator/denominator pairs and named ratio classes. Verify: `uv run pytest
      tests/test_prior.py -q`, plus a test that a task can reach `max_features` columns.
- [ ] 15.2 Remove the effective ~24-column cap so `max_features` is honoured. Verify: a test
      asserting a sampled task with `max_features=64` has substantially more than 24 columns.
- [ ] 15.3 Re-measure §18's four properties (width, logistic-regression difficulty,
      correlation, missingness) against the real panels. Verify: numbers into
      `docs/FINDINGS.md` §18 as a follow-up table, re-derived from the probe script.
      **A lost difficulty match is a regression, not a side effect.**
- [ ] 15.4 Consider rejecting near-separable draws (logistic-regression AUC above ~0.95).
      Verify: report the difficulty distribution before and after, and state whether the
      real-panel range is better covered.
- [ ] 15.5 Retrain at matched compute against the current prior and compare on real panels
      with the paired test. Verify: `results.json` for both, and a finding stating whether
      width was the binding constraint — including if it was not.
