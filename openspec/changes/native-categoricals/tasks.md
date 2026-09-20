# Tasks

- [x] 47.1 **Implement out-of-fold smoothed target encoding** in `inference/categorical.py`.
      Verify: a test shows the naive statistic correlates above 0.9 with the row's own label
      on a column of unique levels while the out-of-fold statistic is constant at the prior —
      without that contrast the test would pass for an encoder that leaks.
      **Done:** `tests/test_categorical.py`, 8 tests, including level-effect recovery so
      leak-freeness cannot be bought by encoding nothing.
- [ ] 47.2 **Measure the encoder against label encoding on one checkpoint.** Verify: both arms
      run under separate TabArena result directories, because results are cached per config
      name and a preprocessing change does not invalidate that cache — a re-run in place
      returns the previous numbers unchanged and would read as "no effect".
- [ ] 47.3 **Re-run the full eligible suite** and report the new mean, rank and the
      categorical/numeric split from §100. Verify: the numeric-only subset's gap is reported
      separately, since that is the part this change cannot move and reporting only the mean
      would credit the encoder for a residual it did not touch.
- [ ] 47.4 **Decide on a learned categorical path** from 47.3's result rather than in advance.
      Verify: the decision is recorded in `docs/DECISIONS.md` with the measured gain that
      triggered it and the threshold that would reverse it.
- [ ] 47.5 **Handle the unseen-level case on real data specifically.** High-cardinality columns
      guarantee query levels absent from context. Verify: the fraction of query cells falling
      back to the prior is reported per dataset — an encoder silently returning the prior for
      most of a column is not encoding that column, and the mean score would hide it.
