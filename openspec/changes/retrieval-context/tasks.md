# Tasks

- [ ] 17.1 Implement a `retrieval` context strategy selecting nearest neighbours on
      normalised features, balanced within class. Verify: a test that a query's context is
      closer to it on average than a uniform sample is.
- [ ] 17.2 Measure inference cost against blind sampling; retrieval must not make scoring
      impractical. Verify: timings into `docs/COMPUTE.md`, measured on the same hardware as
      §23's context sweep.
- [ ] 17.3 Compare all four strategies at matched context size on the UCI panels and
      V4FinBench, with the paired bootstrap from `metrics.py`. Verify: numbers into
      `docs/FINDINGS.md`, re-derived from the run.
- [ ] 17.4 If retrieval loses, record it plainly and state what it implies: context quality
      is not the lever, so §23's flatness points at model capacity and effort should move to
      scale. Verify: a finding that says so without hedging.
