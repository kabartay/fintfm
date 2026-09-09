# Tasks

- [ ] 37.1 Resolve the licence of every classification dataset in
      `inria-soda/tabular-benchmark` and record each one in `README.md`'s licensing section.
      The collection card declares none. Verify: a table of dataset to licence to
      commercial-use verdict, with any dataset whose terms cannot be established **excluded**
      rather than assumed permissive.
- [ ] 37.2 Add a loader for the licence-cleared classification subset, following
      `evaluation/datasets.py`'s conventions and caching under `data/cache/`. Verify: a test
      asserting row and column counts against the published dataset descriptions.
- [ ] 37.3 Score every arm across the suite: our checkpoints, the untrained control, logistic
      regression, LightGBM, CatBoost, XGBoost. Verify: a table with per-dataset results, not
      only aggregates.
- [ ] 37.4 Report native coverage and the **spread** across datasets, not just the mean. §44
      predicts our spread is wide where the boosters' is narrow; the mean would hide it.
      Verify: standard deviation and per-dataset range quoted in the finding.
- [ ] 37.5 Use this suite as the standing gate for every future prior or architecture change,
      replacing "does V4FinBench go up" — a benchmark one column solves cannot diagnose
      anything (§42). Verify: the gate written into `CLAUDE.md` as a standing rule, and the
      next prior or architecture change records its suite-wide spread before and after.
