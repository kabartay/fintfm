# Tasks

- [ ] 45.1 **Install TabArena in its own environment**, in the existing clone at
      `../tabarena`, never inside this project's venv. Verify: `uv run pytest` in *this*
      repository still reports the same test count and the same single expected skip
      afterwards, so the benchmark's dependency tree provably did not leak into fintfm's.
- [ ] 45.2 **Write the model folder and make the registry's own fit-test pass.** TabArena
      fit-tests every registered model through `tests/tabarena/models/test_all_models.py`; no
      per-model test is written. Verify: that test passes for the fintfm entry, which is
      TabArena's own definition of a correctly-integrated model rather than ours.
- [ ] 45.3 **Declare capability honestly and measure coverage.** `_supported_problem_types`
      must be `["binary"]`, and the feature cap must cause a skip rather than a failure.
      Verify: the run reports how many of the suite's tasks were actually scored, and that
      number appears beside every score quoted from it — a benchmark that silently measures
      fewer datasets than it claims is the failure `CLAUDE.md` records under "distrust a jump
      in the skip count".
- [ ] 45.4 **Run TabArena-Lite and place the result against published numbers.** Verify: the
      report states fintfm's score, the published score of at least one foundation model and
      one tuned tree ensemble on the same suite, and the coverage fraction — so the comparison
      is to the field rather than to baselines we ran ourselves.
- [ ] 45.5 **Check BeyondArena's temporal and grouped tasks separately.** These match
      V4FinBench's structure and are the only non-credit temporal data available to this
      project; an aggregate over IID tasks would hide whatever is specific to that regime.
      Verify: temporal/grouped results are reported as their own row, never pooled with IID.
- [ ] 45.6 **Decide on lifting the caps, with evidence.** Multiclass and wider feature support
      are product decisions, not benchmarking conveniences. Verify: the coverage fraction from
      45.3 and the per-regime results from 45.5 are cited in whichever direction the decision
      goes, rather than the caps being lifted because a leaderboard rewards it.
