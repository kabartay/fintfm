# Tasks

- [x] 3.1 Determine whether the UCI ARFF files carry any per-observation date or period
      field. Verify: print the attribute list from each file. If absent, mark this change
      blocked on `second-credit-panel` and say so in `docs/roadmap/NEXT.md`.
      **Done 2026-09-08: NO dates, no periods, no company identifiers — 64 anonymous
      numeric attributes plus a binary class. This change is BLOCKED on
      `second-credit-panel`; it cannot be done on the only dataset we hold. See
      `docs/results/FINDINGS.md` §7.**
- [x] 3.2 **Done (2026-11-08, this commit)** Implement the splitter in `evaluation/`, refusing to run if any identifier appears
      in both windows. Verify: a test constructing an overlapping panel and asserting it
      raises.
      **Evidence: `src/fintfm/evaluation/splits.py::time_based_split` (raises by default on a
      shared identifier; `on_overlap="drop_from_train"` is an explicit opt-in) and
      `tests/test_time_split.py::test_overlapping_panel_raises`. The only prior splitter,
      `experiments/v4_out_of_time.py:76 time_split`, checks year windows only and lets a firm
      sit on both sides, so it was not equivalent; it is left in place. `SurvivalDataset`
      gained an optional `groups` field (V4FinBench `emis_id`).**
- [ ] 3.3 **Progress (2026-11-08): code and SMOKE-TEST run done; the FINDINGS entry is not written because the only checkpoint run was a smoke test.** Report random and time-based numbers side by side in `bench.py`. Verify:
      `uv run fintfm-bench --credit` prints both, and the gap is recorded in
      `docs/results/FINDINGS.md` re-derived from the run output.
      **Implemented as `uv run fintfm-bench --time-split [--max-rows N] [--v4-root DIR]`
      (`bench.py::run_time_vs_random`), not under `--credit`: the `--credit` panels are the UCI
      sets, which have no period column (3.1), so `--credit` output is unchanged. SMOKE-TEST
      (`runs/v0-smoke.pt`, 16 features of 136, 30,000-row V4FinBench sample, train<=2016 /
      test>=2017, 16,859 / 11,824 rows, 110 test positives, 1,180 companies in both windows,
      1,317 train rows dropped): random vs time AUC, logreg 0.6595 / 0.7201, random_forest
      0.8504 / 0.8922, gboost 0.8592 / 0.9127, lightgbm 0.8601 / 0.8978, catboost 0.8579 /
      0.9114, xgboost 0.8372 / 0.8668, fintfm 0.5499 / 0.5321 (degenerate checkpoint). The
      time split scores HIGHER for the baselines on this sample, so no random-split optimism
      shows; with ~110 test positives and the first 16 columns only, this proves the pipeline
      and says nothing about the gap. Remaining: a MEASURED run on the full panel with a real
      checkpoint, then the FINDINGS entry.**
- [ ] 3.4 Update `docs/roadmap/STRATEGY.md` Phase 1 to require the time-split number, since the phase
      currently says "with time-based splits" while the harness does not do it. Verify:
      `grep -n "time-based" docs/roadmap/STRATEGY.md` shows the requirement, and the harness refuses to
      run a time split on a dataset with `has_period_labels=False`.
