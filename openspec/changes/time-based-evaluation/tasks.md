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
- [ ] 3.3 **Progress (2026-11-17, FINDINGS §272, MEASURED, one seed, one cut): on a 100,000-row V4FinBench sample the later window scores higher for six of seven models by AP and all seven by ROC-AUC; test base rates differ (1.467% random, 0.988% time), so the AP gap is not like-for-like. The `--credit` unit timed out at 5 h; the gap comes from the `--time-split` fallback. Open: the literal Verify command, and more than one cut or seed before a gap is quoted with an interval.** **Progress (2026-11-15, run made ready, still open): the Verify's `--credit` panels are
      the UCI sets, so the V4FinBench numbers come from `--time-split` (also accepted with
      `--credit`). The side-by-side block now prints average precision first (random AP, time AP,
      gap), then ROC-AUC, beside the train/test row counts, the companies in both windows, the
      train rows dropped, and the test base rate of each window; `--out FILE.json` writes all of
      it (`details` key; tested in `tests/test_bench_credit.py`). Arms: logreg, random_forest,
      gboost, lightgbm, catboost, xgboost and fintfm. Command for the MEASURED run (the lead
      queues it; check `uptime` first):
      `env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 15 env -u VIRTUAL_ENV uv run --frozen
      --extra bench fintfm-bench --time-split --max-rows N --v4-root
      /Users/morganoko/proj/fintfm/data/cache/v4finbench --model runs/v4-cellattn-labels.pt --out
      runs/time_split_N.json > runs/time_split_N.log 2>&1; echo EXIT=$?`.
      SMOKE-TEST (`--max-rows 6000`, 3,556 train / 2,391 test rows, 51 companies in both windows,
      53 train rows dropped, test base rate 0.878% time vs 1.297% random, 21 test positives):
      480 s wall, 5.6 GB peak RSS (`/usr/bin/time -l`), on a machine at load ~14 beside another
      campaign; the fintfm arm took 237 s (random window) and 187 s (time window), every other
      arm under 11 s. A 30,000-row run (16,859 / 11,824 rows) was killed after 16 min wall with
      the first fintfm arm unfinished (4.4 GB RSS then), so 30,000 rows exceeds the smoke budget.
      ESTIMATE (not measured): fintfm cost is about 40 ms per row of train plus test at 6,000
      rows, so --max-rows 100000 is roughly 2 h per window pair on a loaded machine plus the
      boosters, and the full 1,000,087-row panel roughly 11 h per fintfm arm, 22 h for both
      windows if linear; scaling past 6,000 rows is unmeasured, so run 50,000 first and time it.
      The FINDINGS gap entry still needs that MEASURED run; at 21 test positives the smoke
      numbers say nothing about the gap.**
      **Progress (2026-11-15): `fintfm-bench --credit --time-split` now prints both in one
      run (`run_credit(time_panel=...)`, CLI loads V4FinBench); plain `--credit` states per
      panel "time-based split: NOT AVAILABLE" and a closing "TIME-BASED SPLIT: NOT RUN" line,
      so the absence is never silent; tests in `tests/test_bench_credit.py`. Still missing:
      the FINDINGS gap entry from a MEASURED run with a real checkpoint (not written here).**
      **Progress (2026-11-08): code and SMOKE-TEST run done; the FINDINGS entry is not written because the only checkpoint run was a smoke test.** Report random and time-based numbers side by side in `bench.py`. Verify:
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
- [x] 3.4 **Done (2026-11-15): `docs/roadmap/STRATEGY.md` Phase 1 now carries the requirement
      (line 332, `grep -n "time-based"`); the harness refuses a panel with
      `has_period_labels=False` (`bench.py::run_time_vs_random` raises ValueError, pinned by
      `tests/test_bench_credit.py::test_time_split_refuses_a_panel_without_periods`).** Update `docs/roadmap/STRATEGY.md` Phase 1 to require the time-split number, since the phase
      currently says "with time-based splits" while the harness does not do it. Verify:
      `grep -n "time-based" docs/roadmap/STRATEGY.md` shows the requirement, and the harness refuses to
      run a time split on a dataset with `has_period_labels=False`.
