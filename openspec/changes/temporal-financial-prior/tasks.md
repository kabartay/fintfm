# Tasks

- [x] 2.1 Check whether the five UCI horizon files can be joined into a firm-level panel at
      all — they may not share company identifiers. Verify: report the join rate; if firms
      cannot be linked, this proposal's falsification test needs a different dataset and that
      changes its priority.
      **Done 2026-09-08: they cannot. No company identifier exists in the files, so there is
      nothing to join on (`docs/results/FINDINGS.md` §7). Superseded by V4FinBench (§8), which is a
      genuine company-year panel over 2006-2021 — tasks 2.2 onward now depend on
      `second-credit-panel` task 7.5 rather than being blocked outright.**
- [x] 2.2 **Done (2026-11-15): `fintfm-bench --temporal-gbm` (also `python -m fintfm.evaluation.bench --temporal-gbm`; options `--v4-root`, `--max-rows`, `--seed`) runs `src/fintfm/experiments/temporal_gbm_falsification.py` in a subprocess so LightGBM never shares a process with torch, and prints levels versus levels+deltas+trends with AP and ROC-AUC per fold; `tests/test_bench_temporal_gbm.py` covers the command, the CLI flag and a synthetic-panel end to end; numbers are in FINDINGS §246. The per-horizon (h3 to h5) split is still not run.** **Progress (2026-11-15): numbers re-derived into FINDINGS §246 (falsified on this test: AP lower with trajectory on 3 of 3 folds). Still missing for the literal Verify: a `fintfm.evaluation.bench` variant; the comparison runs as `src/fintfm/experiments/temporal_gbm_falsification.py` because bench imports torch and torch with LightGBM segfaults on macOS. A per-horizon (h3 to h5) split is also not run.** **Progress (2026-11-15): harness `src/fintfm/experiments/temporal_gbm_falsification.py` run on V4FinBench (199,998 of 1,000,087 rows, 3 company-grouped folds, seed 0): deltas and trends did not raise AP (lower on 3 of 3 folds); numbers are in `docs/results/drafts/2.2_2026-11-15.md`, not yet in FINDINGS.md, so this stays open.** Falsification test: gradient boosting on levels versus levels+deltas+trends from a
      real panel. Verify: `uv run python -m fintfm.evaluation.bench` variant reporting both,
      numbers re-derived into `docs/results/FINDINGS.md`. **Do this before writing any generator.**
- [ ] 2.3 Add persistent firm effects and autocorrelated ratios to `prior/financial.py`,
      keeping the macro regime sampled rather than learned. Verify: `uv run pytest
      tests/test_prior.py -q` plus a new test asserting successive years correlate.
- [x] 2.4 **Done (2026-11-15): opt-in `PriorConfig.financial_sample_horizon` (needs `n_horizons`) draws one horizon per financial task, labels `y` as default within it and exposes it as `task.meta["horizon"]`; `tests/test_prior_horizon.py` shows longer horizons give higher base rates and that off reproduces the recorded task.** Sample the prediction horizon per task and expose it. Verify: a test that tasks
      drawn with different horizons produce different base rates in the expected direction.
- [ ] 2.5 Rerun `phase1-prior-ablation` with the temporal prior as a fourth variant. Verify:
      `results.json` shows whether temporal beats cross-sectional at matched compute.
