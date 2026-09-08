# Tasks

- [x] 2.1 Check whether the five UCI horizon files can be joined into a firm-level panel at
      all — they may not share company identifiers. Verify: report the join rate; if firms
      cannot be linked, this proposal's falsification test needs a different dataset and that
      changes its priority.
      **Done 2026-09-08: they cannot. No company identifier exists in the files, so there is
      nothing to join on (`docs/FINDINGS.md` §7). Superseded by V4FinBench (§8), which is a
      genuine company-year panel over 2006-2021 — tasks 2.2 onward now depend on
      `second-credit-panel` task 7.5 rather than being blocked outright.**
- [ ] 2.2 Falsification test: gradient boosting on levels versus levels+deltas+trends from a
      real panel. Verify: `uv run python -m fintfm.evaluation.bench` variant reporting both,
      numbers re-derived into `docs/FINDINGS.md`. **Do this before writing any generator.**
- [ ] 2.3 Add persistent firm effects and autocorrelated ratios to `prior/financial.py`,
      keeping the macro regime sampled rather than learned. Verify: `uv run pytest
      tests/test_prior.py -q` plus a new test asserting successive years correlate.
- [ ] 2.4 Sample the prediction horizon per task and expose it. Verify: a test that tasks
      drawn with different horizons produce different base rates in the expected direction.
- [ ] 2.5 Rerun `phase1-prior-ablation` with the temporal prior as a fourth variant. Verify:
      `results.json` shows whether temporal beats cross-sectional at matched compute.
