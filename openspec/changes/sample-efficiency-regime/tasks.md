# Tasks

- [x] 13.1 Implement the probe with stratified subsampling, a fixed test set, and calibration
      reported at every size. Verify: `uv run pytest tests/test_experiments.py -q`.
      **Done 2026-09-08: `sample_efficiency_probe`, sizes 100-4000, CLI
      `fintfm-ablate --sample-efficiency <ckpt>`**
- [x] 13.2 Add the untrained random-init control to the ablation so a tie is interpretable.
      Verify: a test asserting the control exists with `steps == 0`. **Done 2026-09-08**
- [x] 13.3 Run the probe on the Phase 1 winning checkpoint. Verify:
      `runs/*/sample_efficiency.json` exists and reports a crossover size.
      **Done 2026-09-08 on the financial checkpoint: crossover at n=250 (single seed).**
- [x] 13.4 Record the crossover in `docs/results/FINDINGS.md`, re-derived from that JSON, and state
      plainly whether the model wins anywhere. **If it wins nowhere, say so in the finding and
      in `docs/roadmap/STRATEGY.md`, and do not soften it.**
      **Done 2026-09-08, `FINDINGS` §12. It wins on AUC only at n=100; the real finding is
      that calibration is 2.3-11.7x better at EVERY size.**
- [x] 13.5 Extend the probe to multiple seeds. Verify: crossover reported with a range
      across seeds. **Done 2026-09-08: 5 seeds. AUC win survives only at n=100
      (+0.069 +/- 0.040); calibration advantage stable everywhere.** Taiwan still open below.
- [x] 13.7 **Done (2026-11-16, FINDINGS §270, with a dated note in §12): the ratio is stated per panel and size in §270 and §12 now points to it; no single ratio holds.** **Progress (2026-11-16, FINDINGS §270): the probe was run with `--calibration-frac 0.3` on the
      published binary checkpoint (SHA-256 d231b062...) on Polish 3-year, Taiwanese and a 200,000-row
      V4FinBench pool, one seed per cell. Against `gboost_platt_holdout` and `gboost_isotonic_holdout`
      FinTFM has the lower ECE in 5 and 7 of 13 scorable cells respectively, so the ratio is stated per
      panel in §270 and no single ratio holds. Still owed: the Verify clause also asks that §12 be
      updated with the ratio; §12 carries no pointer to §270 yet, so the box stays open.**
      **Progress (2026-11-15): the probe now carries held-out-slice calibrated boosting arms
      `gboost_platt_holdout` and `gboost_isotonic_holdout` (`calibrated_gboost_holdout_arms` in
      src/fintfm/experiments/prior_ablation.py), calibrated on a stratified slice of the training
      rows only (`--calibration-frac`, default 0.3, recorded per size as `calibration_holdout`),
      beside the existing cross-fitted `gboost_sigmoid`/`gboost_isotonic`; AUC, AP, ECE, Brier and
      the spread fields are reported for each, and a single-class or unstratifiable slice is skipped
      with a recorded reason. Tested with a stub model (no checkpoint). NOT yet run on a real
      checkpoint, so no ratio exists and FINDINGS §12 is not updated. Run:
      `uv run fintfm-ablate --sample-efficiency <checkpoint.pt> --calibration-frac 0.3`; what
      remains: that run, then re-derive the calibration ratio against these arms into §12.**
      **Highest priority follow-up:** compare against a *calibrated* gradient boosting
      baseline (Platt or isotonic on a validation split). The 11.7x calibration advantage is
      currently measured against an uncalibrated incumbent, which is the obvious counter and
      must be pre-empted rather than discovered by a reviewer. Verify: the probe reports a
      third arm (`gboost_calibrated`) and `docs/results/FINDINGS.md` §12 is updated with the ratio
      against it, re-derived from the run.
- [x] 13.8 **Done (2026-11-16, FINDINGS §270): the probe ran on all three panels (Polish 3-year locally,
      Taiwanese and a company-grouped 200,000-row V4FinBench pool on a Kaggle CPU kernel at e475151). The
      calibration comparison is reported per panel and does not replicate: ECE ratio against the
      held-out-slice arms is 0.2x to 5.6x (Polish), 0.4x to 1.7x (Taiwanese), 0.1x to 1.1x (V4FinBench
      pool), one seed, no intervals.**
      **Progress (2026-11-15): the probe takes `--panel {polish3,taiwan,v4finbench}` (default
      `polish3`, as before; `load_probe_panel` in src/fintfm/experiments/prior_ablation.py).
      V4FinBench is subsampled company-grouped to `--v4-pool-rows` company-year rows (default
      200,000) and split company-grouped, so no company has rows on both sides; a panel wider than
      the checkpoint's `max_features` keeps its first columns for every arm. The JSON records
      `panel`, `split`, `panel_rows`, `pool_rows`, `pool_base_rate`, `test_rows`,
      `test_base_rate` and, for V4FinBench, the requested pool, full-panel rows and base rate.
      Tested with monkeypatched loaders and a stub model only; NOT yet run on a real checkpoint,
      so no per-panel calibration ratio exists. Run, one per panel (separate `--out` so the
      files do not overwrite each other):
      `uv run fintfm-ablate --sample-efficiency <checkpoint.pt> --panel polish3 --out runs/se_polish3`;
      `uv run fintfm-ablate --sample-efficiency <checkpoint.pt> --panel taiwan --out runs/se_taiwan`;
      `uv run fintfm-ablate --sample-efficiency <checkpoint.pt> --panel v4finbench --v4-pool-rows 200000 --out runs/se_v4finbench`.
      What remains: those runs, then the per-panel calibration ratio and a plain statement of
      whether it replicates.** Extend the probe to the Taiwan panel and to V4FinBench once ingested. Verify: the
      calibration advantage reported per panel, and stated plainly if it does not replicate.
- [x] 13.6 **Done (2026-11-11, src/fintfm/evaluation/bench.py `run_credit`/`--no-sample-efficiency`, src/fintfm/experiments/prior_ablation.py `ds=` param, tests/test_bench_credit.py::test_credit_output_includes_the_sample_efficiency_probe).** Add the probe to `bench.py`'s default credit output, so full-panel-only numbers
      cannot be reported again by accident. Verify: `fintfm-bench --credit` includes it.
