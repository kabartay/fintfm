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
- [x] 13.4 Record the crossover in `docs/FINDINGS.md`, re-derived from that JSON, and state
      plainly whether the model wins anywhere. **If it wins nowhere, say so in the finding and
      in `docs/STRATEGY.md`, and do not soften it.**
      **Done 2026-09-08, `FINDINGS` §12. It wins on AUC only at n=100; the real finding is
      that calibration is 2.3-11.7x better at EVERY size.**
- [x] 13.5 Extend the probe to multiple seeds. Verify: crossover reported with a range
      across seeds. **Done 2026-09-08: 5 seeds. AUC win survives only at n=100
      (+0.069 +/- 0.040); calibration advantage stable everywhere.** Taiwan still open below.
- [ ] 13.7 **Highest priority follow-up:** compare against a *calibrated* gradient boosting
      baseline (Platt or isotonic on a validation split). The 11.7x calibration advantage is
      currently measured against an uncalibrated incumbent, which is the obvious counter and
      must be pre-empted rather than discovered by a reviewer.
- [ ] 13.8 Extend the probe to the Taiwan panel and to V4FinBench once ingested. Verify: the
      calibration advantage reported per panel, and stated plainly if it does not replicate.
- [ ] 13.6 Add the probe to `bench.py`'s default credit output, so full-panel-only numbers
      cannot be reported again by accident. Verify: `fintfm-bench --credit` includes it.
