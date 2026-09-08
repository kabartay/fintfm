# Tasks

- [x] 13.1 Implement the probe with stratified subsampling, a fixed test set, and calibration
      reported at every size. Verify: `uv run pytest tests/test_experiments.py -q`.
      **Done 2026-09-08: `sample_efficiency_probe`, sizes 100-4000, CLI
      `fintfm-ablate --sample-efficiency <ckpt>`**
- [x] 13.2 Add the untrained random-init control to the ablation so a tie is interpretable.
      Verify: a test asserting the control exists with `steps == 0`. **Done 2026-09-08**
- [ ] 13.3 Run the probe on the Phase 1 winning checkpoint. Verify:
      `runs/*/sample_efficiency.json` exists and reports a crossover size.
- [ ] 13.4 Record the crossover in `docs/FINDINGS.md`, re-derived from that JSON, and state
      plainly whether the model wins anywhere. **If it wins nowhere, say so in the finding and
      in `docs/STRATEGY.md`, and do not soften it.**
- [ ] 13.5 Extend the probe to the Taiwan panel and to multiple seeds, so a crossover is not
      one draw on one dataset. Verify: crossover reported with a range across seeds.
- [ ] 13.6 Add the probe to `bench.py`'s default credit output, so full-panel-only numbers
      cannot be reported again by accident. Verify: `fintfm-bench --credit` includes it.
