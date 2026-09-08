# Tasks

- [x] 1.1 Build the ablation harness with matched-compute enforcement and re-derivable
      `results.json`. Verify: `uv run pytest tests/test_experiments.py -q`.
      **Done 2026-09-08: 4 tests, refuses to report on diverging parameter counts**
- [x] 1.2 Add device selection so it can run on Metal. Verify: `uv run fintfm-ablate --help`
      shows `--device`; `docs/COMPUTE.md` records 0.702 s/step on mps vs 2.476 on cpu at
      identical loss. **Done 2026-09-08**
- [x] 1.3 Fix the device mismatch in the held-out eval path. Verify:
      `uv run pytest tests/test_model.py -q` passes the device-parametrised training test on
      both cpu and mps. **Done 2026-09-08 — this crashed the first run at step 500**
- [ ] 1.4 Complete the 5,000-step × 3-variant run on Metal (~3 h). Verify:
      `runs/phase1-5k/results.json` exists and `summarise()` prints a verdict rather than
      "cannot be evaluated".
- [ ] 1.5 Record the outcome in `docs/FINDINGS.md` as MEASURED, with the delta between
      financial and generic and the honest read of whether the exit condition was met.
      Numbers re-derived from `runs/phase1-5k/results.json`, never retyped.
- [ ] 1.6 If the exit condition is met, rerun at ~12M parameters (`--d-model 384
      --n-layers 8 --steps 20000`, ~1.5-2 days on Metal) to check the effect survives scale.
      If it is not met, open a proposal for the validation-layer-only pivot instead.
- [ ] 1.7 Repeat the winning configuration across at least 3 seeds before any number from it
      is quoted outside this repository. One seed cannot separate a 0.01 AUC difference from
      noise.
