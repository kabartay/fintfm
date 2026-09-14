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
- [x] 1.4 Complete the 5,000-step × 3-variant run on Metal (~3 h). Verify:
      `runs/phase1-5k/results.json` exists and `summarise()` prints a verdict rather than
      "cannot be evaluated". **Done 2026-09-08. Note: launched before the untrained control
      and paired test existed, so the control is absent and the paired test was run
      post-hoc by re-evaluating the saved checkpoints into results-paired.json.**
- [ ] 1.5 Record the outcome in `docs/FINDINGS.md` as MEASURED, with the delta between
      financial and generic and the honest read of whether the exit condition was met.
      Numbers re-derived from `runs/phase1-5k/results.json`, never retyped. Verify:
      `uv run python openspec/tools/validate.py --findings` passes and the finding cites the
      paired test output rather than a win count. **Done 2026-09-08, `FINDINGS` §14: exit
      condition A met (3/6 cells significant, 0/6 against), and the more useful result is
      that mixed p=0.7 dominates on Brier.**
- [ ] 1.8 Re-run with the untrained control included, since this run predates it. Verify:
      `results.json` contains a variant with `steps == 0` and the finding states whether any
      trained variant beats random initialisation.
- [ ] 1.6 If the exit condition is met, rerun at ~12M parameters (`--d-model 384
      --n-layers 8 --steps 20000`, ~1.5-2 days on Metal) to check the effect survives scale.
      If it is not met, open a proposal for the validation-layer-only pivot instead. Verify:
      a second `results.json` at the larger config, or a new directory under
      `openspec/changes/` if the pivot is triggered.
- [ ] 1.7 Repeat the winning configuration across at least 3 seeds before any number from it
      is quoted outside this repository. One seed cannot separate a 0.01 AUC difference from
      noise. Verify: `results.json` per seed, and the finding reports mean +/- std rather than
      a single value.

## Superseded 2026-09-14

Tasks 1.5-1.7 above were written for the original smoke-scale ablation plan and are now
answered by a more rigorous body of work than this proposal specified: §61/§63 (a controlled
two-factor design isolating domain content from base rate, with paired-bootstrap significance),
§73/§75 (the same question re-run on Polish and Taiwan, real independent panels, opposite
ordering from V4FinBench), and §74/§76 (the Bayes-ceiling test and its bisection, which found
the financial prior's cost is not confined to one benchmark — it caps basic signal extraction
everywhere, including on tasks with no relationship to credit at all).

**The exit condition this proposal was written to test is answered, and it is not the simple
yes/no the proposal expected.** The financial prior beats generic on exactly one benchmark
(V4FinBench, +0.098 AP, Holm p=0.002) and loses to it everywhere else measured (two real credit
panels, one exact-Bayes-ceiling synthetic probe). Task 1.6's scale-up question is now secondary
to `cell-attention-and-task-inference`'s architecture question, per the user's explicit
priority (2026-09-14): the cap does not track any prior-content axis bisected so far (§76), so
scaling the financial prior before understanding the cap would very likely just scale the cost
alongside the one narrow benefit.

- [x] 1.5 **Done, superseded** — see above. The honest read of "does a financial prior beat a
      generic one" is now: on one specific benchmark, yes, controlled and significant; as a
      general claim, no, and training on it carries a broad, serious, currently unexplained
      cost. Not the single number this task asked for, and more informative for it.
