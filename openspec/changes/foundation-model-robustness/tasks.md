# Tasks

- [x] 42.1 **Done (2026-11-15): FINDINGS §245, figure docs/results/figures/irrelevant_features_2026-11-15.svg: AUC against total width 2 to 100 for fin00 and fin10 (§74's pair) with gradient boosting, logistic regression and the untrained control, three seeds; fin00 flat (−0.0011 from d = 2 to 100), fin10 +0.0396 from a weak d = 2 point.** **Progress (2026-11-15): harness built, NOT yet run. `src/fintfm/experiments/irrelevant_features.py` (tests: `tests/test_irrelevant_features.py`, SMOKE-TEST only) fixes `y = f(x0, x1)` (`x0 + x1 + 0.5*x0*x1`, thresholded at a 25% rate), appends k independent N(0,1) columns so total width d = 2, 5, 10, 20, 40, 70, 100, and scores each checkpoint, a seeded untrained control, logistic regression and histogram gradient boosting over seeds, writing `irrelevant_features.json`, a printed ROC-AUC/AP table, mean prediction beside observed rate, and the slope (per 10 columns, per decade of width, total change). Command to measure it (check `uptime` first and get a go-ahead; load was 250 when this was written): `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 15 env -u VIRTUAL_ENV uv run --frozen python -m fintfm.experiments.irrelevant_features --models fin00=runs/dl/v4-cellattn-fin00.pt,fin10=runs/dl/v4-cellattn-fin10.pt --out runs/irrelevant_features --seeds 0,1,2 > runs/irrelevant_features.log 2>&1; echo EXIT=$?` Checkpoint paths are those named in FINDINGS §74/§137 and must be confirmed to exist. ESTIMATED CPU time: not derivable from the tiny-model test (2.0 s for 8 in-context fits on a 16-wide model at n=200, on a loaded machine); the real run is 7 widths x 3 seeds x 3 in-context arms = 63 fits at 1,000 context rows, so time `--seeds 0 --widths 2,100` first and scale. Same measurement as 39.13 in `cell-attention-and-task-inference`; one run closes both. Left open: Verify needs the measured curve plotted and its slope reported.** **Irrelevant-feature robustness curve.** No dependency; can start now. Fixed task
      `y=f(x0,x1)`, noise dimensions added up to d=100, on existing checkpoints first (no new
      training). Verify: AUC-vs-d curve plotted for at least the fin00/fin10 pair from §74, to
      see whether the same architectural split shows up here too.
- [ ] 42.2 **Progress (2026-11-15): duplicate of `cell-attention-and-task-inference` task 39.12, whose harness `src/fintfm/experiments/schema_variation.py` is built and not yet run; see that task's Progress note for the run command.** **Schema-variation sweep**, blocked on `mechanism-diverse-prior` task 40.2's
      task-family generators. Verify: recovered feature-importance ranking compared against
      the true relevant set as width grows from 5 to 200.
- [ ] 42.3 **Semantic column information, mode A vs B vs C.** No hard dependency, though most
      informative once real-panel evaluation (§73/§75) is stable. Verify: A/B/C compared on the
      same checkpoint family and the same real panels §73/§75 already measured, since this is
      specifically about transfer to tables with meaningful column names.
- [ ] 42.4 **Active context selection**, blocked on `cross-domain-coverage` task 38.13
      (retrieval's mechanism). Verify: an active-selection strategy is compared against uniform
      and against whatever retrieval fix task 38.13 produces, on the same fold, same paired-
      bootstrap methodology as §69/§71.
