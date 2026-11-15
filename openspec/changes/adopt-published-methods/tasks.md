# Tasks

- [ ] 36.1 **Validate §42's prior fix.** Run `fintfm-capability` on the fixed-prior checkpoint
      against the old one and the untrained control. Verify: the trained model clears the
      untrained control by a wide margin on `linear` and `conjunction`, recorded as a finding.
      **This gates everything below** — the fix is currently believed rather than shown.
- [x] 36.2 **Ensemble over context draws and feature permutations**, TabPFN's standard
      practice and entirely untried here. Cheap, and usually a reliable small gain. Verify:
      AUC and ECE against a single draw on V4FinBench, with the added inference cost stated,
      three seeds.
      **Done 2026-09-10, and the answer is no (`docs/results/FINDINGS.md` §45).** Context-draw
      ensembling changes nothing (0.6841 to 0.6838), and label-swap averaging drops the model
      to chance because the discrimination it removes *was* the label-slot asymmetry.
      Implementation kept as a diagnostic: a checkpoint whose score survives label-swap
      averaging has signal in the evidence rather than in the labelling. Revisit once one
      does.
- [ ] 36.3 **In-context scaling to larger tables**, following TabICL. §33 measured our model
      degrading past 2,000 context rows, and §31 traced the ceiling to pretraining on
      256-1,024-row tasks. Verify: whether pretraining on larger tasks lifts the ceiling, or
      whether the limit is architectural — `docs/infra/COMPUTE.md` prices 2,048-row tasks at 32× per
      step, so cost the experiment before running it.
- [ ] 36.4 **Progress (2026-11-15, third pass): `retrieval_grouping` is now covered (14 harnesses): every arm (the exact reference and each group size) records each horizon as `<arm>/h<k>`, scored or skipped with the `horizon_skip_reason` text, in `coverage` in `retrieval_grouping.json` and a printed line; attempted == scored + skipped is tested in tests/test_coverage_harnesses.py. Ruled out, with reasons: `credit_fdic` and `credit_estonia` (data downloaders and frame builders, they score nothing; `credit_estonia.read_elements` skips a missing yearly element file, which is a data-availability question for the loader, not a scoring skip), `tuned_boosting` (launcher for one split; a failed worker returns `None` and prints, the caller owns the count), `lightgbm_tuned` (scores every trial of one search, no skip path; `inner_split` raises on a single-class side). Still silent, found by grep of src/fintfm/evaluation and src/fintfm/experiments, so the Verify is not met: `sign_inference` (single-class task dropped by `continue`), `correlated_features` (same), `irrelevant_features` (same), `prior_ablation` learning-curve loop (train size above the pool, or a single-class sample, `continue`) and Polish-horizon loop (width mismatch, `continue`), `capability` multiclass probe and calibration cells (`continue` on a class-count shortfall, noted only in a `skipped` list), `context_sweep.paired_strategy_comparisons` (horizon `continue`), and `hybrid_catboost` line 626 (a tuned variant whose worker failed). Left open.** **Progress (2026-11-15, after merges e3be687 and 97e8ff8): 13 harnesses now report `n_attempted`/`n_scored`/`n_skipped` with reasons through `fintfm.evaluation.coverage.Coverage`: bench.run_synthetic, credit_suite, proxy_suite, openml_breadth, prior_score, capability.run, prior_ablation.evaluate_on_credit, v4_protocol.run, v4_out_of_time (`arms_coverage`), term_structure, representation_transfer, context_sweep and hybrid_catboost, each with an attempted == scored + skipped test (tests/test_coverage.py, tests/test_coverage_harnesses.py). Not yet checked: retrieval_grouping, credit_fdic, credit_estonia, tuned_boosting, lightgbm_tuned.** **Progress (2026-11-15): partly met. TabArena runs report a coverage fraction (`eligible_datasets()`, 51 of 51 with `select`, see 46.7) and `fintfm-bench` prints `n_scored/n_tasks` for the synthetic tasks, but not every experiment harness does, and no test asserts attempted = scored + explicitly skipped (grep of `tests/` finds none). Left open.** **Progress (2026-11-15, second pass): shared helper `fintfm.evaluation.coverage.Coverage` (`n_attempted`, `n_scored`, `n_skipped`, `coverage`, reason per skip; `to_dict` raises when attempted differs from scored plus skipped). Wired into `bench.run_synthetic` (returned record and printed line), `credit_suite` (printed line and `<out>.coverage.json`; CSV columns unchanged), `proxy_suite run` (printed line and `<out>.coverage.json`), `openml_breadth.run` (printed line; optional `coverage=` argument) and `prior_score.score_prior` (fields in its JSON, line in its summary). `tests/test_coverage.py` asserts the identity on synthetic inputs, with a single-class split counted as skipped with a reason in the bench, prior-score and proxy-suite cases. Not yet covered, so the Verify (every harness) is not met: `capability` (per-cell `n_seeds_scored` and `skipped` notes only), `prior_ablation.compare_strategies` (skips a too-wide panel with a printed note), `v4_protocol`, `v4_out_of_time` (NaN horizon means), `term_structure`, `representation_transfer`, `context_sweep`, `hybrid_catboost`; `hazard_period_probe` scores every task and cannot skip. `fintfm-bench` writes no JSON, so its record is returned, not stored. Left open.**  **Report native coverage** — the fraction of datasets or tasks scored without
      falling back — alongside accuracy, everywhere. One commercial model running 100% of TabBench while two
      of eight models silently failed on 19% and 29% is a real result, and an average over
      only the easy cases flatters a model exactly as §25 warned. Verify: every experiment
      harness reports a coverage fraction beside its aggregate, and a test asserts that
      attempted tasks equal scored plus explicitly-skipped.
- [x] 36.5 **Done (2026-10-16, D18).** Recorded as `docs/design/DECISIONS.md` D18: opt-in
      dataset-specific fine-tuning shipped in 0.7.0 (§163), off by default, provenance claim
      scoped to zero-shot use; alternatives and reversal condition stated, cross-referenced
      from D2 and D14.
      **Record the fine-tuned-variant decision in `docs/design/DECISIONS.md`**, either way. D2
      forbids training on real data because auditability is the product; Kostrzewa et al. show
      fine-tuning works and transfers across economies. Whether both offerings can coexist is
      a founder's call, and the task is only to write the answer and its reversal condition
      down rather than leave it implicit. Verify: a numbered decision in `docs/design/DECISIONS.md`
      with its alternatives and reversal condition, cross-referenced from D2.
