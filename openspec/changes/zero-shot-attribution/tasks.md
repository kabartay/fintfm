# Tasks

- [x] 12.1 **Done (2026-11-11, docs/research/REFERENCES.md "ExplainerPFN").** Read ExplainerPFN in full and record how attribution targets are generated and
      what fidelity they claim. Verify: entry expanded in `docs/research/REFERENCES.md`; currently
      only the abstract has been read.
- [x] 12.2 **Done (2026-11-15): FINDINGS §244: LightGBM TreeSHAP baseline on Polish 3-year and Taiwan, 5 folds x 3 seeds, seed-to-seed Spearman 0.981 / 0.925, fold-to-fold 0.875 / 0.700, top-10 Jaccard reported; module `src/fintfm/experiments/attribution_baseline.py`.** **Progress (2026-11-15): `src/fintfm/experiments/attribution_baseline.py` fits
      LightGBM per stratified fold and takes exact TreeSHAP from `pred_contrib=True` (no `shap`
      dependency); run as a script, not `-m`, because torch and LightGBM segfault together on
      macOS. MEASURED on Polish 3-year and Taiwan, 5 folds x 3 seeds: seed-to-seed Spearman 0.981
      and 0.925, fold-to-fold 0.875 and 0.700, TreeSHAP about 215 microseconds per row. Numbers
      and command in `docs/results/drafts/12.2_2026-11-15.md`; stays open until they are entered
      in `docs/results/FINDINGS.md`.** Establish the baseline to beat: SHAP over gradient boosting on the UCI panels,
      including runtime. Verify: numbers into `docs/results/FINDINGS.md`, re-derived from the run.
- [x] 12.3 **Done (2026-11-15): `sample_financial_task(return_attributions=True)` adds
      `meta["attribution"]` (one float per column of X, after the column permutation) and
      `meta["attribution_scope"]`. It is a global per-task importance true to the generator,
      not a per-row or per-model explanation: mean absolute change in the sharpness-scaled
      logit when a driver quantity is rotated across rows. Non-driver columns, including
      correlated ones and copies, are zero; a gate column is not attributed. Draws no random
      number, X and y byte-identical to the default. `tests/test_prior_attribution.py`.**
      Emit ground-truth attributions from `prior/financial.py`, which knows the true
      driver weights. Verify: a test that a feature with zero weight receives near-zero
      attribution.
- [ ] 12.4 Add an attribution head and train it jointly. Verify: agreement with SHAP-on-GBM
      reported as a rank correlation per dataset, not as an average.
- [ ] 12.5 Measure attribution stability across context resamples and seeds. Verify: a test
      asserting a stability threshold, chosen and recorded before it is measured.
      **Progress (2026-11-15): thresholds recorded here before any measurement, as constants
      in `src/fintfm/experiments/attribution_stability.py`: on the pinned prior task of the
      test, permutation attribution through the public classifier must reach context-resample
      Spearman mean >= 0.60, seed-to-seed Spearman mean >= 0.70 and seed-to-seed top-5 Jaccard
      mean >= 0.40 (12.2's LightGBM TreeSHAP baseline: seed-to-seed 0.981 / 0.925, fold-to-fold
      0.875 / 0.700 on Polish 3-year / Taiwan). The harness takes any attribution function, so
      the 12.4 head can be passed to it later.**
