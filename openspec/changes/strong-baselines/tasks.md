# Tasks

- [ ] 18.1 Run the boosting family in a torch-free subprocess, exchanging arrays via a temp
      file. Verify: a test that fits LightGBM, CatBoost and XGBoost in a process where torch
      is already imported, and asserts none crashes.
- [ ] 18.2 Demote sklearn's `GradientBoostingClassifier` to a reference row and make the
      strong family the headline comparison. Verify: `fintfm-bench --credit` output lists all
      four with the weak one labelled.
- [ ] 18.3 Announce every skipped baseline rather than dropping it silently. Verify: a test
      that a simulated import failure produces a visible message.
- [ ] 18.4 Re-run findings 12, 16 and 17 against the strong family and **restate their
      margins**, in the finding text itself rather than a footnote. Verify: numbers
      re-derived from the run, and each finding carries an amendment banner.
- [ ] 18.5 Record the OpenMP conflict and its resolution in `docs/COMPUTE.md`, since it is an
      environment fact any future contributor will hit. Verify: the section names the
      bisection that identified it.
