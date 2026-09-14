# Tasks

- [x] 18.1 Run the boosting family in a torch-free subprocess, exchanging arrays via a temp
      file. Verify: a test that fits LightGBM, CatBoost and XGBoost in a process where torch
      is already imported, and asserts none crashes. **Done 2026-09-09:
      `evaluation/boosting.py`, guarded by a parametrised test that imports torch first.**
- [x] 18.2 Demote sklearn's `GradientBoostingClassifier` to a reference row and make the
      strong family the headline comparison. Verify: `fintfm-bench --credit` output lists all
      four with the weak one labelled. **Done 2026-09-09: `bench.py` fits classical baselines
      and our model in-process, the boosting family out-of-process.**
- [x] 18.3 Announce every skipped baseline rather than dropping it silently. Verify: a test
      that a simulated import failure produces a visible message. **Done 2026-09-09: skips
      print the model and the reason; `available_boosting()` reports unavailability.**
- [x] 18.4 Re-run findings 12, 16 and 17 against the strong family and **restate their
      margins**, in the finding text itself rather than a footnote. Verify: numbers
      re-derived from the run, and each finding carries an amendment banner.
      **Done 2026-09-09, `FINDINGS` §27. The window narrows to ~200 obligors, but inside it
      we beat CatBoost on AUC, ECE and Brier skill, and LightGBM/XGBoost post negative skill
      at n=100. Best ECE at every size survives.**
- [ ] 18.6 Repeat §27 across >= 3 seeds. The n=100 AUC win is 0.0005 over CatBoost, which is
      noise; only the ECE and skill margins are substantial. Verify: mean +/- std per arm in
      §27 before the result is quoted outside this repository.
- [x] 18.5 Record the OpenMP conflict and its resolution in `docs/COMPUTE.md`, since it is an
      environment fact any future contributor will hit. Verify: the section names the
      bisection that identified it. **Done 2026-09-09, including that
      `KMP_DUPLICATE_LIB_OK=TRUE` is not sufficient.**
- [ ] 18.6 **Added 2026-09-14, licensing-gated.** Extend the baseline family beyond
      CatBoost/LightGBM/XGBoost to TabICLv2, TabDPT, TabPFN (where licensing permits),
      FT-Transformer, TabM, RealMLP — proposed in an externally-relayed review. **No code or
      weights from TabPFN/TabICL/TabDPT may enter this repository** (standing project rule);
      each candidate's weight licence must be checked separately from its code licence, before
      use, every time — not assumed from a prior check, and not deferred past this task's own
      commit. For ICL-based baselines specifically, compare with the *same context rows*
      fintfm receives, not merely the same train/test split, since context construction has
      already been shown to explain more variance than model choice on this data (§5). Verify:
      a licence note is recorded alongside each new baseline before it is used, and the finding
      states which candidates were excluded on licensing grounds and why.
