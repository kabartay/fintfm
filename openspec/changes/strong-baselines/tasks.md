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
- [ ] 18.6 **Progress (2026-11-15): the runnable Verify is `fintfm.experiments.boosting_seeds` (SMOKE-TEST only so far, nothing MEASURED). §27's script was never committed, so the module re-implements its protocol: Polish 3-year panel, `runs/phase1-5k/mixed.pt`, stratified 70/30 split, stratified training subsample with round(n * default rate) defaults, n = 100/250/1,000/4,000, arms fintfm, CatBoost, LightGBM and XGBoost (boosters out-of-process, library defaults). Each seed changes the split, the subsample and every arm's own seed. It prints average precision first, then ROC-AUC, ECE and Brier skill, each as seed mean +/- std per arm and size, plus a paired bootstrap of fintfm minus CatBoost per n (`paired_auc_difference`, `metric=average_precision_score` for AP, predictions pooled across seeds, `groups` = panel row id so overlapping test rows are resampled whole), Holm across n (`holm_adjusted_p`) separately for AP and ROC-AUC, and writes `boosting_seeds.json`. Command (run from the repo root, queue it, 5 seeds): `uv run python -m fintfm.experiments.boosting_seeds --checkpoint runs/phase1-5k/mixed.pt --seeds 0 1 2 3 4 --sizes 100 250 1000 4000 --n-boot 2000 --out runs/strong-baselines-186`. Estimated 8-10 minutes on this machine, ESTIMATED from a smoke run: 1 seed at n = 100 and 250 took 34 s wall with `--threads 2`. At seed 0 that smoke reproduced §27's LightGBM, CatBoost and XGBoost AUC at n = 100 to four decimals (0.6620, 0.6940, 0.6193); fintfm read 0.6968 against §27's 0.6945, so the fintfm arm is not bit-identical to §27's. Task stays open until the real run is read.** Repeat §27 across >= 3 seeds. The n=100 AUC win is 0.0005 over CatBoost, which is
      noise; only the ECE and skill margins are substantial. Verify: mean +/- std per arm in
      §27 before the result is quoted outside this repository.
- [x] 18.5 Record the OpenMP conflict and its resolution in `docs/infra/COMPUTE.md`, since it is an
      environment fact any future contributor will hit. Verify: the section names the
      bisection that identified it. **Done 2026-09-09, including that
      `KMP_DUPLICATE_LIB_OK=TRUE` is not sufficient.**
- [ ] 18.6 **Progress (2026-11-15): licence note recorded in `docs/research/baseline_licences_2026-11-15.md` for TabICLv2, TabDPT, TabPFN, FT-Transformer, TabM and RealMLP, code and weights separately. TabPFN 3.5, 3, 2.6 and 2.5 weights are excluded on licensing grounds (non-commercial, outputs included). Not closed: no baseline has been added or run, and the finding that states the exclusions in `docs/results/FINDINGS.md` is not yet written.** **Added 2026-09-14, licensing-gated.** Extend the baseline family beyond
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
