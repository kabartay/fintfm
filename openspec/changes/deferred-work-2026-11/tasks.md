# Tasks

## Regression and multiclass (design review DR7.6: overall bar 1,000 Elo, 903 at §224)

- [x] 49.1 **Done (2026-11-14, FINDINGS §231): `reg-ctx4k` hit TabArena's time limit on 7 of 13 datasets and matches §223 within 0.3% on the other 6, so it has no Elo and is not viable; with `reg-head` (§226) neither arm wins and no re-merge follows from them. Partitioned contexts for regression, added under this task, lift regression from 582 to 639 Elo (10 better, 3 worse of 13) and the overall to 909.2, 107th of 110.** **Progress (2026-11-13, FINDINGS §226): `reg-head` hit TabArena's time limit on 6 of 13 datasets and is null on the other 7 (4 better, 3 worse, at most 1.9%); not viable. `reg-ctx4k` was not run (project stopped).** **Score the two queued regression readout arms.** `reg-head` appends the head's
      prediction to the ridge readout (`FINTFM_PROBE` `{"raw": true, "head": true}`) and
      `reg-ctx4k` raises `FINTFM_MAX_CONTEXT` to 4,000 (`runs/local_queue/lite_reg_probe.sh`).
      Both pair against `fintfm_lp_reg_raw_1111` (578 Elo, §223). Verify: a FINDINGS entry with
      per-dataset RMSE ratios over the 13 regression datasets, a sign test and Elo; if either
      arm wins, the re-merged 51-dataset Elo and its position from `scripts/lite_positions.py`.
- [x] 49.2 **Done (2026-11-14, FINDINGS §233): the 480,000-task pure-regression checkpoint (35,300 s on a T4x2) scored on all 13 regression datasets against §223/§227's run, per-dataset RMSE and Elo: 642 against 582 unpartitioned (7 of 13, p 1.0) and 603 against 639 partitioned; merged overall 927.9, 106th of 110. No lift; the 48,000-task checkpoint stays.** **Train a regression checkpoint on the current recipe.** `v4-regression.pt` is a
      6,000-step checkpoint on an earlier recipe (§223), and the readout lever is mostly spent
      there (median RMSE ratio 0.985, §223). Train one with cell attention and labels at no
      fewer than the binary checkpoint's 48,000 tasks, `--max-classes` at least the regressor's
      `n_bins`, with `--checkpoint-every` (CLAUDE.md). State the cost from a measured step time
      (§127, §171) before launch. Verify: the 13 regression datasets scored against §223's run
      in a FINDINGS entry, per-dataset RMSE and Elo, and the merged overall Elo.
- [x] 49.3 **Done (2026-11-13, FINDINGS §227): eight bags tie four (4 better, 4 worse; 874 against 865 Elo); `fintfm_lp_mc_scr_bag8_1113` is the multiclass run in the 905.2 Elo merge.** **Decide the multiclass partition bag count.** Four bags lifted multiclass from 825
      to 865 Elo but was worse on 3 of 8 datasets (§224); `mc-bag8` (8 bags) is queued.
      Verify: `mc-bag8` against `fintfm_lp_mc_scr_bag4_1112` per dataset in a FINDINGS entry;
      the better of the two is the multiclass run in the next merge.
- [ ] 49.4 **Progress (2026-11-16, FINDINGS §262, MEASURED): the 512-column binary checkpoint was rescored on the three wide binary tables with and without §232's readout options (`fintfm_lp_w512_1114`, `fintfm_lp_w512_scr_1114`; per-dataset ranks from the stored 107-method leaderboard). With options: APSFailure 95, Bioresponse 81, kddcup09_appetency 104 of 107, against the entry's 100, 82 and 68. Swapping them into the 952.8 entry lowers it to 943.5 (104th of 110); the no-options run gives 941.2 (105th). The width is not a lift on the wide tables. Still open: the multiclass wide table and QSAR-TID-11 are not rescored (multiclass 512 arm blocked by §234's V4FinBench gate; no 512-column regression checkpoint). Not Done; Verify is met for 3 of 5 tables.** **Progress (2026-11-14, FINDINGS §234): the binary 512-column checkpoint rescored the three wide binary tables (ranks 102/103/91 to 101/102/89 of 107-108, per-dataset ranks from the stored leaderboard); its V4FinBench gate failed on one Holm-significant fold, so the multiclass 512 arm is blocked and QSAR-TID-11 and the multiclass wide table are not rescored. `wide512-scr` (with §232's readout options) is queued.** **Raise the wide tables.** Under `max_features_strategy: select` the five wide tables
      rank 90th to 103rd of 108 (§220, §221). The planned route is DR8.3's 512-column binary
      checkpoint and the multiclass arm `configs/arms/dr75-width512-mc.yaml` (DR7.5 note,
      2026-11-07), both waiting on GPU. Verify: the wide tables rescored with the 512 checkpoints,
      per-dataset ranks from the stored leaderboard, in a FINDINGS entry.

## Binary

- [x] 49.5 **Progress (2026-11-16, FINDINGS §254, §255, MEASURED): binary cross-fitting (`fintfm_lp_bin_scr_xfit_1115`) lifts binary-only Elo 997 to 1027; paired 10 better, 20 worse of 30 (sign p 0.10), and `kddcup09_appetency` (+0.0760 ROC-AUC) carries most of it. Merged entry `fintfm_linprobe_all51_1116` (this binary run, `fintfm_lp_mc_scr_bag8_1113`, `fintfm_lp_reg_rawpart4_pure_1114`): 952.8 Elo, 104th of 110, against 931.9, 106th of 110; TABFLEX_GPU (default) is 977.7 in this fit, a gap of about 25 Elo. Per-dataset scan of the 952.8 entry (stored results re-scored, `runs/median_gain_1116.csv`): moving any single below-median dataset to its median is worth about +11 to +15 overall Elo (rank-based); largest are airline +14.6, anneal +14.6, diamonds +13.6, QSAR_fish_toxicity +13.3, GiveMeSomeCredit +13.3. Airline large-table levers (`FINTFM_PARTITION_SCALE_N`, `FINTFM_PROBE_BUDGET=50000`) all stopped with AutoGluon TimeLimitExceeded on the Mac CPU (baseline fold about 240 s); rerun on a Kaggle T4 is queued (task 49.17). §255: Kaggle T4 cross-fit multiclass 838 against 874 (worse; the device is a confound); regression cross-fit 689 = 689; regression levers log1p 671, quantile 544, wide 689 (identical), blend-cv 635 on 10 datasets and TimeLimitExceeded on the 3 largest. anneal has two out-of-fold rows with true-class p about 1e-7; an out-of-fold-chosen floor e=0.003 moves test log loss 0.2223 to 0.1949 (post hoc on stored predictions; task 49.18). This task stays closed on §232's result; the later numbers belong to the entry (49.6).** **Done (2026-11-14, FINDINGS §232): the binary TabArena-Lite arm ran on §230's not-harmful verdict and lifted binary from 974 to 997 Elo (23 better, 7 worse, p 0.01); overall 925.6, 106th of 110 (`fintfm_linprobe_all51_1114b`). The readout options join the entry's configuration.** **Progress (2026-11-14, FINDINGS §230): V4FinBench verdict recorded. With §219's sampler fix the probe per leaf is a candidate, +0.0580 mean AP on 5 of 5 folds, and the scale/cv/raw options add +0.0449 (0.3015, +0.1029 over the zero-shot head, 5 of 5, Holm-significant on every fold). Not harmful, so the binary TabArena-Lite arm is queued (`bin-probe-scr`, paired against `fintfm_linprobe_squash_part_090`, 974 Elo); open until it reports. A matched raw-only control was not run.** **Score the readout options on binary, V4FinBench first.** The `scale`, `cv` and `raw`
      readout options lifted multiclass and regression on TabArena-Lite (§222, §223) and are
      untested on binary. The `ppfix-v4-f0` to `-f4` folds (`--probe-options`) and
      `ppfix-v4-compare` are queued locally; they also give the verdict on §219's sampler fix
      for the probe per leaf (§218 was −0.0570 mean AP). Verify: a FINDINGS entry with the
      five-fold V4FinBench verdict under the project's rule (mean AP change above +0.0117 with
      at least 4 of 5 folds positive is a candidate, §173); a binary TabArena-Lite arm runs only
      if the verdict is not harmful.

## TabArena

- [ ] 49.6 **Owner decisions (2026-11-16, later): (1) the combined arm's 10 large binary tables run on the Mac, not a T4; a time-limit failure there fails §257's rule as written. (2) Publish the multiclass and regression checkpoints now under Apache-2.0 (owner runs `scripts/publish_weights.py` with HF_TOKEN). (3) fintfm 0.11.0 released (tag v0.11.0 at 6dcacaf). (4) The PR opens only when the settled entry scores above TABFLEX_GPU (default) in the same fit, and only after the owner approves the final text. (5) No hybrid levers in the entry: neither a duplicate-row label lookup for multiclass nor a tree on the regression residuals. (6) The research checkpoint `v4-hazard-ldp.pt` stays private; its jobs run on the Mac. Entry levers merged default-off on the fork's dev branch `fintfm-audit-fixes` (d78c2c4).** 49.6 **Progress (2026-11-16, MEASURED unless stated): the entry is now `fintfm_linprobe_all51_1116`, 952.8 Elo, 104th of 110 (§254), 25 Elo below TABFLEX_GPU 977.7 and 47.2 below the 1,000 bar. PR blockers still open: (1) Elo against TABFLEX_GPU 977.7; (2) the multiclass and regression weights published (owner runs `scripts/publish_weights.py` with HF_TOKEN); (3) fintfm 0.11.0 released, since the wrappers need PartitionedTFMRegressor, ThresholdTFMRegressor and partition `missing_classes`, which 0.10.1 lacks; (4) the reproduction with no env-var overrides (reproduction kernels without overrides in preparation); (5) the owner's yes. In progress today, results not in: multiclass probability floor (49.18), regression random-Fourier-feature readout (49.19), refresh of the PR branch with cross-fitting (49.20), pickle-audit crash (49.21), 0.11.0 release prep, airline levers on a T4 (49.17).** **Progress (2026-11-16): clean PR branch `fintfm-pr-2026-11` on the fork (ecd4ac5, on upstream main bfa8c50; 26 files, model folders + tests + four registration edits; no upstream conflicts; 175 FinTFM tests, both registry fit-tests, audit_warmup and audit_transduction pass; audit_pickle needs a newer AutoGluon). Owner decisions: publish the multiclass and regression checkpoints under Apache-2.0 (owner runs scripts/publish_weights.py with HF_TOKEN); release fintfm 0.11.0 after the cross-fit/lever results settle the entry configuration, because the wrappers need PartitionedTFMRegressor, ThresholdTFMRegressor and partition missing_classes, which 0.10.1 lacks; then bump the pin, run the no-env-var reproduction and decide on opening the PR. Target to beat: TABFLEX_GPU 977.2 (101st of 110); entry 931.9 (106th).** **Progress (2026-11-15, owner asked to prepare a new PR; nothing posted, nothing published): fork `fintfm-audit-fixes` at b48cb22 (pushed to `fork` only): the entry's configuration is now FinTFM-LinProbe's own hyperparameters per problem type (9bb4239; env vars still override), checkpoints are referenced by Hugging Face id + SHA-256 (binary pinned to kabartay/fintfm-binary, public, Apache-2.0; multiclass and regression are named placeholders that raise until published), metadata `verified=False` and display name "FinTFM with a trained linear readout", a proposed search space (c5ed4a4), and cross-fitting of the readout's training rows plus out-of-fold categorical encoding (de27754, `probe_crossfit_folds=5`; the leakage review found the readout trained on rows that read their own labels). PR text drafted in docs/competition/PR_2026-11-draft.md; model cards and a dry-run publisher in docs/models/ and scripts/publish_weights.py. Still open: the owner's bar decision (931.9 Elo, 106th of 110, below 1,000), publishing the two checkpoints with a licence, the cross-fit Lite measurement (queued as `bin-xfit`), and the reproduction run with no env vars.** **Prepare the resubmission package, once the bar is met.** DR7.6's bar is 1,000
      overall Elo (owner, 2026-11-13). The measured runs set their configuration through the
      fork's environment variables (`FINTFM_PARTITION`, `FINTFM_PROBE`, `FINTFM_MAX_CONTEXT`,
      `FINTFM_PARAMS`), so a submission needs them as the entry's own hyperparameters; the
      multiclass (`dr74-mc-s0`) and regression checkpoints must be published, which is the
      owner's decision; the entry must declare that the readout takes the row's features; and
      the search space for the readout entry is undecided (`docs/results/TABARENA.md`,
      `docs/competition/SUBMISSION.md`). Verify: the fork's entry reproduces the merged
      51-dataset Elo with no environment variables set, the checkpoints are public with a
      licence, and the PR text states the label "FinTFM with a trained linear readout".

## Harness defects found by the 2026-11-13 inspection (not fixed; each changes recorded numbers)

- [x] 49.7 **Done (2026-11-13, FINDINGS §229): seven entries rerun under both splits on today's code; coherence holds (40.3% against 0%), accuracy falls 0.01 to 0.04 AUC, no ordering flips; dated notes on §28, §29, §32, §33, §35, §37, §38.** **Progress (2026-11-13): code fixed in b34612d. The out-of-time, context-sweep and retrieval-grouping harnesses default to `time_based_split(on_overlap="drop_from_train")` with a 6-year label embargo (V4FinBench labels reach up to six years ahead); `--split legacy` reproduces the old split. Reruns of §28, §32, §33, §35, §37 and §38 (`-emb` outputs) are queued locally; §29's original output is gone and §30's other checkpoints are unnamed, so those two can only be redone on the current checkpoint.** **Out-of-time split: add a company check and an embargo.**
      `src/fintfm/experiments/v4_out_of_time.py` `time_split` (also used by the context-sweep
      and retrieval-grouping harnesses) does not check companies, so a firm seen in a training
      year can appear in the test years, and a training row's labels span up to six years into
      the test window. `fintfm.evaluation.splits.time_based_split` refuses overlap but these
      harnesses do not use it. Findings that rest on this split include §28 to §35. Verify: the
      harnesses use `time_based_split` (`on_overlap="drop_from_train"`) with an embargo, a test
      shows a firm cannot be on both sides, and the affected findings are rerun with dated notes
      on each.
- [x] 49.8 **Done (2026-11-13, FINDINGS §228): §11 recomputed; the leak moved it about four points (44.0% to 47.8% of steps), and §11's own figure does not reproduce on the stored checkpoint; §11 and the README carry dated notes. Other documents quoting 39% are corrected with 49.7.** **Progress (2026-11-13): code fixed in 07b40a7. The query horizon's context is the 70% training pool, other horizons drop rows identical to a query row; the horizon files carry no firm identifiers, so a firm can still appear in a context and the query with different years' ratios (documented). `--coherence-legacy` reproduces §11. The corrected and legacy reruns are queued locally.** **Term-structure coherence: keep query rows out of the context.**
      `src/fintfm/experiments/prior_ablation.py` `term_structure_coherence` loads the whole
      panel as context, so when the context horizon equals the query horizon the held-out query
      rows and their labels are in the context; horizon files may also share firms. §11's
      figure (11.0% of horizon steps, 39% of firms), quoted in the README, rests on it. Verify:
      a test shows no query row in the context, §11's figure is recomputed in a new FINDINGS
      entry, and §11 and the README carry dated notes if it moves.
- [x] 49.9 **Done (2026-11-13): (a), (b), (e), (f), (g) fixed with tests in `tests/test_defects_49_9.py` (e090300); (c) and (d) fixed with 49.7 (b34612d, `tests/test_v4_split_and_sweep.py`); (h) was already wired (c8c3027 passes `warmup_steps` to schedule-free), only the `_NullSchedule` docstring was stale and is corrected. None changes a recorded number; `tests/test_golden_outputs.py` passes unchanged.** **Smaller defects, fix or record a reason not to.**
      (a) `src/fintfm/modeling/train.py` `_save_training_state` saves the CPU torch RNG only, so
      a resumed CUDA run with dropout differs from an uninterrupted one;
      (b) `capability.py` sweeps skip seeds with a single-class test split silently, so cell
      means use different seed counts with no count recorded;
      (c) `context_sweep.py` records no mean predicted probability beside the observed rate
      (CLAUDE.md requires it) and `max(...)` misorders when a mean AUC is NaN;
      (d) `v4_out_of_time.py` `score_curve` gives an all-NaN horizon column when a horizon is
      single-class in training, which the violation counts then treat as non-violations;
      (e) `FinancialTFMRegressor` and `FinancialTFMQuantileRegressor` set trailing-underscore
      attributes in `__init__`, against the scikit-learn convention;
      (f) `FinancialTFMClassifier.fit` sets `feature_chunk` on a shared model;
      (g) `prior_score.py` catches any `TypeError` around the sampler call;
      (h) `--optimizer schedulefree` gets no warmup (`_NullSchedule`). Verify: each item is fixed
      with a test that fails before and passes after, or marked here as won't-fix with the
      reason; any fix that changes a recorded number gets a FINDINGS entry.

## Owner decisions and release

- [x] 49.10 **Done (2026-11-13, owner): Claim 9 is SINGLE DRAW in `docs/paper/CLAIMS.md`; DR2.4 is two checkpoints.** **Decide Claim 9's tag and DR2.4's checkpoint count.** Recommended on 2026-11-13:
      Claim 9 to SINGLE DRAW (each of its four panels is one 70/30 split, it was measured on the
      pre-cell-attention architecture, and the published checkpoint misses the default rate,
      §174 and the DR4.3 run record), and two checkpoints for DR2.4 (all four mixed-prior arms
      were harmful on V4FinBench, §180, §191). Verify: the owner's decisions are recorded in
      `docs/paper/CLAIMS.md` and in DR2.4's note.
- [x] 49.11 **Done (2026-11-13): Zenodo minted the version DOI 10.5281/zenodo.24183314 for 0.10.1; with the release, CI, the release workflow and PyPI already green, the Verify clause is met.** **Progress (2026-11-13): v0.10.1 published from 47c89aa; CI, the release workflow and the PyPI upload are green (wheel and sdist on PyPI); the Zenodo version DOI was not yet minted when this was written. The TabArena fork pins 0.10.1 (af7da60).** **Release 0.10.1.** The Zenodo record of 0.10.0 describes the model as 107th of 110,
      while that version's archived FINDINGS still says 106th; the correction is §225, on main
      after the release. A patch release carries §225, the 2026-11-13 documentation revision and
      the inspection fixes. Verify: v0.10.1 is published with CI and the release workflow green,
      PyPI shows 0.10.1, and Zenodo has minted its version DOI (`CLAUDE.md`, "Releases").

## Repository leftovers

- [x] 49.12 **Done (2026-11-13): `scripts/draw_logo.py` documents installing Pillow (HPND) ad hoc, since a `logo` extra would change `uv.lock`; `scripts/kaggle/README.md` records the four slug/title mismatches as historical (ids unchanged); of the kernels named below only `ts_dr75_width512_mc_s0` is unpinned (the `task_dr311_*` kernels pin e89def2 or 027e0cc), and it stays unpinned until launch because its gate is unmet; `configs/best.yaml` carries a dated note naming the later entries that supersede §38.** **Small items from the 2026-11-13 inspection.** `scripts/draw_logo.py` needs Pillow,
      which no extra declares; four Kaggle kernels (`task_39_28_replicate_s91`,
      `task_48_3_binned_control`, `task_48_3_quantile_head`, `task_48_9_schedulefree`) have an
      `id` slug that does not match their title; five kernels (`task_dr311_*`,
      `ts_dr75_width512_mc`) carry no pinned commit; `configs/best.yaml` still calls itself the
      best configuration to date (§38). Verify: each is fixed, or noted here with the reason it
      stays.

## Representation controls and the 20-class regression checkpoint (2026-11-15)

- [x] 49.13 **Done (2026-11-15): FINDINGS §242. `fintfm_lp_reg_c20part4_1115`: 699 Elo against §235's 689, better on 5 of 13 (sign p = 0.58, median RMSE +0.17%), 13 of 13 better than §240's raw-only control; re-merged entry `fintfm_linprobe_all51_c20_1115` 932.8 Elo, 106th of 110, against 931.9, 106th of 110. Null; the entry stays on §235's run.** **Progress (2026-11-15): running.** Kernel `scripts/kaggle/ts_reg_pure_c20_s0`
      (68718fc; 6,000 steps x 8 tasks per step, DDP on 2 GPUs = 48,000 tasks, `--max-classes 20`,
      `--p-regression 1.0`) trained on Kaggle in about 1.2 h; checkpoint fetched to
      `runs/kaggle-fintfm-ts-reg-pure-c20-s0/reg-pure-c20-s0/reg-pure-c20-s0.pt`, stored
      `max_classes` 20. Lite run `fintfm_lp_reg_c20part4_1115` queued (`reg-c20part4`, four
      partition bags, ridge readout with `raw`, `FINTFM_REG_BINS=20` from fork dcd46c2; a first
      attempt failed on all 13 tables because `FINTFM_PARAMS` does not take `n_bins`).
      **Score the 20-class regression checkpoint on TabArena-Lite regression.** Regression
      tasks are binned into the head's class count, so 20 classes give a finer target grid than
      the 10 of the current checkpoint. Verify: a FINDINGS entry with per-dataset RMSE over the
      13 regression datasets, a sign test and Elo, paired against `fintfm_lp_reg_rawpart4_pure_1114`
      (689, §235) and against the raw-only control `fintfm_lp_reg_rawonly4_1115` (537, §240); if
      it beats 689, the re-merged 51-dataset Elo and its position from
      `scripts/lite_positions.py` with the field size.
- [ ] 49.14 **Progress (2026-11-15): not run: its precondition (49.13 moves Elo) did not hold (§242, sign p = 0.58, position unchanged). Left open under the keep-deferred rule; it applies again only if a later 20-class arm shows a gain.** **Separate the checkpoint from the bin count, only if 49.13 moves Elo.** 49.13
      changes the checkpoint and `n_bins` together, so a gain cannot be assigned to either.
      Score the 20-class checkpoint at `FINTFM_REG_BINS=10` with 49.13's configuration
      otherwise unchanged. Verify: per-dataset RMSE and Elo against 49.13's run and against
      §235's run; the FINDINGS entry says which of the two changes carries the difference. Not
      run if 49.13 is null or negative (then there is nothing to assign).
- [x] 49.15 **Done (2026-11-15): FINDINGS §243. `fintfm_lp_mc_rawonly8_1115` (§237's configuration with `no_rep`): 829 Elo against the entry run's 874, representation better on 7 of 8 (sign p = 0.07), median log loss 15.5% lower; README and CLAIMS now say for multiclass what §239/§240 say for binary and regression.** **The raw-only control for multiclass.** Binary (§239) and regression (§240) each
      have a matched control with `no_rep` (a tree with a linear readout per leaf on the
      squashed raw features, no FinTFM representation); multiclass, the entry's 874 Elo
      (`fintfm_lp_mc_scr_bag8_1113`), has none, so it is unknown whether multiclass's Elo is
      the representation's. Run §237's multiclass configuration with `no_rep` added to
      `FINTFM_PROBE`. Verify: a FINDINGS entry with per-dataset log loss, a sign test and Elo
      against the entry's multiclass run, and README, CLAIMS and FINDINGS say for multiclass
      what §239/§240 say for binary and regression.
- [ ] 49.16 **48.24's size control, if context size is ever revisited.** 48.24 was falsified
      (§241) without its pre-registered 4,000-row uniform context control
      (`configs/arms/48.24-uniform-ctx4000.yaml`; `runs/4824/uni4000` is empty), so the
      majority-downsampled arm's +0.0041 mean AP is confounded with doubling the context. The
      verdict does not depend on it and §83/§84 measured context as nearly flat, so this is
      low priority. Verify: five-fold V4FinBench AP of the uniform 4,000-row arm against
      `runs/lrsweep/eval-published` and against `runs/4824/maj4000`, clustered comparison, in
      a FINDINGS entry that states how much of §241's difference is context size.

- [x] 49.17 **Done (2026-11-16, FINDINGS §257, MEASURED, Kaggle T4): airline 1-AUC control 0.05165, depth 0.04858, budget 0.02449, both 0.02590; both levers on all 30 binary datasets 15 better / 8 worse / 7 tied against the CPU entry run (sign p 0.21), binary-only Elo 1046 against 1027; merged entry `fintfm_linprobe_all51_1116b` 965.6 Elo, 104th of 110 (TABFLEX_GPU 978.0 in the same fit). Decision: the entry adopts both levers for binary. Device noted (T4 against CPU).** **Airline large-table levers on a GPU, then decide the entry.** On the Mac CPU
      `FINTFM_PARTITION_SCALE_N` and `FINTFM_PROBE_BUDGET=50000` all stopped with AutoGluon
      TimeLimitExceeded (first bagged fold longer than its share of 3,600 s; baseline fold about
      240 s), so they are unmeasured. Kaggle T4 kernel `scripts/kaggle/task_lite_air_levers`
      (0897bff) runs airline control, depth, budget and both, then both levers on all 30 binary
      datasets. Result pending (2026-11-16). Verify: a FINDINGS entry with per-dataset ROC-AUC
      and Elo for control against each lever on airline and on the 30 binary datasets (device
      noted, since §255 shows a T4 confound), a sign test, the re-merged 51-dataset Elo and
      position from `scripts/lite_positions.py` with the field size, and a stated decision on
      whether the entry adopts either lever.
- [x] 49.18 **Done (2026-11-16, FINDINGS §269): the option is merged default-off with tests (fork dev branch `fintfm-audit-fixes`, d78c2c4); all 8 multiclass tables run (SDSS17 on the Mac): 4 better, 3 worse, 1 tied, sign test p 1.00; same fit 966.1 against 965.4 Elo (111 methods); scored alone 966.1, 103rd of 110. Adopted under §265's rule as a configuration choice, not a measured improvement; the floor is chosen on out-of-fold rows, never test labels.** **Progress (2026-11-16, FINDINGS §265, MEASURED, Kaggle CPU): paired against a same-machine control, the floor (`FINTFM_PROB_FLOOR=cv`) is better on 4, worse on 2, tied on 1 of 7 multiclass tables (sign test p 0.69; anneal 0.2217 to 0.2018 log loss). SDSS17 hit the time limit on Kaggle CPU and runs on the Mac (`mc-floor-sdss17`); adoption rule pre-registered in §265.** **Multiclass probability floor, then a Lite multiclass paired run.** `anneal` has two
      out-of-fold rows with true-class p about 1e-7; an out-of-fold-chosen floor e=0.003 moves
      test log loss 0.2223 to 0.1949 (post hoc on stored predictions, §255). An option is in
      progress in a fork (2026-11-16). Verify: the option merged with a test, a TabArena-Lite
      multiclass run on the 8 datasets paired against `fintfm_lp_mc_scr_bag8_1113` (874 Elo),
      per-dataset log loss, sign test, Elo and the re-merged overall position, in a FINDINGS
      entry that states how the floor is chosen without test labels.
- [ ] 49.19 **Progress (2026-11-16, FINDINGS §265, MEASURED, Kaggle CPU): paired against a same-machine control, the RFF readout is better on 6 of 7 regression tables, worse on concrete (+7.8% RMSE), sign test p 0.13; airfoil -24.9%. The six large tables hit the time limit on Kaggle CPU and run on the Mac (`reg-rff-large6`); adoption rule pre-registered in §265.** **Regression nonlinear readout (random Fourier features), then a Lite regression
      paired run.** The regression readout is linear; a random-Fourier-feature readout
      prototype is in progress in a fork (2026-11-16). Verify: the prototype merged with a test,
      a TabArena-Lite regression run on the 13 datasets paired against
      `fintfm_lp_reg_rawpart4_pure_1114` (689 Elo), per-dataset RMSE, sign test, Elo and the
      re-merged overall position, in a FINDINGS entry; a null result is recorded, not dropped.
- [ ] 49.20 **Progress (2026-11-17): PR branch `fintfm-pr-2026-11` moved to bb41dff (pushed to `fork` only, no PR): the multiclass probability floor is ported (only the floor; ensemble, RFF, one-hot and spline stay on the dev branch) and the entry defaults per problem type now match the runs merged in `fintfm_linprobe_all51_1116f_floor8`: binary partition depth scaled with n and readout budget 50,000 (§257), cross-fit 5; multiclass floor `cv` and cross-fit 0 (§269, §255); regression cross-fit 0. 168 FinTFM-LinProbe tests pass. Still open: the multiclass and regression checkpoints are unpublished placeholders (SHA-256 match the measured files), the problem-type gate needs `FINTFM_ALLOW_*`, and the branch's defaults are not yet re-measured without env overrides.** **Progress (2026-11-16): the premise was wrong; nothing needed porting. On `fintfm-pr-2026-11` (ecd4ac5) the FinTFM model folders and the seven `test_fintfm_*` files are byte-identical to the dev tip 6981265 (`git diff --stat` empty), so the branch already sets `probe_crossfit_folds=5` by default and also carries the default-off regression and large-table levers. Rerun on ecd4ac5: 175 FinTFM tests pass; registry tests (`-k "fintfm or FinTFM or registry or warmup"`) 69 pass; audit_transduction binary OK (max deviation 1.25e-06, K=5 active); audit_warmup ok. Open only for the no-env-var reproduction of the 952.8 merge (49.6).** **The PR branch carries cross-fitting as its default.** Branch `fintfm-pr-2026-11` was
      cut before binary cross-fitting lifted the entry to 952.8 Elo (§254); a refresh onto it is
      in progress (2026-11-16). Verify: the branch's entry sets `probe_crossfit_folds` by
      default for the problem types where it was measured (binary), its tests pass, the fork
      fit-tests pass, and the branch reproduces the 952.8 Elo merge with no environment
      variables (the same check as 49.6's).
- [x] 49.21 **Done (2026-11-16, MEASURED): `audit_pickle` passes on the PR branch (ecd4ac5) for FinTFM and FinTFM-LinProbe, binary, in a fresh environment with the branch's own AutoGluon floor (1.6.4b20261115): `model.pkl` grows 136 and 88 B per row against the 1,024 B limit. The crash was the development venv's older nightly (1.6.4b20260919), not the wrappers. Recorded in `docs/competition/SUBMISSION.md` ("Pickle audit"); regression needs the published checkpoint to be audited by a maintainer.** **The pickle audit passes or is documented.** `audit_pickle` crashed on the PR branch
      (needs a newer AutoGluon, 49.6); a crash investigation is in progress (2026-11-16). Verify:
      `audit_pickle` passes on the PR branch, or the crash's cause and the AutoGluon version it
      needs are written into `docs/competition/SUBMISSION.md` with the reason the entry is still
      safe to submit.
- [ ] 49.22 **Progress (2026-11-16): the first gate kernel (fork 169402a) fitted nothing: 2 of 135 model tests failed. Both were test defects, not model defects: `predict_proba` returns float32, so the in-model mean agrees with the mean of separate members to float32 resolution, not 1e-9 (each member alone matches its separate fit exactly); and the partitioned entry scores each routed leaf in its own call (14, 6, 6, 14 rows), so a shared pass is one call per leaf, not one of 40. Fixed on the fork (6ca0219); gate relaunched without its airline timing run. The control arms showed Kaggle's 4-core CPU 2-6x slower than the Mac (6 of 13 regression and 1 of 8 multiclass tables stopped at the 3,600 s limit; the rest reproduce the Mac runs to about 1e-4), so §257's combined arm runs its 20 tables under 700 s Mac fit time on Kaggle CPU and its 10 large tables on a T4 or the Mac, merged into one run.** **Binary readout ensemble on TabArena-Lite (pre-registered in FINDINGS §256).** Averaging
      stored binary predictions of the cross-fit and non-cross-fit readouts (chosen on validation rows)
      re-scores the entry at 965.6 Elo; with a raw-only member, 987.8 (MEASURED re-score, not a fitted
      model). A fork option `probe_ensemble` that fits the members inside one model is in progress
      (2026-11-16). Verify: the reproduction check of §256 passes, the 30-dataset Lite arm(s) run,
      and a FINDINGS entry applies §256's gate (adopt or not) with the merged entry's Elo and position.
- [ ] 49.23 **Progress (2026-11-16): option merged default-off on the fork's dev branch `fintfm-audit-fixes` (d9f1358): `probe_spline` / `FINTFM_PROBE_SPLINE` (`on` = 5 quantile knots, degree 3, linear extrapolation, at most 64 expanded columns chosen by distinct training values, label-free); knots fitted on the readout's training rows only, one transformer per readout shared across cross-fit folds and leaves (knots read X, never labels); applies to binary, multiclass and regression readouts; 270 FinTFM fork tests pass. The Lite binary arm is not yet run.** **Spline expansion of the readout's raw numeric columns (binary), then a Lite binary
      arm.** The readout already reads the raw columns beside the representation (`raw: true`); an
      outside tabular competition (2026-11-16, Kaggle Playground S6E11, a single synthetic dataset)
      measured +0.00028 CV ROC-AUC, 5 of 5 folds, from splines on the numeric columns of a logistic
      model. That is a lead, not evidence about TabArena. Add a default-off fork option that
      replaces each raw numeric column with a natural cubic spline basis (knots at training-row
      quantiles, fitted on the readout's training rows only, so no test row sets a knot), with
      tests, and run the 30 binary Lite tables on top of the settled binary entry, after §257's
      combined arm and 49.19 settle (the Mac is the bottleneck; large tables run one per queued
      job). Pre-registered rule, the same as §265's: adopt only if every binary table completes
      without the 3,600 s time limit and the merged entry scores above the entry without it in one
      fit (`scripts/lite_same_fit.py`). Not a hybrid: no second model, only a wider design matrix
      for the same readout. Verify: the option is merged with tests, the 30-table arm runs, and a
      FINDINGS entry applies the rule (adopt or not) with the merged entry's Elo and position.
- [ ] 49.24 **Progress (2026-11-16, FINDINGS §271, MEASURED, Kaggle CPU): without the floor, one-hot
      indicators are better on 4 of 7 multiclass tables, worse on MIC (+1.4%), identical on 2 with no
      applicable column; anneal log loss 0.2217 to 0.1101; sign test p 0.375. SDSS17 stopped at the time
      limit. The one-hot plus floor arm and its rule are pre-registered in §271.** **One-hot categorical
      indicators in the multiclass readout (`probe_onehot`), then a Lite multiclass arm on top of the floor
      entry.** fintfm encodes a multiclass target's categorical columns as level frequencies, so the
      readout cannot see a level that is not frequency-distinct. Verify: the one-hot plus floor arm runs on
      all 8 multiclass tables (SDSS17 on the Mac), and a FINDINGS entry applies §271's rule (adopt or not)
      with the merged entry's Elo and position in one fit.
