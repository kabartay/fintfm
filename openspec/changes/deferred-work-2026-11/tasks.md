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
- [ ] 49.4 **Progress (2026-11-14, FINDINGS §234): the binary 512-column checkpoint rescored the three wide binary tables (ranks 102/103/91 to 101/102/89 of 107-108, per-dataset ranks from the stored leaderboard); its V4FinBench gate failed on one Holm-significant fold, so the multiclass 512 arm is blocked and QSAR-TID-11 and the multiclass wide table are not rescored. `wide512-scr` (with §232's readout options) is queued.** **Raise the wide tables.** Under `max_features_strategy: select` the five wide tables
      rank 90th to 103rd of 108 (§220, §221). The planned route is DR8.3's 512-column binary
      checkpoint and the multiclass arm `configs/arms/dr75-width512-mc.yaml` (DR7.5 note,
      2026-11-07), both waiting on GPU. Verify: the wide tables rescored with the 512 checkpoints,
      per-dataset ranks from the stored leaderboard, in a FINDINGS entry.

## Binary

- [x] 49.5 **Done (2026-11-14, FINDINGS §232): the binary TabArena-Lite arm ran on §230's not-harmful verdict and lifted binary from 974 to 997 Elo (23 better, 7 worse, p 0.01); overall 925.6, 106th of 110 (`fintfm_linprobe_all51_1114b`). The readout options join the entry's configuration.** **Progress (2026-11-14, FINDINGS §230): V4FinBench verdict recorded. With §219's sampler fix the probe per leaf is a candidate, +0.0580 mean AP on 5 of 5 folds, and the scale/cv/raw options add +0.0449 (0.3015, +0.1029 over the zero-shot head, 5 of 5, Holm-significant on every fold). Not harmful, so the binary TabArena-Lite arm is queued (`bin-probe-scr`, paired against `fintfm_linprobe_squash_part_090`, 974 Elo); open until it reports. A matched raw-only control was not run.** **Score the readout options on binary, V4FinBench first.** The `scale`, `cv` and `raw`
      readout options lifted multiclass and regression on TabArena-Lite (§222, §223) and are
      untested on binary. The `ppfix-v4-f0` to `-f4` folds (`--probe-options`) and
      `ppfix-v4-compare` are queued locally; they also give the verdict on §219's sampler fix
      for the probe per leaf (§218 was −0.0570 mean AP). Verify: a FINDINGS entry with the
      five-fold V4FinBench verdict under the project's rule (mean AP change above +0.0117 with
      at least 4 of 5 folds positive is a candidate, §173); a binary TabArena-Lite arm runs only
      if the verdict is not harmful.

## TabArena

- [ ] 49.6 **Prepare the resubmission package, once the bar is met.** DR7.6's bar is 1,000
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

- [ ] 49.13 **Progress (2026-11-15): running.** Kernel `scripts/kaggle/ts_reg_pure_c20_s0`
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
- [ ] 49.14 **Separate the checkpoint from the bin count, only if 49.13 moves Elo.** 49.13
      changes the checkpoint and `n_bins` together, so a gain cannot be assigned to either.
      Score the 20-class checkpoint at `FINTFM_REG_BINS=10` with 49.13's configuration
      otherwise unchanged. Verify: per-dataset RMSE and Elo against 49.13's run and against
      §235's run; the FINDINGS entry says which of the two changes carries the difference. Not
      run if 49.13 is null or negative (then there is nothing to assign).
- [ ] 49.15 **The raw-only control for multiclass.** Binary (§239) and regression (§240) each
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
