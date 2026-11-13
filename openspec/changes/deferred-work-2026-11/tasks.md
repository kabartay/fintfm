# Tasks

## Regression and multiclass (design review DR7.6: overall bar 1,000 Elo, 903 at §224)

- [ ] 49.1 **Score the two queued regression readout arms.** `reg-head` appends the head's
      prediction to the ridge readout (`FINTFM_PROBE` `{"raw": true, "head": true}`) and
      `reg-ctx4k` raises `FINTFM_MAX_CONTEXT` to 4,000 (`runs/local_queue/lite_reg_probe.sh`).
      Both pair against `fintfm_lp_reg_raw_1111` (578 Elo, §223). Verify: a FINDINGS entry with
      per-dataset RMSE ratios over the 13 regression datasets, a sign test and Elo; if either
      arm wins, the re-merged 51-dataset Elo and its position from `scripts/lite_positions.py`.
- [ ] 49.2 **Train a regression checkpoint on the current recipe.** `v4-regression.pt` is a
      6,000-step checkpoint on an earlier recipe (§223), and the readout lever is mostly spent
      there (median RMSE ratio 0.985, §223). Train one with cell attention and labels at no
      fewer than the binary checkpoint's 48,000 tasks, `--max-classes` at least the regressor's
      `n_bins`, with `--checkpoint-every` (CLAUDE.md). State the cost from a measured step time
      (§127, §171) before launch. Verify: the 13 regression datasets scored against §223's run
      in a FINDINGS entry, per-dataset RMSE and Elo, and the merged overall Elo.
- [ ] 49.3 **Decide the multiclass partition bag count.** Four bags lifted multiclass from 825
      to 865 Elo but was worse on 3 of 8 datasets (§224); `mc-bag8` (8 bags) is queued.
      Verify: `mc-bag8` against `fintfm_lp_mc_scr_bag4_1112` per dataset in a FINDINGS entry;
      the better of the two is the multiclass run in the next merge.
- [ ] 49.4 **Raise the wide tables.** Under `max_features_strategy: select` the five wide tables
      rank 90th to 103rd of 108 (§220, §221). The planned route is DR8.3's 512-column binary
      checkpoint and the multiclass arm `configs/arms/dr75-width512-mc.yaml` (DR7.5 note,
      2026-11-07), both waiting on GPU. Verify: the wide tables rescored with the 512 checkpoints,
      per-dataset ranks from the stored leaderboard, in a FINDINGS entry.

## Binary

- [ ] 49.5 **Score the readout options on binary, V4FinBench first.** The `scale`, `cv` and `raw`
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

- [ ] 49.7 **Out-of-time split: add a company check and an embargo.**
      `src/fintfm/experiments/v4_out_of_time.py` `time_split` (also used by the context-sweep
      and retrieval-grouping harnesses) does not check companies, so a firm seen in a training
      year can appear in the test years, and a training row's labels span up to six years into
      the test window. `fintfm.evaluation.splits.time_based_split` refuses overlap but these
      harnesses do not use it. Findings that rest on this split include §28 to §35. Verify: the
      harnesses use `time_based_split` (`on_overlap="drop_from_train"`) with an embargo, a test
      shows a firm cannot be on both sides, and the affected findings are rerun with dated notes
      on each.
- [ ] 49.8 **Term-structure coherence: keep query rows out of the context.**
      `src/fintfm/experiments/prior_ablation.py` `term_structure_coherence` loads the whole
      panel as context, so when the context horizon equals the query horizon the held-out query
      rows and their labels are in the context; horizon files may also share firms. §11's
      figure (11.0% of horizon steps, 39% of firms), quoted in the README, rests on it. Verify:
      a test shows no query row in the context, §11's figure is recomputed in a new FINDINGS
      entry, and §11 and the README carry dated notes if it moves.
- [ ] 49.9 **Smaller defects, fix or record a reason not to.**
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
- [ ] 49.11 **Release 0.10.1.** The Zenodo record of 0.10.0 describes the model as 107th of 110,
      while that version's archived FINDINGS still says 106th; the correction is §225, on main
      after the release. A patch release carries §225, the 2026-11-13 documentation revision and
      the inspection fixes. Verify: v0.10.1 is published with CI and the release workflow green,
      PyPI shows 0.10.1, and Zenodo has minted its version DOI (`CLAUDE.md`, "Releases").

## Repository leftovers

- [ ] 49.12 **Small items from the 2026-11-13 inspection.** `scripts/draw_logo.py` needs Pillow,
      which no extra declares; four Kaggle kernels (`task_39_28_replicate_s91`,
      `task_48_3_binned_control`, `task_48_3_quantile_head`, `task_48_9_schedulefree`) have an
      `id` slug that does not match their title; five kernels (`task_dr311_*`,
      `ts_dr75_width512_mc`) carry no pinned commit; `configs/best.yaml` still calls itself the
      best configuration to date (§38). Verify: each is fixed, or noted here with the reason it
      stays.
