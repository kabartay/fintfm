DRAFT, not posted. Do not open a TabArena pull request from this text until every box in "Before posting" is ticked.

# Draft text: TabArena resubmission of FinTFM (follows autogluon/tabarena#616)

Rewritten 2026-11-17. Every figure is from `docs/results/FINDINGS.md` (section in brackets) and is
TabArena-Lite, repeat 0, fold 0, run from the tabarena fork. None of it is the public leaderboard or a
full TabArena run. Code statements are read from the PR branch `fintfm-pr-2026-11` (fork commit
e5cb05b), folder `packages/tabarena/src/tabarena/models/fintfm_linprobe/` unless another path is given.

**Standing from stored runs (§293).** Entry `fintfm_g292_emr`, all 51 Lite datasets, none
excluded or imputed: **1,023.0 Elo (+70 / -80), 99th of 110**. RandomForest (default) 1,000.0, 100th.
TABFLEX_GPU 977.7. The owner's bar for resubmitting is 1,000 Elo (DR7.6). The entry clears it by 23.0
Elo, which is inside the interval, and the 10,000-row and 5,000-row gate lines were drawn after
results were seen (§286, §287, §290, §292), so this is a configuration choice on one fold. These are
stored-run figures. The reproduction from the PR branch with no `FINTFM_*` variables completed 51 of
51 datasets and scores 1,018.7 Elo (+73 / -79), 99th of 110 (FINDINGS §297). It uses AutoGluon
1.6.4b20261109 (the PR branch's floor); the stored runs used 1.6.4b20260919.

## What #616 said, and where this draft answers it

The pull request autogluon/tabarena#616 (closed 2026-09-25, withdrawn by us before the maintainers'
full run) received three concerns from the TabArena collaborator who reviewed it, and one positive
finding.

| # | Reviewer concern in #616 | State now |
| --- | --- | --- |
| 1 | The entry declared binary only. Excluding datasets or problem types because the model does worse there is cherry-picking and is not eligible. | Answered. The entry declares binary, multiclass and regression and covers every Lite dataset. No exclusion. Per-type figures are reported below. |
| 2 | Datasets were missing, and "there are failures that should be fixed". The reviewer asked that "new submissions ... cover all datasets". | Answered on the stored runs: 51 of 51 datasets, none excluded, none imputed (§290, §221). Wide tables run through `max_features_strategy: select`, not a new checkpoint. Reproduction coverage: `<REPRO: datasets completed of 51, failed splits>`. |
| 3 | The model does not beat a tuned KNN or other simple baselines, so it does not meet the competitiveness bar. | Partly answered. Stored entry: 1,023.0 Elo, 99th of 110, above RandomForest (default) at 1,000.0 and above TABFLEX_GPU at 977.7. The tuned-and-ensembled KNN figure is from the older 110-method board (965.3 on 2026-11-15, `../tabarena/eval/fintfm_linprobe_all51_1115/tabarena_leaderboard.csv`); it is not re-read against the 1,009.8 board, so re-read it before posting. It remains far below tuned tree ensembles (`GBM (tuned + ensemble)` 1379.3 on the 2026-11-15 board, `openspec/changes/tabarena-integration/tasks.md` 45.4). The margin over RandomForest is inside the interval. |
| 4 | Positive: "Based on the submission's code and numbers you shared, I think everything is correct." | The wrapper has changed a lot since that review (partitioning, readout, cross-fitting, per-type checkpoints, size gates). The changes are listed below so the reviewer can see what needs a fresh read. |

In #616 we wrote that we would "come back with full coverage from the start". That is now true.

## Draft pull request body (follows the TabArena template)

### Summary

FinTFM is a tabular foundation model pretrained on a synthetic prior only. This entry, labelled
"FinTFM with a trained linear readout", covers all 51 TabArena-Lite datasets (30 binary, 8
multiclass, 13 regression). It is not zero-shot: it fits a linear readout (logistic regression, ridge
or random-Fourier-feature ridge for regression) on training rows inside each partition leaf. On
TabArena-Lite (repeat 0, fold 0) the stored runs score 1,023.0 Elo, 99th of 110 (§293); the
reproduction from this branch scores 1,018.7 Elo, 99th of 110. The wrapper does not follow
the template's rule on `X_val` / `y_val` and internal cross-validation; the deviations are declared in
Details.

<details>
<summary>Details</summary>

**What the reviewers need to know first**

- **Not zero-shot.** The readout is fitted on the training split. The label is "FinTFM with a trained
  linear readout".
- **Checkpoints, one per problem type, all public (Apache-2.0, ungated since 2026-11-17), pinned by
  Hugging Face revision and SHA-256** in `models/fintfm/checkpoints.py`:
  - binary: `kabartay/fintfm-binary`, `v4-cellattn-labels.pt`, revision `f116bfd4...`;
  - multiclass: `kabartay/fintfm-multiclass`, `dr74-mc-s0.pt`, revision `b292efa8...`;
  - regression: `kabartay/fintfm-regression`, `v4-binned48_3-control.pt`, revision `787a3446...`.
- **Package.** `fintfm` 0.12.0 is on PyPI (released 2026-11-17, Zenodo DOI 10.5281/zenodo.24325794),
  Apache-2.0. Problem types: binary, multiclass, regression. Paper and docs: https://github.com/kabartay/fintfm.
  The PR branch's `info.py` and `pyproject.toml` still pin `fintfm==0.11.0`; they must be changed to
  0.12.0 (the 0.11.0 release does not hold the multiclass and regression checkpoints), see "Before
  posting".
- **No environment variables.** Every lever is an entry default per problem type in `model.py`
  (`_entry_checkpoint`, `_entry_partition`, `_entry_bags_max_rows`, `_entry_probe`,
  `_entry_probe_spline`, `_entry_spline_max_rows`, `_entry_probe_onehot`, `_entry_reg_readout`,
  `_entry_reg_rff_max_rows`, `_entry_prob_floor`, `_entry_crossfit_folds`). `FINTFM_*` variables still
  exist as optional overrides (`_env_entry_overrides`); the reproduction removes them.
- **Numbers are TabArena-Lite, repeat 0, fold 0.** Not a full TabArena run, not the public leaderboard.

**Results**

| quantity | stored runs | reproduction from this branch | source |
| --- | --- | --- | --- |
| Overall, 51 of 51 datasets | 1,023.0 Elo (+70 / -80), 99th of 110 | 1,018.7 (+73 / -79), 99th of 110 | §293, §297 |
| RandomForest (default), same board | 1,000.0, 100th | 1,000.0, 100th | §293, §297 |
| TABFLEX_GPU, same board | 977.7 | 978.0 | §293, §297 |
| Binary, 30 datasets, problem-type Elo | 1,090.3 (+110 / -104), 92nd of 108 | 1,083.4 (+109 / -100), 92nd of 108 | §291, §297 |
| Multiclass, 8 datasets | 947.1 (+100 / -253), 98th of 108 | 947.1 (+100 / -253), 98th of 108 | §293, §297 |
| Regression, 13 datasets | 845.1 (+127 / -219), 97th of 103 | 845.1 (+127 / -219), 97th of 103 | §293, §297 |
| Coverage | 51 of 51, none imputed | 51 of 51, none imputed | §293, §297 |

The per-type stored figures for the current entry must be re-read with `scripts/lite_positions.py`
before posting; the values above are the earlier entries and are marked so.

**Stored runs are what the reproduction must confirm.** The stored entry was assembled from runs on
different machines (Kaggle CPU, Kaggle T4, a Mac) and different AutoGluon nightlies. §283 shows two
Mac reruns reproducing stored metrics exactly (relative difference 0.0). The PR-branch reproduction
(no `FINTFM_*` variables; each table on the kind of machine its stored run used) is the check a
maintainer could make. Its result replaces the stored-run figures as the entry's standing (§290):
49 of 51 datasets within 0.3% of the stored metric, anneal -0.56%, and
taiwanese_bankruptcy_prediction +22.2% on Kaggle CPU, which reproduces exactly on the Mac where it was
stored (+0.01%), so that difference is the machine (§297).

**What changed since #616** (the withdrawn entry was binary only, zero-shot head, 94th of 95, mean
ROC-AUC 0.7823 over 27 datasets, `docs/competition/SUBMISSION.md`, §122). The README's change table
(§225 to §290) lists each step with its Elo. In short:

- partitioned contexts (a tree routes rows to leaves, each leaf has its own context, with bags);
- a linear readout per leaf, with the row's own squashed features appended;
- one checkpoint per problem type;
- column selection for wide tables;
- cross-fitted readout for binary (§254), binary depth scaling and a 50,000-row readout budget
  (§257), a multiclass probability floor (§269), one-hot indicators for multiclass and an RFF readout
  for regression fits of at most 5,000 rows (§283), four readout bags for binary fits of at most
  5,000 rows (§286), spline features for binary fits of at most 10,000 rows (§290), spline features for
  multiclass fits of at most 10,000 rows and for every regression fit (§293).

Each step after §254 is a configuration choice on one fold inside the Elo interval, and the size lines
were drawn after results were seen.

**Controls the reviewers should read beside the Elo values.** Each problem type has a matched control
in which the readout sees only the squashed raw features, with no FinTFM representation (`no_rep`): a
tree with a linear model per leaf. These controls were run against earlier entries, not the current
one.

| problem type | entry compared | raw-only control | representation better on | source |
| --- | --- | --- | --- | --- |
| binary | 997 Elo (earlier entry) | 1009 Elo | 24 of 30 datasets by ROC-AUC, but the Elo intervals overlap | §239 |
| multiclass | 874 Elo (earlier entry) | 829 Elo | 7 of 8 datasets (sign test p = 0.07) | §243 |
| regression | 689 Elo (earlier entry) | 537 Elo | 12 of 13 datasets (sign test p = 0.003) | §240 |

**The binary Elo is not evidence for the representation.** A tree with a logistic regression per leaf
and no FinTFM representation scored 1009 against 997 (§239). The representation is supported on
multiclass and regression, not on binary Elo. This is TabArena breadth only, not a credit result.

**Deviation from the template's implementation rule (`X_val` / `y_val`, no internal CV).** The template
says `_fit` uses the `X_val` / `y_val` that TabArena passes, with no internal cross-validation,
hold-out split or tuning on training data. This wrapper does not meet that. What the code does:

1. `FinTFMLinearProbeModel._fit` accepts `X_val` and `y_val` and passes them to the parent
   `FinTFMModel._fit` (`model.py`), which ignores them (docstring: "`X_val`/`y_val` and `time_limit` are
   ignored", `fintfm/model.py`). Nothing is tuned or stopped on the validation split. The model has no
   training loop; the context is the training rows.
2. The binary and multiclass readout picks its regularisation `C` by internal cross-validation on
   training rows: `_new_classifier_probe` builds `LogisticRegressionCV(Cs=np.logspace(-3, 2, 6), cv=3,
   scoring="neg_log_loss")` when the probe option `cv` is set (it is in the binary and multiclass
   entry, `_entry_probe`). Folds drop below 3 only if the rarest class has fewer than 3 rows.
3. The binary readout is cross-fitted with 5 folds (`_entry_crossfit_folds`, `_crossfit_apply`,
   `_crossfit_representation`): readout rows are represented by a context that leaves them out.
   Multiclass and regression do not cross-fit (0 folds).
4. The multiclass probability floor is chosen on out-of-fold training rows (`_entry_prob_floor` =
   `"cv"`, `_collect_floor_oof`, `_finish_prob_floor`; `StratifiedKFold` plus `cross_val_predict` on up
   to `PROB_FLOOR_MAX_ROWS` readout rows). The code comment says it never uses validation or test rows.
5. Regression with the RFF readout fits `RFFRidgeCV` (`_fit_ridge`), which selects ridge alphas by its
   own cross-validation on training rows. Regression blending by CV (`reg_readout_blend="cv"`) exists
   but is off in the entry.
6. Training matrix: AutoGluon's own preprocessing runs with out-of-fold categorical encoding when
   cross-fitting is on (`preprocess`).

All of these use training rows only. None uses the validation split. They are part of the measured
method: removing them changes the entry. Options for the owner: change the wrapper to select `C` and
the floor on the `X_val` / `y_val` TabArena passes (a new measurement, not the entry measured here), or
post with this declaration and let the maintainers decide. Decision pending.

**Search space.** `hpo.py` declares five categorical hyperparameters (`probe_binary`,
`probe_multiclass`, `probe_regression`, `partition_binary`, `partition_multiclass`) with 3, 3, 2, 2
and 3 values. The product of the options is at most 3 x 3 x 2 x 2 x 3 = 108 combinations, and a
problem type samples only its own keys, so fewer. This is under the template's "about 200
configurations", no sampled config has been run on TabArena, and the file calls the space "PROPOSED,
not a tuned one". `info.py` sets `can_hpo=False`; the raw results hold the default config only
(`manual_configs=[{}]`). `hpo.py` also still says the 932 Elo entry was made with cross-fitting off,
which is outdated. The proposed `probe_crossfit_folds` space (5 or 0) is in
`proposed_search_space_additions`, not in `search_space`. The template's own Results item asks for the
default plus about 25 random configurations "where the method has a search space"; none were run.
Decision pending: declare "no search space" (`can_hpo=False`, as is) or run a sampled set.

**Wiring (`time_limit`, resources, seed, pickling).**
- `time_limit`: accepted and forwarded to the parent, which ignores it. There is no early stop.
  A fit near the limit can still be killed by AutoGluon; the stored runs recorded `TimeLimitExceeded`
  only on busy Mac sessions (§286), and the kernels give each dataset its own 2-hour timeout.
- `num_cpus` / `num_gpus`: a granted GPU selects `cuda` when available, else CPU
  (`fintfm/model.py` `_choose_device`); MPS is never chosen automatically, and `_forced_device`
  honours an override variable. `num_cpus` is passed to
  the parent and not otherwise used. `info.py` declares `compute="cpu"`; the nine binary tables above
  10,000 rows ran on a T4 in the stored runs, so the hardware statement below must say so.
- Seed: `seed_name = "random_state"`; `self.random_seed` seeds the row draws, bagging, the cross-fit
  and floor folds.
- Pickling: the entry references checkpoints by pinned dict, not by loaded object. The audit below was
  run on an earlier commit and must be re-run.
- Optional imports (`fintfm`, `torch`, `sklearn`) are inside methods. No global monkey-patching was
  found in the files read.

**Cost.** `<REPRO: summary.json per-dataset wall times, total>`. Stored runs: longest single fit 3,262 s
on a Mac (`polish_companies_bankruptcy`, §290), regression 269 min for 13 datasets on a loaded CPU
(§235).

**Hardware and entry-point script.** Stored runs: Kaggle CPU, Kaggle T4 (nine binary tables above
10,000 rows) and a Mac, from the fork. Reproduction: four Kaggle kernels from
`scripts/kaggle/repro_pr_kernel_template.py` (two CPU binary kernels for the 21 tables of at most
10,000 rows, one T4 binary kernel for the 9 larger tables, one CPU multiclass and regression kernel).
`<REPRO: CPU model, RAM, GPU, command, kernel versions>`. Optional: link to the `results.pkl`
directory, `<REPRO: link>`.

**Licences.** `fintfm` and all three checkpoints are Apache-2.0. `info.py` records `license="Apache-2.0"`,
`commercial_use=True`, `verified=False`.

</details>

### Tests

Not run yet on this branch for the resubmission; to be filled with the commands actually run:
`ruff check . && ruff format --check .`; `pytest -m models -k FinTFM`;
`python -P -m tabarena.tools.audit_pickle --model FinTFM-LinProbe`; the model's own tests in
`models/fintfm_linprobe/tests/`. Earlier pickle audit (2026-11-16, commit ecd4ac5, fintfm 0.10.1,
AutoGluon 1.6.4b20261115, binary only) passed: `model.pkl` growth 88 B per row against the 1,024 B flag
limit (`docs/competition/SUBMISSION.md`). Regression could not be audited then because its checkpoint
was not public; it is now. Not claimed for the current commit.

### Model or system submission

Kind: model. A new entry; it does not replace a leaderboard entrant (the #616 entry was withdrawn before
any run).

**Files**

- [ ] Model: `models/fintfm_linprobe/{__init__,model,hpo,info}.py` exist (plus `rff.py`,
  `target_transform.py`, and tests in `models/fintfm_linprobe/tests/`). Not met: the lazy entry in
  `models/__init__.py` has `FinTFMModel` only, not `FinTFMLinearProbeModel`; the extra
  `fintfm_linprobe` is not in `extended` (only `tabarena[fintfm]` is); `website/website_format.py` lists
  `"FINTFM"` only; `tests/tabarena/models/smoke_configs.py` has `"FinTFM-LinProbe": ModelSmokeTest(problem_types=("binary",))`
  (binary only, but the entry declares three types). Also `tests/tabarena/models/test_fintfm_*.py` sit
  under `tests/`, where the template says model tests must not go: check whether these are ours.
- [ ] Logo: none prepared. `info.py` names organization from `reference_url` only; check whether the
  leaderboard shows a logo for it, else attach a square SVG or PNG of at least 128 px.

**Implementation**

- [ ] `_fit` uses the `X_val` / `y_val` TabArena passes, no internal cross-validation, and about 200
  configurations. **Not ticked.** See the declared deviations and the search-space paragraph above.
- [ ] `time_limit`, `num_cpus` / `num_gpus` and the seed wired through; optional imports inside the
  wrapper; the model pickles. Partly: see "Wiring". `time_limit` is accepted and ignored; the pickle
  audit must be re-run.
- [ ] Exact pin, extra matches `pip_extra`, checkpoints pinned by revision, licence and problem types
  stated. Partly: `fintfm==0.12.0` is released but the PR branch still pins `0.11.0` in `info.py`
  (`pip_extra`) and `pyproject.toml`. Checkpoints are pinned by revision and SHA-256 and are ungated.
- [ ] Hosted API: not applicable.
- [ ] `ruff` clean and `pytest -m models -k FinTFM` passes. Not run for this PR.

**Results**

- [ ] TabArena-Lite with the official pipeline: the default config only; no sampled configs were run
  (no search space measured). Validation protocol: the arena default. `<REPRO: confirm official
  validation protocol was used, official_validation_protocol not set to False>`.
- [ ] Hardware and entry-point script stated: see Details, `<REPRO>` placeholders.
- [ ] Optional: link to the run's `expname` folder with `results.pkl` files.
- [ ] Sign-off on the maintainer re-run: `<GitHub handle, owner to supply>`.

`<REPRO: leaderboard table printed by the run, hardware, command>`

---

By submitting this pull request, I confirm that you can use, modify, copy, and redistribute this
contribution, under the terms of your choice.

## Before posting (all must be true)

Done:

- [x] Weights public: binary, multiclass and regression checkpoints are Apache-2.0, ungated, pinned by
      revision and SHA-256 in the fork's `checkpoints.py`.
- [x] `fintfm` 0.12.0 released on PyPI (2026-11-17, DOI 10.5281/zenodo.24325794).
- [x] Entry defaults on the PR branch (`fintfm-pr-2026-11`, e5cb05b): every lever is a per-type default;
      no environment variable is needed.
- [x] Coverage: 51 of 51 datasets on the stored runs, none excluded or imputed.
- [x] Stored entry at 1,023.0 Elo, 99th of 110, clears the owner's 1,000 bar (DR7.6) by 23.0, inside the
      interval.

Remaining:

- [ ] **Reproduction numbers.** The four Kaggle kernels finish (the T4 kernel waits for weekly GPU
      quota), the results are merged with `merge_fintfm_lite.py --expect 51`, and the Elo and position
      are scored with `scripts/lite_positions.py`. Fill every `<REPRO: ...>`. If the reproduction falls
      below 1,000 Elo, the owner decides again under DR7.6. The AutoGluon nightly differs from the
      stored runs.
- [ ] **`X_val` / internal-CV question.** Owner decision: change the wrapper to use the validation
      split TabArena passes, or post with the declared deviations (internal `LogisticRegressionCV`,
      5-fold cross-fit for binary, multiclass probability floor on out-of-fold rows, `RFFRidgeCV`).
      Do not tick the template box either way unless the wrapper changes.
- [ ] **Search space.** Decide: keep `can_hpo=False` with the 108-combination proposed space, or run a
      sampled set (the template asks about 200 configs and the Lite default plus about 25 configs).
      Fix the stale comment in `hpo.py`.
- [ ] **Pin.** `info.py` `pip_extra` and `pyproject.toml` (`fintfm`, `fintfm_linprobe`) from
      `fintfm==0.11.0` to `0.12.0`, and the PR-branch commit used by the kernels re-checked against it.
- [ ] **Registration files.** Lazy entry for `FinTFMLinearProbeModel` in `models/__init__.py`, the
      `fintfm_linprobe` extra in `extended`, the family in `website/website_format.py`, the smoke
      override covering multiclass and regression, and the `tests/` placement checked.
- [ ] **Tests run on the PR branch.** `ruff check`, `ruff format --check`, `pytest -m models -k FinTFM`,
      and the model's own tests, with output kept.
- [ ] **`audit_pickle`** re-run on the PR-branch head for binary, multiclass and regression
      (the 2026-11-16 audit was binary only, commit ecd4ac5).
- [ ] **Hardware and script** filled in from the kernels' `summary.json`; the nine large binary tables
      ran on a T4 and `info.py` says `compute="cpu"`.
- [ ] **Sign-off handle** and, if the leaderboard shows none, a logo.
- [ ] **Numbers re-read.** Per-type Elo and the KNN and tree-ensemble rows re-read from the board
      current at posting (the field was 110 methods on 2026-11-17).
- [ ] **Support status.** `docs/competition/SUBMISSION.md` says the project is no longer actively
      supported from mid-November 2026. Decide who answers the reviewers.
- [ ] **No confidential or peer-product names** in the posted text.
- [ ] **The owner's yes.**
