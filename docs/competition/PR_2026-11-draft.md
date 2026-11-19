DRAFT, not posted. Do not open a TabArena pull request from this text until every box in "Before posting" is ticked.

# Draft text: TabArena resubmission of FinTFM (follows autogluon/tabarena#616)

Rewritten 2026-11-19. Every figure is from `docs/results/FINDINGS.md` (section in brackets) and is
TabArena-Lite, repeat 0, fold 0, run from the tabarena fork. None of it is the public leaderboard or a
full TabArena run. Code statements are read from the PR branch `fintfm-pr-2026-11` (fork commit
8b3ea4b), folder `packages/tabarena/src/tabarena/models/fintfm_linprobe/` unless another path is given.

**Standing (§303).** The entry reproduced from the PR branch with no `FINTFM_*` variable, all 51 Lite
datasets, none excluded or imputed: **1,019.6 Elo (+72 / -75), 99th of 110**. RandomForest (default)
1,000.0, 101st; TABFLEX_GPU 978.0, 102nd. The owner's bar for resubmitting is 1,000 Elo (DR7.6). The
entry clears it by 19.6 Elo, which is inside the interval, and the 10,000-row and 5,000-row gate lines
were drawn after results were seen (§286, §287, §290, §292), so this is a configuration choice on one
fold. The stored runs it reproduces score 1,024.0 (§301). AutoGluon 1.6.4b20261109 (the PR branch's
floor); the stored runs used 1.6.4b20260919.

**Owner decisions, 2026-11-19.** Post with the internal-CV deviations declared (no wrapper change);
default config only (`can_hpo=False`), the search space stays a proposal; report taiwanese_bankruptcy_prediction
with both machines' values; the owner signs off as `kabartay` and answers the review himself.

## What #616 said, and where this draft answers it

The pull request autogluon/tabarena#616 (closed 2026-09-25, withdrawn by us before the maintainers'
full run) received three concerns from the TabArena collaborator who reviewed it, and one positive
finding.

| # | Reviewer concern in #616 | State now |
| --- | --- | --- |
| 1 | The entry declared binary only. Excluding datasets or problem types because the model does worse there is cherry-picking and is not eligible. | Answered. The entry declares binary, multiclass and regression and covers every Lite dataset. No exclusion. Per-type figures are reported below. |
| 2 | Datasets were missing, and "there are failures that should be fixed". The reviewer asked that "new submissions ... cover all datasets". | Answered: the reproduction from the PR branch completed 51 of 51 datasets with no failed split, none imputed (§297, §303). Wide tables run through `max_features_strategy: select`, not a new checkpoint. |
| 3 | The model does not beat a tuned KNN or other simple baselines, so it does not meet the competitiveness bar. | Partly answered. 1,019.6 Elo, above RandomForest (default) at 1,000.0, TABFLEX_GPU at 978.0 and KNN (tuned + ensemble) at 965.5 on the same board (§303). It remains far below tuned tree ensembles (`GBM (tuned + ensemble)` 1,380.1). The margin over RandomForest is inside the interval. |
| 4 | Positive: "Based on the submission's code and numbers you shared, I think everything is correct." | The wrapper has changed a lot since that review (partitioning, readout, cross-fitting, per-type checkpoints, size gates). The changes are listed below so the reviewer can see what needs a fresh read. |

In #616 we wrote that we would "come back with full coverage from the start". That is now true.

## Draft pull request body (follows the TabArena template)

### Summary

FinTFM is a tabular foundation model pretrained on a synthetic prior only. This entry, labelled
"FinTFM with a trained linear readout", covers all 51 TabArena-Lite datasets (30 binary, 8
multiclass, 13 regression). It is not zero-shot: it fits a linear readout (logistic regression, ridge
or random-Fourier-feature ridge for regression) on training rows inside each partition leaf. On
TabArena-Lite (repeat 0, fold 0), reproduced from this branch with no environment overrides, it scores
1,019.6 Elo, 99th of 110. The wrapper does not follow the template's rule on `X_val` / `y_val` and
internal cross-validation; the deviations are declared in Details.

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
- **Package.** `fintfm==0.12.0` on PyPI (released 2026-11-17, Zenodo DOI 10.5281/zenodo.24325794),
  Apache-2.0, pinned in `info.py` (`pip_extra`) and in the `fintfm` and `fintfm_linprobe` extras.
  Problem types: binary, multiclass, regression. Paper and docs: https://github.com/kabartay/fintfm.
- **No environment variables.** Every lever is an entry default per problem type in `model.py`
  (`_entry_checkpoint`, `_entry_partition`, `_entry_bags_max_rows`, `_entry_probe`,
  `_entry_probe_spline`, `_entry_spline_max_rows`, `_entry_probe_onehot`, `_entry_reg_readout`,
  `_entry_reg_rff_max_rows`, `_entry_prob_floor`, `_entry_crossfit_folds`, `_entry_reg_clip_margin`).
  `FINTFM_*` variables still exist as optional overrides (`_env_entry_overrides`); the reproduction
  removes them.
- **Numbers are TabArena-Lite, repeat 0, fold 0.** Not a full TabArena run, not the public leaderboard.

**Results** (reproduction from this branch; each per-type figure is from a leaderboard over that
problem type's datasets only)

| quantity | value | source |
| --- | --- | --- |
| Overall, 51 of 51 datasets | 1,019.6 Elo (+72 / -75), 99th of 110 | §303 |
| RandomForest (default), same board | 1,000.0, 101st | §303 |
| TABFLEX_GPU (default), same board | 978.0, 102nd | §303 |
| KNN (tuned + ensemble), same board | 965.5, 104th | §303 |
| Binary, 30 datasets | 1,083.4 (+109 / -100), 92nd of 108 | §297 |
| Multiclass, 8 datasets | 947.1 (+100 / -253), 98th of 108 | §297 |
| Regression, 13 datasets | 852.6 (+119 / -221), 97th of 103 | §303 |
| Coverage | 51 of 51, no failed split, none imputed | §297, §303 |

**How the reproduction matches the stored runs.** The stored entry was assembled from runs on
different machines (Kaggle CPU, Kaggle T4, a Mac) and AutoGluon nightlies. The reproduction from the
PR branch (no `FINTFM_*` variable) gives 49 of 51 datasets within 0.3% of the stored metric (§297).
anneal is -0.56%. taiwanese_bankruptcy_prediction is +22.2% worse on Kaggle CPU, and on the Mac, where
its stored run was fitted, it reproduces to +0.01%, so that difference is the machine. The figures
above keep the Kaggle CPU value.

**What changed since #616** (the withdrawn entry was binary only, zero-shot head, 94th of 95, mean
ROC-AUC 0.7823 over 27 datasets, `docs/competition/PR_2026-09.md`, §122). The README's change table
(§225 to §303) lists each step with its Elo. In short:

- partitioned contexts (a tree routes rows to leaves, each leaf has its own context, with bags);
- a linear readout per leaf, with the row's own squashed features appended;
- one checkpoint per problem type;
- column selection for wide tables;
- cross-fitted readout for binary (§254), binary depth scaling and a 50,000-row readout budget
  (§257), a multiclass probability floor (§269), one-hot indicators for multiclass and an RFF readout
  for regression fits of at most 5,000 rows (§283), four readout bags for binary fits of at most
  5,000 rows (§286), spline features for binary fits of at most 10,000 rows (§290), spline features for
  multiclass fits of at most 10,000 rows and for every regression fit (§293), and regression
  predictions clipped to the training target's range widened by 10% (§295, §301).

Each step after §254 is a configuration choice on one fold inside the Elo interval, and the size lines
were drawn after results were seen. The clip removes a failure, not a tuning gain: on diamonds the
spline readout predicted as far as -19 million for a few off-manifold rows, and the clip takes its
test RMSE from 249,744 to 957; no regression table got worse by more than 0.1% (§301).

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

**Declared deviation from the template's implementation rule (`X_val` / `y_val`, no internal CV).**
The template says `_fit` uses the `X_val` / `y_val` that TabArena passes, with no internal
cross-validation, hold-out split or tuning on training data. This wrapper does not meet that, and we
submit it as measured rather than change the method. What the code does:

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
7. The regression clip records the training target's minimum and maximum at fit (`_clip_bounds`).

All of these use training rows only. None uses the validation split. They are part of the measured
method: removing them changes the entry. If the maintainers prefer, `C` and the floor could be
selected on the `X_val` / `y_val` TabArena passes instead; that would be a new measurement, not the
entry reported here.

**Search space: none submitted.** `info.py` sets `can_hpo=False` and the results hold the default
config only (`manual_configs=[{}]`). `hpo.py` keeps a PROPOSED space, not a tuned one: five
categorical hyperparameters (`probe_binary`, `probe_multiclass`, `probe_regression`,
`partition_binary`, `partition_multiclass`) with 3, 3, 2, 2 and 3 values, at most 108 combinations, and
a further proposal (`probe_crossfit_folds`, 5 or 0) in `proposed_search_space_additions`. No sampled
config has been run on TabArena.

**Wiring (`time_limit`, resources, seed, classes, pickling).**
- `time_limit`: accepted and forwarded to the parent, which ignores it. There is no early stop.
  AutoGluon's own per-fold projection can still stop a fit near the limit; on Kaggle's 4-CPU sessions
  QSAR-TID-11 now sits at it (§301).
- `num_cpus` / `num_gpus`: a granted GPU selects `cuda` when available, else CPU
  (`fintfm/model.py` `_choose_device`); MPS is never chosen automatically, and `FINTFM_DEVICE` forces a
  device. `info.py` declares `compute="gpu"`: the nine binary tables above 10,000 training rows were
  fitted on a T4, the rest on CPU. The readout's BLAS calls run under `threadpoolctl` at
  `min(4, num_cpus)` threads (`blas_threads="auto"`): without it, OpenBLAS on an 8-vCPU host
  oversubscribed and three large tables hit the time limit (§296-§299).
- Classes: the multiclass checkpoint represents at most 10 classes, declared as `ag.max_classes = 10`.
- Seed: `seed_name = "random_state"`; `self.random_seed` seeds the row draws, bagging, the cross-fit
  and floor folds.
- Pickling: the entry references checkpoints by pinned dict, not by loaded object. The audit passes on
  all three problem types (see Tests).
- Optional imports (`fintfm`, `torch`, `sklearn`) are inside methods. No global monkey-patching was
  found in the files read.

**Cost** (fit plus inference time recorded in the 51 `results.pkl` files, §303): 18.8 hours in total.
Kaggle CPU, 34 tables: 11.4 h of training, longest hiva_agnostic 2,829 s. Hugging Face T4, 9 tables:
3.4 h, longest kddcup09_appetency 3,353 s. Mac CPU, 8 tables: 3.4 h, longest diamonds 2,591 s.
Inference: 38 minutes in total.

**Hardware and entry-point script.** The reproduction used the kernels generated from
`scripts/kaggle/repro_pr_kernel_template.py` (fintfm repository). Each installs the fork at the PR
branch and `fintfm==0.12.0`, removes every `FINTFM_*` variable, and fits one dataset per process with
the registry entry's default config:

```python
experiments = TabArenaV0pt1ExperimentBundle(models=[("FinTFM-LinProbe", 0)]).build_experiments()
TabArenaContext().build_and_run_jobs(experiments, expname=..., subset="lite",
    build_kwargs={"dataset_names": [dataset]}, new_result_prefix="[New] ", debug_mode=True)
```

The arena's default validation protocol was used (no `official_validation_protocol` override).
Machines:
- Kaggle CPU sessions (4 CPUs, 31 GB RAM): 21 binary tables of at most 10,000 rows, 7 multiclass and
  6 regression tables.
- Hugging Face Jobs `t4-medium` (8 vCPU, 30 GB RAM, Tesla T4, torch 2.15.0+cu132): the 9 binary tables
  above 10,000 rows. Five of them ran with `OPENBLAS_NUM_THREADS=4` set in the job, before the
  adapter's own limit existed; with the adapter's limit and no variable set, three of them were rerun
  and matched exactly (§299).
- A Mac (Apple M3 Max, 16 cores, 64 GB, CPU only; torch 2.15.0, numpy 2.5.5, scikit-learn 1.9.1):
  SDSS17 and 7 regression tables, where the stored runs were fitted.

The regression tables were fitted at the clip commit (fork fe2b072, the same model code as this
branch), the others at 65c21d5 (§297, §301). The `results.pkl` files are available on request.

**Licences.** `fintfm` and all three checkpoints are Apache-2.0. `info.py` records `license="Apache-2.0"`,
`commercial_use=True`, `verified=False`.

</details>

### Tests

Run on the PR branch on 2026-11-19 (fork 9c7db11 to 8b3ea4b; outputs kept):

- `ruff check` and `ruff format --check` (ruff 0.14.0) on `models/fintfm`, `models/fintfm_linprobe`
  and `tests/tabarena/models`: clean.
- The model's own tests (`models/fintfm/tests/`, `models/fintfm_linprobe/tests/`): 414 passed,
  1 skipped.
- `pytest -m models -k FinTFM`: passes. On a machine without CUDA the FinTFM-LinProbe smoke fit is
  skipped because `info.py` declares `compute="gpu"`; run with `compute="cpu"` it passes on binary,
  multiclass and regression. `test_model_many_class` skips it on its declared 10-class limit.
- `python -P -m tabarena.tools.audit_pickle --model FinTFM-LinProbe --problem-type {binary,multiclass,regression}`:
  "OK: nothing that scales with the training rows sits in model.pkl" on all three. **Note for the
  maintainers:** the unpatched tool crashes on this model. scipy's `BSpline` (inside the spline
  readout) keeps the array namespace module in `_xp`, `tensor_nbytes` walks into that module and on
  through `sys.modules`, and stops on a `urllib3` mapping that refuses iteration. We ran it with
  module objects skipped in the walk (`isinstance(value, types.ModuleType)`); the same one-line guard
  would fix the tool for any model holding scipy splines.

### Model or system submission

Kind: model. A new entry; it does not replace a leaderboard entrant (the #616 entry was withdrawn before
any run).

**Files**

- [x] Model: `models/fintfm_linprobe/{__init__,model,hpo,info}.py` (plus `rff.py`,
  `target_transform.py`), tests in `models/fintfm_linprobe/tests/` and `models/fintfm/tests/`. Lazy
  entry in `models/__init__.py`, the `fintfm_linprobe` extra in `extended`, the family in
  `website/website_format.py`, and a smoke override covering all three problem types.
- [ ] Logo: none prepared. `info.py` names the organization from `reference_url` only; attach a square
  SVG or PNG of at least 128 px if the leaderboard shows none.

**Implementation**

- [ ] `_fit` uses the `X_val` / `y_val` TabArena passes, no internal cross-validation, and about 200
  configurations. **Not ticked.** See the declared deviations and "Search space: none submitted".
- [x] `time_limit`, `num_cpus` / `num_gpus` and the seed wired through; optional imports inside the
  wrapper; the model pickles. `time_limit` is accepted and ignored (see Wiring).
- [x] Exact pin (`fintfm==0.12.0`), extra matches `pip_extra`, checkpoints pinned by revision and
  SHA-256 and ungated, licence and problem types stated.
- [ ] Hosted API: not applicable.
- [x] `ruff` clean and `pytest -m models -k FinTFM` passes (see Tests).

**Results**

- [x] TabArena-Lite with the official pipeline and the default validation protocol: the default config
  only; no sampled configs (no search space submitted).
- [x] Hardware and entry-point script stated: see Details.
- [ ] Optional: link to the run's `expname` folder with `results.pkl` files (available on request).
- [x] Sign-off on the maintainer re-run: kabartay.

Leaderboard rows around the entry, as printed by the reproduction (TabArena-Lite, 110 methods; §303):

| position | method | Elo |
| --- | --- | --- |
| 97 | NN_TORCH (default) | 1,037.1 |
| 98 | XRFM_GPU (default) | 1,031.7 |
| **99** | **[New] TA-FINTFM-LINPROBE (default)** | **1,019.6** |
| 100 | XT (default) | 1,008.8 |
| 101 | RF (default) | 1,000.0 |
| 102 | TABFLEX_GPU (default) | 978.0 |

---

By submitting this pull request, I confirm that you can use, modify, copy, and redistribute this
contribution, under the terms of your choice.

## Before posting (all must be true)

Done:

- [x] Weights public: binary, multiclass and regression checkpoints are Apache-2.0, ungated, pinned by
      revision and SHA-256 in the fork's `checkpoints.py`.
- [x] `fintfm` 0.12.0 released on PyPI (2026-11-17, DOI 10.5281/zenodo.24325794) and pinned on the
      PR branch.
- [x] Entry defaults on the PR branch: every lever is a per-type default; no environment variable is
      needed.
- [x] Reproduction: 51 of 51, 1,019.6 Elo, 99th of 110, above the 1,000 bar (DR7.6) by 19.6, inside
      the interval (§303).
- [x] Owner decisions (2026-11-19): declare the internal CV; default config only; taiwanese with both
      machines' values; sign-off and review by kabartay.
- [x] Registration files, test placement and the smoke override (fork 81e94d1, 8b3ea4b).
- [x] Tests and `audit_pickle` on all three problem types, outputs kept (`runs/pr_checks_9c7db11/`).
- [x] Hardware, script and cost filled in.
- [x] Support status: `docs/competition/SUBMISSION.md` records that the owner answers this review.

Remaining:

- [ ] **Logo**, only if the leaderboard shows none for the organization.
- [ ] **Numbers re-read** from the board current at posting (the field was 110 methods on 2026-11-19).
- [ ] **No confidential or peer-product names** in the posted text (checked 2026-11-19; re-check after
      any edit).
- [ ] **The owner's yes.**
