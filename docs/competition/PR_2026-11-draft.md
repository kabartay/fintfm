DRAFT, not posted. Do not open a TabArena pull request from this text until every box in "Before posting" is ticked.

# Draft text: TabArena resubmission of FinTFM (follows autogluon/tabarena#616)

Rewritten 2026-11-19 after reading recent model PRs upstream (#700, #716, #729/#730, #735). Every
figure is from `docs/results/FINDINGS.md` (section in brackets) and is TabArena-Lite, repeat 0,
fold 0, run from the tabarena fork; none of it is the public leaderboard or a full TabArena run. The
PR branch is `fintfm-pr-2026-11` at fork commit 6f353cf (upstream `main` merged in at eee00bb).

**Standing (§303).** Reproduced from the PR branch with no `FINTFM_*` variable, 51 of 51 datasets:
**1,019.6 Elo (+72 / -75), 99th of 110**, 19.6 above RandomForest (default) and the owner's
resubmission bar (DR7.6), inside the interval. The stored runs it reproduces score 1,024.0 (§301).

**Owner decisions, 2026-11-19.** Post with the internal-CV deviation declared (no wrapper change);
default config only (`can_hpo=False`); report taiwanese_bankruptcy_prediction with both machines'
values; keep `fintfm==0.12.0` (no package change since; not 1.0.0); the owner signs off as `kabartay`
and answers the review himself.

## What #616 said, and where this draft answers it

The pull request autogluon/tabarena#616 (closed 2026-09-25, withdrawn before the maintainers' full
run; record in [`PR_2026-09.md`](PR_2026-09.md)) received three concerns and one positive finding.

| # | Reviewer concern in #616 | State now |
| --- | --- | --- |
| 1 | Binary only; excluding problem types where the model does worse is cherry-picking. | Answered: binary, multiclass and regression, every Lite dataset, no exclusion. |
| 2 | Datasets missing; "failures that should be fixed"; new submissions should "cover all datasets". | Answered: 51 of 51 reproduced from the branch, no failed split, none imputed (§297, §303). |
| 3 | Does not beat a tuned KNN or other simple baselines. | Partly: above KNN (tuned + ensemble) 965.5, TABFLEX_GPU 978.0 and RandomForest (default) 1,000.0 on the same board, but inside the interval for RF, and far below tuned tree ensembles (GBM (tuned + ensemble) 1,380.1) (§303). |
| 4 | Positive: "I think everything is correct." | The wrapper has changed a lot since; the body lists what needs a fresh read. |

## Pull request body

Title: **Add FinTFM-LinProbe: FinTFM with a trained linear readout (binary, multiclass, regression)**

---

## Summary

Adds FinTFM-LinProbe, a new model entry for FinTFM, a tabular foundation model pretrained on a
synthetic prior only. It is not zero-shot: a linear readout (logistic regression for
classification, ridge or random-Fourier-feature ridge for regression) is fitted on the network's
representation of the training rows, one per partition leaf, and also reads the row's own features.
It covers all 51 TabArena-Lite datasets with one public checkpoint per problem type and the default
config only. Reproduced from this branch on TabArena-Lite it scores **1,019.6 Elo, 99th of 110**.
This follows #616 (zero-shot, binary only, declined on competitiveness and coverage).

**Benchmark exposure.** The checkpoints were trained on synthetic tables only. The readout's
settings were not: every lever of the default config was chosen on the TabArena-Lite fold-0 test
splits, and several size thresholds (5,000 and 10,000 training rows) were drawn after seeing those
results. The Lite score is therefore development-exposed, each step after the first was inside the
Elo interval, and the full run may land lower. Please take this into account when
evaluating or listing the entry. [Measurement log](https://github.com/kabartay/fintfm/blob/main/docs/results/FINDINGS.md)
(section numbers below refer to it).

**One declared deviation.** `X_val` / `y_val` are not used, but the readout selects its
regularisation, its binary cross-fit and its multiclass probability floor by internal
cross-validation on the training rows. Details below. We submit the method as measured rather than
change it; if you prefer selection on `X_val`, that would be a new measurement.

<details>
<summary>Details</summary>

**Model and pins**

- `fintfm==0.12.0` (PyPI, Apache-2.0, [DOI 10.5281/zenodo.24325794](https://doi.org/10.5281/zenodo.24325794)),
  pinned in `info.py` and the `fintfm_linprobe` extra (listed in `extended`).
- Checkpoints, public and ungated, Apache-2.0, pinned by revision and SHA-256 in
  `models/fintfm/checkpoints.py`:
  binary `kabartay/fintfm-binary@f116bfd43a2b15c65ed3551ea8c38e3364629ddc`,
  multiclass `kabartay/fintfm-multiclass@b292efa80fbebf759802a980bafe9b5bdcc626bd`,
  regression `kabartay/fintfm-regression@787a3446685cdcd76eaf079637c50bc4c6fb5a04`.
- Every setting is a per-problem-type default in `model.py` (`_entry_*`); no environment variable is
  needed. `FINTFM_*` variables remain as optional overrides and were removed for the reproduction.

**What the default config does** (each a configuration choice on Lite fold 0)

- Partitioned contexts: a shallow tree routes rows to leaves, each leaf has its own context and
  readout, in bags (4 for binary fits of at most 5,000 rows, 8 for multiclass, 4 for regression).
- Readout inputs: the representation plus the row's squashed features; spline features for binary
  and multiclass fits of at most 10,000 rows and every regression fit; one-hot indicators for
  multiclass categorical columns of at most 64 levels; column selection for tables wider than the
  checkpoint.
- Regression: a random-Fourier-feature ridge for fits of at most 5,000 rows, a ridge otherwise, and
  predictions clipped to the training target's range widened by 10%. The clip removes a failure,
  not a tuning gain: on diamonds a few off-manifold rows were predicted near -19 million, and the clip
  takes its test RMSE from 249,744 to 957, with no table worse by more than 0.1% (§295, §301).

**Declared deviation: internal cross-validation on training rows**

1. Binary and multiclass: `LogisticRegressionCV(Cs=np.logspace(-3, 2, 6), cv=3,
   scoring="neg_log_loss")` picks `C` (`_new_classifier_probe`).
2. Binary: the readout is cross-fitted with 5 folds, so each readout row is represented by a context
   that leaves it out (`_entry_crossfit_folds`); categorical encoding is out-of-fold to match.
3. Multiclass: a probability floor is chosen on out-of-fold training predictions (`_entry_prob_floor`).
4. Regression with the random-Fourier-feature readout: `RFFRidgeCV` selects its ridge alpha.
5. Regression: the clip bounds are the training target's range.

None of these reads the validation split; `X_val` / `y_val` and `time_limit` are accepted and unused,
as the class docstring states.

**Wrapper decisions**

- Device: a granted GPU selects CUDA, otherwise CPU; MPS is never chosen automatically
  (`_choose_device`; `FINTFM_DEVICE` forces one). `info.py` declares `compute="gpu"`: the nine binary
  tables above 10,000 training rows were fitted on a T4, the rest on CPU.
- Threads: the readout's BLAS calls run at `min(4, num_cpus)` threads (`blas_threads="auto"`,
  `threadpoolctl`). Without it, OpenBLAS on an 8-vCPU host oversubscribed and three large tables
  hit the time limit; with it they finish with identical scores (§296-§299).
- Classes: the multiclass checkpoint represents at most 10 classes, declared as `ag.max_classes = 10`.
- Seed: `seed_name = "random_state"` seeds the row draws, bagging, cross-fit and floor folds.
- `time_limit` does not bound the fit (no training loop); AutoGluon's per-fold projection still
  applies. On 4-CPU machines QSAR-TID-11 sits near the limit (§301).
- Checkpoints are referenced by pinned dict, not loaded objects; optional imports stay inside the
  wrapper; nothing is patched globally.

**Results** (reproduction from this branch, §297, §303)

| | Elo | position |
| --- | --- | --- |
| all 51 datasets | 1,019.6 (+72 / -75) | 99th of 110 |
| binary, 30 datasets (binary-only board) | 1,083.4 (+109 / -100) | 92nd of 108 |
| multiclass, 8 datasets (multiclass-only board) | 947.1 (+100 / -253) | 98th of 108 |
| regression, 13 datasets (regression-only board) | 852.6 (+119 / -221) | 97th of 103 |

Against the stored development runs, 49 of 51 datasets are within 0.3%. anneal is -0.56%.
taiwanese_bankruptcy_prediction is +22.2% worse on Kaggle CPU and within 0.01% on the Mac where it
was first fitted, so that difference is the machine; the table keeps the Kaggle CPU value.

**Controls.** A matched control replaces the FinTFM representation with the squashed raw features
alone (a tree with a linear model per leaf). Against earlier versions of the entry, the
representation is better on 12 of 13 regression datasets (sign test p = 0.003) and 7 of 8
multiclass datasets (p = 0.07), but not on binary Elo: the control scored 1,009 against the entry's
997 there (§239, §240, §243). These are general-tabular results; on real credit panels FinTFM
currently loses to gradient boosting (§216).

**Hardware and entry point.** One dataset per process with the registry entry's default config
through `TabArenaV0pt1ExperimentBundle(models=[("FinTFM-LinProbe", 0)])` and
`TabArenaContext().build_and_run_jobs(..., subset="lite")`, the default validation protocol, AutoGluon
1.6.4b20261109:

- Kaggle CPU sessions (4 CPUs, 31 GB): 21 binary tables of at most 10,000 rows, 7 multiclass, 6
  regression.
- Hugging Face Jobs `t4-medium` (8 vCPU, 30 GB, Tesla T4, torch 2.15.0+cu132): the 9 larger binary
  tables. Five ran with `OPENBLAS_NUM_THREADS=4` set in the job, before the adapter had its own limit;
  three of them were rerun with the adapter's limit and no variable, and matched exactly (§299).
- A Mac (Apple M3 Max, 64 GB, CPU only): SDSS17 and 7 regression tables.
- Check of GPU dependence: credit_card_clients_default on a RunPod A40 scored within 2e-6 (relative)
  of the HF T4 (§300).

Fit plus inference over the 51 tables: 18.8 hours (Kaggle CPU 11.4 h of training, longest
hiva_agnostic 2,829 s; T4 3.4 h, longest kddcup09_appetency 3,353 s; Mac 3.4 h, longest diamonds
2,591 s). The `results.pkl` files are available on request.

</details>

## Tests

Run on this branch (outputs kept):

- `ruff check` and `ruff format --check` (ruff 0.14.0) on the touched files: clean.
- The model's own tests (`models/fintfm/tests/`, `models/fintfm_linprobe/tests/`): 414 passed,
  1 skipped.
- `pytest -m models -k FinTFM`: passes. Without CUDA the FinTFM-LinProbe smoke fit is skipped
  (`compute="gpu"`); with `compute="cpu"` it passes on binary, multiclass and regression.
  `test_model_many_class` skips it on its declared 10-class limit.
- `python -P -m tabarena.tools.audit_pickle --model FinTFM-LinProbe --problem-type <type>`:
  "nothing that scales with the training rows sits in model.pkl" for all three types. The unpatched
  tool crashes on this model: scipy's `BSpline` (in the spline readout) keeps the array-namespace
  module in `_xp`, `tensor_nbytes` walks into it and through `sys.modules`, and stops on a `urllib3`
  mapping that refuses iteration. We ran it with module objects skipped in the walk
  (`isinstance(value, types.ModuleType)`); the same guard would fix the tool for any model holding
  scipy splines.
- `python -P -m tabarena.tools.audit_transduction --model FinTFM-LinProbe --problem-type <type>`:
  "every probe row gets the same prediction in every setting" for binary, multiclass and regression
  (largest difference 8e-7).
- `python -P -m tabarena.tools.audit_warmup --model FinTFM-LinProbe --problem-type <binary|regression>`:
  warm-up status ok, no failed step, no new packages in the timed fit or predict. The first binary
  fit loads 20 submodules of `narwhals`, which the warm-up has already imported.

## Model or system submission

Kind: model. A new entry; it does not replace a leaderboard entrant (#616 was withdrawn before any
run).

**Files**

- [x] Model: `models/fintfm_linprobe/{__init__,model,hpo,info}.py` (plus `rff.py`,
  `target_transform.py`), a lazy entry in `models/__init__.py`, the `fintfm_linprobe` extra in
  `packages/tabarena/pyproject.toml` (also in `extended`), the family in `website/website_format.py`,
  a smoke override for all three problem types, and model tests in `models/fintfm_linprobe/tests/`
  and `models/fintfm/tests/`.
- [ ] Logo: none attached.

**Implementation**

- [ ] `_fit` uses the `X_val` / `y_val` TabArena passes, no internal cross-validation, and about 200
  configurations. **Not met:** see "One declared deviation". No search space is submitted
  (`can_hpo=False`); `hpo.py` keeps an unmeasured proposal of at most 108 combinations.
- [x] `num_gpus`, `num_cpus` (BLAS threads) and the seed wired through; `time_limit` does not apply
  (no training loop); imports inside the wrapper; nothing patched globally; the model pickles.
- [x] Exact pin `fintfm==0.12.0` matching `pip_extra`; checkpoints pinned by full revision and
  SHA-256; licence (Apache-2.0, `commercial_use=True`), links and problem types stated.
- [ ] Hosted API: not applicable.
- [x] `ruff` clean and `pytest -m models -k FinTFM` passes.

**Results**

- [x] TabArena-Lite with the official pipeline and default validation protocol, default config only.
- [x] Hardware and entry point stated (Details).
- [ ] Optional: `results.pkl` folder (available on request).
- [x] Sign-off on the maintainer re-run: @kabartay.

| position | method | Elo |
| --- | --- | --- |
| 97 | NN_TORCH (default) | 1,037.1 |
| 98 | XRFM_GPU (default) | 1,031.7 |
| **99** | **[New] TA-FINTFM-LINPROBE (default)** | **1,019.6** |
| 100 | XT (default) | 1,008.8 |
| 101 | RF (default) | 1,000.0 |
| 102 | TABFLEX_GPU (default) | 978.0 |

By submitting this pull request, I confirm that you can use, modify, copy, and redistribute this
contribution, under the terms of your choice.

---

## Before posting (all must be true)

Done:

- [x] Weights public, Apache-2.0, ungated, pinned by full revision and SHA-256.
- [x] `fintfm` 0.12.0 on PyPI and pinned on the PR branch; no package change since (2026-11-19).
- [x] Every lever a per-type default; no environment variable needed.
- [x] Reproduction: 51 of 51, 1,019.6 Elo, 99th of 110, above the 1,000 bar (DR7.6), inside the
      interval (§303).
- [x] Owner decisions recorded above (2026-11-19).
- [x] Registration files, smoke override, tests moved next to the models (fork 81e94d1, 8b3ea4b).
- [x] Upstream `main` merged into the PR branch without conflict (eee00bb); class docstring states
      the unused inputs (6f353cf), as the maintainers asked of #700.
- [x] Tests and `audit_pickle` on all three problem types (`runs/pr_checks_9c7db11/`); after the
      upstream merge, tests again plus `audit_transduction` on all three and `audit_warmup`
      (`runs/pr_checks_eee00bb/`).
- [x] Hardware, entry point and cost filled in.
- [x] Support status recorded in [`SUBMISSION.md`](SUBMISSION.md).

Remaining:

- [ ] Numbers re-read from the board current at posting (110 methods on 2026-11-19).
- [ ] No confidential or peer-product names in the posted text (checked 2026-11-19; re-check after
      any edit).
- [ ] When opening the PR: from `kabartay/tabarena:fintfm-pr-2026-11`, with **"Allow edits by
      maintainers" ticked** (#729 had to be redone as #730 because it was not), and the body above
      the "Before posting" line only.
- [ ] **The owner's yes.**
