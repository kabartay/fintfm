# Tasks

## Phase 0 — decisions for the owner

- [ ] AU0.1 **LendingClub conflicts with the no-benchmark-data rule.** It is in BeyondArena
      (`lending_club_1m`); §155 knew and chose a bounded forfeit (exclude that task from any
      FinTFM-R BeyondArena number), and §157's fold-0 pilot trained on it. The standing rule is
      never to train on TabArena or BeyondArena datasets. Decide: retire it (mark
      `EXCLUDED_BENCHMARK` in `docs/research/data_candidates.csv`, make
      `prior/real_lendingclub.py` refuse to load, make `check_overlap.py` fail on any `IN_USE`
      row with a `benchmark_match`), or record the exception in DECISIONS. No shipped FinTFM
      checkpoint is affected.
      Verify: the decision is recorded in DECISIONS.md, and `data_candidates.csv` agrees with it.
- [ ] AU0.2 **Freddie Mac wording disagrees in three places**: `CLAUDE.md:374` (none until a
      commercial licence is signed), §153 (free tier permits internal use) and
      `docs/research/data_candidates.csv:3` ("noncommercial project", `IN_USE`). Owner to
      reconcile; `CLAUDE.md` is hand-edited.
      Verify: one wording, the same in all three places.
- [ ] AU0.3 **`CLAUDE.md` is public and names `temelion.ai` and a local `/Users/...` path**
      (`CLAUDE.md:14-16,199`). Owner's file. The AXIOM harness licence was removed from three
      public files in b78a52c; it remains in git history.
      Verify: `git grep -n temelion` and `git grep -n /Users/` print nothing outside history.

## Phase 1 — correctness (code)

- [x] AU1.1 **Done (2026-10-16).** `pad` now reads context rows only;
      `test_query_chunking_is_exact_with_a_context_empty_column`. Near-zero-std Z left as is
      (changes ordinary outputs; follow-up). **Padding mask reads query rows.**
      `modeling/model.py:233` computes
      `pad = isnan(X).all(dim=1)` over context and query rows, so one query with a value in a
      column that is all-NaN in the context un-pads that column for every row in the chunk, and
      chunked prediction stops being exact (breaks the docstrings at `classifier.py:197`,
      `quantile_regressor.py:124`). Fix: compute `pad` from `X[:, :n_ctx]`; set Z to 0 where
      the context std is ~0. Verify: a test that a query's output is identical alone and in a
      chunk with a mate that has a value in a context-empty column. Note: the fix also changes
      pretraining, so record whether it changes any shipped checkpoint's V4 scores.
- [ ] AU1.2 **Retrieval positive floor falls short.** `inference/retrieval.py:165` takes the
      `short` nearest positives overall, which include the ones already picked, so the result
      holds about max(have, short) positives, not `min_positive`, and the context can shrink.
      Fix: exclude `picked` before choosing `extra`. Verify: a test with 2 positives in the kNN
      set and floor 8 ends with ≥8 positives and exactly `k` rows.
- [ ] AU1.3 **Rank transform crashes on ±inf.** `inference/preprocess.py:152-159,185`: the
      default `feature_transform="rank"` passes infinities to `QuantileTransformer`, which
      raises; the `power` branch already clamps. Fix: reuse that clamp. Verify: fit and
      transform on a column with ±inf.
- [ ] AU1.4 **Label-swap ensembling scrambles multiclass.** `classifier.py:633` reverses the
      probability columns even when `_flip` returned `y` unchanged (≠2 classes), so each class
      is averaged with another; through the regressor it reverses bin order. Fix: swap only
      when binary. Verify: a 3-class test where label swap on equals label swap off.
- [ ] AU1.5 **Non-finite regression targets go to edge bins.** `inference/binning.py:141`
      (`np.digitize`) puts NaN and +inf in the top bin and -inf in bin 0; `regressor.py:112`
      then trains on them. Fix: raise on non-finite `y` in `FinancialTFMRegressor.fit`.
      Verify: a test that NaN or ±inf targets raise.
- [ ] AU1.6 **`fit` replaces `self.model` with the fine-tuned copy** (`classifier.py:450`). A
      second `fit`, or `sklearn.clone` of a fitted estimator, fine-tunes the already tuned
      weights again and can carry one fold's rows into another. No current caller does this.
      Fix: keep `self.model` as passed; store the tuned copy as `model_`. Verify: fitting twice
      gives the same weights as fitting once.
- [ ] AU1.7 **Fine-tuning edge cases.** (a) `n_ctx` ignores the pool size (`classifier.py:461`),
      so small datasets train on duplicated rows at a context size inference never uses: clamp
      to the pool. (b) One group, or two groups with query fraction ≥0.75, empties the context
      pool and raises: clamp `n_q_groups` to `[1, len(uniq) - 1]`, clear error under 2 groups.
      (c) `pos_frac` at or near 1 divides by zero, above 1 gives a negative size: validate
      `0 < pos_frac < 1`. (d) `groups` length is not checked against `y`. (e) `finetune_context=0`
      silently means `max_context`. Verify: one test per case.
- [ ] AU1.8 **Fine-tuning resets the caller's GPU RNG.** `classifier.py:467-468`:
      `fork_rng(devices=[])` restores CPU state only, while `manual_seed` reseeds CUDA/MPS.
      Scores are unaffected (column identities are drawn on CPU, dropout 0). Fix: fork the
      active device too, or use a local `torch.Generator`.
      Verify: a test on CUDA or MPS that the caller's device RNG state is unchanged by `fit`.
- [ ] AU1.9 **Fine-tuning context ignores the inference strategy.** `classifier.py:438-440`
      always draws a uniform context; with `balanced`/`hybrid`/`retrieval`/`prototype` or
      `ensemble_label_swap`, inference sees a distribution the tuned weights never saw. Fix:
      warn or raise unless uniform and no swap, or draw with the same strategy.
      Verify: a test that a non-uniform strategy with fine-tuning on warns or raises.
- [ ] AU1.10 **Regressor `clone` drops every classifier setting.** `regressor.py:74-83` takes
      `**classifier_kwargs`, which `get_params` ignores, so `clone`/`GridSearchCV` silently
      revert to defaults. Fix: one explicit `classifier_params: dict`. Verify: `clone(r)`
      round-trips the settings.
- [ ] AU1.11 **Smaller estimator issues.** `__init__` loads and moves the model (so
      `set_params(device=...)` leaves it on the old device); no `n_features_in_`; predict
      before fit raises `AttributeError`, not `NotFittedError`; no gradient clipping in
      fine-tuning (pretraining clips at 1.0); MPS cache not cleared after fine-tuning; quantile
      regressor can return crossing quantiles (`np.sort` the grid) and goes all-NaN on a NaN
      target; `_pooled_prior_shift` assumes equal shares for multiclass retrieval
      (`classifier.py:514`).
      Verify: one test per item, each failing before its fix.
- [ ] AU1.12 **Tests that cannot fail.** `tests/test_prior.py:16` (`... or True`),
      `tests/test_model.py:296` (`isnan or isfinite`), `tests/test_prior_score.py:61`
      (`n_scored >= 0`). The wall-clock bound at `tests/test_finetune.py:97` is flaky on a
      loaded machine: assert on operation count or loosen it. `inference/preprocess.py` has no
      direct tests. Re-check `.pytest_cache/lastfailed` (model permutation and padding
      invariance, two task-family cases) when the machine is free.
      Verify: each rewritten assertion fails when its condition is broken by hand.

- [ ] AU1.13 **Graph reuse distorts the prior mixture.** `prior/mixture.py:387-395` draws the
      source to decide whether a slot is grouped, then `sample_task` draws it again, and
      `p_trivial`, `p_crossed`, `p_task_family` are left out of the check, while each SCM slot
      fills `scm_reuse_graph` tasks. Agent measurement at p_fin 0.7, p_tree 0.2, p_fam 0.2:
      SCM share 0.198 at reuse 1 against 0.558 at reuse 4. Any comparison across reuse values
      (task 48.12, §113) mixed two changes. Fix: draw the source once and dispatch. Verify: a
      test that source shares match within binomial error at reuse 1 and 4; re-read §113.
- [ ] AU1.14 **Some prior tasks get labels unrelated to X.** A single-class draw is repaired by
      flipping 5% of rows at random. `prior/real_mortgage.py:141-144` has no minimum-positives
      step, so at a 0.048% rate most 256-row mortgage tasks get random labels;
      `prior/real_edgar.py:134-142` likewise at ~3% of 256-row tasks. Fix: resample instead of
      flipping; minimum-positives enrichment in every real loader.
      Verify: a test that no real-loader task has labels flipped at random; positive count ≥ the floor.
- [ ] AU1.15 **EDGAR context/query leakage.** 73% of EDGAR rows share a `cik` with another row;
      row-wise sampling (`prior/real_edgar.py:131`) puts the same firm in context and query,
      which teaches lookup. Fix: split by firm, as `_finetune_rows` does.
      Verify: a test that no `cik` appears in both context and query of an EDGAR task.
- [ ] AU1.16 **Rare classes appear only in query rows.** Random cut-points (`prior/scm.py:248-250`,
      `prior/tree.py:143-145`) leave a class with ≤2 rows in ~31% of 256-row tasks; the loss is
      then taken on labels absent from the context. Fix: a minimum count per class in context.
      Verify: sample 1,000 256-row tasks; every class present in the query is present in the context.
- [ ] AU1.17 **Learnability filter rejects every regression task.** `is_learnable` reads the
      placeholder `y=zeros` (`prior/scm.py:340`, `prior/learnability.py:47-49`). Fix: skip
      regression or judge on `y_continuous`.
      Verify: with `p_learnability_filter > 0`, regression tasks pass at their true rate in a test.
- [ ] AU1.18 **Evaluation shares the training RNG** (`modeling/train.py:518`): changing
      `eval_every` changes the training tasks. Fix: a separate generator for evaluation.
      Verify: two runs differing only in `eval_every` draw identical training batches.
- [ ] AU1.19 **TabArena adapter fills NaN with 0** (`tabarena/.../models/fintfm/model.py:147`,
      the local TabArena fork). The model has its own missingness mask; filling erases it and
      collides with real zeros. Its docstring (`:43-47`) says the checkpoint was trained "plus a
      generic random-graph SCM"; verify against the checkpoint's recorded prior config and
      correct. Fix: pass NaN through. Verify: TabArena-Lite A/B on the binary datasets.

## Phase 2 — evaluation rigour

- [ ] AU2.1 **Bootstrap resamples rows, not companies.** `evaluation/metrics.py:238`; test
      folds hold ~5.3 rows per company, so CIs and p-values are somewhat narrow. Fix: save
      `company` in `predictions_fold*.npz`, add a cluster bootstrap, report both for §159-§164.
      Verify: the cluster and row bootstraps reported side by side for §163; a test on synthetic clustered data shows the cluster CI is wider.
- [ ] AU2.2 **Tuned baselines are selected on validation ROC-AUC** (`v4_protocol.py:383,440`)
      while AP is the headline. Fix: select on validation AP, record both.
      Verify: the run JSON records the selection metric, and a test selects on AP.
- [ ] AU2.3 **Relabel the booster numbers.** CatBoost's 0.4005 is its library default of 1,000
      iterations; the `--tune` grid caps at 200 (`v4_protocol.py:101`), so the "untuned
      boosters understate the field" banner (`v4_protocol.py:575`) is likely wrong for
      CatBoost. LightGBM and XGBoost ran 100 trees. Fix the banner and every place quoting
      0.4005 as "untuned CatBoost" without the iteration count.
      Verify: `grep -rn 0.4005 docs README.md` shows the iteration count beside every hit.
- [ ] AU2.4 **Pilot arms are ranked on fold-0 test** (`scripts/candidates/pilot_compare.py:52-63`).
      Validation predictions (`y_val`, `val_<arm>`) are already saved: switch the pilot to them,
      and note in §157/§162 that fold 0 selected the arm.
      Verify: `pilot_compare.py --split val` exists and is the default.
- [ ] AU2.5 **`operational_status` looks like a collection-time snapshot.** Constant within
      99.9% of companies (209 of 188,065 vary), so for earlier years it may carry future
      information; not in `DROP_COLUMNS` (`v4_protocol.py:67`). Check the upstream V4FinBench
      drop list, then run every arm without it as an ablation. Also note that `Insolvency_flag`
      covers 3,586 of 3,587 positives.
      Verify: the ablation is written up in FINDINGS with per-fold AP for every arm.
- [ ] AU2.6 **Pairing checks compare labels only.** `scripts/compare_v4_scores.py:58`,
      `scripts/c5_verdict.py:56` check `y_true` equality; also compare the `v4_protocol.json`
      configs minus `model`. Floor bootstrap p-values at 1/(drawn+1) (`metrics.py:249`) and fix
      the "redraw" comment at `:239`.
      Verify: a test that a config mismatch makes both scripts exit non-zero.
- [ ] AU2.7 **Re-derive §159-§163 from the raw runs** and correct any mismatch (in progress
      2026-10-15; results appended here when the check reports).
      Verify: every §159-§163 figure is listed beside its recomputed value.

- [ ] AU2.8 **Cell attention's credited gain may be confounded with the prior mix.** §80 says
      the only difference between arms is `n_cell_blocks`/`cell_labels`, but the doc review
      found the cell arm is `v4-cellattn-fin10` (`p_financial` 1.0, batch 4, `n_rows` 256/512;
      its values equal §97 row A) and the old arm `v4-colid-fin07` (`p_financial` 0.7; §83).
      §91's +0.0337 is likewise fin10-cellattn minus fin07-colid. If confirmed from the two
      checkpoints' recorded configs, the architecture-only effect is unmeasured, and the
      "+0.042 AP from cell attention" in README.md needs the qualifier. Fix: confirm from the
      checkpoints, then score a matched pair (same prior, same budget) or restate the claim.
      Verify: the two checkpoints' recorded prior configs quoted in FINDINGS, then a matched pair or a restated claim.
- [ ] AU2.9 **§159's 0.0073 bar is not seed noise.** §160 showed it is mostly the max_classes
      10-against-2 head effect; seed noise at matched head size is 0.0009 mean AP. §163 kept the
      bar and cleared it by 0.0003, with a fold-to-fold SD of Δ of 0.0054. The bar errs
      conservative, so no verdict flips, but its size is not a noise estimate. Also: fine-tuning's
      own variance (seed and context draw) is outside the bootstrap CIs; §163's Stage A lr choice
      was a 4×10⁻⁶ AP tie and used fold 0's validation rows, which are fold 4's test rows; §163's
      "helps" rule is applied by hand, not in `scripts/c5_verdict.py`; and §162-§164 judge many
      arms against one bar with no family-wise correction. Re-derived numbers all match
      (2026-10-15). Fix: re-derive a yardstick from matched-head replicates (S1 gives one), add
      a `helps` mode to `c5_verdict.py`, and state the family of comparisons in §164's write-up.
      Verify: `c5_verdict.py helps` reproduces §163's verdict; FINDINGS states the new yardstick
      with its source runs.

## Phase 3 — documents

- [ ] AU3.1 **§162's confirmation result sits under §163.** `docs/results/FINDINGS.md`
      ~11000-11029 belongs to §162; update the stale status lines at ~10890 (§162 "no result
      exists"), ~10968 and ~11029 ("§163, running"). §158 appears before §157.
      Verify: §162 holds its own result, and no status line says pending or running for a finished run.
- [ ] AU3.2 **CLAIMS.md and LIMITATIONS.md stop at §121 and §114.** Add §157-§164: fine-tuning,
      the full-panel booster gap with its iteration counts, logistic regression beating FinTFM
      only at the 200k cap, the FinTFM-R nulls, and that a fine-tuned model no longer "never
      trains on your data". Resolve `LIMITATIONS.md:21-23` against `CLAIMS.md:290` (§80) and the
      stale `LIMITATIONS.md:25`.
      Verify: every § from 157 to 164 is cited in CLAIMS.md or LIMITATIONS.md.
- [ ] AU3.3 **"No gradient steps at fit time" is now true only by default.** `README.md:23-25`,
      `README.md:211`, the `pyproject.toml` description, `src/fintfm/__init__.py` docstring.
      Verify: `grep -rn "gradient steps" README.md pyproject.toml src/fintfm/__init__.py` shows the qualifier at each hit.
- [ ] AU3.4 **Counts and versions.** README test counts disagree (`:60` 407, `:205` "246 tests;
      1 skip"); derive the count and state the sync line it assumes. README BibTeX
      `version = {0.5.5}` (`:484`). No `__version__` in `src/fintfm/__init__.py`. CHANGELOG
      [Unreleased] is empty despite 76551e2 and 5273275. Check CHANGELOG LightGBM 0.2666 vs the
      per-fold mean.
      Verify: README counts equal `pytest --collect-only`; `tests/test_version_agreement.py` covers BibTeX and `__version__`.
- [ ] AU3.5 **openspec lag.** `adopt-published-methods/tasks.md:28` (36.5) needs a DECISIONS
      entry for fine-tuning; `fintfm-r/tasks.md:160` still says "EDGAR scoring"; §164 has no task.
      Verify: the three items closed or updated with their evidence.
- [ ] AU3.6 **Kaggle checkpoint URL is not pinned.** `scripts/kaggle/task_164_*.py:15` use
      `resolve/main`; pin the revision as the README does (the SHA-256 check already guards it).
      Verify: `grep -n resolve/main scripts/kaggle` prints nothing.
- [ ] AU3.7 **Licence checker fails locally** on `text-unidecode` (Artistic/GPL, via `kaggle` →
      `python-slugify`); CI does not see it because it syncs only `--extra bench`. Decide:
      drop the `kaggle` extra from the project, or record why a dev-only tool is acceptable.
      Verify: `uv run python scripts/check_licences.py` passes locally, or the exception is documented.
- [ ] AU3.8 **Documentation review findings** from the seven doc agents (appended below as they
      report).
      Verify: each doc agent's list applied or explicitly left alone, and `check_doc_links.py` passes.
- [ ] AU3.9 **Check the 2026-10-15 peer sweep against primary sources** (RELATED_WORK.md, last
      section) before any figure in it is quoted, then update the earlier entries it
      contradicts (TabPFN-3.5 rank and Elo, Mitra, Kumo, LimiX-2, best real-data model, the
      TabSwift "data points against 2D") and §114's scope (48k tasks, not scale in general).
      Verify: every figure in that section carries a primary-source citation, and the earlier entries agree with it.

## Phase 4 — cheap diagnostics of the CatBoost gap (hours, CPU or one Kaggle session)

- [ ] AU4.1 **Is the gap boosting rounds?** LightGBM at 1,000 trees and lr 0.05, CatBoost at
      100 iterations, same folds. If LightGBM approaches 0.40, the gap is rounds over all
      ~2,150 training positives, not categoricals (CatBoost gets no `cat_features` here; check
      `get_all_params()` for its boosting type).
      Verify: per-fold AP for both arms in FINDINGS, paired against the existing booster runs.
- [ ] AU4.2 **Context composition, inference only.** Uniform beat balanced/hybrid in §29/D9,
      but a minimum-positive floor on a uniform context (~7 positives at 2,000 rows) and
      k-means prototype undersampling (the V4FinBench authors' 0.3 ratio) were never scored on
      the full panel. Comparator: zero-shot uniform.
      Verify: five-fold AP against zero-shot uniform, paired bootstrap, §159's rule.
- [ ] AU4.3 **Context length beyond training.** Pretraining contexts reach at most ~921 rows
      (`prior/mixture.py:446` with the shipped `n_rows` choices; verify from the checkpoint
      config), inference uses 2,000. Score `max_context` 900 with `n_ensemble` 3 against 2,000.
      Verify: the checkpoint's recorded `n_rows` choices quoted, and five-fold AP for both arms.
- [ ] AU4.4 **Hybrid ceiling.** Out-of-fold FinTFM logits (inner company-grouped K-fold) as a
      CatBoost `baseline` and as a feature; compare against CatBoost alone. Measures a system,
      not FinTFM, and is reported as such.
      Verify: five-fold AP for CatBoost alone and CatBoost plus FinTFM, with no outer-test rows in the inner folds (asserted).

## Phase 5 — strength levers (GPU, ordered by expected gain × feasibility)

- [ ] AU5.1 **Fine-tuning recipe at published scale.** Our 200 steps at 1e-5 is the bottom of
      the published range (Rubachev et al. 2025, arXiv 2506.08982: full fine-tune, lr sweep
      5e-6..5e-4, 1,024 queries per step, early stopping on validation every 10 steps, patience
      16). Arms: lr {1e-5, 3e-5, 1e-4}, up to 3,000 steps, early stopping on inner-validation
      companies. Depends on §164.
      Verify: five-fold AP against §163's arm under §159's rule, inner-validation selection only.
- [ ] AU5.2 **Pretraining throughput.** bf16 autocast (training is fp32, `modeling/train.py:471-484`),
      and an attention path that allows flash attention (the bool row mask blocks it). Measure
      tasks per GPU-hour before and after; no accuracy claim.
      Verify: tasks per GPU-hour measured before and after on the same kernel; outputs equal within fp tolerance on a fixed batch.
- [ ] AU5.3 **Capacity where the bottleneck is.** The shipped model has ~886k parameters (agent
      measurement; verify) and pools up to 136 cell tokens into one 128-d row vector after one
      cell block (`model.py:537-553`). Peers keep per-cell tokens (TabPFN v2) or several summary
      tokens per row (TabICL); verify their sizes from primary sources first. Arms: more cell
      blocks; 4 summary tokens per row; expose `n_heads`.
      Verify: peer sizes quoted from primary sources; each arm scored at matched tasks under §159's rule.
- [ ] AU5.4 **Train on the context lengths inference uses** (up to 4,096 rows; query-to-context
      attention keeps cost linear in queries).
      Verify: the trained `n_rows` range covers inference's `max_context`; five-fold AP against the current checkpoint.
- [ ] AU5.5 **Class-index permutation in pretraining** (`collate`): SCM classes are ordinal and
      class 1 is always the rare default in the financial prior, so the model can learn a
      class-index prior.
      Verify: a test that class indices are permuted per task; label-symmetry probe before and after.
- [ ] AU5.6 **Match pretraining and inference preprocessing.** Pretraining z-scores; inference
      defaults to rank-to-normal. Draw a random transform per pretraining task and ensemble over
      transforms at inference.
      Verify: a test that pretraining draws each transform; five-fold AP against the current checkpoint.
- [ ] AU5.7 **Stronger default ensembling.** `n_ensemble` is 1 in the library and V4 configs;
      add feature-shuffle and transform members, and feature-subset ensembling to lift the
      `max_features` hard limit (`classifier.py:373`).
      Verify: TabArena-Lite and V4FinBench scores against `n_ensemble` 1 and the current 8.
- [ ] AU5.8 **Large-n methods from the literature**, each its own arm: TabPFN-kNN/LoCalPFN-style
      retrieval with per-class neighbours (arXiv 2406.05207; code has no licence file, so
      reimplement); a tree partition or forest of PFNs (tabpfn-extensions `rf_pfn`,
      Apache-2.0); BoostPFN (arXiv 2503.01256, MIT).
      Verify: each arm's five-fold AP against §163's arm under §159's rule.
- [ ] AU5.9 **Hazard context carries only the binary label** (`model.py:751-759,787-797`): embed
      the default period and a censored token for context rows.
      Verify: a hazard test where the term structure is learnable only from context periods.
- [ ] AU5.10 **Optimiser nits.** Weight decay also hits LayerNorm, biases and embeddings
      (`train.py:407`); no final LayerNorm before pooling in two encoders; warmup fixed at 500
      steps with no flag.
      Verify: a test that LayerNorm, biases and embeddings are excluded from weight decay.
- [ ] AU5.11 **Prior diversity gaps** (agent sampling, verify each): the binary checkpoint was
      trained at `p_financial 1.0` (§91/§104), where tasks are binary, ~1% categorical and have
      a median logistic-regression AUC of 0.57; categoricals are always ordered 2-7 level codes
      (`prior/scm.py:261-264`); missingness is only MCAR and never label-informative; SCM and
      tree marginals are near-Gaussian; the task-family generator always emits exactly 136
      Gaussian columns (`mixture.py:229-236`); tree splits pick columns by raw std
      (`prior/tree.py:79-82,136`). Arm: a TabArena checkpoint on SCM + tree + families with
      `p_financial ≤ 0.3`, more tasks, after AU1.13.
      Verify: the prior sampler's statistics (class balance, categoricals, missingness, kurtosis) reported before and after.
- [ ] AU5.12 **Per-member inference diversity** (peer recipes read from installed TabPFN 9.0.0,
      TabICL 2.2.0, TabDPT 1.3.0, LimiX code; all reimplemented, no code copied): first a
      label-symmetry probe on the current checkpoint (§45 found −0.62 on an older one), then
      class permutation as a member axis, alternating preprocessing per member (rank vs robust
      scaling with soft clip or power, SVD components in one arm), outlier clipping, and
      balanced feature subsets to cover tables over 136 features (unlocks the 5 excluded
      TabArena datasets). Local TabArena-Lite T+E gains for peers whose defaults lack this:
      TabPFNv2 +92 Elo, TabDPT +129; RealTabPFN-2.5, which has it by default, −3.
      Verify: the label-symmetry probe result recorded first; each member axis scored on TabArena-Lite.
- [ ] AU5.13 **Fill the TabArena search space** (`tabarena/.../models/fintfm/hpo.py` is empty)
      once AU5.12 exists; compute-heavy (11 s per 1k rows locally), so GPU only.
      Verify: a TabArena T+E run recorded as its own row, not as the default.
- [ ] AU5.14 **TabArena context size**: TabArena-Lite at `max_context` 500 and 750 against 1000
      (trained range ends ~921 rows).
      Verify: TabArena-Lite Elo at 500, 750 and 1,000 recorded.
- [ ] AU5.15 **Pretraining volume first: 48k tasks → at least 2M.** The strongest evidence in
      the 2026-10-15 sweep (LightPFN's 256k → 7.68M draws, +1.66 AUC points; LimiX-2 and Kumo
      scale curves at 10^7+ tasks). Prerequisites: AU1.13, AU5.2, pre-generated task dumps,
      `--checkpoint-every`. Cost to be measured on one Kaggle session before committing. After
      it, re-test the closed architecture verdicts, all measured at 48k tasks.
      Verify: tasks per GPU-hour and cost measured on one Kaggle session before the full run; then §159's rule on V4FinBench and TabArena-Lite.
- [ ] AU5.16 **Value embedding**: Fourier or RBF plus ECDF-rank features in place of the MLP on
      (z, missing) (LimiX-2M RaBEL, TabPFN-3.5). One module, behind a flag.
      Verify: a flag, a test that it is off by default, and five-fold AP at matched tasks.
- [ ] AU5.17 **Optimiser and attention scaling from TabICLv2**: Muon, QASSMax (length-aware
      softmax scaling), early target inclusion; also the fix for AU4.3 if context length is the
      problem.
      Verify: each change behind a flag, scored at matched tasks.
- [ ] AU5.18 **Score the TabArena imbalanced subset** (v0.1.9), the credit-relevant slice, with
      the current checkpoint as a baseline.
      Verify: the imbalanced-subset Elo recorded in TABARENA.md with its date.
