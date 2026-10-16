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
- [x] AU1.2 **Done (2026-10-16).** Top-up draws only unpicked positives and drops the
      farthest negatives; `test_min_positive_tops_up_positives_already_in_the_neighbourhood`.
      **Retrieval positive floor falls short.** `inference/retrieval.py:165` takes the
      `short` nearest positives overall, which include the ones already picked, so the result
      holds about max(have, short) positives, not `min_positive`, and the context can shrink.
      Fix: exclude `picked` before choosing `extra`. Verify: a test with 2 positives in the kNN
      set and floor 8 ends with ≥8 positives and exactly `k` rows.
- [x] AU1.3 **Done (2026-10-16).** `rank` and `power` share one fitted ±inf clamp;
      `tests/test_preprocess.py::test_infinities_are_clamped_to_the_finite_extremes`.
      **Rank transform crashes on ±inf.** `inference/preprocess.py:152-159,185`: the
      default `feature_transform="rank"` passes infinities to `QuantileTransformer`, which
      raises; the `power` branch already clamps. Fix: reuse that clamp. Verify: fit and
      transform on a column with ±inf.
- [x] AU1.4 **Done (2026-10-16).** `_member_proba` swaps only when the task is binary;
      `tests/test_classifier.py::test_label_swap_leaves_a_multiclass_task_unchanged`.
      **Label-swap ensembling scrambles multiclass.** `classifier.py:633` reverses the
      probability columns even when `_flip` returned `y` unchanged (≠2 classes), so each class
      is averaged with another; through the regressor it reverses bin order. Fix: swap only
      when binary. Verify: a 3-class test where label swap on equals label swap off.
- [x] AU1.5 **Done (2026-10-16).** `FinancialTFMRegressor.fit` raises on NaN/±inf `y`;
      `test_non_finite_targets_are_refused`. **Non-finite regression targets go to edge bins.** `inference/binning.py:141`
      (`np.digitize`) puts NaN and +inf in the top bin and -inf in bin 0; `regressor.py:112`
      then trains on them. Fix: raise on non-finite `y` in `FinancialTFMRegressor.fit`.
      Verify: a test that NaN or ±inf targets raise.
- [x] AU1.6 **Done (2026-10-16).** `model` stays as passed, the tuned copy is `model_` and
      prediction runs through it;
      `tests/test_finetune.py::test_refitting_fine_tunes_from_the_model_passed_in_not_the_tuned_copy`.
      **`fit` replaces `self.model` with the fine-tuned copy** (`classifier.py:450`). A
      second `fit`, or `sklearn.clone` of a fitted estimator, fine-tunes the already tuned
      weights again and can carry one fold's rows into another. No current caller does this.
      Fix: keep `self.model` as passed; store the tuned copy as `model_`. Verify: fitting twice
      gives the same weights as fitting once.
- [x] AU1.7 **Done (2026-10-16).** Context clamped to the smallest pool, query groups to
      `[1, len(uniq) - 1]` with an error under two groups, `pos_frac` and `finetune_context`
      validated, `groups` length checked; one test per case in `tests/test_finetune.py`
      (`test_finetune_context_never_exceeds_the_context_pool` and the four after it).
      **Fine-tuning edge cases.** (a) `n_ctx` ignores the pool size (`classifier.py:461`),
      so small datasets train on duplicated rows at a context size inference never uses: clamp
      to the pool. (b) One group, or two groups with query fraction ≥0.75, empties the context
      pool and raises: clamp `n_q_groups` to `[1, len(uniq) - 1]`, clear error under 2 groups.
      (c) `pos_frac` at or near 1 divides by zero, above 1 gives a negative size: validate
      `0 < pos_frac < 1`. (d) `groups` length is not checked against `y`. (e) `finetune_context=0`
      silently means `max_context`. Verify: one test per case.
- [x] AU1.8 **Done (2026-10-16).** `_forked_rng` restores the CUDA and MPS generators as well
      as the CPU one; `tests/test_finetune.py::test_finetuning_leaves_the_callers_device_rng_alone`
      (run on MPS, skips without an accelerator).
      **Fine-tuning resets the caller's GPU RNG.** `classifier.py:467-468`:
      `fork_rng(devices=[])` restores CPU state only, while `manual_seed` reseeds CUDA/MPS.
      Scores are unaffected (column identities are drawn on CPU, dropout 0). Fix: fork the
      active device too, or use a local `torch.Generator`.
      Verify: a test on CUDA or MPS that the caller's device RNG state is unchanged by `fit`.
- [x] AU1.9 **Done (2026-10-16).** `_finetune` warns (`UserWarning`) unless the strategy is
      uniform and label swap is off;
      `tests/test_finetune.py::test_finetuning_warns_when_inference_reads_another_context_distribution`.
      **Fine-tuning context ignores the inference strategy.** `classifier.py:438-440`
      always draws a uniform context; with `balanced`/`hybrid`/`retrieval`/`prototype` or
      `ensemble_label_swap`, inference sees a distribution the tuned weights never saw. Fix:
      warn or raise unless uniform and no swap, or draw with the same strategy.
      Verify: a test that a non-uniform strategy with fine-tuning on warns or raises.
- [x] AU1.10 **Done (2026-10-16).** Explicit `classifier_params`; bare keywords still work
      with a `DeprecationWarning`; `test_clone_keeps_the_classifier_settings`.
      **Regressor `clone` drops every classifier setting.** `regressor.py:74-83` takes
      `**classifier_kwargs`, which `get_params` ignores, so `clone`/`GridSearchCV` silently
      revert to defaults. Fix: one explicit `classifier_params: dict`. Verify: `clone(r)`
      round-trips the settings.
- [x] AU1.11 **Done (2026-10-16).** Classifier part in 683f9c4 (`n_features_in_`, `NotFittedError`, opt-in `finetune_clip`, MPS cache, eval mode restored in `finally`, per-class pooled prior shift; device placement moved to `fit` in f9ec061); quantile part in b1ca937 (sorted grid, non-finite targets refused, width check). Tests in `test_classifier.py`, `test_finetune.py`, `test_quantile_regressor.py`.
      **Smaller estimator issues.** `__init__` loads and moves the model (so
      `set_params(device=...)` leaves it on the old device); no `n_features_in_`; predict
      before fit raises `AttributeError`, not `NotFittedError`; no gradient clipping in
      fine-tuning (pretraining clips at 1.0); MPS cache not cleared after fine-tuning; quantile
      regressor can return crossing quantiles (`np.sort` the grid) and goes all-NaN on a NaN
      target; `_pooled_prior_shift` assumes equal shares for multiclass retrieval
      (`classifier.py:514`).
      Verify: one test per item, each failing before its fix.
- [x] AU1.12 **Done (2026-10-16).** `tests/test_preprocess.py` now exists (from AU1.3), and the full suite on the merged fixes ran green, 450 passed and 8 skipped, including the permutation, padding-invariance and task-family tests `.pytest_cache/lastfailed` had listed. **Partly done (2026-10-16).** The three assertions are rewritten and the
      fine-tuning bound now forbids `np.isin` (backstop 60 s); each fails when its condition is
      broken by hand. Still open: direct tests for `inference/preprocess.py` (left to the
      preprocess work) and the `.pytest_cache/lastfailed` re-check on a free machine.
      **Tests that cannot fail.** `tests/test_prior.py:16` (`... or True`),
      `tests/test_model.py:296` (`isnan or isfinite`), `tests/test_prior_score.py:61`
      (`n_scored >= 0`). The wall-clock bound at `tests/test_finetune.py:97` is flaky on a
      loaded machine: assert on operation count or loosen it. `inference/preprocess.py` has no
      direct tests. Re-check `.pytest_cache/lastfailed` (model permutation and padding
      invariance, two task-family cases) when the machine is free.
      Verify: each rewritten assertion fails when its condition is broken by hand.

- [x] AU1.13 **Done (2026-10-16).** Each task's source is drawn once by `_draw_source` and
      dispatched; SCM tasks are grouped only afterwards; reuse-1 stream unchanged
      (`tests/test_mixture.py`). **Graph reuse distorts the prior mixture.** `prior/mixture.py:387-395` draws the
      source to decide whether a slot is grouped, then `sample_task` draws it again, and
      `p_trivial`, `p_crossed`, `p_task_family` are left out of the check, while each SCM slot
      fills `scm_reuse_graph` tasks. Agent measurement at p_fin 0.7, p_tree 0.2, p_fam 0.2:
      SCM share 0.198 at reuse 1 against 0.558 at reuse 4. Any comparison across reuse values
      (task 48.12, §113) mixed two changes. Fix: draw the source once and dispatch. Verify: a
      test that source shares match within binomial error at reuse 1 and 4; re-read §113.
- [x] AU1.14 **Done (2026-10-16).** Single-class draws are topped up with real rows
      (`real_panel.ensure_both_classes`), never relabelled; every real loader takes the
      mixture's positive floor; a one-class panel raises (`tests/test_real_*.py::test_single_class_*`).
      **Some prior tasks get labels unrelated to X.** A single-class draw is repaired by
      flipping 5% of rows at random. `prior/real_mortgage.py:141-144` has no minimum-positives
      step, so at a 0.048% rate most 256-row mortgage tasks get random labels;
      `prior/real_edgar.py:134-142` likewise at ~3% of 256-row tasks. Fix: resample instead of
      flipping; minimum-positives enrichment in every real loader.
      Verify: a test that no real-loader task has labels flipped at random; positive count ≥ the floor.
- [x] AU1.15 **Done (2026-10-16).** EDGAR tasks take one row per `cik` (`real_panel.draw_rows`),
      so every context/query split is firm-disjoint
      (`tests/test_real_edgar.py::test_no_firm_appears_in_both_context_and_query`).
      **EDGAR context/query leakage.** 73% of EDGAR rows share a `cik` with another row;
      row-wise sampling (`prior/real_edgar.py:131`) puts the same firm in context and query,
      which teaches lookup. Fix: split by firm, as `_finetune_rows` does.
      Verify: a test that no `cik` appears in both context and query of an EDGAR task.
- [x] AU1.16 **Done (2026-10-16).** New `PriorConfig.min_ctx_per_class` (default 0, unchanged
      behaviour; config `prior.min_ctx_per_class`, `--min-ctx-per-class`) moves whole rows
      into the context once the split is drawn (`base.ensure_context_classes`). Applied in
      `sample_batch`, not in scm.py/tree.py, since only the batch knows `n_ctx`
      (`tests/test_mixture.py::test_every_query_class_is_in_the_context_with_a_class_floor`).
      **Rare classes appear only in query rows.** Random cut-points (`prior/scm.py:248-250`,
      `prior/tree.py:143-145`) leave a class with ≤2 rows in ~31% of 256-row tasks; the loss is
      then taken on labels absent from the context. Fix: a minimum count per class in context.
      Verify: sample 1,000 256-row tasks; every class present in the query is present in the context.
- [x] AU1.17 **Done (2026-10-16).** `is_learnable` judges a regression task on `y_continuous`
      split at its median, at the same AUC floor; regression prior tasks now pass at a
      non-zero rate (`tests/test_prior_learnability.py`).
      **Learnability filter rejects every regression task.** `is_learnable` reads the
      placeholder `y=zeros` (`prior/scm.py:340`, `prior/learnability.py:47-49`). Fix: skip
      regression or judge on `y_continuous`.
      Verify: with `p_learnability_filter > 0`, regression tasks pass at their true rate in a test.
- [x] AU1.18 **Done (2026-10-16).** Evaluation draws from its own generator (`[seed, 1]`, saved
      in the resume state); CHANGELOG notes training streams differ from 0.7.1 with evaluation
      on (`tests/test_train.py::test_evaluation_does_not_change_the_training_batches`).
      **Evaluation shares the training RNG** (`modeling/train.py:518`): changing
      `eval_every` changes the training tasks. Fix: a separate generator for evaluation.
      Verify: two runs differing only in `eval_every` draw identical training batches.
- [ ] AU1.19 **TabArena adapter fills NaN with 0** (`tabarena/.../models/fintfm/model.py:147`,
      the local TabArena fork). The model has its own missingness mask; filling erases it and
      collides with real zeros. Its docstring (`:43-47`) says the checkpoint was trained "plus a
      generic random-graph SCM"; verify against the checkpoint's recorded prior config and
      correct. Fix: pass NaN through. Verify: TabArena-Lite A/B on the binary datasets.

## Phase 2 — evaluation rigour

- [x] AU2.1 **Done (2026-10-16).** `paired_auc_difference(groups=)` resamples whole companies (`groups=None` byte-identical, pinned); new runs save `company_test`/`company_val` codes; `compare_v4_scores.py --cluster [--companies]` prints both bootstraps; `scripts/v4_fold_companies.py` rebuilt the old runs' companies (verified against every saved `y_true`/`y_val`, 5.3 rows per company) into `runs/audit-cluster/`. §163 side by side, n_boot 2000: row CIs reproduce §163's table, company CIs are at most 0.0006 wider per end, Holm p unchanged except fold 4 (0.138 → 0.232); no fold changes significance. Tests: `test_cluster_bootstrap_is_wider_on_clustered_data`, `test_singleton_groups_reproduce_the_row_bootstrap`, `test_row_bootstrap_is_unchanged_by_cluster_support`, `test_predictions_carry_the_company_of_every_row`, `tests/test_cluster_comparison.py`. §159-§162 and §164 not yet re-reported with the company bootstrap.
      **Bootstrap resamples rows, not companies.** `evaluation/metrics.py:238`; test
      folds hold ~5.3 rows per company, so CIs and p-values are somewhat narrow. Fix: save
      `company` in `predictions_fold*.npz`, add a cluster bootstrap, report both for §159-§164.
      Verify: the cluster and row bootstraps reported side by side for §163; a test on synthetic clustered data shows the cluster CI is wider.
- [x] AU2.2 **Done (2026-10-16).** `run(select_metric="ap")` and `--select-metric` (default `ap`, recorded in the run config; `auc` reproduces earlier runs); `selection_scores` logs val AP and ROC-AUC for every pick. Tests: `test_tuning_selects_on_the_requested_metric`, `test_an_unknown_selection_metric_is_refused`.
      **Tuned baselines are selected on validation ROC-AUC** (`v4_protocol.py:383,440`)
      while AP is the headline. Fix: select on validation AP, record both.
      Verify: the run JSON records the selection metric, and a test selects on AP.
- [x] AU2.3 **Done (2026-10-16).** The untuned banner names the iteration counts and the 200-iteration grid cap; README, CHANGELOG and §163 quote 0.4005 with "1,000 iterations", and §163's "tuning can only widen the gap" is corrected in place.
      **Relabel the booster numbers.** CatBoost's 0.4005 is its library default of 1,000
      iterations; the `--tune` grid caps at 200 (`v4_protocol.py:101`), so the "untuned
      boosters understate the field" banner (`v4_protocol.py:575`) is likely wrong for
      CatBoost. LightGBM and XGBoost ran 100 trees. Fix the banner and every place quoting
      0.4005 as "untuned CatBoost" without the iteration count.
      Verify: `grep -rn 0.4005 docs README.md` shows the iteration count beside every hit.
- [x] AU2.4 **Done (2026-10-16).** `pilot_compare.py --split val` is the default and refuses older runs that saved no validation predictions; `--split test` keeps the old behaviour. Smoke-tested on synthetic npz files; the script has no test module.
      **Pilot arms are ranked on fold-0 test** (`scripts/candidates/pilot_compare.py:52-63`).
      Validation predictions (`y_val`, `val_<arm>`) are already saved: switch the pilot to them,
      and note in §157/§162 that fold 0 selected the arm.
      Verify: `pilot_compare.py --split val` exists and is the default.
- [ ] AU2.5 **Partly done (2026-10-16).** Upstream checked: their drop list
      (`docs/benchmark_protocol.md`, `preprocessing.py` at 59d33ea) equals ours and keeps both
      `operational_status` and `Insolvency_flag` as features; the paper lists the latter as
      "Total_liab > Total_assets". `--drop-columns` added (`test_drop_columns_*`). Still open:
      the ablation run and its FINDINGS write-up.
      **`operational_status` looks like a collection-time snapshot.** Constant within
      99.9% of companies (209 of 188,065 vary), so for earlier years it may carry future
      information; not in `DROP_COLUMNS` (`v4_protocol.py:67`). Check the upstream V4FinBench
      drop list, then run every arm without it as an ablation. Also note that `Insolvency_flag`
      covers 3,586 of 3,587 positives.
      Verify: the ablation is written up in FINDINGS with per-fold AP for every arm.
- [x] AU2.6 **Done (2026-10-16).** `check_protocol` in `compare_v4_scores.py` (used by `c5_verdict.py`) refuses runs whose horizon, row cap, feature count, fold seed or split count differ; bootstrap p is floored at 1/(drawn+1) and the comment says skip. Tests: `test_fold_diffs_refuses_a_different_protocol`, `test_bootstrap_p_is_never_exactly_zero`.
      **Pairing checks compare labels only.** `scripts/compare_v4_scores.py:58`,
      `scripts/c5_verdict.py:56` check `y_true` equality; also compare the `v4_protocol.json`
      configs minus `model`. Floor bootstrap p-values at 1/(drawn+1) (`metrics.py:249`) and fix
      the "redraw" comment at `:239`.
      Verify: a test that a config mismatch makes both scripts exit non-zero.
- [x] AU2.7 **Done (2026-10-15).** Every §159-§163 figure recomputed from the npz and json files matches (claim-by-claim table in the audit's re-derivation report); the method issues it found are AU2.9.
      **Re-derive §159-§163 from the raw runs** and correct any mismatch (in progress
      2026-10-15; results appended here when the check reports).
      Verify: every §159-§163 figure is listed beside its recomputed value.

- [ ] AU2.8 **Confirmed 2026-10-16, still open.** No checkpoint records its prior config, but the
      arm names and §78/§84 settle it: the cell arm is `v4-cellattn-fin10` (`p_financial` 1.0, batch 4,
      `n_rows` 256/512) and the old arm `v4-colid-fin07` (`p_financial` 0.7); no `colid-fin10` or
      `cellattn-fin07` was ever trained. The matched pair needs one training run (old architecture at
      `p_financial` 1.0 on §84's protocol, or the reverse). Original text follows.
      **Cell attention's credited gain may be confounded with the prior mix.** §80 says
      the only difference between arms is `n_cell_blocks`/`cell_labels`, but the doc review
      found the cell arm is `v4-cellattn-fin10` (`p_financial` 1.0, batch 4, `n_rows` 256/512;
      its values equal §97 row A) and the old arm `v4-colid-fin07` (`p_financial` 0.7; §83).
      §91's +0.0337 is likewise fin10-cellattn minus fin07-colid. If confirmed from the two
      checkpoints' recorded configs, the architecture-only effect is unmeasured, and the
      "+0.042 AP from cell attention" in README.md needs the qualifier. Fix: confirm from the
      checkpoints, then score a matched pair (same prior, same budget) or restate the claim.
      Verify: the two checkpoints' recorded prior configs quoted in FINDINGS, then a matched pair or a restated claim.
- [ ] AU2.9 **Update 2026-10-16: "the bar errs conservative" may not hold for pretraining arms.**
      From §160's one matched replicate pair (per-fold Δ +0.0145, −0.0113, +0.0075, −0.0066,
      +0.0003), SD 0.0104, so the noise of a five-fold mean is about 0.0047 and 1.96σ about 0.009
      (ESTIMATED from one pair), above 0.0073. Fine-tuning arms share one checkpoint, so their
      noise is the fine-tuning seed gap (§164's S1: +0.0004 and −0.0013 on folds 0-1 so far),
      not this. Method agreed: σ = SD(per-fold Δ)/√5 pooled over replicate pairs, yardstick
      t(0.975, df)·σ, separate σ for pretraining and fine-tuning arms, Holm across arms.
      `c5_verdict.py helps` landed (ee5df44). **Partly done (2026-10-16), still open.** `c5_verdict.py helps` applies §163's rule
      (two-sided, `rule_helps`) and reproduces §163's verdict on `runs/ft` against
      `control_rescore.json` (+0.0076, 5/5 positive: helps); tests
      `test_helps_rule_needs_the_mean_and_four_folds_of_its_sign`,
      `test_helps_rule_reproduces_section_163s_verdict`. Open: the re-derived yardstick (waits for
      §164's `S1`; `YARDSTICK` is unchanged) and the family of comparisons in §164's write-up.
      Original text follows.
      **§159's 0.0073 bar is not seed noise.** §160 showed it is mostly the max_classes
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

- [x] AU3.1 **Done (2026-10-15, 25da290).** §162's confirmation block moved back under §162; the stale status lines in §162 and §163 updated.
      **§162's confirmation result sits under §163.** `docs/results/FINDINGS.md`
      ~11000-11029 belongs to §162; update the stale status lines at ~10890 (§162 "no result
      exists"), ~10968 and ~11029 ("§163, running"). §158 appears before §157.
      Verify: §162 holds its own result, and no status line says pending or running for a finished run.
- [x] AU3.2 **Done (2026-10-16).** CLAIMS.md Claims 16-18 (fine-tuning, FinTFM-R nulls, ablations and context) plus Claim 6/10 updates; LIMITATIONS.md five new sections and dated notes on the §80 LR and confound lines (the stale `:21-25` text already carried 4a136d6's superseded note); §157-§164 each cited in both files.
      **CLAIMS.md and LIMITATIONS.md stop at §121 and §114.** Add §157-§164: fine-tuning,
      the full-panel booster gap with its iteration counts, logistic regression beating FinTFM
      only at the 200k cap, the FinTFM-R nulls, and that a fine-tuned model no longer "never
      trains on your data". Resolve `LIMITATIONS.md:21-23` against `CLAIMS.md:290` (§80) and the
      stale `LIMITATIONS.md:25`.
      Verify: every § from 157 to 164 is cited in CLAIMS.md or LIMITATIONS.md.
- [x] AU3.3 **Done (2026-10-15, 5450410).** README (two places), the `pyproject.toml` description and the package docstring now say no gradient steps *by default*, with fine-tuning opt-in.
      **"No gradient steps at fit time" is now true only by default.** `README.md:23-25`,
      `README.md:211`, the `pyproject.toml` description, `src/fintfm/__init__.py` docstring.
      Verify: `grep -rn "gradient steps" README.md pyproject.toml src/fintfm/__init__.py` shows the qualifier at each hit.
- [x] AU3.4 **Done (2026-10-16).** README counts and BibTeX fixed in 5450410; `fintfm.__version__` now reads the installed distribution, and `test_package_and_readme_citation_state_the_same_version` pins it and the BibTeX to `pyproject.toml`.
      **Counts and versions.** README test counts disagree (`:60` 407, `:205` "246 tests;
      1 skip"); derive the count and state the sync line it assumes. README BibTeX
      `version = {0.5.5}` (`:484`). No `__version__` in `src/fintfm/__init__.py`. CHANGELOG
      [Unreleased] is empty despite 76551e2 and 5273275. Check CHANGELOG LightGBM 0.2666 vs the
      per-fold mean.
      Verify: README counts equal `pytest --collect-only`; `tests/test_version_agreement.py` covers BibTeX and `__version__`.
- [x] AU3.5 **Done (2026-10-16).** DECISIONS.md D18 (cross-referenced from D2 and D14) closes 36.5; fintfm-r C5 carries EDGAR's stable null (§161) prepended; §164 is lender-adaptation task 16.6.
      **openspec lag.** `adopt-published-methods/tasks.md:28` (36.5) needs a DECISIONS
      entry for fine-tuning; `fintfm-r/tasks.md:160` still says "EDGAR scoring"; §164 has no task.
      Verify: the three items closed or updated with their evidence.
- [x] AU3.6 **Done (2026-10-16).** Both §164 kernels fetch revision f116bfd43a2b15c65ed3551ea8c38e3364629ddc (HF's linked etag equals the pinned SHA-256). EN was already running from the old URL; its SHA-256 check covers it.
      **Kaggle checkpoint URL is not pinned.** `scripts/kaggle/task_164_*.py:15` use
      `resolve/main`; pin the revision as the README does (the SHA-256 check already guards it).
      Verify: `grep -n resolve/main scripts/kaggle` prints nothing.
- [x] AU3.7 **Done (2026-10-16).** Kept the `kaggle` extra and recorded `text-unidecode` in `KNOWN_GOOD` (dual Artistic / GPLv2+, taken under Artistic; reached only through the optional kaggle CLI, never imported by fintfm).
      **Licence checker fails locally** on `text-unidecode` (Artistic/GPL, via `kaggle` →
      `python-slugify`); CI does not see it because it syncs only `--extra bench`. Decide:
      drop the `kaggle` extra from the project, or record why a dev-only tool is acceptable.
      Verify: `uv run python scripts/check_licences.py` passes locally, or the exception is documented.
- [x] AU3.8 **Done (2026-10-15).** All seven doc reviews applied (5450410, 41c7d88, 16b6626, 25da290, 4a136d6); items deliberately left are listed in those commits' reports and in AU0.2.
      **Documentation review findings** from the seven doc agents (appended below as they
      report).
      Verify: each doc agent's list applied or explicitly left alone, and `check_doc_links.py` passes.
- [x] AU3.9 **Done (2026-10-16).** Every figure re-read from the raw TabArena CSV (Space commit
      `0bc60d4`, v0.1.9.7) and the arXiv/Hub PDFs via `pdftotext`, each now cited by
      table/page; six errors corrected (O'Prior 40k tasks not 400k, TabICLv2 ≈100 Elo each,
      LightPFN rule-prior +0.78 and short-budget claim, the 1780/1703 attribution, imputation
      date); earlier entries carry dated updates; ten entries added to REFERENCES.md.
      Unverifiable: Kumo-Medium's 62M (derived from checkpoint size, marked ESTIMATED) and
      O'Prior's code licence (repo not found). §114's scope note in FINDINGS.md is left to its
      separate change.
      **Check the 2026-10-15 peer sweep against primary sources** (RELATED_WORK.md, last
      section) before any figure in it is quoted, then update the earlier entries it
      contradicts (TabPFN-3.5 rank and Elo, Mitra, Kumo, LimiX-2, best real-data model, the
      TabSwift "data points against 2D") and §114's scope (48k tasks, not scale in general).
      Verify: every figure in that section carries a primary-source citation, and the earlier entries agree with it.

## Phase 4 — cheap diagnostics of the CatBoost gap (hours, CPU or one Kaggle session)

- [ ] AU4.1 **Prepared (2026-10-16), not launched.** `--booster-params` adds fixed-parameter
      booster arms (`test_booster_params_*`); CPU kernel
      `scripts/kaggle/task_au41_boost_rounds/`, commit pin still a placeholder.
      **Is the gap boosting rounds?** LightGBM at 1,000 trees and lr 0.05, CatBoost at
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
- [ ] AU4.4 **Implemented behind `python -m fintfm.experiments.hybrid_catboost` (2026-10-16, a3865e0); measurement open.**
      **Hybrid ceiling.** Out-of-fold FinTFM logits (inner company-grouped K-fold) as a
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
- [ ] AU5.2 **Implemented behind `--attn-impl split` (2026-10-16, e18d122); measurement open.**
      Exact unmasked route (context attention plus a sentinel key carrying each query's own
      key, no mask in either call), in the row encoder and the row-within-feature stage;
      equal to `masked` within 1e-5 fp32 (tests/test_model.py `*attn_impl_split*`). This
      supersedes the "not implemented" note below. CPU timing at load 200-350 on 16 cores
      showed no difference beyond noise; the flash benefit must be measured on CUDA.
      `--benchmark-steps` now reads `--task-dump` (e18d122).
      **`--amp fp16` with loss scaling added for GPUs without native bf16 (T4, P100) (2026-10-16, 6ec5534); measurement open.**
      **Implemented behind `--amp bf16` and `--benchmark-steps` (2026-10-16, 9c3b615); measurement open.**
      Flash attention not implemented: CUDA's flash kernel takes no attention mask at all, and
      `MultiheadAttention` already turns the bool mask into an additive float one, so a float
      mask changes nothing. An exact route (FlexAttention with this mask, or context and query
      attention split and merged by log-sum-exp) replaces the encoder layers in `model.py`.
      **Pretraining throughput.** bf16 autocast (training is fp32, `modeling/train.py:471-484`),
      and an attention path that allows flash attention (the bool row mask blocks it). Measure
      tasks per GPU-hour before and after; no accuracy claim.
      Verify: tasks per GPU-hour measured before and after on the same kernel; outputs equal within fp tolerance on a fixed batch.
- [ ] AU5.3 **fintfm-train flags `--n-heads`, `--n-cell-heads` and `--n-summary-tokens` wired (2026-10-16, 6ec5534); measurement open.**
      **Hooks implemented behind `n_cell_heads` and `n_summary_tokens` (2026-10-16, 0a3b8c8); measurement open; fintfm-train flags and `--n-heads` not yet wired; more cell blocks is the existing `n_cell_blocks`.** **Capacity where the bottleneck is.** The shipped model has ~886k parameters (agent
      measurement; verify) and pools up to 136 cell tokens into one 128-d row vector after one
      cell block (`model.py:537-553`). Peers keep per-cell tokens (TabPFN v2) or several summary
      tokens per row (TabICL); verify their sizes from primary sources first. Arms: more cell
      blocks; 4 summary tokens per row; expose `n_heads`.
      Verify: peer sizes quoted from primary sources; each arm scored at matched tasks under §159's rule.
- [ ] AU5.4 **Implemented behind `--max-query-rows` / `prior.max_query_rows` (2026-10-16,
      b1084ce); measurement open.** Contexts to 4,096 rows were already reachable through
      `--n-rows-choices` (4552 gives `int(0.9 * 4552) == 4096`); the flag caps the query rows
      that come with them. Row attention is a dense masked N x N, so cost grows with context
      plus query, not linearly in queries as stated below.
      **Train on the context lengths inference uses** (up to 4,096 rows; query-to-context
      attention keeps cost linear in queries).
      Verify: the trained `n_rows` range covers inference's `max_context`; five-fold AP against the current checkpoint.
- [ ] AU5.5 **Implemented behind `--permute-classes` / `prior.permute_classes` (2026-10-16,
      b1084ce); measurement open.**
      **Class-index permutation in pretraining** (`collate`): SCM classes are ordinal and
      class 1 is always the rare default in the financial prior, so the model can learn a
      class-index prior.
      Verify: a test that class indices are permuted per task; label-symmetry probe before and after.
- [ ] AU5.6 **Implemented behind `--feature-transforms` / `prior.feature_transforms`
      (2026-10-16, b1084ce); measurement open.**
      **Match pretraining and inference preprocessing.** Pretraining z-scores; inference
      defaults to rank-to-normal. Draw a random transform per pretraining task and ensemble over
      transforms at inference.
      Verify: a test that pretraining draws each transform; five-fold AP against the current checkpoint.
- [ ] AU5.7 **Implemented behind `max_features_strategy: subsample` and `ensemble_transforms`
      (2026-10-16, b3da945); measurement open.** Round-robin column subsets per member cover
      every feature of a table wider than `max_features`, raising the member count to
      `ceil(F / size)` when `n_ensemble` is below it; the default `raise` is unchanged.
      **Stronger default ensembling.** `n_ensemble` is 1 in the library and V4 configs;
      add feature-shuffle and transform members, and feature-subset ensembling to lift the
      `max_features` hard limit (`classifier.py:373`).
      Verify: TabArena-Lite and V4FinBench scores against `n_ensemble` 1 and the current 8.
- [ ] AU5.8 **Implemented behind `--large-n partition|boost` and `inference.retrieval_per_class` (2026-10-16, b53ee51); measurement open.**
      **Large-n methods from the literature**, each its own arm: TabPFN-kNN/LoCalPFN-style
      retrieval with per-class neighbours (arXiv 2406.05207; code has no licence file, so
      reimplement); a tree partition or forest of PFNs (tabpfn-extensions `rf_pfn`,
      Apache-2.0); BoostPFN (arXiv 2503.01256, MIT).
      Verify: each arm's five-fold AP against §163's arm under §159's rule.
- [ ] AU5.9 **Implemented behind `hazard_period_context` at inference (2026-10-16, f69f5a4): `FinancialTFMClassifier.fit(X, y, periods=...)` stores context periods and `predict_term_structure` passes them; required on a flag-on checkpoint, ignored on a flag-off one; the V4FinBench harnesses pass `ds.period`; measurement open.**
      **fintfm-train flag `--hazard-period-context` wired (2026-10-16, 6ec5534); measurement open; inference gap unchanged.**
      **Implemented behind `hazard_period_context` (2026-10-16, 0a3b8c8); measurement open; the classifier cannot yet pass context periods at inference.** **Hazard context carries only the binary label** (`model.py:751-759,787-797`): embed
      the default period and a censored token for context rows.
      Verify: a hazard test where the term structure is learnable only from context periods.
- [ ] AU5.10 **Implemented behind `--no-decay-norm-bias` and `--warmup-steps` (2026-10-16, 9c3b615); measurement open.**
      The final-LayerNorm-before-pooling nit is not done (model.py).
      **Optimiser nits.** Weight decay also hits LayerNorm, biases and embeddings
      (`train.py:407`); no final LayerNorm before pooling in two encoders; warmup fixed at 500
      steps with no flag.
      Verify: a test that LayerNorm, biases and embeddings are excluded from weight decay.
- [ ] AU5.11 **"Before" statistics measured with `scripts/prior_stats.py` (2026-10-16,
      a4806e2), 300 tasks per mixture; no default changed.** Two claims below do not hold for
      the financial prior: its missingness is label-informative in 11.7% of tasks (Fisher,
      Bonferroni 0.01) and its marginals are heavy-tailed (median excess kurtosis 4.06).
      **Prior diversity gaps** (agent sampling, verify each): the binary checkpoint was
      trained at `p_financial 1.0` (§91/§104), where tasks are binary, ~1% categorical and have
      a median logistic-regression AUC of 0.57; categoricals are always ordered 2-7 level codes
      (`prior/scm.py:261-264`); missingness is only MCAR and never label-informative; SCM and
      tree marginals are near-Gaussian; the task-family generator always emits exactly 136
      Gaussian columns (`mixture.py:229-236`); tree splits pick columns by raw std
      (`prior/tree.py:79-82,136`). Arm: a TabArena checkpoint on SCM + tree + families with
      `p_financial ≤ 0.3`, more tasks, after AU1.13.
      Verify: the prior sampler's statistics (class balance, categoricals, missingness, kurtosis) reported before and after.
- [ ] AU5.12 **Implemented behind `ensemble_class_shift`, `ensemble_transforms` (new `squash`
      transform) and `ensemble_svd_components`, with `max_features_strategy: subsample` for
      wide tables (2026-10-16, b3da945); measurement open.** Label-symmetry probe
      (`scripts/label_symmetry_probe.py`), MEASURED on `runs/v4-cellattn-labels.pt`, Pearson
      r of `p` against `1 - p_swapped`, pooled over queries (median per task): synthetic
      financial 0.891 (0.687), 24 tasks, 6,144 queries; synthetic SCM 0.955 (0.872), 24 tasks,
      6,144 queries; Polish 3y 0.490 (0.445), 3 draws of 512 context and 512 query rows;
      Taiwan 0.829 (0.839), same draws. No task negative, against §45's −0.62 on an older
      checkpoint; the Polish panel is the least symmetric.
      **Per-member inference diversity** (peer recipes read from installed TabPFN 9.0.0,
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
- [ ] AU5.15 **Implemented behind `scripts/dump_prior_tasks.py --stream per_shard --workers K`
      (2026-10-16, 3705a71); measurement open.** K=2 writes the K=1 dump array for array,
      compressed or not (tests/test_prior_levers.py). Throughput at load 514-521 on 16 cores,
      128 batches of the reference recipe: K=1 4.94 batches/s, K=4 3.43 (compressed 2.36), so
      no speed-up on a saturated machine; repeat on an idle one.
      **Task-dump prerequisite implemented behind `--task-dump` and
      `scripts/dump_prior_tasks.py` (2026-10-16, b1084ce); measurement open.**
      **Pretraining volume first: 48k tasks → at least 2M.** The strongest evidence in
      the 2026-10-15 sweep (LightPFN's 256k → 7.68M draws, +1.66 AUC points; LimiX-2 and Kumo
      scale curves at 10^7+ tasks). Prerequisites: AU1.13, AU5.2, pre-generated task dumps,
      `--checkpoint-every`. Cost to be measured on one Kaggle session before committing. After
      it, re-test the closed architecture verdicts, all measured at 48k tasks.
      Verify: tasks per GPU-hour and cost measured on one Kaggle session before the full run; then §159's rule on V4FinBench and TabArena-Lite.
- [ ] AU5.16 **fintfm-train flags `--value-embedding`, `--value-basis-size`, `--value-basis-learnable` and `--value-ecdf` wired (2026-10-16, 6ec5534); measurement open.**
      **Implemented behind `value_embedding` and `value_ecdf` (2026-10-16, 0a3b8c8); measurement open.** **Value embedding**: Fourier or RBF plus ECDF-rank features in place of the MLP on
      (z, missing) (LimiX-2M RaBEL, TabPFN-3.5). One module, behind a flag.
      Verify: a flag, a test that it is off by default, and five-fold AP at matched tasks.
- [ ] AU5.17 **fintfm-train flags `--attn-length-scale` and `--attn-length-ref` wired (2026-10-16, 6ec5534); measurement open.**
      **Implemented behind flags (2026-10-16); measurement open.** Attention half:
      `attn_length_scale` (0a3b8c8; fixed log-length scale, no learned per-query factor).
      Optimiser half: `--optimizer muon` (9c3b615). Early target inclusion not done.
      **Optimiser and attention scaling from TabICLv2**: Muon, QASSMax (length-aware
      softmax scaling), early target inclusion; also the fix for AU4.3 if context length is the
      problem.
      Verify: each change behind a flag, scored at matched tasks.
- [ ] AU5.18 **Score the TabArena imbalanced subset** (v0.1.9), the credit-relevant slice, with
      the current checkpoint as a baseline.
      Verify: the imbalanced-subset Elo recorded in TABARENA.md with its date.
