# Tasks

## Phase 0 — decisions for the owner

- [x] AU0.1 **Done (2026-10-20): retired** (`docs/design/DECISIONS.md` D19). `data_candidates.csv` marks it `EXCLUDED_BENCHMARK` and `check_overlap.py` fails on any `IN_USE` or `SHORTLIST` row with a `benchmark_match`; the sampler raises `RetiredSourceError`, `p_real_lendingclub` and a `real_lendingclub` weight are refused before any draw, and the generic panel route refuses a `lendingclub` directory (`tests/test_real_lendingclub.py`, `tests/test_real_panel.py`, `tests/test_data_candidates.py`).
      **LendingClub conflicts with the no-benchmark-data rule.** It is in BeyondArena
      (`lending_club_1m`); §155 knew and chose a bounded forfeit (exclude that task from any
      FinTFM-R BeyondArena number), and §157's fold-0 pilot trained on it. The standing rule is
      never to train on TabArena or BeyondArena datasets. Decide: retire it (mark
      `EXCLUDED_BENCHMARK` in `docs/research/data_candidates.csv`, make
      `prior/real_lendingclub.py` refuse to load, make `check_overlap.py` fail on any `IN_USE`
      row with a `benchmark_match`), or record the exception in DECISIONS. No shipped FinTFM
      checkpoint is affected.
      Verify: the decision is recorded in DECISIONS.md, and `data_candidates.csv` agrees with it.
- [x] AU0.2 **Done (2026-10-20).** All three now say §153's reading of the terms: free tier, internal research use and noncommercial public release of results, for as long as fintfm is free; the fee-based agreement the day it is monetised; never correlated to individuals, never staged to a third-party host. `CLAUDE.md` (the FinTFM-R row and the rule) was the one out of line; §153, `data_candidates.csv:3`, README and DECISIONS D17 already agreed.
      **Freddie Mac wording disagrees in three places**: `CLAUDE.md:374` (none until a
      commercial licence is signed), §153 (free tier permits internal use) and
      `docs/research/data_candidates.csv:3` ("noncommercial project", `IN_USE`). Owner to
      reconcile; `CLAUDE.md` is hand-edited.
      Verify: one wording, the same in all three places.
- [x] AU0.3 **Done (2026-10-20).** `CLAUDE.md` no longer names the other employer's domain or a home-directory path; this task's own text is redacted the same way (the one exception to prepend-only, since the Verify forbids the string anywhere), and FINDINGS' one home-directory path reads `<venv>/...`. Both greps print nothing outside history.
      **`CLAUDE.md` is public and names another employer's domain and a local home-directory path**
      (`CLAUDE.md:14-16,199`). Owner's file. The AXIOM harness licence was removed from three
      public files in b78a52c; it remains in git history.
      Verify: `git grep` for that domain and for the macOS home-directory prefix prints nothing outside history.

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
- [x] AU1.19 **Done (2026-10-19).** The fork's adapter passes NaN through (93821e5, branch `fintfm-audit-fixes`, local); the Verify A/B is in `docs/results/TABARENA.md` (DR3.5): no measurable difference over 27 binary datasets, so passthrough is kept. The docstring (`model.py:43-47`) now says the binary checkpoint is financial prior only (`--p-financial 1.0`), which matches `configs/published/v4-cellattn-labels.yaml` (`p_financial: 1.0`, `max_classes: 2`); the generic SCM is mixed in only for the multiclass checkpoint (`model.py:207-211`).
      **TabArena adapter fills NaN with 0** (`tabarena/.../models/fintfm/model.py:147`,
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
- [x] AU2.5 **Done (2026-11-03): FinTFM's own ablation is §200 (zero-shot, −0.0022 mean AP, 4 of 5 folds down, one Holm-significant) and §202 (fine-tuned, −0.0023, 5 of 5 down, none Holm-significant), both per fold; with §167's boosters every arm is written up. FinTFM depends on the column by about 0.002 AP against CatBoost's 0.034.** **Booster ablation done (2026-10-16, §167):** without `operational_status` CatBoost loses 0.0343 mean AP (5 of 5 folds), LightGBM 0.0210, LR 0.0064. FinTFM's own ablation is still open.
      **Partly done (2026-10-16).** Upstream checked: their drop list
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

- [x] AU2.8 **Done (2026-10-18, §172).** At a matched prior and protocol the cell-attention block is worth +0.0520 mean AP, 5 of 5 folds significant; the confound did not inflate §80. **Confirmed 2026-10-16, still open.** No checkpoint records its prior config, but the
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
- [x] AU2.9 **Done (2026-10-19, §173).** Re-derived yardstick 0.0117 for pretraining arms; `c5_verdict.py helps` landed earlier. **Update 2026-10-16: "the bar errs conservative" may not hold for pretraining arms.**
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

- [x] AU4.1 **Done (2026-10-16, §167).** Yes, it is rounds: LightGBM at 1,000 trees reaches 0.3908 mean AP against 0.2666 at its 100-tree default (5 of 5 folds up), CatBoost at 100 iterations falls to 0.3746 from 0.4005.
      **Prepared (2026-10-16), not launched.** `--booster-params` adds fixed-parameter
      booster arms (`test_booster_params_*`); CPU kernel
      `scripts/kaggle/task_au41_boost_rounds/`, commit pin still a placeholder.
      **Is the gap boosting rounds?** LightGBM at 1,000 trees and lr 0.05, CatBoost at
      100 iterations, same folds. If LightGBM approaches 0.40, the gap is rounds over all
      ~2,150 training positives, not categoricals (CatBoost gets no `cat_features` here; check
      `get_all_params()` for its boosting type).
      Verify: per-fold AP for both arms in FINDINGS, paired against the existing booster runs.
- [ ] AU4.2 **Progress (2026-11-05, `docs/results/drafts/v4_inference_kernels_2026-11.md`): the full-panel prototype arm is prepared on Kaggle GPU, not pushed, not run.** Kernel `scripts/kaggle/task_v4inf_au42proto_f01234/`, pinned 62a73c3, `configs/arms/au42-prototype.yaml`; control `runs/lrsweep/eval-published`; ESTIMATED 3 to 4 GPU-h (k-means time unmeasured); rule §159's with §173's 0.0117 bar; prediction: not a candidate. SMOKE-TEST passed on CPU. **Progress (2026-11-03, `docs/results/drafts/inference_tasks_2026-11.md`): floor half answered by §200, null** (floor16 +0.0055 mean AP, 3 folds up, 2 down, below 0.0117). **Open: prototype undersampling on the full panel** (only §162's capped fold 0 exists); command and estimate in the draft, `configs/arms/au42-prototype.yaml`, not run (memory). **Context composition, inference only.** Uniform beat balanced/hybrid in §29/D9,
      but a minimum-positive floor on a uniform context (~7 positives at 2,000 rows) and
      k-means prototype undersampling (the V4FinBench authors' 0.3 ratio) were never scored on
      the full panel. Comparator: zero-shot uniform.
      Verify: five-fold AP against zero-shot uniform, paired bootstrap, §159's rule.
- [ ] AU4.3 **Progress (2026-11-05, `docs/results/drafts/v4_inference_kernels_2026-11.md`): the five-fold V4FinBench `max_context` 900, 3-member arm is prepared on Kaggle GPU, not pushed, not run.** Kernel `scripts/kaggle/task_v4inf_ctx900e3_f01234/`, pinned 62a73c3, `configs/arms/proxy/au43-ctx900-ens3.yaml`; control `runs/lrsweep/eval-published`; ESTIMATED 2.4 GPU-h; rule as AU4.2; prediction: null or harmful. SMOKE-TEST passed on CPU. **Progress (2026-11-03, `docs/results/drafts/inference_tasks_2026-11.md`): the checkpoint does not record `n_rows` choices; the recorded training command's `--n-rows-choices 256,512,1024` gives contexts of 76-921 rows.** Proxy, MEASURED: `max_context` 900 with 3 members against the default, 11 wins, 5 losses, p 0.21, not a candidate; §162's 500-row 8-member arm was −0.0029 on V4FinBench. Open: the five-fold V4FinBench arm (command in the draft, about 7.5 h ESTIMATED, not run). **Context length beyond training.** Pretraining contexts reach at most ~921 rows
      (`prior/mixture.py:446` with the shipped `n_rows` choices; verify from the checkpoint
      config), inference uses 2,000. Score `max_context` 900 with `n_ensemble` 3 against 2,000.
      Verify: the checkpoint's recorded `n_rows` choices quoted, and five-fold AP for both arms.
- [x] AU4.4 **Done (§201): null, a system result.** CatBoost 0.4005; + FinTFM as baseline 0.4044 (+0.0039, 4/5 folds), as feature 0.4023 (+0.0018, 2/5); nothing survives Holm. Leak assert keyed on (country, company) held. **Implemented behind `python -m fintfm.experiments.hybrid_catboost` (2026-10-16, a3865e0); measurement open.**
      **Hybrid ceiling.** Out-of-fold FinTFM logits (inner company-grouped K-fold) as a
      CatBoost `baseline` and as a feature; compare against CatBoost alone. Measures a system,
      not FinTFM, and is reported as such.
      Verify: five-fold AP for CatBoost alone and CatBoost plus FinTFM, with no outer-test rows in the inner folds (asserted).

## Phase 5 — strength levers (GPU, ordered by expected gain × feasibility)

- [ ] AU5.1 **Progress (2026-11-04, `docs/results/drafts/au5.1_2026-11.md`, 0606ceb): early stopping on inner-validation companies implemented behind `finetune_val_frac` (default 0, byte-identical to before; held-out companies come from the fold's own training rows, never the test fold or the protocol's validation split), selected on validation cross-entropy; arms `au51-lr1e5`, `au51-lr3e5`, `au51-lr1e4` (cap 3,000 steps, val_frac 0.1, eval every 10 steps, patience 16) as configs under `configs/arms/` and nine Kaggle kernels `scripts/kaggle/task_au51_*` (folds 0-1, 2-3, 4 per arm), unpinned until pushed; pre-registered under §159 / §164 Rule 3 (mean ΔAP above 0.0073, at least 4 of 5 folds positive, against §163's arm); cost ESTIMATED at 7.1 GPU-h per arm if it stops near step 600, 27.6 at the cap (5.7 s/step and about 1,400 s scoring per fold, MEASURED in the L600 log); not run.** **Fine-tuning recipe at published scale.** Our 200 steps at 1e-5 is the bottom of
      the published range (Rubachev et al. 2025, arXiv 2506.08982: full fine-tune, lr sweep
      5e-6..5e-4, 1,024 queries per step, early stopping on validation every 10 steps, patience
      16). Arms: lr {1e-5, 3e-5, 1e-4}, up to 3,000 steps, early stopping on inner-validation
      companies. Depends on §164.
      Verify: five-fold AP against §163's arm under §159's rule, inner-validation selection only.
- [x] AU5.2 **Done (2026-11-05, closed by the owner on the recommendation below).** **Recommended for closing (2026-11-02, `docs/results/drafts/preregs_2026-11.md`): the Verify is met by §170 and its addendum** (split attention 1.21x at fp32 and 1.10x under fp16 in tasks per hour on one Kaggle kernel, logits within 1.2e-4 and 9.4e-4 of fp32 masked); no run needed. **Implemented behind `--attn-impl split` (2026-10-16, e18d122); measurement open.**
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
- [ ] AU5.3 **Narrowed (2026-11-02, `docs/results/drafts/preregs_2026-11.md`): one arm, `au53-cell2-s0` (`--n-cell-blocks 2`, a weight-tied second cell pass at d05ea19), pre-registration drafted and kernel `scripts/kaggle/ts_au53_cell2_s0/` written; not trained.** Heads and width are superseded by §202 (DR8.2 width null) and summary tokens by DR6.3's readout tokens, harmful at screening (§178). **fintfm-train flags `--n-heads`, `--n-cell-heads` and `--n-summary-tokens` wired (2026-10-16, 6ec5534); measurement open.**
      **Hooks implemented behind `n_cell_heads` and `n_summary_tokens` (2026-10-16, 0a3b8c8); measurement open; fintfm-train flags and `--n-heads` not yet wired; more cell blocks is the existing `n_cell_blocks`.** **Capacity where the bottleneck is.** The shipped model has ~886k parameters (agent
      measurement; verify) and pools up to 136 cell tokens into one 128-d row vector after one
      cell block (`model.py:537-553`). Peers keep per-cell tokens (TabPFN v2) or several summary
      tokens per row (TabICL); verify their sizes from primary sources first. Arms: more cell
      blocks; 4 summary tokens per row; expose `n_heads`.
      Verify: peer sizes quoted from primary sources; each arm scored at matched tasks under §159's rule.
- [ ] AU5.4 **Arm `au54-ctx2k-s0` drafted (2026-11-02, `docs/results/drafts/preregs_2026-11.md`): `--n-rows-choices 256,512,1024,2223 --max-query-rows 512`, contexts to 2,000 rows, kernel `scripts/kaggle/ts_au54_ctx2k_s0/` written; not trained.** **Implemented behind `--max-query-rows` / `prior.max_query_rows` (2026-10-16,
      b1084ce); measurement open.** Contexts to 4,096 rows were already reachable through
      `--n-rows-choices` (4552 gives `int(0.9 * 4552) == 4096`); the flag caps the query rows
      that come with them. Row attention is a dense masked N x N, so cost grows with context
      plus query, not linearly in queries as stated below.
      **Train on the context lengths inference uses** (up to 4,096 rows; query-to-context
      attention keeps cost linear in queries).
      Verify: the trained `n_rows` range covers inference's `max_context`; five-fold AP against the current checkpoint.
- [x] AU5.5 **Done (2026-11-04, `docs/results/drafts/au5.5_2026-11.md`): the "after" label-symmetry probe ran on CPU with §165's settings on `dr74-mc-s0` and `dr74-mc-s1` (trained with `--permute-classes`, provenance checked) against `v4-multiclass.pt` re-measured in the same session (MEASURED, 24 tasks per synthetic prior, 3 draws per real panel, probe seed 0). Median per-task r rises from 0.564 to 0.917 and 0.931 on the financial prior and from 0.757 to 0.960 and 0.955 on SCM; the control's 3 of 24 negative SCM tasks fall to 0 at both seeds. Polish 3-year pooled r rises from 0.726 to 0.951 and 0.953, Taiwanese from 0.887 to 0.981 and 0.964. Caveats: §165's own "before" was on `v4-cellattn-labels.pt`; the after checkpoints differ from the control in training protocol as well as permutation (§185); the probe is binary-only; no accuracy effect is claimed.** **Recommended for closing without a GPU run (2026-11-02, `docs/results/drafts/preregs_2026-11.md`):** the test exists; the probe "before" is §165, and "after" can run on CPU on `dr74-mc-s0`/`dr74-mc-s1` (trained with `--permute-classes`) against `v4-multiclass.pt`. A binary credit arm is not funded: in credit the default is always class 1. **Implemented behind `--permute-classes` / `prior.permute_classes` (2026-10-16,
      b1084ce); measurement open.**
      **Class-index permutation in pretraining** (`collate`): SCM classes are ordinal and
      class 1 is always the rare default in the financial prior, so the model can learn a
      class-index prior.
      Verify: a test that class indices are permuted per task; label-symmetry probe before and after.
- [x] AU5.6 **Done (2026-11-05, closed as superseded by the owner on the recommendation below).** **Recommended for closing as superseded (2026-11-02, `docs/results/drafts/preregs_2026-11.md`): §88 measured that the inference rank transform does the invariance work (§87's mismatch "needs no retraining"; marginal augmentation about 0.001); no arm drafted.** **Implemented behind `--feature-transforms` / `prior.feature_transforms`
      (2026-10-16, b1084ce); measurement open.**
      **Match pretraining and inference preprocessing.** Pretraining z-scores; inference
      defaults to rank-to-normal. Draw a random transform per pretraining task and ensemble over
      transforms at inference.
      Verify: a test that pretraining draws each transform; five-fold AP against the current checkpoint.
- [x] AU5.7 **Done (2026-11-05, closed by the owner: both halves of Verify measured, the adapter's 8 members stay).** **Progress (2026-11-04, `docs/results/drafts/inference_tasks_2026-11.md`): the TabArena-Lite half is measured: `n_ensemble` 1 against the adapter's 8 is worse on 20 of 27 datasets (sign test p 0.019), mean ROC-AUC −0.00445, Elo 696 against 756 (`fintfm_lite_ens1_1104` against `fintfm_lite_nan_1019`, `scripts/compare_lite_arms.py`); with §196 both halves of Verify are met; the adapter's 8 stays; proposed for closure.** **Progress (2026-11-03, `docs/results/drafts/inference_tasks_2026-11.md`): the V4FinBench half is §196 (`ens8` +0.0058 mean AP, 3 of 5 folds up); `ens8-tf` against `ens8` on the proxy 12 wins, 4 losses, p 0.077.** Open: TabArena-Lite at `n_ensemble` 1 against the adapter's 8, which needs an override in the fork's adapter first. **Implemented behind `max_features_strategy: subsample` and `ensemble_transforms`
      (2026-10-16, b3da945); measurement open.** Round-robin column subsets per member cover
      every feature of a table wider than `max_features`, raising the member count to
      `ceil(F / size)` when `n_ensemble` is below it; the default `raise` is unchanged.
      **Stronger default ensembling.** `n_ensemble` is 1 in the library and V4 configs;
      add feature-shuffle and transform members, and feature-subset ensembling to lift the
      `max_features` hard limit (`classifier.py:373`).
      Verify: TabArena-Lite and V4FinBench scores against `n_ensemble` 1 and the current 8.
- [ ] AU5.8 **Progress (2026-11-05, `docs/results/drafts/v4_inference_kernels_2026-11.md`): per-class retrieval and boost are prepared on Kaggle GPU, not pushed, not run.** Kernels `scripts/kaggle/task_v4inf_pc16_f01234/` (`configs/arms/au58-retrieval-pc16.yaml`, ESTIMATED 2 to 5.5 GPU-h) and `task_v4inf_boost_{f0,f12,f34}/` (`configs/arms/au58-boost.yaml`, residual pool capped at 100,000 rows, so a null does not close the uncapped method; f0 is the timing probe, about 18 GPU-h in all, ESTIMATED); pinned 62a73c3; controls `runs/ft/B_lr1e5` (the task's) and the default; rule §159's with §173's bar. SMOKE-TEST passed on CPU. **Progress (2026-11-03, `docs/results/drafts/inference_tasks_2026-11.md`): partition against §163's fine-tuned arm, MEASURED from saved predictions, +0.0299 mean AP, 5 of 5 folds, 4 Holm-significant (company bootstrap); the partition third of Verify is met.** Open: `--large-n boost` and `retrieval_per_class` (`configs/arms/au58-retrieval-pc16.yaml`) on V4FinBench; commands and estimates in the draft, not run. **Implemented behind `--large-n partition|boost` and `inference.retrieval_per_class` (2026-10-16, b53ee51); measurement open.**
      **Large-n methods from the literature**, each its own arm: TabPFN-kNN/LoCalPFN-style
      retrieval with per-class neighbours (arXiv 2406.05207; code has no licence file, so
      reimplement); a tree partition or forest of PFNs (tabpfn-extensions `rf_pfn`,
      Apache-2.0); BoostPFN (arXiv 2503.01256, MIT).
      Verify: each arm's five-fold AP against §163's arm under §159's rule.
- [ ] AU5.9 **Progress (2026-11-07, `docs/results/drafts/au5.9_2026-11.md`): the hold is resolved by the code fix, not by float32. `HazardHead.hazards` now computes projection, sigmoid and clamp in float32 with autocast disabled, so `log1p(-h)` never sees a half-precision hazard; `_autocast` no longer refuses a hazard head and still refuses the quantile head (its pinball loss is not cast to float32). Tests: bf16 NLL equals float32 within 1e-6 where bf16 rounds the hazard to one, and a 3-step CPU `--amp bf16 --n-horizons 6` smoke has finite loss (SMOKE-TEST; CUDA fp16 not run locally). Kernels reset to `UNPINNED` with their `--amp fp16` flags unchanged; the coordinator pins after the push.** **Held (2026-11-07): both kernels removed from the Kaggle lanes before launch.** `ts_au59_hazard_ctrl_s0` and `ts_au59_hazard_ctx_s0` pass `--amp fp16` with `--n-horizons 6`, and `_autocast` (src/fintfm/modeling/train.py, at 62a73c3 and on main) refuses any amp mode for a model with a hazard head, so both would stop at the first training step. Open: either train in float32 (`--amp none`, about 2.4x the GPU time by §86's 1.90 against 0.80 s/step, ESTIMATED) or make the hazard NLL compute in float32 under autocast and lift the refusal, then re-pin. **Progress (2026-11-04, `docs/results/drafts/au5.9_2026-11.md`, 3a4911c): the Verify task and scorer are built and the pipeline is proven by a SMOKE-TEST; the flag-on pretraining arm is prepared, not run.** `fintfm.experiments.hazard_period_probe` generates tasks whose early and late regimes share X and binary y and differ only in default periods (K = 6), scores a checkpoint with and without context periods against a periods-blind floor (shape RMSE 0.2965), and applies a pre-registered pass rule (below the floor, paired 95% bootstrap interval below zero, strong at or below 0.148). A 40-step CPU flag-on checkpoint loads, accepts `periods=` and scores (SMOKE-TEST, no quality claim). Arms `au59-hazard-ctx-s0` and the flag-off control `au59-hazard-ctrl-s0` (published recipe, `--n-horizons 6`, T4x2, ESTIMATED 2.2 to 2.6 h for both) are configured with unpinned kernels; measurement open. **Progress (2026-11-03, `docs/results/drafts/inference_tasks_2026-11.md`): not runnable at inference; no flag-on checkpoint exists** and the published one is classification-only. Needs a `--n-horizons --hazard-period-context` pretraining run and a synthetic task built for the Verify. **Implemented behind `hazard_period_context` at inference (2026-10-16, f69f5a4): `FinancialTFMClassifier.fit(X, y, periods=...)` stores context periods and `predict_term_structure` passes them; required on a flag-on checkpoint, ignored on a flag-off one; the V4FinBench harnesses pass `ds.period`; measurement open.**
      **fintfm-train flag `--hazard-period-context` wired (2026-10-16, 6ec5534); measurement open; inference gap unchanged.**
      **Implemented behind `hazard_period_context` (2026-10-16, 0a3b8c8); measurement open; the classifier cannot yet pass context periods at inference.** **Hazard context carries only the binary label** (`model.py:751-759,787-797`): embed
      the default period and a censored token for context rows.
      Verify: a hazard test where the term structure is learnable only from context periods.
- [x] AU5.10 **Done (2026-11-05, closed without a run by the owner on the recommendation below).** **Recommended for closing without a run (2026-11-02, `docs/results/drafts/preregs_2026-11.md`):** the Verify test exists (`tests/test_train.py::test_decay_norm_bias_false_exempts_norms_biases_and_embeddings`); exempting decay of 0.01 cannot plausibly reach the 0.0117 yardstick (§173) one 48k arm resolves. Final-LayerNorm nit not done, not funded. **Implemented behind `--no-decay-norm-bias` and `--warmup-steps` (2026-10-16, 9c3b615); measurement open.**
      The final-LayerNorm-before-pooling nit is not done (model.py).
      **Optimiser nits.** Weight decay also hits LayerNorm, biases and embeddings
      (`train.py:407`); no final LayerNorm before pooling in two encoders; warmup fixed at 500
      steps with no flag.
      Verify: a test that LayerNorm, biases and embeddings are excluded from weight decay.
- [ ] AU5.11 **Progress (2026-11-05, `docs/results/drafts/au5.11_2026-11c.md`): two Kaggle arms prepared and pre-registered, NOT launched, no checkpoint trained:** `au511-diverse-s0` (p_financial/p_tree/p_task_family 0.3, unordered categoricals, mixed missingness, DR2.4 bands, plus `heavy_tail_column_prob` 0.15, `tree_split_by: uniform`, `missing_label`) and `au511-mixture-s0` (same mixture, new flags off, to attribute the result). Kernels `scripts/kaggle/ts_au511_{diverse,mixture}_s0/`, pinned at 62a73c3 (contains 7dfd663), 48,000 tasks, seed 0, control `e1-v48-s0` (0.1975), rule §175 (0.0117, 4 of 5 folds); prediction: harmful on V4FinBench, as §180's general trunk was. Prior generation costs 6x (mixture) and 11x (diverse) the control's per batch (SMOKE-TEST on a loaded Mac, ratio only), so a session is ESTIMATED at 1.5 to 2.6 h per arm. SMOKE-TEST: the arm config trains 4 CPU steps; `tests/test_au511_arm.py` passes. **Progress (2026-11-04, `docs/results/drafts/au5.11_2026-11b.md`, 7dfd663): the three proposed flags are implemented, all off by default and byte-identical to 63500ad when off (SHA-256 over 80 tasks pinned in `tests/test_prior_diversity.py`): `heavy_tail_column_prob` (sinh(a z)/a), `tree_split_by: uniform`, `missing_label` (mnar_label). On `broader_flags` (300 tasks, MEASURED): kurtosis share above 3 rises from 0.097 to 0.151, label-informative missingness in binary tasks from 0.057 to 0.090 (difference untested), median kurtosis unchanged. No training run; the arm checkpoint is not trained.** **Progress (2026-11-04, `docs/results/drafts/au5.11_2026-11.md`, MEASURED, 300 tasks per mixture, seed 0, CPU): the "before" reproduces in the same session (shipped p_financial 1.0: label-informative missingness 11.7% of binary tasks, median excess kurtosis 4.06). The arm mixture (p_financial 0.3, p_tree 0.3, p_task_family 0.3) gives 4.7% and median -0.03; adding the existing flags `categorical_mode=unordered`, `missing_mode=mixed` and `feature_ratio_bands` gives 5.7%, median -0.03, categorical levels p95 54 (max 64) and features p50 31.5. Still open: heavy-tailed marginals (no flag), tree splits by raw std (`prior/tree.py:82-84`), Gaussian-only families; smallest code changes proposed in the draft, none implemented. The arm checkpoint is not trained.** **"Before" statistics measured with `scripts/prior_stats.py` (2026-10-16,
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
- [ ] AU5.12 **Progress (2026-11-05, `docs/results/drafts/v4_inference_kernels_2026-11.md`): the five-fold V4FinBench run of `ensemble_transforms=("rank", "squash")` with 8 members is prepared on Kaggle GPU, not pushed, not run.** Kernels `scripts/kaggle/task_v4inf_tfsq8_{f01,f23,f4}/`, pinned 62a73c3, config `configs/arms/au512-ens8-tfsq.yaml`; reference `runs/v4-lever-ens8` (§196's ens8, all five folds saved) and the default; ESTIMATED 14.4 GPU-h (8 x 0.36 h per fold, from a MEASURED one-member T4 timing). Rule (the task's Verify states none; confirmed by the owner 2026-11-05): five-fold mean AP over ens8 above +0.0117 (§173) with at least 4 of 5 folds positive, else null; prediction: null against ens8. SMOKE-TEST passed on CPU. Queued first in both Kaggle lanes (2026-11-05). **Progress (2026-11-05, `docs/results/drafts/inference_tasks_2026-11.md`): the TabArena-Lite half, each axis against the 8-member default on 27 binary datasets, MEASURED:** class shift 10 better / 17 worse (sign p 0.25, mean ROC-AUC −0.00611, Elo 716 against 756); SVD 16 components 17 / 10 (p 0.25, +0.00306, Elo 782); transforms `rank, squash` 20 / 6 / 1 (p 0.009, +0.01023, Elo 807). Transforms are a Lite gain only; open: a five-fold V4FinBench run of `ensemble_transforms=("rank", "squash")` before any default changes. Class shift and SVD are nulls on Lite. **Progress (2026-11-03, `docs/results/drafts/inference_tasks_2026-11.md`): member axes on the proxy against 8 plain members, none a candidate:** class shift 11 wins, 5 losses (p 0.21, MEASURED); label-swap twin 9/7 (p 0.80) and transforms 12/4 (p 0.077) re-read from §187's CSVs; SVD 16 components 12/4 (p 0.077, MEASURED), a lead carried by two narrow tables (wilt, banknote-authentication) that V4FinBench's 130 columns leave no room for (at most 6 components). Open: the TabArena-Lite runs, which need the fork adapter to expose these settings. **Implemented behind `ensemble_class_shift`, `ensemble_transforms` (new `squash`
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
- [ ] AU5.13 **Progress (2026-11-07): rebased again, local only.** Fork branch `fintfm-hpo-space-r2` (8ffdef1, pushed to the fork as a new branch `fork/fintfm-hpo-space-r2` on 2026-11-07, so no force-push of `fintfm-hpo-space` is needed; the worktree is removed) sits on `fintfm-audit-fixes` 86ab7af. The one conflict was two partition paths: `FINTFM_PARTITION` (86ab7af) now feeds the same `_wrap_large_n` wrapper as the `large_n`/`partition_*` hyperparameters, and asking for both is refused; device moves use `_inner_estimators`, which now also reaches the partition's template. Fork FinTFM tests 37 passed, 0 failed (all five `test_fintfm_*.py` files, run against the worktree's source), ruff clean. Not sized yet: the T+E run fits many configurations per dataset, and no TabArena-Lite default arm has a T4 timing; `task_lite_partition` (Fix 4, queued on Kaggle, partition and default arms over the 27 binary datasets) will give one. Plan agreed with the owner (2026-11-07): read the T4 timing of the default arm from `task_lite_partition`; if a full T+E does not fit Kaggle, run a reduced T+E (a fixed number of seeded draws, for example 8) recorded as its own row labelled reduced and not compared with the leaderboard's T+E rows; if even that is ESTIMATED above about 15 GPU-h, defer AU5.13 with the reason written here (most inference levers were null on V4FinBench, and TabArena is breadth only). The kernel pins `FORK_COMMIT` 8ffdef1. **Progress (2026-11-05): the search space is committed in the fork and rebased onto the current adapter.** Branch `fintfm-hpo-space` (3965939; `fork/fintfm-hpo-space` holds 3965939 as of 2026-11-07, checked against the fork; since then `fintfm-audit-fixes` has moved to 86ab7af, which adds `FINTFM_PARTITION` and an upstream merge, so the T+E run needs the branch rebased again first) sits on `fintfm-audit-fixes` be201d7, keeping `FINTFM_PARAMS`, `FINTFM_MINORITY_POSITIVE` and `FINTFM_DEVICE` beside the `large_n` partition wrapper. The fork's FinTFM tests pass against that branch's code (26, including `test_fintfm_hpo.py`), ruff clean; all 10 sampled keys are `FinancialTFMClassifier` arguments except `large_n` and `partition_*`, which the wrapper consumes. The patch file named below no longer exists. Open: the T+E run, not started. **Search space written (2026-11-02); not yet committed in the fork, and the T+E run
      is open.** The fork's `hpo.py` was not empty (it held an unmeasured space over the ensemble
      member axes); it is replaced by a seeded ConfigSpace space over the levers FINDINGS shows
      moving a score: `n_ensemble` 4/8/16 (§187, §196), `max_context` 1,000/2,000 (AU5.14,
      §187), `context_strategy` uniform/retrieval (§187, §196), `context_positive_floor`
      none/0/16 (§200), `large_n` partition with depth 1-3 and 1 or 4 bags (§169),
      `finetune_steps` 0/200 (§163), `feature_transform` rank/power (§129). Conditions and
      forbidden clauses keep every sampled config inside what fintfm accepts. The wrapper gains
      `large_n`/`partition_*` and moves a fine-tuned copy (`model_`) on `_set_device`, which it
      missed before. The keys past v0.7.1 need fintfm from source. Fork tests 10 of 10 pass
      (`tests/tabarena/models/test_fintfm_hpo.py`), ruff 0.14.0 clean. The fork could not be
      branched from this session, so the change is the patch `~/proj/tabarena-fintfm-hpo-space.patch`
      for branch `fintfm-hpo-space` off `fintfm-audit-fixes` (5cb019c).
      **Fill the TabArena search space** (`tabarena/.../models/fintfm/hpo.py` is empty)
      once AU5.12 exists; compute-heavy (11 s per 1k rows locally), so GPU only.
      Verify: a TabArena T+E run recorded as its own row, not as the default.
- [x] AU5.14 **Done (2026-10-26, `docs/results/TABARENA.md`): 702, 745 and 756 Elo at 500, 750 and 1,000, 94th of 95 each; the default stays 1,000.** **TabArena context size**: TabArena-Lite at `max_context` 500 and 750 against 1000
      (trained range ends ~921 rows).
      Verify: TabArena-Lite Elo at 500, 750 and 1,000 recorded.
- [ ] AU5.15 **Progress (2026-11-08, `docs/results/drafts/au5.15_2026-11.md`): tasks per GPU-hour is already MEASURED (2026-11-05 kernel, 1 T4, 300 steps: live 36,000, `--task-dump` 40,934; stream 41,014 from a 50-step benchmark only), so no second session is needed for the Verify's measurement clause. ESTIMATED GPU-hours at batch 8: 500k 13.9 live / 12.2 dump; 1M 27.8 / 24.4; 2M 55.6 / 48.9, which is 1.85 / 1.63 Kaggle weeks at 30 GPU-h per week, about 7 / 6 nine-hour sessions with resume. AU1.13 and AU5.2 are done. Missing: a measured cross-session resume (state writing is measured, resuming is not), a 300-step single-GPU stream run, the 48k-task control with seed replicates, the owner's go-ahead.** **Progress (2026-11-05, `docs/results/drafts/au5.15_2026-11.md`): throughput kernel run on a Kaggle T4x2 (commit 62a73c3, fp16 split, batch 8); all stages exited 0 and checkpointing worked in every 300-step run including DDP (resume not exercised). MEASURED: per_shard generation K=1/2/4 = 76.5/148/162 tasks/s on 4 vCPUs; single-GPU 300-step runs: live 0.80 s/step = 36,000 tasks per GPU-hour (checkpoints cost nothing measurable), `--task-dump` 0.70 s/step = 40,934, two-GPU DDP stream 0.50 s/step = 28,800 per GPU-hour (1.6x wall-clock, 0.8x per GPU-hour); 50-step benchmarks: live 38,503, dump 42,519, stream (2 workers) 41,014 tasks/h; against 1.90 s/step fp32 (§86) the recipe is about 2.4x faster. ESTIMATED: 1M tasks about 24.4 GPU-hours on one GPU with stream or dump (27.8 live, 34.7 DDP at 17.4 h wall), 2M double; a full dump is about 296 GB per million tasks, so the stream loader is the practical choice. The measurement clause of the Verify is met; open: §159's scoring on V4FinBench then TabArena-Lite after the full run.** **Progress (2026-11-04, `docs/results/drafts/au5.15_2026-11.md`, e1db9a5): one-session throughput kernel `scripts/kaggle/task_au515_throughput/` written and smoke-tested on CPU (SMOKE-TEST, script path only); not launched, nothing measured. It measures per_shard generation at K=1, 2, 4, single-GPU live, `--task-dump` and stream training, and checkpointed 300-step runs including two-GPU DDP, in about 25 to 30 minutes (ESTIMATED). `--task-dump` is refused with `--ddp`, so a dump run is single-GPU (§170: 37,367 tasks/h fp16 split; ESTIMATED 53.5 GPU-h for 2M tasks). Launch after setting COMMIT to a pushed commit.** **Implemented behind `scripts/dump_prior_tasks.py --stream per_shard --workers K`
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
- [ ] AU5.16 **Arm `au516-rbf-s0` drafted (2026-11-02, `docs/results/drafts/preregs_2026-11.md`): `--value-embedding rbf`, 16 fixed centres, ECDF channel left out; kernel `scripts/kaggle/ts_au516_rbf_s0/` written; not trained.** **fintfm-train flags `--value-embedding`, `--value-basis-size`, `--value-basis-learnable` and `--value-ecdf` wired (2026-10-16, 6ec5534); measurement open.**
      **Implemented behind `value_embedding` and `value_ecdf` (2026-10-16, 0a3b8c8); measurement open.** **Value embedding**: Fourier or RBF plus ECDF-rank features in place of the MLP on
      (z, missing) (LimiX-2M RaBEL, TabPFN-3.5). One module, behind a flag.
      Verify: a flag, a test that it is off by default, and five-fold AP at matched tasks.
- [ ] AU5.17 **Two arms drafted (2026-11-02, `docs/results/drafts/preregs_2026-11.md`): `au517-lenscale-s0` (`--attn-length-scale`) and `au517-muon-s0` (`--optimizer muon`), kernels `scripts/kaggle/ts_au517_lenscale_s0/` and `ts_au517_muon_s0/` written; not trained.** **fintfm-train flags `--attn-length-scale` and `--attn-length-ref` wired (2026-10-16, 6ec5534); measurement open.**
      **Implemented behind flags (2026-10-16); measurement open.** Attention half:
      `attn_length_scale` (0a3b8c8; fixed log-length scale, no learned per-query factor).
      Optimiser half: `--optimizer muon` (9c3b615). Early target inclusion already exists as `cell_labels` (task 39.2; design review 2026-10-16).
      **Optimiser and attention scaling from TabICLv2**: Muon, QASSMax (length-aware
      softmax scaling), early target inclusion; also the fix for AU4.3 if context length is the
      problem.
      Verify: each change behind a flag, scored at matched tasks.
- [x] AU5.18 **Done (2026-11-06, `docs/results/TABARENA.md`, "The imbalanced subset, every split"): Elo 1018.2 (+142.1/-332.8), 85th of 96, over the 7 covered datasets, all 9 splits (63 fits, none imputed), MEASURED; above `RF (default)` 1000.0, below `LR (default)` 1086.7; same per-dataset standing as DR3.6 (above RF on 3 of 7, LR on 2 of 7).** **Run script ready (2026-11-02, `scripts/tabarena_imbalanced_run.py`); run open.**
      Default config, every split of the 7 covered binary datasets of the subset's 15 (63 fits;
      `--dry-run` confirms the count and the 8 exclusions), Elo over those 7 only, no imputation
      of FinTFM. Cost ESTIMATED from the 2026-10-19 Lite run's split-0 times on these 7 (MEASURED
      2,010 s fit plus 879 s predict on MPS): about 7 to 10 h, and under 1 GB of process memory
      per fit by the harness's own record, which does not count MPS allocations.
      **Score the TabArena imbalanced subset** (v0.1.9), the credit-relevant slice, with
      the current checkpoint as a baseline.
      Verify: the imbalanced-subset Elo recorded in TABARENA.md with its date.
