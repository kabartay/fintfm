# Tasks

## Phase DR1 — integrity, no output may change

- [x] DR1.1 **Done (2026-10-16).** `tests/test_golden_outputs.py` against
      `tests/fixtures/golden_v4cellattn.npz` (SHA-256 per output, 7 outputs, term structure
      asserted refused) and `golden_random_init.npz` (9 CI outputs); 1-ulp perturbation
      asserted to fail every one; `scripts/regen_golden.py` rebaselines, `scripts/pre-push`
      runs it. **Golden-output fingerprint of the published checkpoint.** A test that scores
      `runs/v4-cellattn-labels.pt` (SHA-256 d231b062...) on a seeded 2,000 x 136 prior table plus
      512 queries for: default; `n_ensemble` 2 with label swap; retrieval; 3 fine-tuning steps on
      CPU; the three `transform_representation` stages; `predict_term_structure`. Exact equality,
      skipped when the checkpoint is absent, rebaselined only when the torch version changes, run
      from a local pre-push hook. Random-initialisation fingerprints of the classifier, quantile
      regressor and term structure run in CI.
      Verify: the test fails when any one of those outputs is perturbed by 1 ulp.
- [x] DR1.2 **Done (2026-10-16).** Checkpoints carry a `provenance` record (`fintfm.provenance`,
      configured and realised shares from `prior.mixture.source_shares`), `v4_protocol.json` a
      `provenance` block, dump manifests a `code_fingerprint` (override
      `--task-dump-allow-code-mismatch`); `runs/MANIFEST.json` by `scripts/runs_manifest.py`.
      Tests `test_train.py::test_two_step_checkpoint_states_its_normalised_source_shares` and
      `test_prior_levers.py::test_dump_written_before_a_prior_edit_is_refused_after_it`.
      **Provenance in every artefact.** `FinancialTFM.save` writes the prior, training and
      model configs, git SHA and dirty flag, package and torch versions, tasks seen
      (steps x batch) and the task-dump id; `load` ignores the key so outputs are unchanged. Run
      JSONs record the checkpoint SHA-256, commit and `clf.get_params()`. Task-dump manifests carry
      a hash of `prior/*.py` and refuse a mismatched dump. A committed `runs/MANIFEST.json` lists
      the checkpoints FINDINGS cites.
      Verify: a 2-step training run's checkpoint states its normalised source shares, and a dump
      written before a prior edit is refused after it.
- [x] DR1.3 **Done (2026-10-16).** `FinancialTFMClassifier.from_config` passes every
      `inference` field; the V4 protocol, out-of-time, context-sweep and retrieval-grouping
      harnesses use it, so the four ignored keys arrive. New keys `ensemble_label_swap`,
      `ensemble_feature_frac`, `finetune_clip`; `winsor_quantile` and `transform_subsample`
      became classifier arguments feeding `FeatureConditioner`. The `model` and `train`
      sections carry every `ModelConfig`/`TrainConfig` field (`d_ff: null` = 4 x `d_model`),
      the `prior` section every prior flag, and every `fintfm-train` flag but `--out` defaults
      to `None`. The sections are mirrored, not generated, with drift tests iterating
      `inspect.signature`, `fields()` and the parser's actions. Tests
      `test_config.py::test_every_inference_key_reaches_the_classifier`,
      `test_train.py::test_published_recipe_builds_the_published_checkpoints_model` and
      `test_v4_protocol.py::test_from_config_scores_exactly_as_the_old_explicit_construction`.
      **Configuration from one source.** `FinancialTFMClassifier.from_config`; generate
      `InferenceConfig`, `ModelConfigDefaults` and the `fintfm-train` flags from dataclass fields;
      drift tests iterate `inspect.signature` and `fields()`; every `fintfm-train` flag defaults to
      `None`; commit `configs/published/v4-cellattn-labels.yaml`. Fix the silently ignored keys:
      `correct_prior`, `retrieval_groups`, `retrieval_min_positive`, `prototype_minority_ratio`
      reach the V4 harness; `ensemble_label_swap`, `ensemble_feature_frac`, `finetune_clip` get
      config keys; delete `winsor_quantile` and `transform_subsample` or give them a consumer.
      Verify: a test that sets every inference key to a non-default value and asserts the built
      classifier carries it; the published recipe reproduces from the YAML alone.
- [x] DR1.4 **Done (2026-10-16).** `FinancialTFM.predict_context` (logits, term structure,
      representation) over one `pack_context` helper, used by the classifier's three paths,
      the quantile regressor and the capability probes (unpadded, `pad_features=False`, bit
      for bit as before); `predict_term_structure` now draws column identities at
      `random_state`; `context_summary_` replaces the `_ctx_y`/`_ctx_rate`/`_full_rate` reads
      in context_sweep, v4_out_of_time, bench and homecredit_submit. Tests
      `test_classifier.py::test_every_inference_caller_goes_through_predict_context`,
      `test_hazard.py::test_classifier_term_structure_draws_column_identities_at_random_state`,
      `test_model.py::test_predict_context_is_forward_on_the_packed_task`; golden fingerprints
      unchanged (no `random_state != 0` term-structure entry exists, so none was rebaselined).
      **One inference path.** `FinancialTFM.predict_context(...)` with one padding helper,
      used by `predict_proba`, `predict_term_structure`, `transform_representation`, the quantile
      regressor and the capability probes; `predict_term_structure` takes the same
      `column_id_seed` as `predict_proba`; a public `context_summary_` replaces the private
      `_ctx_y`/`_ctx_rate` reads.
      Verify: DR1.1's fingerprints unchanged except `predict_term_structure` at
      `random_state != 0`, which now matches `predict_proba`'s seed, recorded as a deliberate fix.
- [x] DR1.5 **Done (2026-10-16).** `FinancialTFMClassifier._validate_combination` (at
      `fit`) refuses fine-tuning a quantile or hazard checkpoint, class shift with label
      swap on a binary task, and SVD members with no width left; it warns on fine-tuning
      beside any ensemble axis (binary class shift named as a label swap). The binned
      regressor refuses class shift over more than two bins unless the checkpoint's
      recorded prior had `permute_classes`; the large-n wrappers validate before fitting;
      `train()`/`benchmark()` call `check_model_matches_prior`; `encode_rows` refuses a
      table wider than `max_features`. `tests/test_lever_combinations.py`: every refusal
      names both levers, and greedy pairwise covers of 15 model and 20 classifier rows
      (complete over every valid pair) pass; golden fingerprints unchanged.
      **Lever-combination validator and covering tests.** Refuse fine-tuning on a
      non-classification checkpoint; warn on any ensemble member axis during fine-tuning (class
      shift on a binary task is a label swap); raise on SVD with zero components after
      subsampling; cross-check `ModelConfig` against `PriorConfig` in `train()`. A pairwise
      covering array of about 20 model and 25 classifier lever configurations.
      Verify: each refused combination raises with a message naming both levers; every covering
      configuration runs, gives finite outputs, chunks exactly and returns valid probabilities.
- [x] DR1.6 **Done (2026-10-20).** `CLAUDE.md`'s "Count the tasks" section now gives 125,000 steps at batch 8: about 66 T4-hours (§86) or about $28 of L4 time (§127) per million tasks, ESTIMATED from the measured step times, with a note that $13 predates cell attention.
      **Correct `CLAUDE.md`'s cost figure** (owner's file): about 66 T4-hours or about $28 of
      L4 time per million tasks at the current recipe (§86, §127), not $13.
      Verify: the owner has edited it, or declined.
- [x] DR1.7 **Done (2026-10-16, f209e29).** rms_match scaling (0.2·√max(n, m)), lr shared with AdamW, weight decay 0.01, Q/K/V orthogonalised as separate blocks (Keller's write-up; nothing per head is described there), fp32 Newton-Schulz without native bf16. **Verify amended 2026-10-16:** the step-1 comparison was ill-posed, since AdamW's first step is lr·sign(g) (RMS exactly lr) while 0.2·√max(n, m) matches AdamW's steady state; the check is now update RMS within 2x of AdamW's steady-state RMS on full-rank gradients (`tests/test_muon.py`). Measured on the tiny model (SMOKE-TEST): full-rank updates 0.14-0.24·lr against AdamW's steady 0.229·lr.
      **Muon at peer settings before any use.** 0.2·√max(n, m) update scaling, shared
      lr 8e-4 to 1e-3, weight decay 0.01, Newton-Schulz in fp32 on GPUs without bf16, and the fused
      QKV projection orthogonalised per head block, as TabICLv2, LightPFN and Mitra-v2 describe.
      Verify: per-element update RMS at step 1 within 2x of AdamW's on the same batch.

## Phase DR3 — measurements that need no pretraining

- [x] DR3.1 **Done (2026-10-16, §169).** Helps: zero-shot mean AP 0.1986 → 0.2362 (+0.0375), 5 of 5 folds positive, 4 significant on a company bootstrap; AP within flagged rows +0.0387.
      **V4: partitioned contexts against uniform** (`--large-n partition`, depth 2, leaf
      minimum 5%), zero-shot, five folds, company bootstrap; report AP within flagged rows beside
      headline AP. Comparator: the zero-shot control. Rule: §163's (mean delta above the bar, 4 of 5
      folds the same sign).
      Verify: per-fold AP for both arms in FINDINGS, with the flagged-row AP.
- [ ] DR3.2 **Progress (2026-11-04, `docs/results/drafts/dr3.2_2026-11.md`): arm 1 (8 members against 1 at 2,000 rows, full panel, five folds, MEASURED) is §196: mean AP 0.1986 to 0.2044 (+0.0058), 3 of 5 folds positive, below §173's 0.0117 bar and the 4-of-5 requirement, so null under the rule; per-fold AP is in §196. Open: the `ensemble_class_shift` arm, `configs/arms/dr32-ens8-cshift.yaml`, about 35 h on CPU (ESTIMATED from §196's about 7 h per fold), not run.** **V4: 8 ensemble members against 1** at 2,000 rows, and a second arm adding
      `ensemble_class_shift`. Same rule.
      Verify: per-fold AP for each arm in FINDINGS.
- [x] DR3.3 **Done (§200): null.** Exact +0.0024 (fold SD 0.0091 vs 0.0177), floor16 +0.0055 (SD 0.0250), both below 0.0117 and split by fold; default stays null. **Implemented behind `context_positive_floor` (2026-10-16, 6917d41); measurement open.** `null` (default) is today's uniform draw; an integer k holds exactly max(k, round(max_context x pool rate)) positives, rest uniform from negatives, with the D5 correction read from the actual context rate. Arms: `configs/arms/dr33-exact.yaml` (0) and `configs/arms/dr33-floor16.yaml` (16); tests in `tests/test_classifier.py` (`*positive_floor*`) and `tests/test_large_n.py`.
      **V4: a fixed positive count in the uniform context** (exactly round(n x rate), or a
      floor of 16, with the base-rate correction). Report the fold-to-fold SD beside the mean.
      Verify: per-fold AP and SD in FINDINGS.
- [x] DR3.4 **Done (2026-11-05, §209): at equal labels the probe ties the head on V4FinBench, −0.0005 AP cross-fitted (3 of 5 folds positive, every fold within ±0.0048), so §130-§136's reading is label count, not representation; the 20,000-row probe reproduces §131 (+0.0227, 4 of 5); consequence written in §209 and RELATED_WORK.** **Prior-side control measured (§203, 2026-11-03): at equal labels the head beats the probe on 200 prior tasks, −0.0046 AP, 75 of 200; the real-data run below is still open.** **Implemented behind `--probe-rows context` (2026-10-16, 138e7c3); measurement open.** `representation_transfer` fits the probe, raw-feature and MLP arms on the classifier's `context_indices_` and keeps the 20k stratified arms under a `_stratified` suffix; default `stratified` is §130 bit for bit (`tests/test_representation_transfer.py`). Cross-fitted since (2026-10-19, dccaa0a): a context row scored against a context holding its labelled copy reads its label, so `context` represents each of 5 folds against the other 4/5 (`restrict_context`) and scores the head the same way (`own_head_crossfit`); the leaky variant is `--probe-rows context-leaky`, diagnosis only. §130's own sample shared 53-75 of its 20,000 rows (0.27-0.38%) with the context, 4-13 of its 2,128-2,189 positives (0.19-0.61%) per fold, reconstructed from the code's seeds.
      **§130's probe fitted on the 2,000 context rows only.** If the probe's advantage over
      the head falls to 0.005 AP or less, §130-§136's reading is label count, not representation.
      Verify: the result and its consequence for §130-§136 written in FINDINGS.
- [x] DR3.5 **Done (2026-11-05, `docs/results/TABARENA.md`, sections "Context size" and "Inference arms against the 8-member default"): every lever named here has its per-dataset paired differences recorded, MEASURED:** `max_context` 500 12/14/1 (p 0.85, −0.00048) and 750 14/8/5 (p 0.29, +0.00096), null, default 1,000 stays; `ensemble_class_shift` 10/17/0 (p 0.25, −0.00611), null; `ensemble_transforms=("rank", "squash")` 20/6/1 (p 0.009, +0.01023, Elo 807 against 756), a general-tabular gain that changes nothing until V4FinBench agrees, a run AU5.12 now holds; NaN against zero fill null (2026-10-19). **Progress (2026-11-05, `docs/results/drafts/inference_tasks_2026-11.md`): class shift and transforms on Lite, each against the 8-member default, MEASURED:** `ensemble_class_shift` 10 better / 17 worse of 27 (sign p 0.25, −0.00611 ROC-AUC), null; `ensemble_transforms=("rank", "squash")` 20 / 6 / 1 (p 0.009, +0.01023, Elo 807 against 756), a Lite gain that changes nothing until V4FinBench agrees (AU5.12 holds that run). Open: copy the `max_context` and these rows into TABARENA.md. **Progress (2026-11-03, `docs/results/drafts/inference_tasks_2026-11.md`): `max_context` 500 and 750 paired per dataset from the existing Lite results:** 500 better on 12, worse on 14 (p 0.85, −0.00048 ROC-AUC); 750 better on 14, worse on 8 (p 0.29, +0.00096); null, default stays; to be copied into TABARENA.md. Open: class shift and transforms on Lite (adapter override needed); on the proxy neither is a candidate. **Partly done (2026-10-19): NaN against zero fill measured, null** (`docs/results/TABARENA.md`, "NaN passed through against NaN filled with 0"): 21 of 27 binary datasets identical, NaN better on 2 of the 6 that differ, mean +0.00008 ROC-AUC, sign test p = 0.69; NaN passthrough kept. Open: `ensemble_class_shift`, `ensemble_transforms`, and `max_context` 500 and 750.
      **TabArena-Lite, same day, same fintfm source:** NaN against zero fill (fork branch
      `fintfm-audit-fixes`), then `ensemble_class_shift`, then `ensemble_transforms=("rank",
      "squash")`, then `max_context` 500 and 750 against 1,000. Paired per-dataset differences,
      not Elo, decide.
      Verify: per-dataset paired differences recorded in `docs/results/TABARENA.md`.
- [x] DR3.6 **Done (2026-10-16).** `docs/results/TABARENA.md` "The imbalanced subset": Elo
      914.1 (+127.3/-343.6), 87th of 95, over the 7 of 15 subset datasets FinTFM ran (8 are
      multiclass, regression or over 136 features), re-scored from §122's held Lite frame with
      `scripts/tabarena_subset_elo.py` (no fits; all 27 reproduce §122's 764.6); results predate
      AU1.19 (NaN filled with 0).
      **Score the TabArena imbalanced subset** from results already held (minutes).
      Verify: the subset Elo in `docs/results/TABARENA.md` with its date.
- [x] DR3.7 **Done (2026-10-16).** FINDINGS §168 states each fact with its recomputed count,
      with pointers from §35 and §147; the review's 8,171 cross-company rows (6 positives) did
      not reproduce (1,081 rows, 0 positives on the harness's 130 features) and §168 says so.
      **Disclose the V4 data facts** in the benchmark section of FINDINGS: the insolvency
      gate, positives only in final rows, base-rate drift by year, the 8,171 cross-company
      identical rows, the 130 (not 136) features after the silent bool drop, and that the harness
      imputes so the missingness mask is never exercised (correct §147's reading).
      Verify: each fact stated with its MEASURED count and a pointer from §35 and §147.

- [x] DR3.8 **Done (2026-10-16).** Pointers in FINDINGS §118 and §141; dated §166 qualifiers
      at every §116 credit citation in README.md, CLAIMS.md, LIMITATIONS.md and RELATED_WORK.md,
      on D16's §141 row in DECISIONS.md, and on the roadmap's tree-prior lines.
      **Carry §166 into the documents.** Add pointers to §118 and §141; correct the
      README's "the tree prior ships off because it hurts credit", `docs/paper/CLAIMS.md` and
      `docs/paper/LIMITATIONS.md` wherever §116 is cited as a credit loss; mark D-entries that rest
      on it.
      Verify: `grep -rn "§116" README.md docs/paper docs/design` shows the §166 qualifier at each
      hit that cites it as a credit result.

- [x] DR3.9 **Done (2026-10-16).** README, CLAIMS (Claim 6 update and Claim 16's table, now
      with the 1,000-tree LightGBM row), LIMITATIONS and CHANGELOG [Unreleased] quote §167 with
      the feature set (published features, with `operational_status`); the older tuned-booster
      figures from §69-§82 are left as historical, a different protocol.
      **Re-baseline every quoted booster figure on §167.** The field on V4FinBench is
      about 0.39-0.40 mean AP (CatBoost defaults 0.4005, LightGBM at 1,000 trees 0.3908); the
      100-tree LightGBM and XGBoost defaults are under-trained and must not be quoted as the
      field. State the feature set (with or without `operational_status`) beside every booster
      number.
      Verify: README, CLAIMS, LIMITATIONS and CHANGELOG quote §167's figures with feature set.
- [x] DR3.10 **Done (§202): fine-tuned −0.0023 AP without the column (5 of 5 folds negative, none survives Holm), zero-shot −0.0022 (§200); CatBoost loses 0.034 (§167).** **Zero-shot done (§200):** −0.0022 AP without the column (CatBoost loses 0.034, §167); fine-tuned arm open. **FinTFM without `operational_status`** (zero-shot and fine-tuned arms), paired
      against the published-feature runs, so both models' dependence on the column is known.
      Verify: per-fold AP for both arms in FINDINGS beside §167.

- [ ] DR3.11 **Progress (2026-11-05, §208, `docs/results/drafts/dr3.11_2026-11.md`): fine-tuning per leaf measured on all five folds, null.** Against §169's partition (`runs/dr31/partition`), paired company bootstrap, Holm over five folds, MEASURED: per-fold AP 0.2301, 0.2265, 0.2463, 0.2510, 0.2514 against 0.2241, 0.2262, 0.2411, 0.2370, 0.2524; mean ΔAP +0.0049 (ROC-AUC +0.0009), 4 of 5 folds positive, only fold 3 surviving Holm (p 0.002). Under §163's rule (mean above +0.0073 with at least 4 of 5 positive) this is null: the mean is below the bar. Folds 0-1 reproduce `runs/dr311_ftleaf_cmp.txt`. Open: 8 members, depth 1 and 3 with fine-tuning, bagging over 4-8 trees, TabArena-Lite. **Fine-tuning per leaf, folds 0-1 only (2026-11-02, Kaggle `task_dr311_ftleaf`, `runs/dr311_ftleaf_cmp.txt`): +0.0060 and +0.0003 AP against §169's partition, neither surviving Holm; folds 2-4 at the next quota reset before any verdict.** **Depth done (2026-10-17, §169 addendum):** depth 1, 2, 3 give 0.2347, 0.2361, 0.2357. **Follow up §169.** Partitioned contexts with fine-tuning per leaf; with 8 members;
      depth 1 and 3; bagging over 4-8 trees; and on TabArena-Lite (must not lose on numeric-only
      datasets). Each against §169's arm under §163's rule.
      Verify: per-fold AP for each arm in FINDINGS.

## Phase DR4 — an evaluation loop that does not overlap TabArena

- [x] DR4.1 **Done (2026-10-20).** `docs/results/overlap_report.json` (generated 2026-10-19T22:02Z by `python -m fintfm.evaluation.overlap --report ... --fetch`, log `runs/dr41/overlap_report.log`) compares all 45 candidates (the 9 SUITE datasets, the 35 proxy-suite tasks and the credit-g control) with all 193 references (51 TabArena-v0.1, 142 BeyondArena) by row hashes and column fingerprints: 0 suite datasets overlap, and the control overlaps TabArena 46918 credit-g, as it must. `tests/test_overlap_report.py` pins it offline (candidate set, full coverage, no overlap, control). Blind spot stated in the report: all-categorical candidates (kr-vs-kp, tic-tac-toe, dresses-sales) hash no rows. Earlier: **Partly done (2026-10-19): reopened** because the row-hash check covers only the 27 of 51 TabArena-v0.1 datasets cached locally and no BeyondArena contents, and it is a network CLI check, not the test the Verify names; the selection guard is complete. Remaining: fetch the 24 uncached TabArena datasets and BeyondArena's contents and pin the overlap result in a test. Earlier the same day: Row-hash check (`overlap.content_fingerprint`, `openml_breadth --row-hash`): 0 of 9 SUITE datasets overlap the 27 of 51 TabArena-v0.1 datasets cached locally (BeyondArena not comparable by contents, no ids); `SELECTION_FORBIDDEN` guard in pilot_compare, label_symmetry_probe, openml_breadth and the proxy runner (`tests/test_selection_guard.py`).
      **Partly done (2026-10-16), left open.** The six datasets are out of `SUITE`;
      `tests/test_openml_breadth_overlap.py` checks OpenML id and name against TabArena-v0.1 and
      name against BeyondArena (which lists no ids); §165 carries a dated note. Open: the row-hash
      comparison the Verify names, which needs the datasets' contents, and a rule against choosing
      arms on the Polish and Taiwanese panels.
      **Remove the TabArena overlap from the decision loop.** Drop bank-marketing, churn,
      blood-transfusion, credit-g, qsar-biodeg and diabetes from `openml_breadth.py`'s SUITE, and
      stop choosing arms on the Polish and Taiwanese panels; mark §165 as measured on TabArena
      datasets.
      Verify: a test that the SUITE shares no OpenML id, name or row hash with TabArena-v0.1 or
      BeyondArena.
- [ ] DR4.2 **Progress (2026-11-04, spearman-verify-n6, `docs/results/drafts/dr4.2_2026-11b.md`): the Verify was computed on 6 distinct checkpoints (published, e1-v480-s0, priorctl, treeclean, treeprior, priorwide; 16 tasks x 3 splits; Lite Elo from the fork's own `compare`, reproducing 756/763/842): proxy Elo against Lite Elo rho = +0.60 (exact two-sided p 0.24, bootstrap 95% [-0.64, +1.00]), mean AP rho +0.03, mean ROC-AUC rho -0.31, all MEASURED. Threshold 0.8 NOT met and not decidable at n = 6; the proxy stays unvalidated and decides nothing. Open: proxy-score priorctl-s1, treeclean-s1 and widerprior for n = 9, and add checkpoints far apart on both scores to widen the spread.** **Progress (2026-11-04, `docs/results/drafts/dr4.2_2026-11.md`): the Spearman Verify is not computable yet; 2 distinct checkpoints (published, `e1-v480-s0`) have both a full proxy run (12 exist) and a TabArena-Lite Elo (756 and 842 on record); 7 older 3.4 MB checkpoints have Lite results on disk but no proxy run. Cheapest route to n = 6: proxy-score 4 of them (ESTIMATED 0.2 to 0.8 h of CPU) and compute their Lite Elo from existing results with the fork's leaderboard code.** **Partly done (2026-10-19): runner, Elo, sign test and row hash built, left open.**
      `proxy_suite run` scores FinTFM (`from_config`) and cached RF/ExtraTrees/Linear/KNN/LightGBM
      on OpenML's own splits (repeat 0, folds 0-2; 34 tasks are 1x10-fold CV, energy_efficiency
      10x10) into a per-split CSV; `compare A B` gives Bradley-Terry Elo (RF = 1000, task
      bootstrap) and a per-task sign test (`tests/test_proxy_suite_runner.py`). `--row-hash`:
      0 of 35 suite datasets overlap the 27 TabArena-v0.1 datasets cached locally; the positive
      control (OpenML credit-g 31 against TabArena 46918) reads row and column share 1.000.
      SMOKE-TEST only: 2 tasks (breast-w, energy_efficiency) on `v4-cellattn-labels.pt`. Open:
      the Spearman Verify, which needs full runs of 6 or more checkpoints.
      **Partly done (2026-10-16), licences verified at source, left open.** Owner
      decision 2026-10-16: OpenML's `Public` does not admit a task; each of the 77 candidates
      left by the overlap filters was traced to its original source and that page's licence
      recorded with URL, text as stated, date, verdict and reason (`SOURCE_LICENCES`). 53 pass
      (49 UCI pages stating CC BY 4.0, pc1 on Zenodo CC BY 4.0, numerai28.6 and kings_county on
      Kaggle CC0, Fashion-MNIST MIT); 24 are excluded (no licence stated or page unreachable,
      Delve and socmob non-commercial, an R package's GPL, Kaggle CC0 over data scraped from
      sofifa.com and baseball-reference.com). Suite: 18 binary / 8 multiclass / 9 regression
      (35), of which 16 / 7 / 9 (32) are within the 136-feature cap; five of the eight
      multiclass tasks are views of UCI Multiple Features. `tests/test_proxy_suite.py` checks
      every suite task has an included source entry and no rejected one is eligible. Open: the
      row hash, the runner, the sign test and the Spearman Verify.
      Earlier the same day: built, not run:
      `fintfm.evaluation.proxy_suite` records all 107 CC18/CTR23 tasks with OpenML licences
      (re-fetched, 0 changed) and excludes by TabArena id, name against both arenas, a
      same-source list and licence; `--dry-run` prints every exclusion, `tests/test_proxy_suite.py`
      checks the suite against both lists. Only 10 regression tasks survive: all 72 CC18 datasets
      carry OpenML's default `Public`, a research-use grant, so no binary or multiclass task
      passes until licences are verified at source (owner decision). Open: that, the row hash,
      the runner, the sign test and the Spearman Verify.
      **A proxy Elo suite:** about 45 OpenML-CC18/CTR23 tasks outside TabArena and
      BeyondArena (excluded by id, name and row hash; CC-BY, CC0 or public licences only), 30
      binary / 8 multiclass / 13 regression, three splits, run through TabArena's own custom-task
      pipeline against cached RF, ExtraTrees, Linear, KNN and GBDT baselines; Elo anchored at RF
      (default) = 1000, plus a paired sign test against a reference checkpoint.
      Verify: Spearman 0.8 or higher against Lite Elo across 6 or more existing checkpoints
      before the proxy is used to decide anything.
- [ ] DR4.3 **Progress (2026-11-05): the owner admitted Estonia and FDIC** (`include=True` in `credit_suite.DATASETS`): Estonia with its survivor-conditioned label stated in the licence reason, so a score there is not quoted as bankruptcy-prediction skill; FDIC as a named exception to DR4.2's stated-at-source rule, public domain on statute (17 U.S.C. section 105), FinTFM only (SEC-filer overlap at least 6.72% of rows). Open: their FinTFM, logistic-regression and LightGBM runs at full scale. **Progress (2026-11-04, `docs/results/drafts/dr4.3_2026-11b.md`): overlap and selection checks, MEASURED.** FDIC SEC-filer overlap against the 2023q1-q4 EDGAR filer list only (`www.sec.gov` returned HTTP 403): the exact-name rule flags 49,318 of 733,528 rows (6.72%) and 20 of 2,303 positive rows (0.87%), a floor since most 2008-2012 failures' parents are absent from a 2023 list; a looser rule flags 11.25% of rows and 5.82% of positive rows; FDIC stays FinTFM-only. Row hashes (50,000-row samples) of FDIC and Estonia against 51 TabArena-v0.1 datasets and V4FinBench: row and column share 0.0000, positive controls 1.0. Estonia: positives file fuller, not emptier, reports (emptiness AUC 0.2513; low total assets 0.168), and without totals, signs and headcount AUC stays 0.82-0.85 while AP falls (LR 0.4726 to 0.2659, LightGBM 0.6314 to 0.4729; SMOKE-TEST scale), so its ROC-AUC near 0.90 is a population-mix effect consistent with the survivor conditioning, not quotable as bankruptcy-prediction skill. Both datasets stay not admitted. **Progress (2026-11-04, `docs/results/drafts/dr4.3-fdic_2026-11.md`): FDIC bank failures built, not admitted.** `fintfm.evaluation.credit_fdic` (BankFind API, 733,528 bank-quarter rows from 2000, 2,303 positive rows = 0.314%, MEASURED) with a 4-quarter failure label, 35-day reporting lag and 401-day embargo; evaluation-only. The API page and FDIC website policies state no licence, so public domain rests on 17 U.S.C. §105 and admission waits on an owner decision. FinTFM-only until the SEC-EDGAR filer overlap check is done. 80% of 581 failures since 2000 fall in 2008-2012, so shared split 0 tests the crisis tail (873 positive rows) and splits 1 and 2 hold 109 and 66 test positives; `crisis_split` tests 2008-2011 (1,771). SMOKE-TEST of logistic regression and LightGBM on 5,000 rows per split only; no FinTFM number yet. **Progress (2026-11-04, `docs/results/drafts/dr4.3-estonia_2026-11.md`): Estonian register built, gated off.** The licence passes (RIK links CC BY 4.0, read 2026-11-04), but the open files hold only firms currently registered, so the label (bankruptcy within 365 days of filing, 1,251,575 firm-years, 260 positives, MEASURED) is survivor-conditioned: firms bankrupt and then deleted are absent, and 129,890 deleted-firm reports have no label. Registered as `estonia` with include=False and EVAL_ONLY; SMOKE-TEST on 5,000 enriched rows (5.20% base rate): logistic regression 0.4726 and LightGBM 0.6314 mean AP, not a result. Owner decides whether to admit it; a default label needs deletion dates and reasons the open data lacks. **Measured on two datasets (2026-10-21, §174, §177):** ULB (FinTFM 0.6830 mean AP, logistic regression 0.7489) and SBA 7(a), added 2026-10-20 as evaluation-only (FinTFM 0.1095, logistic regression 0.1295, LightGBM 0.3861); FinTFM is below logistic regression on both. Open: Estonian register (label to build) and FDIC failures, and a paired comparison once three or more datasets are in. **Partly done (2026-10-19, ebc8112, 40ad109): built, not run.** `fintfm.evaluation.credit_suite` has loaders, three fixed splits per dataset (PPDai out-of-time on `loan_time`; Bondora out-of-time on `LoanDate`, borrower-grouped, 366-day embargo; ULB stratified, its `Time` being seconds into a two-day capture), FinTFM, logistic regression and LightGBM, and one CSV row per dataset, split and model (`tests/test_credit_suite.py`). All three are `EVAL_ONLY` in `real_panel.py` and `data_candidates.csv`, so the FinTFM-R sampler refuses them. Blocked: re-read at source on 2026-10-19, no licence passes DR4.2's rule (PPDai's CC BY 4.0 is a researcher's figshare grant, not the platform's; Bondora's CC0 is a Kaggle uploader's, Bondora's own page states none; ULB is ODbL-1.0 plus DbCL-1.0, and ODbL's share-alike is outside the admitted list), so `run()` refuses all three until the owner sets `include=True`. Checkpoints from the §157 FinTFM-R pilot arms saw these panels and are not held out here. Owner decision (2026-10-19): ULB admitted, since its grant is the collector's and scores are ODbL Produced Works, public with `ODBL_NOTICE`; PPDai and Bondora stay out. Open: the Verify run on ULB with the published checkpoint (estimated 2.4 to 5.5 hours of CPU, not measured end to end), and licence-clean credit-default datasets to replace PPDai and Bondora, since ULB is card fraud, not default. Search of 2026-10-19 (licences read at source, no data downloaded): SBA 7(a) and 504 FOIA (U.S. Government Works, public domain; `ApprovalDate`, no borrower id; drop `ChargeOffDate`, `GrossChargeOffAmount`, `PaidInFullDate`, which give the label away; open loans are EXEMPT) and the FDIC failures list (public domain; financials state no licence) pass; the Estonian e-Business Register (CC BY 4.0 from RIK) passes but needs a firm-year bankruptcy label and a check that deleted firms are kept; a Moroccan household survey passes but is 2,000 rows without a date. Unverified: CDFI TLR (no licence at source) and the Peruvian FMOD set on Zenodo (researchers' grant over a lender's data). Next: download SBA 7(a) to measure rows and charge-offs, then the row-hash check before admitting it.
      **A credit suite outside V4FinBench:** PPDai (CC BY 4.0), Bondora (CC0 mirror), ULB
      fraud (DbCL-1.0), three splits, evaluation only (which removes PPDai from FinTFM-R's training
      queue).
      Verify: the suite runs end to end on the published checkpoint with runtimes recorded.
- [x] DR4.4 **Done (2026-10-19, §173).** Pretraining yardstick 0.0117 mean AP (per-fold seed SD 0.0087, 15 df, three fp32 replicates plus E1's fp16 pair); fine-tuning noise SD 0.0008 (§164).
      **The noise yardstick from at least three matched replicates** (AU2.9's method), with
      separate pretraining and fine-tuning values and Holm across arms.
      Verify: FINDINGS states both yardsticks with source runs and degrees of freedom.

## Phase DR5 — throughput, then a volume test

- [x] DR5.1 **Done (2026-10-17, §170).** No single-GPU setting clears 1.3x (split 1.21x, split plus batch width 1.29x); DDP on a T4x2 is about 1.9x; fp16 runs out of memory at batch 4 where fp32 needs 6.2 GB (DR5.5).
      **E0 throughput.** `fintfm-train --benchmark-steps 50` at the reference recipe on a
      Kaggle T4 (fp32, `--amp fp16`, `--attn-impl split`) and an HF L4 (fp32, `--amp bf16`, split),
      with rows pinned at 1,024 and mixed; prior generation on each host's CPUs at 1 and 2
      workers. Adopt a setting only if outputs agree on a fixed batch and tasks per GPU-hour rise
      by 1.3x or more.
      Verify: a FINDINGS entry with tasks per GPU-hour for each setting and host.
- [x] DR5.5 **Done (2026-10-17, §170 addendum).** fp16 masked 2.38x and split 2.62x over fp32 on a T4, at lower memory; logits agree within 9.4e-4. **Partly done (2026-10-17): fix landed; GPU measurement open.** The cell
      stages' heads are 12 wide (`d_cell` 48 over 4 heads). On a T4 the memory-efficient
      kernel needs a multiple of 8 in fp16 but not in fp32, and the masks keep flash out, so
      fp16 ran on the math kernel, which upcasts to fp32 and stores every score matrix: the
      1.13 GiB is the column encoder's (4,096 x 4, 136, 136) fp32 scores, the 256 MiB the
      row-within-feature stage's (4 x 4 x 4, 1,024, 1,024). `aligned_attention` zero-pads
      half-precision heads to 8 and the cell stages run through `_prenorm_stack` in fp16 or
      bf16; fp32 is untouched (`tests/test_golden_outputs.py` identical). Tests in
      `tests/test_model.py`; the CUDA one, `test_fp16_peak_memory_is_at_most_fp32s_on_cuda`,
      skips here. Open: E0 v4 (`scripts/kaggle/task_e0_throughput`, `COMMIT` placeholder)
      on a T4. **Fix fp16's attention memory.** Under `--amp fp16` every arm ran out of memory
      in `multi_head_attention_forward`, even at batch 4 where fp32 peaks at 6.2 GB (§170).
      Find which call falls back to the math kernel (mask dtype or form under autocast,
      `need_weights`, or the split path's padded head dimension), fix it, and re-run E0's fp16
      arms.
      Verify: a CUDA test asserting fp16 peak memory at or below fp32's on a fixed batch, and
      fp16 tasks per hour measured on a T4 against §170's fp32 rows.
- [x] DR5.2 **Done (2026-10-21): the CUDA/NCCL run is E1 (§171).** All three finished E1 arms trained with `--loader stream --ddp` on a Kaggle T4x2 (`ddp: 2 processes, 8 tasks per step` in each `train.log`); `e1-v480-s0` ran 60,000 steps with all 60 step checkpoints written in about 31,870 s, about 0.53 s per step at global batch 8 in fp16, and scored to completion (kernel total 34,012 s). Earlier: **Implemented behind `--loader stream` and `--ddp` (2026-10-16, 96c1e39); measurement open.**
      `src/fintfm/modeling/loader.py`; the Verify tests pass (`tests/test_loader.py`: stream
      batch i equals a `per_shard`, `shard_size=1` dump's batch i; one and two workers give
      the same stream; a two-process gloo run averages the ranks' gradients and resumes
      exactly). Open: a CUDA/NCCL run on a Kaggle T4x2, which nothing here has exercised.
      **A streaming loader:** batch i drawn from `default_rng([seed, i])` in an
      IterableDataset with workers and prefetch, so resume needs no RNG state and DDP ranks take
      disjoint indices; dumps kept only for byte-identity checks. Use both GPUs of a Kaggle T4x2.
      Verify: a test that the loader's batch i equals the dump's batch i, and that two workers
      give the same stream as one.
- [x] DR5.3 **Done (2026-10-24, §188): null on V4FinBench at both seeds** (Δ -0.0109 and -0.0103, mean -0.0106 against the 0.0117 yardstick); positive on the unvalidated proxy at both seeds (+168, +186 Elo), which §171 did not make decisive. No 2M run at this size and prior. Earlier: **Interim (2026-10-21, §176; 2026-10-23, §181): seed 0 is null on V4FinBench (Δ_0 = -0.0109) and positive on the proxy (+168 Elo, 14 of 16 tasks, p = 0.004), so seed 1 runs** (launched after the quota reset).
      **E1: does volume help?** 48,000 against 480,000 tasks, seeds 0 and 1, one pinned
      commit, everything else fixed, the 48k arm retrained rather than reused (§158). Volume
      helps if the 480k arm's five-fold mean V4FinBench AP is positive at both seeds and their
      average exceeds the DR4.4 pretraining yardstick; hurts under the mirror; otherwise null.
      Secondary: DR4.2 proxy and a fixed 512-task synthetic set. Pre-register the outcome actions
      (2M run, mixed-prior 2M run, or neither) before launching.
      Verify: the pre-registration committed before the first arm trains; the verdict in FINDINGS.
- [x] DR5.4 **Done (2026-11-05, closed as superseded by the owner on the recommendation below).** **Recommended for closing as superseded (2026-11-02, `docs/results/drafts/preregs_2026-11.md`): the Verify test passes; the open 2M-task branched run is unfunded by §188, and §202 closed scale at this size.** **Implemented behind `--schedule wsd` and `--eval-set-seed` (2026-10-16, 96c1e39); measurement open.**
      A wsd run keeps `<out>.stable.state` at its last stable step; the Verify test passes
      (`tests/test_train.py::test_wsd_stable_state_resumes_with_a_longer_horizon_to_the_
      uninterrupted_weights`, 12 -> 20 steps, equal weights). Open: a 2M-task run branched
      to a longer horizon.
      **An extensible schedule:** warmup-stable-decay with decay branches, so a 2M-task
      run can continue to 10M; a fixed held-out synthetic set for in-training evaluation instead
      of fresh tasks per evaluation.
      Verify: a test that resuming a stable-phase checkpoint with a longer horizon reproduces the
      uninterrupted run's weights.

## Phase DR2 — prior v2

- [x] DR2.1 **Done (2026-10-16).** `PriorConfig.source_weights` (`prior.source_weights`,
      `fintfm-train --source-weights tree=0.25,scm=0.4,...`): one `rng.random()` per task over
      the normalised weights, unknown names, inert `p_*`/`real_panels` and timeless sources
      under `n_horizons` refused; `None` (default) keeps the sequential stream.
      `source_shares` records `mode` with configured and realised shares. Tests
      `test_mixture.py::test_weighted_draw_realises_the_normalised_weights_at_10000_draws`,
      `::test_weighted_shares_agree_at_reuse_one_and_four`, the stored fingerprints in
      `test_mixture.py` and `test_prior_levers.py`, and
      `test_train.py::test_source_weights_flag_trains_and_the_checkpoint_states_the_weighted_shares`.
      **Normalised source weights.** Replace the sequential source draw with one
      categorical draw over normalised weights, behind a flag that keeps today's stream when
      off; record the realised shares in every checkpoint (DR1.2).
      Verify: realised shares within binomial error of the weights at 10,000 draws; default
      stream byte-identical.
- [x] DR2.2 **Done (2026-10-18).** `PriorConfig.financial_sign_flip_prob`
      (`prior.financial_sign_flip_prob`, `--financial-sign-flip-prob`, default 0.0, pinned at
      0.0 in the published recipe): per task each exposed ratio column (ratio family, named
      ratios, the gated ratio and copies of them) is negated with this probability, drawn
      last, so the label, gate and NaN pattern are the unflipped task's. MEASURED
      (`scripts/prior_stats.py --checks`' financial_gated checks, 300 tasks, gate probability
      1, flip 0.5): worst column `net_profit/revenue` at 61.5% same-direction over 96
      significant tasks (`long_term_debt/equity` 58.5% over 65), so every column with >= 50
      significant tasks is at or below 75%; tree AP 0.6052 against rank-logistic 0.5017
      (+0.1035, tree wins 64.8% of 264 scorable tasks). The flip shifts the stream of later
      tasks, so that is a different 300-task draw; on the recorded draw (which reproduces
      87.3% and 0.6304/0.4647 at flip 0) negating the ratios post hoc gives worst 60.7%
      (`margin`, 56 tasks) and tree 0.6334 against 0.4651 (+0.1683). The inference `rank`
      conditioner is odd under negation to within 3.4e-4 on ratio columns (float32 rounding
      in the normal tail; up to 0.0093 on tied non-ratio columns such as `sector`), with
      identical NaNs, and the model's context z-score is exactly odd; `power` (Yeo-Johnson)
      is not symmetric, up to 1.0 on a few cells. Default stream byte-identical (SHA-256 of
      the default and gated financial streams unchanged; `test_prior_v2.py`).
      **Partly done (2026-10-18).** Source `"gated"` (`prior/gated.py`, via
      `source_weights` only) and `PriorConfig.financial_gate_prob` (`prior.financial_gate_prob`,
      `--financial-gate-prob`, default 0.0): gate drawn from negative equity, negative working
      capital, coverage < 1 or a random ratio threshold, economic side on a coin, driver score
      inside, LogU(0.002, 0.05) leak share outside; the gated ratio is always exposed. MEASURED
      (`scripts/prior_stats.py --checks`, 300 tasks, gate probability 1): tree AP 0.6304 against
      rank-logistic 0.4647 (+0.1657, tree wins 70.7% of 266 scorable tasks; ungated financial
      -0.0339). **Open:** `long_term_debt/equity` points the same way in 87.3% of the 71 tasks
      where it is significant (Spearman p < 0.01); every other column is at or below 63.9%. An
      exclusion gate (81%, `equity/total_assets`) and a thin-equity tail (83%) were measured and
      are worse; see `_GATE_ECONOMIC_PROB`. Needs an owner decision on the equity axis. Default
      stream unchanged (`test_prior_v2.py`, `test_mixture.py`, `test_prior_levers.py`).
      **A gated-label task source** (a threshold on 1-2 columns, then a score inside the
      gate) and a gated variant of the financial label (equity < 0, working capital < 0,
      coverage < 1 or a random ratio threshold, with a small leakage rate outside the gate),
      justified from balance-sheet insolvency tests and randomised so no global rule exists
      (§47).
      Verify: on 300 sampled tasks a tree beats rank-logistic regression on AP for the gated
      financial source, and no column points the same way in more than 75% of tasks.
- [x] DR2.3 **Done (2026-10-18).** `prior/knobs.py`, each off by default (`prior.*` and the
      `fintfm-train` flag of the same name): `feature_ratio_bands`, `p_binary`,
      `imbalance_draws`, `categorical_mode=unordered`, `missing_mode=mixed`,
      `tie_column_prob`, `irrelevant_frac_max`, `duplicate_frac_max`, `label_flip_max`, on
      SCM, regression, tree, task families and gated (width bands on financial too).
      MEASURED (`scripts/prior_stats.py --checks`, 300 tasks per source): every distribution
      check passes; v2 generation 0.126 s per batch of 8 at 256/512/1024 rows and 136
      features (default 0.094 s, load average 318). Default stream unchanged; tests in
      `test_prior_v2.py`.
      **Knob distributions:** feature-to-row ratio bands, binary probability 0.7, explicit
      imbalance draws, unordered and high-cardinality categoricals with Zipf frequencies,
      MCAR/MAR/MNAR missingness, ties and zero inflation, irrelevant and duplicated columns,
      label noise (the review's spec, §1.5 and DR2 in `DESIGN_REVIEW_2026-10-16.md`).
      Verify: `scripts/prior_stats.py` reports each distribution within its target on 300 tasks
      per source; generation stays at or below 0.25 s per batch.
- [x] DR2.4 **Done (2026-10-24, §180 verdict harmful on V4FinBench for all four; §191 on the proxy: trunk arms best, (c) 764 Elo, level with KNN).** Owner decision open: one checkpoint or two. Earlier: **Pre-registered (2026-10-23, §180); kernels `scripts/kaggle/ts_dr24_{a,b,c,d}_s0/` pinned at 5e0fad0, launch at the weekly quota reset.** `--new-prior-phase` (5e0fad0) makes the trunk-then-credit curriculum a recorded two-phase resume.
      **Prior v2 A/B at fixed tasks:** financial v2 alone, general trunk alone, trunk then
      credit, and interleaved, 48k tasks each, matched `max_classes`, two seeds for any arm that
      passes. V4FinBench first, then TabArena-Lite.
      Verify: per-arm per-fold AP and Lite paired differences in FINDINGS under DR4.4's yardstick.

## Phase DR6 — architecture v2

- [x] DR6.1 **Done (2026-11-05, closed by the owner on the recommendation below).** **Recommended for closing (2026-11-02, `docs/results/drafts/preregs_2026-11.md`): the Verify is met (2d22ba9's identical-output test; step time in §170, 1.29x against split alone's 1.21x at fp32).** Not added to the 2026-11 arms, to keep each one lever from its control. **Implemented behind `pad_to_batch_width` (2026-10-16, 2d22ba9); measurement open.**
      Column ids are drawn at 136 and sliced when on, so a task's outputs match full padding
      (max abs logit gap 3.6e-7, eval, fixed `column_id_seed`) and a widest-136 batch is byte
      for byte the same. MEASURED on a 300-task sample of the reference recipe at batch 8: the
      widest task averages 124.3 of 136 features, so it removes 8.3% of cell tokens (15.1% of
      column-attention pairs), not half; 46.4% only at batch 1. CPU step time at load 33-72 on
      16 cores, 3 steps per arm, two rounds: 28.6 s and 25.6 s full against 28.4 s and 21.5 s
      narrow, a difference inside the noise (those steps were 4.2% narrower).
      **Per-batch feature width** (pad to the batch's widest task, not 136), removing about
      half of all cell tokens from financial tasks.
      Verify: identical outputs on a fixed batch whose widest task is 136; step time measured.
- [x] DR6.2 **Done (2026-10-21, §178): harmful at screening under §175's rule, not adopted** (-0.0186 mean AP, 5 of 5 folds negative; -0.0089 against the other 48k seed, so read as no gain), 13% faster to train. **Implemented behind `cell_row_attn: induced` (2026-10-18, 6916a89); measurement open.**
      `n_inducing` (default 64) learned vectors per layer attend to each feature's context
      cells only, then every cell attends to the k summaries (Set Transformer ISAB, context-only);
      default `full` is today's stage, unchanged (Phase 5 fingerprint, golden outputs incl. the
      published checkpoint). Tests: no query reaches another row, query and feature chunking
      exact, no cell-width call is rows x rows, padded columns ignored, fp16 head widths.
      MEASURED on CPU (4 threads, load 135-385 on 16 cores, batch 1, 136 features, recipe
      widths, 3/4 context, mean of 2 steps after a warm-up), forward+backward s and peak RSS GB
      for induced / full split / full masked: N 512: 1.74 / 2.01 / 2.19 s, 1.31 / 1.34 / 1.24 GB;
      N 1,024: 2.75 / 4.94 / 5.06 s, 2.12 / 2.29 / 2.01 GB; N 2,048: 5.145 / 10.05 / 13.015 s,
      3.66 / 3.98 / 3.76 GB. CPU's fused attention never materialises the N x N scores, so peak
      memory is linear in N for all three here; the quadratic memory of §79 is the CUDA math
      path and needs measuring there. 929,714 parameters against 885,650.
      **Inducing-point, context-only cell stage** behind a flag (gather from labelled
      context cells through k inducing points), linear in rows.
      Verify: at matched N up to 1,024 it is within the yardstick of full attention; memory at
      N = 4,096 measured.
- [x] DR6.3 **Done (2026-10-21, §178): harmful at screening, not adopted** (-0.0261 mean AP, 5 of 5 folds negative and Holm-significant; -0.0165 against the other 48k seed). **Implemented behind `readout: cls` and `head_mlp` (2026-10-18, 6916a89); measurement
      open.** `n_readout` (default 4) learned tokens join each row's column attention: tokens
      and unpadded cells attend to each other, padded cells stay masked as keys, the tokens skip
      the row-within-feature stage and carry their state between blocks, and their concatenated
      outputs are the row vector. `head_mlp` is d_model -> d_ff -> out. Defaults `pool` and
      false are today's model. Parameters at the recipe: 885,650 pool; 898,130 cls K=4; 922,898
      cls K=8; 952,466 pool + head_mlp; 964,946 cls K=4 + head_mlp.
      **Readout tokens that take part in feature attention** (re-specify AU5.3), feeding a
      2-layer head.
      Verify: §137's parity probe at k = 2 rises above 0.60 AUC (chance today).
- [ ] DR6.4 **Progress (2026-11-07, `docs/results/drafts/dr6.4_2026-11.md`): `scripts/dr64_arm_verdict.py` checks a fetched arm, writes its V4FinBench scoring command, applies the §175 rule against the same-seed control and composes the verdicts under the DR6.4 rule; tested on synthetic outputs only, no arm has a result. The two AU5.9 kernels pass `--amp fp16` with a hazard head, which `_autocast` refuses.** **Progress (2026-11-05, `docs/results/drafts/dr6.4_2026-11.md`, rule fixed, no bundle yet): no lever qualifies.** Every screened pretraining lever is null or harmful at its one seed and none has a seed-1 arm (DR6.2 −0.0186 and DR6.3 −0.0261, §178; DR2.4 a to d −0.0212, −0.1262, −0.0546, −0.0358, §191; volume −0.0109 and −0.0103, §188; depth −0.0047, §195; width +0.0003, §202); the seven 2026-11 arms (AU5.17 lenscale and muon, AU5.4 ctx2k, AU5.3 cell2, AU5.16 rbf, AU5.9 hazard, DR8.3 width512) have no result. Composition rule, fixed before results: a lever enters if promising (above 0.0117, at least 4 of 5 folds) at seed 0 against e1-v48-s0 and at seed 1 against e1-v48-s1, harmful at neither, composable, judged on V4FinBench; two or more give the bundle, one a single-lever test, none no bundle. Arms dr64-v1 and dr64-bundle at seeds 0 and 1, 240,000 tasks each (a matched v1 must be trained); about 20 to 22 GPU-h ESTIMATED from 0.525 s per step (§171) if the bundle keeps the control's step time, more with cell2 or ctx2k. Built when every pending arm and its seed-1 replicate has reported. **Deferred (2026-11-02, `docs/results/drafts/preregs_2026-11.md`): no bundle exists, since §178 excluded DR6.2 and DR6.3; to be re-specified as the 2026-11 screening levers promising at both seeds, about 22 GPU-h ESTIMATED for four 240k arms.** **v2 bundle against v1** at matched 240,000 tasks, two seeds: adopt if five-fold mean
      AP is at least +0.010 with 4 of 5 folds positive at both seeds; then leave-one-out on DR6.2
      and DR6.3 only.
      Verify: the verdict in FINDINGS.

## Phase DR7 — TabArena

- [x] DR7.1 **Done (2026-10-16).** ROADMAP Phase E and TABARENA_BAR set the bar at about
      1,010 Elo with the pool's baseline rows, each verified in the fork's
      `fintfm_lite_tafintfm/tabarena_leaderboard.csv` (FinTFM 764.6 there).
      **Correct the resubmission bar** in `docs/roadmap/ROADMAP.md` and
      `docs/roadmap/TABARENA_BAR.md`: about 1,010 Elo in the §122 pool (RF (default) parity), not
      Linear (default) at 936.
      Verify: both files state the new bar with the pool's baseline rows.
- [x] DR7.2 **Done: closed without asking (owner decision, 2026-10-21).** §122's decline had two grounds, and the coverage question only touches one: the binary model ranked 94 of 95, below every linear and nearest-neighbour baseline, and DR3.5 measured it at 756 Elo on 2026-10-19 against a resubmission bar of about 1,010 (`docs/roadmap/TABARENA_BAR.md`). Until the model clears that bar, the coverage precedent decides nothing, so no message goes to the maintainers; competitiveness comes first, and the question is reopened only with a number at the bar.
      **Ask the maintainers** whether LightPFN's classification-only precedent (admitted
      2026-10-11 with regression imputed) applies, before building regression coverage.
      Verify: their answer recorded in `docs/results/TABARENA.md`.
- [x] DR7.3 **Done (2026-10-23, §181):** `v4-multiclass.pt` scores 836 proxy Elo (6 of 8 tasks; RF 1,000), the binary checkpoint 561 (16 of 18). **Owner decision (2026-10-21): FinTFM covers binary and multiclass.** Baseline the
      multiclass checkpoint where arms may be chosen: `runs/v4-multiclass.pt` (max_classes 10,
      cell labels, one cell block) on the proxy suite's 8 multiclass tasks (DR4.2 runner), beside
      the binary checkpoint's proxy Elo, so every later multiclass arm has a control that is not
      TabArena. §121's TabArena rank (93.4 of 95 over 7 datasets) is reported, never selected on.
      Verify: the proxy Elo and per-task scores for `v4-multiclass.pt` recorded in FINDINGS.
- [x] DR7.4 **Done (§202). Volume step null (§202): 4 wins, 2 losses per task, 12 of 18 splits (p 0.24); 48,000 tasks stays the multiclass volume.** **Seed 1 a candidate too (2026-10-26, §197): 16 of 18 splits, 5 of 6 tasks; permutation enters the multiclass recipe. Next: the volume step, pre-registered as §198 with its kernel, at the next quota reset.** **TabArena-Lite, reporting only (2026-10-26, TABARENA.md): 306 Elo, 94th of 95 on 7 datasets, no gain.** **Seed 0 a candidate (2026-10-24, §186): 6 of 6 tasks, 17 of 18 splits, 994 proxy Elo against the control's 808.** Next: the seed-1 replicate at the next quota reset, then the volume step.
      **A multiclass arm at the current recipe:** E1's 48k recipe (§171) with
      `max_classes 10`, the generic SCM mixed in as `v4-multiclass.pt` was trained, and
      `--permute-classes` (AU5.5), one seed, against DR7.3's control on the proxy suite's
      multiclass tasks by paired sign test; then at the volume E1 recommends, if it recommends
      more. Blocked by DR7.3 and GPU quota.
      Verify: the arm's proxy result and the decision in FINDINGS, pre-registered before training.
- [ ] DR7.5 **Progress (2026-11-07, `docs/results/drafts/dr8.3_2026-11.md` section 7): the multiclass 512 arm is prepared, not launched.** `configs/arms/dr75-width512-mc.yaml` is DR8.3's arm plus DR7.4's four multiclass keys (`max_classes` 10, `p_financial` 0.5, `scm_legacy`, `permute_classes`), kernel `scripts/kaggle/ts_dr75_width512_mc_s0/` unpinned with the refuse guard, 48,000 tasks at global batch 4. SMOKE-TEST: the prior emits up to 512 columns and 2 to 10 classes (27% of tasks above 136 columns), and a CPU training smoke ran. ESTIMATED 2.2 to 4 T4 GPU-hours. Launch gate (binary arm: Step 0 cost, V4FinBench mean AP at least 0.1810 with no Holm-significant fold loss, clean training) is written in the draft. Still open: the launch and the TabArena-Lite rerun. **Progress (2026-11-04, `docs/results/drafts/dr8.3_2026-11.md`): still blocked on DR8.3's checkpoint, now prepared but not trained.** The 512 checkpoint is binary only; DR7.5's 4 excluded multiclass datasets and DR7.6's joint declaration also need a multiclass 512 checkpoint (DR7.4 recipe, max_classes 10). The Lite rerun with no width exclusion runs only after the wide proxy tasks and V4FinBench sanity pass. **Progress (2026-11-03, `docs/results/drafts/inference_tasks_2026-11.md`): no inference lever left** (§182 subsampling, §190 selection); blocked on DR8.3's `max_features` 512 checkpoint. **Measured on the proxy (2026-10-23, §182): subsampling covers the wide tasks but scores them near the floor** (mfeat-factors log-loss 1.99 against a 2.30 uniform guess). Open: proper wide-data handling (a wider checkpoint or column selection), then the TabArena-Lite run.
      **No excluded datasets in the declared types.** The reviewer called excluding datasets
      a model does poorly on cherry-picking (§122), and §106's width cap excluded 3 binary and 4
      multiclass datasets. Cover them with `max_features_strategy: subsample` (AU5.7) at
      inference, measured on the proxy suite's widest tasks first.
      Verify: TabArena-Lite runs every binary and multiclass dataset with no capability exclusion.
- [ ] DR7.6 **Resubmit only at the bar:** binary and multiclass declared together, scored on
      TabArena-Lite, and a resubmission only when that Elo reaches about 1,010 (TABARENA_BAR).
      Today's binary model is 756 (DR3.5), so the binary gap is the larger part of the work.
      Blocked by DR7.4, DR7.5 and a binary model near the bar.
      Verify: the declared-type Elo recorded in `docs/results/TABARENA.md`, and the resubmission
      decision beside it.

## Phase DR8 — new approaches (2026-10-23)

Filed after §176-§182: volume is the one lever that moved FinTFM (+140 proxy Elo at 480k tasks),
DR6.2/DR6.3 did not, and boosters lead by 3.5x on SBA.

- [x] DR8.1 **Done (2026-11-03, `docs/results/drafts/inference_tasks_2026-11.md`): Verify met. Every §183 arm has its per-task sign test in FINDINGS (§187; `ft200` 16/0 on split 0 in §192), and both proxy candidates were scored on V4FinBench (§194 harmful, §196 not adopted). The retrieval collapse's cause stays a diagnostic, tested next by AU5.8's per-class retrieval arm.** **Proxy sweep done (§187): `ens8` and `retrieval` candidates. V4FinBench (2026-10-25, §194): `retrieval` collapses to 0.0126 mean AP against 0.1986, 5 of 5 folds, not adopted; the contexts are 9.74% positive against a 0.35% pool, drawn from the same 540 of 2,128 positives (fold 0 diagnostic); why those sit near every centroid is open. `ens8` (2026-10-26, §196): +0.0058 mean AP, three folds up and two down, not adopted; a lead for DR6.4.** **Inference-lever sweep on the proxy** (pre-registered §183): 8-member ensembles,
      label swap, mixed transforms, an 8,000-row context, retrieval, 200 fine-tuning steps; a
      proxy candidate is scored on V4FinBench before it is believed.
      Verify: each arm's per-task sign test against the default in FINDINGS.
- [x] DR8.2 **Done (§202): width null too, +0.0003 AP on V4FinBench and 10 wins, 6 losses (p 0.45) on the proxy; with depth (§195) null, no larger arm at 480,000 tasks.** **Depth closed (2026-10-26, §195): null on V4FinBench (-0.0047 mean AP, §193) and on the proxy (8 wins, 8 losses); no deeper arm. Open: the width arm, pre-registered as §199 with its two kernels, at the next quota reset.** **Model size at volume.** §114's wide scale-up lost at 48,000 tasks, where a larger
      model is data-starved; repeat depth (Nori's 16 layers at width 128) and width at 480,000
      tasks against `e1-v480-s0`, pre-registered with §173's yardstick. Blocked by GPU quota and
      E1's seed-1 verdict.
      Verify: the pre-registration and verdict in FINDINGS.
- [ ] DR8.3 **Progress (2026-11-04, `docs/results/drafts/dr8.3_2026-11.md`, arm prepared, not run): the financial prior could not emit more than about 370 columns (20 accounts, 380 ordered ratios; 363-375 returned for 500 asked, MEASURED), fixed behind a top-up that leaves tasks within the old reach bit-identical, and tested; the default width draw at 512 puts 74% of tasks above 136 columns, so the arm uses three feature-ratio bands (29% above 136, median 69, MEASURED). `configs/arms/dr83-width512.yaml` changes three keys of the published recipe; kernel `scripts/kaggle/ts_dr83_width512_s0` is unpinned (T4x2, per-GPU batch 2, 12,000 steps, the same 48,000 tasks, so a batch confound against the control; per-GPU batch 4 is ESTIMATED at 14-18 GB, over a T4). Open: launch, the padded-inference cost of a 512 checkpoint on a GPU, the multiclass 512 checkpoint, then the per-task proxy comparison and V4FinBench sanity under the draft's rule.** **Progress (2026-11-03, `docs/results/drafts/inference_tasks_2026-11.md`): no inference lever left; the remaining arm is a pretraining run.** **Column selection measured (2026-10-24, §190): no gain, and a loss on Internet-Advertisements**; built behind `max_features_strategy: select` (3a63d7b). Open: the `max_features` 512 checkpoint.
      **Wide data handled, not subsampled.** §182: subsampling scores wide tasks near the
      floor. Compare a column-selection front end (mutual information or a booster's importance,
      fitted on the training split only) against a checkpoint trained at `max_features` 512, on
      the proxy's wide tasks first.
      Verify: per-task results for both on madelon, Internet-Advertisements and mfeat-factors.
- [ ] DR8.4 **Progress (2026-11-04, `docs/results/drafts/dr8.4_2026-11.md`, MEASURED): the tuned LightGBM control is run on the credit suite (seeded search of up to 30 configs on an inner split of each training split; SBA's inner split without embargo, the test split keeps it). SBA: tuned 0.4021 mean AP against untuned 0.3861 (+0.0159, 3 of 3) and against §189's logit-as-feature hybrid 0.3971 (+0.0049, 3 of 3), so §189's feature gain is smaller than a tuning gain. ULB: tuned 0.8299 against the init-score hybrid's 0.8534 (hybrid +0.0235, 3 of 3). Open: a tuned booster plus the FinTFM logit (needs out-of-fold logits §189 did not save), and V4FinBench.** **Credit suite measured (2026-10-24, §189):** the logit as a LightGBM feature adds +0.011 AP on SBA (3 of 3 splits); as an init score it gives ULB's best and most stable result (0.853 mean). Open: a tuned LightGBM control, and V4FinBench (capped, about 4.5 to 6 h of CPU).
      **FinTFM plus booster for credit** (AU4.4's `hybrid_catboost`, built, unmeasured):
      on SBA and ULB (credit suite) and V4FinBench, against LightGBM alone, so the practical
      credit answer is measured, not assumed.
      Verify: credit-suite and V4FinBench rows for the hybrid in FINDINGS.
