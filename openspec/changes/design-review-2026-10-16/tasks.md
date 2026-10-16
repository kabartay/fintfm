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
- [ ] DR1.3 **Configuration from one source.** `FinancialTFMClassifier.from_config`; generate
      `InferenceConfig`, `ModelConfigDefaults` and the `fintfm-train` flags from dataclass fields;
      drift tests iterate `inspect.signature` and `fields()`; every `fintfm-train` flag defaults to
      `None`; commit `configs/published/v4-cellattn-labels.yaml`. Fix the silently ignored keys:
      `correct_prior`, `retrieval_groups`, `retrieval_min_positive`, `prototype_minority_ratio`
      reach the V4 harness; `ensemble_label_swap`, `ensemble_feature_frac`, `finetune_clip` get
      config keys; delete `winsor_quantile` and `transform_subsample` or give them a consumer.
      Verify: a test that sets every inference key to a non-default value and asserts the built
      classifier carries it; the published recipe reproduces from the YAML alone.
- [ ] DR1.4 **One inference path.** `FinancialTFM.predict_context(...)` with one padding helper,
      used by `predict_proba`, `predict_term_structure`, `transform_representation`, the quantile
      regressor and the capability probes; `predict_term_structure` takes the same
      `column_id_seed` as `predict_proba`; a public `context_summary_` replaces the private
      `_ctx_y`/`_ctx_rate` reads.
      Verify: DR1.1's fingerprints unchanged except `predict_term_structure` at
      `random_state != 0`, which now matches `predict_proba`'s seed, recorded as a deliberate fix.
- [ ] DR1.5 **Lever-combination validator and covering tests.** Refuse fine-tuning on a
      non-classification checkpoint; warn on any ensemble member axis during fine-tuning (class
      shift on a binary task is a label swap); raise on SVD with zero components after
      subsampling; cross-check `ModelConfig` against `PriorConfig` in `train()`. A pairwise
      covering array of about 20 model and 25 classifier lever configurations.
      Verify: each refused combination raises with a message naming both levers; every covering
      configuration runs, gives finite outputs, chunks exactly and returns valid probabilities.
- [ ] DR1.6 **Correct `CLAUDE.md`'s cost figure** (owner's file): about 66 T4-hours or about $28 of
      L4 time per million tasks at the current recipe (§86, §127), not $13.
      Verify: the owner has edited it, or declined.
- [x] DR1.7 **Done (2026-10-16, f209e29).** rms_match scaling (0.2·√max(n, m)), lr shared with AdamW, weight decay 0.01, Q/K/V orthogonalised as separate blocks (Keller's write-up; nothing per head is described there), fp32 Newton-Schulz without native bf16. **Verify amended 2026-10-16:** the step-1 comparison was ill-posed, since AdamW's first step is lr·sign(g) (RMS exactly lr) while 0.2·√max(n, m) matches AdamW's steady state; the check is now update RMS within 2x of AdamW's steady-state RMS on full-rank gradients (`tests/test_muon.py`). Measured on the tiny model (SMOKE-TEST): full-rank updates 0.14-0.24·lr against AdamW's steady 0.229·lr.
      **Muon at peer settings before any use.** 0.2·√max(n, m) update scaling, shared
      lr 8e-4 to 1e-3, weight decay 0.01, Newton-Schulz in fp32 on GPUs without bf16, and the fused
      QKV projection orthogonalised per head block, as TabICLv2, LightPFN and Mitra-v2 describe.
      Verify: per-element update RMS at step 1 within 2x of AdamW's on the same batch.

## Phase DR3 — measurements that need no pretraining

- [ ] DR3.1 **V4: partitioned contexts against uniform** (`--large-n partition`, depth 2, leaf
      minimum 5%), zero-shot, five folds, company bootstrap; report AP within flagged rows beside
      headline AP. Comparator: the zero-shot control. Rule: §163's (mean delta above the bar, 4 of 5
      folds the same sign).
      Verify: per-fold AP for both arms in FINDINGS, with the flagged-row AP.
- [ ] DR3.2 **V4: 8 ensemble members against 1** at 2,000 rows, and a second arm adding
      `ensemble_class_shift`. Same rule.
      Verify: per-fold AP for each arm in FINDINGS.
- [ ] DR3.3 **V4: a fixed positive count in the uniform context** (exactly round(n x rate), or a
      floor of 16, with the base-rate correction). Report the fold-to-fold SD beside the mean.
      Verify: per-fold AP and SD in FINDINGS.
- [ ] DR3.4 **§130's probe fitted on the 2,000 context rows only.** If the probe's advantage over
      the head falls to 0.005 AP or less, §130-§136's reading is label count, not representation.
      Verify: the result and its consequence for §130-§136 written in FINDINGS.
- [ ] DR3.5 **TabArena-Lite, same day, same fintfm source:** NaN against zero fill (fork branch
      `fintfm-audit-fixes`), then `ensemble_class_shift`, then `ensemble_transforms=("rank",
      "squash")`, then `max_context` 500 and 750 against 1,000. Paired per-dataset differences,
      not Elo, decide.
      Verify: per-dataset paired differences recorded in `docs/results/TABARENA.md`.
- [ ] DR3.6 **Score the TabArena imbalanced subset** from results already held (minutes).
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
- [ ] DR3.10 **FinTFM without `operational_status`** (zero-shot and fine-tuned arms), paired
      against the published-feature runs, so both models' dependence on the column is known.
      Verify: per-fold AP for both arms in FINDINGS beside §167.

## Phase DR4 — an evaluation loop that does not overlap TabArena

- [ ] DR4.1 **Partly done (2026-10-16), left open.** The six datasets are out of `SUITE`;
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
- [ ] DR4.2 **A proxy Elo suite:** about 45 OpenML-CC18/CTR23 tasks outside TabArena and
      BeyondArena (excluded by id, name and row hash; CC-BY, CC0 or public licences only), 30
      binary / 8 multiclass / 13 regression, three splits, run through TabArena's own custom-task
      pipeline against cached RF, ExtraTrees, Linear, KNN and GBDT baselines; Elo anchored at RF
      (default) = 1000, plus a paired sign test against a reference checkpoint.
      Verify: Spearman 0.8 or higher against Lite Elo across 6 or more existing checkpoints
      before the proxy is used to decide anything.
- [ ] DR4.3 **A credit suite outside V4FinBench:** PPDai (CC BY 4.0), Bondora (CC0 mirror), ULB
      fraud (DbCL-1.0), three splits, evaluation only (which removes PPDai from FinTFM-R's training
      queue).
      Verify: the suite runs end to end on the published checkpoint with runtimes recorded.
- [ ] DR4.4 **The noise yardstick from at least three matched replicates** (AU2.9's method), with
      separate pretraining and fine-tuning values and Holm across arms.
      Verify: FINDINGS states both yardsticks with source runs and degrees of freedom.

## Phase DR5 — throughput, then a volume test

- [ ] DR5.1 **E0 throughput.** `fintfm-train --benchmark-steps 50` at the reference recipe on a
      Kaggle T4 (fp32, `--amp fp16`, `--attn-impl split`) and an HF L4 (fp32, `--amp bf16`, split),
      with rows pinned at 1,024 and mixed; prior generation on each host's CPUs at 1 and 2
      workers. Adopt a setting only if outputs agree on a fixed batch and tasks per GPU-hour rise
      by 1.3x or more.
      Verify: a FINDINGS entry with tasks per GPU-hour for each setting and host.
- [ ] DR5.2 **A streaming loader:** batch i drawn from `default_rng([seed, i])` in an
      IterableDataset with workers and prefetch, so resume needs no RNG state and DDP ranks take
      disjoint indices; dumps kept only for byte-identity checks. Use both GPUs of a Kaggle T4x2.
      Verify: a test that the loader's batch i equals the dump's batch i, and that two workers
      give the same stream as one.
- [ ] DR5.3 **E1: does volume help?** 48,000 against 480,000 tasks, seeds 0 and 1, one pinned
      commit, everything else fixed, the 48k arm retrained rather than reused (§158). Volume
      helps if the 480k arm's five-fold mean V4FinBench AP is positive at both seeds and their
      average exceeds the DR4.4 pretraining yardstick; hurts under the mirror; otherwise null.
      Secondary: DR4.2 proxy and a fixed 512-task synthetic set. Pre-register the outcome actions
      (2M run, mixed-prior 2M run, or neither) before launching.
      Verify: the pre-registration committed before the first arm trains; the verdict in FINDINGS.
- [ ] DR5.4 **An extensible schedule:** warmup-stable-decay with decay branches, so a 2M-task
      run can continue to 10M; a fixed held-out synthetic set for in-training evaluation instead
      of fresh tasks per evaluation.
      Verify: a test that resuming a stable-phase checkpoint with a longer horizon reproduces the
      uninterrupted run's weights.

## Phase DR2 — prior v2

- [ ] DR2.1 **Normalised source weights.** Replace the sequential source draw with one
      categorical draw over normalised weights, behind a flag that keeps today's stream when
      off; record the realised shares in every checkpoint (DR1.2).
      Verify: realised shares within binomial error of the weights at 10,000 draws; default
      stream byte-identical.
- [ ] DR2.2 **A gated-label task source** (a threshold on 1-2 columns, then a score inside the
      gate) and a gated variant of the financial label (equity < 0, working capital < 0,
      coverage < 1 or a random ratio threshold, with a small leakage rate outside the gate),
      justified from balance-sheet insolvency tests and randomised so no global rule exists
      (§47).
      Verify: on 300 sampled tasks a tree beats rank-logistic regression on AP for the gated
      financial source, and no column points the same way in more than 75% of tasks.
- [ ] DR2.3 **Knob distributions:** feature-to-row ratio bands, binary probability 0.7, explicit
      imbalance draws, unordered and high-cardinality categoricals with Zipf frequencies,
      MCAR/MAR/MNAR missingness, ties and zero inflation, irrelevant and duplicated columns,
      label noise (the review's spec, §1.5 and DR2 in `DESIGN_REVIEW_2026-10-16.md`).
      Verify: `scripts/prior_stats.py` reports each distribution within its target on 300 tasks
      per source; generation stays at or below 0.25 s per batch.
- [ ] DR2.4 **Prior v2 A/B at fixed tasks:** financial v2 alone, general trunk alone, trunk then
      credit, and interleaved, 48k tasks each, matched `max_classes`, two seeds for any arm that
      passes. V4FinBench first, then TabArena-Lite.
      Verify: per-arm per-fold AP and Lite paired differences in FINDINGS under DR4.4's yardstick.

## Phase DR6 — architecture v2

- [ ] DR6.1 **Per-batch feature width** (pad to the batch's widest task, not 136), removing about
      half of all cell tokens from financial tasks.
      Verify: identical outputs on a fixed batch whose widest task is 136; step time measured.
- [ ] DR6.2 **Inducing-point, context-only cell stage** behind a flag (gather from labelled
      context cells through k inducing points), linear in rows.
      Verify: at matched N up to 1,024 it is within the yardstick of full attention; memory at
      N = 4,096 measured.
- [ ] DR6.3 **Readout tokens that take part in feature attention** (re-specify AU5.3), feeding a
      2-layer head.
      Verify: §137's parity probe at k = 2 rises above 0.60 AUC (chance today).
- [ ] DR6.4 **v2 bundle against v1** at matched 240,000 tasks, two seeds: adopt if five-fold mean
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
- [ ] DR7.2 **Ask the maintainers** whether LightPFN's classification-only precedent (admitted
      2026-10-11 with regression imputed) applies, before building regression coverage.
      Verify: their answer recorded in `docs/results/TABARENA.md`.
