# Tasks

- [x] 31.1 Cheapest falsification first: score the existing checkpoint with contexts filtered
      to firms whose outcome is known across the whole grid, versus unfiltered. Verify: if
      horizon-3 AUC moves materially on filtering alone, timing evidence in the context is
      confirmed to matter before any architecture work is done.
      **Done 2026-09-09: confounded, not supportive (`docs/results/FINDINGS.md` §31).** The filter
      moved mean AUC +0.006 but the gain was largest at h0 (+0.0102) and smallest at h3
      (+0.0018) — the opposite of the prediction. `n_observed` is censored by default itself,
      so the filter removes defaulters and drops the context rate from 1.54% to 0.19%; the
      gain is §29's base-rate effect, not an observation-depth effect. No cheap unconfounded
      test exists. Tasks 31.2+ now sit behind `retrieval-context`.
- [x] 31.2 **Done (2026-11-11, src/fintfm/inference/classifier.py, tests/test_context_labels.py: test_selected_context_labels_equal_the_selected_rows_labels, test_omitting_or_adding_n_observed_leaves_the_binary_path_byte_identical; `periods` already existed under AU5.9, so this adds `n_observed` and the public `context_period_` / `context_n_observed_`).** Extend `FinancialTFMClassifier.fit` to accept `period` and `n_observed`, carried
      through context selection intact. Verify: a test asserting the selected context's period
      distribution matches the selected rows, and that omitting them leaves the binary path
      byte-identical.
- [x] 31.3 **Done (2026-11-15: the embedding itself landed earlier as `hazard_period_context` under AU5.9, src/fintfm/modeling/model.py `_label_embedding` and `period_embed`, `K + 1` states with index 0 the censored token, off by default and absent in old checkpoints; the classifier and `predict_term_structure` already pass context periods. This closes the Verify with tests/test_hazard.py: test_period_label_indices_cover_the_k_plus_one_states_including_censored, test_period_context_term_structure_is_invariant_to_context_row_order, plus the existing zero-at-init and flag-off equality tests; tests/test_train.py: test_hazard_period_context_trains_three_cpu_steps).** Embed the period label in `FinancialTFM.term_structure` over a `K + 1` state grid.
      Verify: `uv run pytest tests/test_hazard.py -q`, plus a test that permuting context rows
      leaves predictions unchanged (the equivariance guarantee must survive the new embedding).
- [ ] 31.4 Emit `period` into the pretraining context, not only the query targets. Verify: a
      test that a task's context period labels agree with its binary labels.
- [ ] 31.5 Pretrain the matched pair and run the pre-registered comparison. Verify:
      `fintfm-v4oot` on both checkpoints, numbers into `docs/results/FINDINGS.md` with the pre-registered
      prediction quoted and marked hit or miss **before** any interpretation.
- [ ] 31.6 Only if 31.5 succeeds: per-horizon base-rate correction, now that per-horizon
      context rates are recoverable. Verify: ECE at horizons 2-3 against §28's scalar-shift
      numbers (0.0056 and 0.0114), monotonicity still asserted.
