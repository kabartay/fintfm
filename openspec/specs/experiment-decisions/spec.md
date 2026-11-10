# Experiment decisions

## Purpose

Between 2026-10-10 and 2026-11-10 more than forty arms were screened. Each verdict depends
on a few fixed rules. If a rule moves after the results are seen, a null turns into a
candidate. This spec fixes the rules and names the code that applies them.

## Requirements

**D1 — An arm is a candidate only if its mean AP change on V4FinBench exceeds the yardstick
and at least 4 of 5 folds move the same way.** The mirror image is harmful, and anything else
is null. The yardstick is +0.0117, the per-fold seed noise of a 48,000-task pretraining run
(`FINDINGS` §173). Pretraining arms use it, and so do inference arms from §211 on. Verdicts
before §211 on inference and fine-tuning arms used §158's +0.0073.

- *Enforced by:* `tests/test_dr64_arm_verdict.py::test_thresholds_are_section_175s`,
  `tests/test_dr64_arm_verdict.py::test_judge_boundaries`,
  `tests/test_c5_verdict.py::test_yardstick_is_section_158s_measured_gap` and
  `tests/test_c5_verdict.py::test_helps_rule_needs_the_mean_and_four_folds_of_its_sign`.
  Which yardstick a given verdict applies is checked in review; no test pins the switch at §211.

**D2 — Paired comparisons resample companies, not rows, and adjust across folds with Holm.**
A company's rows are correlated, so a row bootstrap understates the interval.

- *Enforced by:* `tests/test_metrics.py::test_cluster_bootstrap_is_wider_on_clustered_data`,
  `tests/test_metrics.py::test_singleton_groups_reproduce_the_row_bootstrap`,
  `tests/test_cluster_comparison.py::test_every_reconstructed_test_company_lies_in_one_fold` and
  `tests/test_v4_protocol.py::test_predictions_carry_the_company_of_every_row`.

**D3 — Two runs are paired only if they share the protocol and the rows**: horizon, row cap,
feature count, fold seed and split count.

- *Enforced by:* `tests/test_c5_verdict.py::test_fold_diffs_refuses_a_different_protocol` and
  `tests/test_c5_verdict.py::test_fold_diffs_refuses_mismatched_rows`.

**D4 — A bootstrap p-value is never exactly zero.** It is floored at 1/(resamples + 1), so a
Holm-adjusted 0 cannot appear.

- *Enforced by:* `tests/test_c5_verdict.py::test_bootstrap_p_is_never_exactly_zero`.

**D5 — No arm is chosen on a benchmark dataset.** The Polish and Taiwanese bankruptcy panels
are TabArena datasets, so the arm-selection tools refuse them.

- *Enforced by:* `tests/test_selection_guard.py::test_the_forbidden_panels_are_tabarena_datasets`,
  `tests/test_selection_guard.py::test_guard_refuses_then_flags_for_reporting` and
  `tests/test_selection_guard.py::test_pilot_compare_refuses_a_bankruptcy_run`.

**D6 — A pre-registered arm cannot run until it is pinned to a commit.** This means the code
that produces a result is fixed before the result exists.

- *Enforced by:* `tests/test_au51_arms.py::test_unpinned_kernel_refuses_to_run`,
  `tests/test_dr311_arms.py::test_unpinned_kernel_refuses_to_run_and_refuses_a_cpu_run` and
  `tests/test_masked_column.py::test_kernel_differs_from_its_control_by_the_lever_alone_and_cannot_launch`.

**D7 — An arm differs from its control only in the setting it tests.**

- *Enforced by:* `tests/test_au51_arms.py::test_arm_config_differs_from_the_others_only_in_the_learning_rate`,
  `tests/test_dr311_arms.py::test_arm_changes_one_setting_of_the_partition`,
  `tests/test_dr83_arm.py::test_arm_differs_from_the_published_recipe_in_exactly_three_keys`,
  `tests/test_au511_arm.py::test_arm_file_differs_from_published_in_prior_keys_only`,
  `tests/test_4825_arm.py::test_arm_file_is_the_published_recipe_plus_the_two_train_keys` and
  `tests/test_au59_arms.py::test_arm_and_control_differ_only_in_the_flag_and_from_the_recipe_in_two_keys`.

**D8 — V4FinBench decides. The proxy suite and TabArena-Lite do not.** The proxy suite was meant
to stand in for TabArena. Its rank correlation with TabArena is rho +0.717 on nine checkpoints,
below the 0.8 bar (§216), so no proxy result promotes an arm. A configuration that helps only on
TabArena-Lite is reported as Lite-only; an example is probe plus squash plus partition, at
1,000.2 Elo (§215).

- *Enforced by:* `tests/test_dr42_spearman.py::test_correlate_takes_nine_rows` and
  `tests/test_dr42_spearman.py::test_n9_matches_scipy_on_random_data` for the correlation itself.
  Review enforces the labelling.

## What would legitimately change this spec

A new noise measurement that replaces §173 with more seeds, recorded as a FINDINGS section
**before** any arm is judged against it. Another option is a proxy suite that clears the 0.8
bar on a fresh set of checkpoints. Moving a threshold after seeing an arm's result is never
legitimate.
