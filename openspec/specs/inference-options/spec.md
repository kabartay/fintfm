# Inference options

## Purpose

`FinancialTFMClassifier` carries about thirty inference options: context strategies, class
balance, ensembles, partitioned contexts, fine-tuning, fitted calibration. Most were measured
null or harmful (`METHODS.md`), and they stay in the code so the measurements can be repeated.
An option that leaks into the default path would change every published number without a
trace, so this spec states what holds when they are off and what each one promises when on.

## Requirements

**I1 — Every inference option is off by default, and off reproduces the published outputs.**
The packaged defaults build the plain classifier, and the published checkpoint's outputs are
pinned bit for bit.

- *Enforced by:* `tests/test_golden_outputs.py::test_checkpoint_fingerprint`,
  `tests/test_config.py::test_from_config_at_the_packaged_defaults_is_the_plain_classifier`,
  `tests/test_classifier.py::test_levers_default_off_and_the_default_member_is_the_plain_model`,
  `tests/test_large_n.py::test_the_default_configuration_wraps_nothing`,
  `tests/test_classifier.py::test_calibration_off_is_byte_identical_to_the_baseline`,
  `tests/test_classifier.py::test_context_positive_floor_off_is_the_uniform_draw_byte_for_byte`,
  `tests/test_finetune.py::test_off_by_default_is_byte_identical` and
  `tests/test_finetune.py::test_early_stopping_off_by_default_changes_nothing`.

**I2 — Every inference key in the configuration reaches the classifier and its wrappers**, so a
setting written in a YAML file is the setting that runs.

- *Enforced by:* `tests/test_config.py::test_every_inference_key_reaches_the_classifier`,
  `tests/test_config.py::test_large_n_section_matches_the_wrapper_defaults` and
  `tests/test_large_n.py::test_configuration_reaches_the_wrapper_arguments`.

**I3 — Combinations that would run but misbehave are refused before anything is fitted**, and
a misspelt option is an error, not a silent fallback.

- *Enforced by:* `tests/test_lever_combinations.py::test_classifier_cover_is_pairwise_complete`,
  `tests/test_lever_combinations.py::test_the_large_n_wrappers_refuse_before_fitting_anything`,
  `tests/test_lever_combinations.py::test_class_shift_with_label_swap_is_refused_on_a_binary_task_only`,
  `tests/test_classifier.py::test_context_positive_cap_refuses_what_it_cannot_mean`,
  `tests/test_classifier.py::test_context_positive_floor_refuses_what_it_cannot_mean` and
  `tests/test_classifier.py::test_ensemble_transforms_rejects_a_misspelling_rather_than_running_rank`.

**I4 — Every inference caller goes through `FinancialTFM.predict_context`**, so the
invariances in `model-invariances` hold on every path.

- *Enforced by:* `tests/test_classifier.py::test_every_inference_caller_goes_through_predict_context`.

**I5 — A context whose class balance was changed stays calibrated.** A positive floor, a
positive cap and a majority-downsampled context each read their own positive rate into the
base-rate correction (`calibrated-prediction` C2).

- *Enforced by:* `tests/test_classifier.py::test_context_positive_floor_stays_calibrated_with_correct_prior`,
  `tests/test_classifier.py::test_context_positive_cap_reaches_every_member_and_stays_calibrated`
  and `tests/test_classifier.py::test_majority_downsample_context_reads_its_own_rate_in_the_base_rate_correction`.

**I6 — Partitioned contexts route each query to its own leaf's context**, leaves too small or
with one class fall back to the global context, and bags are averaged.

- *Enforced by:* `tests/test_large_n.py::test_partition_routes_each_query_to_its_own_leaf_context`,
  `tests/test_large_n.py::test_partition_small_or_single_class_leaves_fall_back_to_the_global_context`,
  `tests/test_large_n.py::test_partition_bags_draw_distinct_subsets_and_average_them` and
  `tests/test_large_n.py::test_partition_fits_predicts_and_is_deterministic`.

**I7 — The fitted-calibration slice never reaches what it calibrates.** It is held out before
conditioning and context selection, by whole groups when groups are given, and the fitted
temperature keeps the ranking.

- *Enforced by:* `tests/test_classifier.py::test_calibration_slice_never_reaches_the_stored_context`,
  `tests/test_classifier.py::test_calibration_slice_holds_out_whole_groups` and
  `tests/test_classifier.py::test_fitted_temperature_preserves_ranking_and_is_valid`.

**I8 — Fine-tuning never trains on what it is judged on.** Query companies never appear in the
context, validation companies never supply training tasks, early stopping restores the best
weights, and the caller's random state is left alone.

- *Enforced by:* `tests/test_finetune.py::test_query_groups_never_appear_in_the_context`,
  `tests/test_finetune.py::test_validation_companies_never_supply_training_tasks`,
  `tests/test_finetune.py::test_early_stopping_restores_the_best_weights`,
  `tests/test_finetune.py::test_finetuning_leaves_the_callers_torch_rng_alone` and
  `tests/test_finetune.py::test_refitting_fine_tunes_from_the_model_passed_in_not_the_tuned_copy`.

**I9 — A representation probe never sees a row's own label through its representation**, and
the probe per partition leaf leaves the global arms unchanged.

- *Enforced by:* `tests/test_representation_transfer.py::test_a_context_rows_own_label_never_reaches_its_cross_fitted_representation`,
  `tests/test_representation_transfer.py::test_partition_keeps_the_default_runs_global_arms_bit_for_bit`
  and `tests/test_representation_transfer.py::test_partition_with_every_leaf_falling_back_is_the_global_probe`.

## What would legitimately change this spec

Promoting an option to the default. That needs a candidate verdict under `experiment-decisions`
D1 on V4FinBench, a new golden fixture regenerated with `scripts/regen_golden.py`, and a
changelog entry naming the outputs that change. A TabArena-Lite gain alone is not enough
(`experiment-decisions` D8).
