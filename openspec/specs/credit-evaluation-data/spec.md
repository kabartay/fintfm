# Credit evaluation data

## Purpose

The credit suite has six real datasets. They have different licences and labels, and some of
them overlap the public benchmarks. A dataset with the wrong licence makes the results
unpublishable. One that overlaps a benchmark or leaks across time makes them wrong. This spec
states the gates every real dataset passes before it is scored, on top of
`pretraining-provenance` P1 and P4.

## Requirements

**L1 — Evaluation-only sources can never train a model.** ULB card fraud, SBA 7(a), FDIC
bank failures, the Estonian register, PPDai and Bondora are refused both when a prior is parsed
and when it samples.

- *Enforced by:* `tests/test_real_panel.py::test_evaluation_only_sources_are_refused_at_parse_and_at_sampling`,
  `tests/test_real_panel.py::test_evaluation_only_sources_match_the_registry` and
  `tests/test_credit_suite.py::test_suite_keys_are_all_evaluation_only`.

**L2 — A dataset is admitted by the licence at its original source, checked before any file is
read.** Being public on OpenML or Kaggle does not admit it. PPDai and Bondora stay excluded.

- *Enforced by:* `tests/test_credit_suite.py::test_an_unadmitted_licence_is_refused_before_any_file_is_read`,
  `tests/test_proxy_suite.py::test_openml_public_alone_does_not_admit_a_task` and
  `tests/test_proxy_suite.py::test_every_suite_task_has_a_verified_source_licence`.

**L3 — No benchmark dataset enters an evaluation suite**, even when its licence would admit it.

- *Enforced by:* `tests/test_credit_suite.py::test_a_tabarena_panel_is_refused_even_when_admitted`,
  `tests/test_proxy_suite.py::test_suite_shares_no_openml_id_or_name_with_tabarena_or_beyondarena`
  and `tests/test_overlap_report.py::test_report_compared_every_reference`.

**L4 — LendingClub is retired as a training source** (`docs/design/DECISIONS.md` D19), because
it is BeyondArena's `lending_club_1m`.

- *Enforced by:* `tests/test_real_lendingclub.py::test_the_sampler_refuses_even_a_valid_panel`,
  `tests/test_real_lendingclub.py::test_a_positive_p_real_lendingclub_is_refused_before_any_draw`
  and `tests/test_real_lendingclub.py::test_a_real_lendingclub_source_weight_is_refused`.

**L5 — Out-of-time splits respect the label window, the embargo and the borrower**, and a time
split that shares an identifier across train and test is refused.

- *Enforced by:* `tests/test_credit_suite.py::test_out_of_time_splits_respect_the_window_the_embargo_and_the_borrower`,
  `tests/test_credit_suite.py::test_sba_out_of_time_split_is_embargoed_by_60_months`,
  `tests/test_credit_fdic.py::test_crisis_split_trains_before_and_respects_the_embargo_and_groups`
  and `tests/test_time_split.py::test_overlapping_panel_raises`.

**L6 — A hybrid arm never sees its own row's FinTFM logit.** Out-of-fold logits are
cross-fitted with whole borrowers per fold, and a leaking split is refused before any fit.

- *Enforced by:* `tests/test_hybrid_credit.py::test_out_of_fold_logits_never_use_their_own_row`,
  `tests/test_hybrid_credit.py::test_a_constructed_leak_is_refused_before_any_fit`,
  `tests/test_hybrid_catboost.py::test_a_leaking_split_is_refused_before_any_fit` and
  `tests/test_hybrid_catboost.py::test_inner_contexts_never_hold_an_outer_val_or_test_row`.

**L7 — Every label states its window, and a caveat travels with the dataset.** Examples are
SBA's 60-month charge-off, FDIC's failure within four quarters, and the Estonian panel's
survivor conditioning.

- *Enforced by:* `tests/test_credit_suite.py::test_sba_label_is_a_60_month_charge_off_with_undisbursed_and_undated_loans_dropped`,
  `tests/test_credit_fdic.py::test_label_window_lag_and_censoring_by_hand` and
  `tests/test_credit_estonia.py::test_estonia_is_registered_evaluation_only_and_admitted_with_its_caveat`.

## What would legitimately change this spec

A new dataset joins by adding a registry entry with its licence as read at the original source,
its label window and its overlap check, together with tests of the same kind as above. Deferred
datasets stay out until their terms can be read: FICO HELOC is one (task 48.28), since its
data-use agreement page is gone.
