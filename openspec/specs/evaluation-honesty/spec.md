# Evaluation honesty

## Purpose

In an ML repository a wrong number does not crash. It looks like a result. Three bugs in one
afternoon each produced a plausible headline that was meaningless (`CLAUDE.md`), and all
three were caught only by running the pipeline rather than reading it.

## Requirements

**E1 — Every reported number carries how it was produced:** SMOKE-TEST, MEASURED, SIMULATED
or ESTIMATED.

- *Enforced by:* review, and `openspec/tools/validate.py --findings`, which checks each
  numbered finding in `docs/FINDINGS.md` declares a status.

**E2 — Discrimination is never reported alone.** Every credit result carries calibration
(Brier, ECE) and minority recall beside AUC, because AUC cannot see the failure that matters.

- *Enforced by:* `CreditMetrics.summary()`, which renders all of them together.

**E3 — A win count is not a result.** Comparisons across variants and panels use a paired
bootstrap with family-wise correction. Only 22 of 406 pairwise comparisons were significant
in this domain (`FINDINGS` §9), so raw tallies manufacture winners.

- *Enforced by:* `tests/test_metrics.py::test_paired_auc_difference_detects_a_real_gap_and_ignores_a_fake_one`
  and `::test_holm_bonferroni_is_stricter_than_raw_alpha`.

**E4 — Undefined metrics are NaN, never invented.** A single-class split has no AUC.

- *Enforced by:* `tests/test_metrics.py::test_auc_is_nan_for_single_class_rather_than_invented`.

**E5 — A NaN must not silently poison an aggregate.** Skipped cells are reported as skipped.

- *Enforced by:* `bench.py`'s `N/M scorable` reporting.

**E6 — Results are re-derivable.** Any number in a document names where it came from, and
experiment output carries the config and git commit. Numbers are never copied between
documents.

- *Enforced by:* `experiments/prior_ablation.py` writing `results.json` with provenance;
  `tests/test_experiments.py::test_ablation_matches_compute_and_writes_rederivable_results`.

**E7 — Matched compute is enforced, not asserted.** An ablation refuses to report if
parameter counts diverge across variants.

- *Enforced by:* the same test.

**E8 — An untrained control is included by default**, so a tie between priors can be read.

- *Enforced by:* `tests/test_experiments.py` asserting the control exists with `steps == 0`.
