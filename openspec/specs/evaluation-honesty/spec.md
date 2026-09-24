# Evaluation honesty

## Purpose

In an ML repository a wrong number does not crash. It looks like a result. Three bugs in one
afternoon each produced a plausible headline that was meaningless (`CLAUDE.md`), and all
three were caught only by running the pipeline rather than reading it.

## Requirements

**E1 — Every reported number carries how it was produced:** SMOKE-TEST, MEASURED, SIMULATED
or ESTIMATED.

- *Enforced by:* review, and `openspec/tools/validate.py --findings`, which checks each
  numbered finding in `docs/results/FINDINGS.md` declares a status.

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
- *Earned its place:* the control showed a random-weight model reaches AUC 0.726, above the
  generic-prior variant, which reframed the whole Phase 1 reading (`docs/results/FINDINGS.md` §15).

**E9 — Calibration is reported as *skill against a feature-free baseline*, never raw, and a
model with no discriminative content is flagged.** On a 4-7% base rate a constant
base-rate predictor scores ECE 0.0002 — better than every trained model in this project —
and Brier within 1-2% of the best. Raw calibration numbers therefore cannot carry an
argument (`docs/results/FINDINGS.md` §17).

- *Enforced by:* `CreditMetrics.brier_skill` and `.is_degenerate`, rendered by `summary()`;
  `tests/test_metrics.py::test_constant_base_rate_predictor_is_flagged_degenerate` and
  `::test_summary_always_shows_skill_beside_raw_brier`.
- *This spec requirement is the direct result of four findings overstating their case* for
  want of a reference row.
