# Pretraining provenance

## Purpose

A benchmark number from this model cannot be inflated by memorising the benchmark, and that
is checkable by a third party rather than asserted. The leakage literature measures
contamination at up to 32 points of MAPE (`docs/FINDINGS.md` §1), and every competitor
pretraining on real tables carries public-benchmark numbers open to that critique.

This is the project's cheapest durable advantage and the easiest to destroy by accident.

## Requirements

**P1 — No real data may reach the pretraining path.** The only source of pretraining data is
`fintfm.prior`. Real datasets are loaded exclusively in `fintfm.evaluation`.

- *Enforced by:* `openspec/tools/validate.py --provenance`, and
  `tests/test_provenance.py::test_no_real_data_in_pretraining_path`.

**P2 — The financial prior stays parametric.** It samples a macro regime rather than learning
real crisis history. Realism may be increased by enriching generative structure (more
accounting identities, more regimes), **never** by conditioning the generator on a real
dataset.

- *Rationale:* fitting to real panels reintroduces the global-pattern memorisation the
  leakage literature warns about, and would improve benchmarks while silently voiding P1.
- *Enforced by:* P1's import check, plus review. Not fully mechanisable.

**P3 — Trained weights never enter git.** `.gitignore` excludes `*.pt`.

- *Enforced by:* `tests/test_provenance.py::test_no_weights_tracked`.

**P4 — Every real dataset carries its licence and attribution in code**, not in a comment.

- *Enforced by:* `CreditDataset.licence` and `.attribution` are required fields;
  `tests/test_metrics.py::test_dataset_loaders_declare_period_labels_honestly`.

## What would legitimately change this spec

Only a decision to abandon the auditability claim, which is D2 in `docs/DECISIONS.md` and
would need to be argued there first, not here.
