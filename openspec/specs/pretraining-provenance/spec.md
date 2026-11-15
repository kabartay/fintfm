# Pretraining provenance

## Purpose

A benchmark number from this model cannot be inflated by memorising the benchmark, and that
is checkable by a third party rather than asserted. The leakage literature measures
contamination at up to 32 points of MAPE (`docs/results/FINDINGS.md` §1), and every competitor
pretraining on real tables carries public-benchmark numbers open to that critique.

This is the project's cheapest durable advantage and the easiest to destroy by accident.

## Requirements

**P1 — No real data may reach the pretraining path for FinTFM.** The only source of
pretraining data for the synthetic-only family is `fintfm.prior`'s generators. Real datasets
are otherwise loaded exclusively in `fintfm.evaluation`.

- *Enforced by:* `openspec/tools/validate.py --provenance`.

**Exception, named by D17 (2026-10-06), not a relaxation of the rule above.**
`prior/real_edgar.py` reads a real SEC EDGAR panel — it is FinTFM-**R**'s data source, a
second, explicitly real-data family opened as an addition to this project, not a change to
FinTFM. It is listed in `validate.py`'s `PROVENANCE_EXEMPT`, which (a) makes the file
inspectable rather than silently excluded, and (b) is itself checked: the exemption is only
sound while `PriorConfig.p_real_edgar` defaults to `0.0`, so a FinTFM checkpoint (anything
that does not explicitly set it) never reaches this file, and the validator fails if that
default ever changes. P1 therefore holds exactly as stated for FinTFM; it does not, and was
never claimed to, hold for FinTFM-R. See `CLAUDE.md`'s "Two families" section for the
claim-labelling discipline this makes necessary.

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

**P5 — Adapters and fine-tuned weights are a separate artefact, never the base model.**
An adapter, or fine-tuned weights, trained on a lender's book or on any other real data
inherits the FinTFM-**R** side of the provenance table in `CLAUDE.md`'s "Two families"
section. It never carries the synthetic-only claim, and it is labelled as fitted to that
data wherever it is named. The base checkpoint's claim is unchanged: the base stays frozen
and shared, and no real data enters its pretraining. Adapters are per-lender, never merged
into a shipped base model, because merging one lender's adapter would carry their book into
everyone else's predictions and void P1.

- *Enforced by:* review, and the frozen-base test required by `lender-adaptation` task 16.2
  (base weights unchanged after fitting). Not yet mechanised: a checkpoint's provenance
  record does not state whether real data was used in fine-tuning (see the note in
  `openspec/changes/lender-adaptation/tasks.md` 16.5).

#### Scenario: a lender adapter is fitted

- **Given** a frozen FinTFM base checkpoint with a synthetic-only claim
- **When** an adapter is fitted on one lender's real book
- **Then** the adapter is stored and named as that lender's artefact, with the FinTFM-R
  label, and the base checkpoint's file, hash and synthetic-only claim are unchanged
- **And** no release, model card or leaderboard row describes the adapter, or any model
  that includes it, as synthetic-only
- **And** the adapter is not merged into any base model that is shipped to others.

## What would legitimately change this spec

Only a decision to abandon the auditability claim, which is D2 in `docs/design/DECISIONS.md` and
would need to be argued there first, not here.
