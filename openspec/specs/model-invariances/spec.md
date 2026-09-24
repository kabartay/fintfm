# Model invariances

## Purpose

A table has no canonical column order, so an answer that depends on column position is
wrong. The first architecture failed this by construction (`docs/design/ARCHITECTURE.md`), and it
was the main limit on transfer.

## Requirements

**M1 — Column-order invariance.** Permuting features must not change any prediction.

- *Enforced by:* `tests/test_model.py::test_predictions_are_invariant_to_column_permutation`.

**M2 — Padding-width invariance.** A narrow table must score identically regardless of the
padded width, and regardless of where the padding sits.

- *Enforced by:* `tests/test_model.py::test_padding_width_does_not_change_predictions`.

**M3 — Query isolation.** A query row's prediction must not depend on any other query row, so
a prediction never depends on which rows share its batch. Self-attention is permitted; a
query carries no label, so it leaks nothing.

- *Enforced by:* `tests/test_model.py::test_query_rows_never_see_each_other`.

**M4 — Context-only normalisation.** Feature statistics come from context rows alone, so a
query cannot influence its own normalisation.

- *Enforced by:* `normalize_features` signature and
  `tests/test_model.py::test_normalize_features_masks_missing`.

**M5 — No feature-index embeddings.** Column identity comes from the data distribution, never
from position. Adding a learned per-column embedding would violate M1.

- *Enforced by:* M1's test, which such an embedding would break.

**M6 — Device parity.** Every code path must work on every available device. A path exercised
only on CPU is untested, not working — this cost a three-hour run (`CLAUDE.md`).

- *Enforced by:*
  `tests/test_model.py::test_training_completes_on_each_available_device`, parametrised over
  available devices.
