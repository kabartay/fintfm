# FinTFM (synthetic-only) binary checkpoint: `v4-cellattn-labels.pt`

LICENCE: Apache-2.0 (owner decision 2026-11-16, matching the code and the public binary checkpoint)

Status: **already public** at `kabartay/fintfm-binary` (checked read-only on 2026-11-15; the
repository's metadata declares `apache-2.0`). `CLAUDE.md` said at the time that trained
weights are not published, so the licence on that repository was the first thing for the owner
to reconcile (see `docs/models/WEIGHTS_LICENCE_OPTIONS.md`); the owner decided on 2026-11-16,
as the licence line above records. Nothing here uploads anything.

| | |
| --- | --- |
| family | **FinTFM (synthetic-only)**. Never saw a real table (D2/D14 claim intact). |
| local path | `runs/v4-cellattn-labels.pt` (gitignored, never committed) |
| size | 3,585,704 bytes |
| SHA-256 | `d231b062483c9261aae4c503ad5359db295fc7206fa0183f51763e4953c518d2` |
| Hub copy | LFS SHA-256 and size identical to the local file (public API, 2026-11-15) |
| parameters | 885,650 (`d_cell 48, d_model 128, n_layers 4, n_col_layers 2`, cell attention, per-cell labels, `max_features 136`, `max_classes 2`) |
| used by | the binary arm of the TabArena entry (`FinTFM-LinProbe`, §232) |

## Intended use

In-context binary tabular classification with a table supplied as context, or as a frozen
representation for a per-leaf linear readout (FinTFM-LinProbe). Credit-risk framing; research
and benchmarking. Not a general tabular model.

## Training data

Synthetic prior only: the financial prior (`--p-financial 1.0`), 6,000 steps x batch 8 =
**48,000 tasks**, seed 0, T4. Recipe: `configs/published/v4-cellattn-labels.yaml`, reproducing
the command in `docs/results/FINDINGS.md` §127 and §141 (cell-attention protocol §91). The
training commit is not recorded in the checkpoint (no `provenance` field); the recipe file is
the record. No real data at any stage of pretraining.

## Evaluation (all MEASURED, TabArena-Lite repeat 0 fold 0, CPU)

- Entry run `fintfm_lp_bin_scalecvraw_1114`: **997 Elo** on 30 binary datasets, against 974 Elo
  for the probe-only run, better on 23 of 30 (§232).
- Whole entry across the three problem types: 931.9 Elo (§235).
- Elo is not a ranking claim: positions are quoted only from `scripts/lite_positions.py` with
  the field size beside them.

## Limitations

- **The binary entry's Elo is not evidence for FinTFM** (§239). The raw-only control, a decision
  tree with a logistic regression per leaf that uses none of this checkpoint's representation,
  scores **1009 Elo against 997**; the representation wins on 24 of 30 datasets (sign test
  p < 0.01, mean +0.0043 ROC-AUC) but loses on the small noisy tables that move Elo most.
  V4FinBench showed +0.0040 ROC-AUC from the representation, better on 2 of 5 folds (§238).
- Binary only, at most 136 features. One seed, 48,000 tasks.
- Discrimination loses to tuned gradient boosting on real corporate-default panels; the
  repository's claims ledger (`docs/paper/CLAIMS.md`) records what holds.
- Weights licence: this card records the owner's decision of 2026-11-16 (Apache-2.0); it does
  not reopen it.

LICENCE: Apache-2.0 (owner decision 2026-11-16, matching the code and the public binary checkpoint)
