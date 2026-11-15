# FinTFM (synthetic-only) multiclass checkpoint: `dr74-mc-s0.pt`

LICENCE: OWNER DECISION

Status: **not public** (`kabartay/fintfm-multiclass` returns no model to the public API,
2026-11-15). Nothing here uploads anything.

| | |
| --- | --- |
| family | **FinTFM (synthetic-only)**. No real data. |
| local path | `runs/kaggle-ts-dr74-mc-s0/dr74-mc-s0/dr74-mc-s0.pt` (gitignored, never committed) |
| size | 3,598,528 bytes |
| SHA-256 | `14d5b4d561fbea6e35c7a30dc2644cb8f64f437c25d62da087ee960717cc8f4d` |
| parameters | 888,090 (same architecture as the binary checkpoint, `max_classes 10`) |
| used by | the multiclass arm of the TabArena entry: default checkpoint of `runs/local_queue/lite_mc_probe.sh`, run `fintfm_lp_mc_scr_bag8_1113` |

## Intended use

Frozen representation for a per-leaf readout on multiclass tabular tasks of up to 10 classes and
136 features (FinTFM-LinProbe with partition bags). Research and benchmarking.

## Training data

Synthetic prior only: `--max-classes 10 --p-financial 0.5 --scm-legacy --permute-classes`,
so half the tasks come from the financial prior and half from the generic structural-causal-model
prior. 6,000 steps at a global batch of 8 = **48,000 tasks**, seed 0, Kaggle T4x2 with DDP,
fp16, split attention (`scripts/kaggle/ts_dr74_mc_s0/`; pre-registered in §185; fintfm pinned
at a901cdc per §185, package version 0.7.1 per the stored provenance; the stored `commit`
field is empty). Recipe copy: `runs/kaggle-ts-dr74-mc-s0/dr74-mc-s0/recipe.yaml`. No real data.

## Evaluation (all MEASURED)

- Proxy suite, 6 of 8 multiclass tasks scored (`letter` and `mfeat-factors` exceed the caps):
  994 Elo against 808 for the earlier multiclass checkpoint; 6 task wins, 0 losses (§186).
- TabArena-Lite multiclass, entry run `fintfm_lp_mc_scr_bag8_1113`: **874 Elo** over 8 datasets
  (§227, §243); four bags gave 865 (§224). Entry overall: 931.9 Elo (§235).

## Limitations

- **Raw-only control (§243).** Without the representation (same configuration, `no_rep`) the
  entry scores **829 Elo against 874**; the representation is better on 7 of 8 datasets (sign
  test p = 0.07), median log loss 15.5% lower, and worse on `hiva_agnostic` by 2.0%. Unlike
  binary, the multiclass Elo is carried by the representation, on eight datasets and one seed.
- Eight datasets, one repeat, one fold; the Elo interval for the control is +151 / -344.
- Digit-feature tables (`mfeat-*`) stay far behind linear and boosted models (§186).
- More than 10 classes or 136 features is out of range.
- Weights licence is not decided in this card.

LICENCE: OWNER DECISION
