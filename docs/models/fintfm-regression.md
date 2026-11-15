# FinTFM (synthetic-only) pure-regression checkpoint: `v4-binned48_3-control.pt`

LICENCE: OWNER DECISION

Status: **not public** (`kabartay/fintfm-regression` returns no model to the public API,
2026-11-15). Nothing here uploads anything.

The file name says "control" because it began as the matched binned-head control for the
pinball-quantile experiment (§148). It is the checkpoint the entry uses for regression.

| | |
| --- | --- |
| family | **FinTFM (synthetic-only)**. No real data. |
| local path | `runs/task48-3-kaggle/v4-binned48_3-control.pt` (gitignored, never committed) |
| size | 3,595,783 bytes |
| SHA-256 | `31d5b3ef49e89bbe405fc85ab1d746a9fd6c3a1c5282e1770111902cc4af526c` |
| parameters | 888,090 (binned head, `max_classes 10`) |
| used by | the regression arm of the TabArena entry, run `fintfm_lp_reg_rawpart4_pure_1114`; its queue entry `reg-part-bag4` sets `REG_CKPT` to this path (the default `REG_CKPT` of `lite_reg_probe.sh` is the older mixed `v4-regression.pt`, which the entry does not use) |

## Intended use

Frozen representation for a per-leaf ridge readout on regression tasks (FinTFM-LinProbe with four
partition bags). Research and benchmarking.

## Training data

Synthetic prior only: `--p-regression 1.0` (every task a synthetic regression task), binned head,
`--max-classes 10`, 6,000 steps x batch 8 = **48,000 tasks**, seed 0, Kaggle GPU, `fintfm==0.5.6`
from PyPI (`scripts/kaggle/task_48_3_binned_control/`; §148). The checkpoint stores no commit
(no `provenance` field); the kernel script and the pinned package version are the record. No
real data.

## Evaluation (all MEASURED, TabArena-Lite regression, repeat 0 fold 0, CPU, 13 datasets)

- First use in place of the mixed checkpoint, one bag: 582 Elo against 578, RMSE lower on 8 of
  13 datasets (§227).
- Entry run `fintfm_lp_reg_rawpart4_pure_1114`: **689 Elo**, better than the one-bag run on 12
  of 13 datasets (sign test p < 0.01) (§235).

## Limitations

- **Raw-only control (§240).** Without the representation (`no_rep`, effectively four bagged
  trees with a ridge per leaf) the entry scores **537 Elo against 689**; the representation is
  better on 12 of 13 datasets (p = 0.003), median RMSE 11.4% lower, largest on airfoil_self_noise
  where raw only is 88.6% worse. Food_Delivery_Time, QSAR_fish_toxicity and QSAR-TID-11 differ
  by under 1%.
- A 480,000-task pure-regression checkpoint did not lift regression (§233); the 48,000-task one
  is what the entry uses. One seed per arm, one fold.
- Weights licence is not decided in this card.

LICENCE: OWNER DECISION
