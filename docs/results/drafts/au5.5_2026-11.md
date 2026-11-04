# AU5.5: label-symmetry probe, before and after `--permute-classes` (2026-11-04)

**Result.** Verify met on the probe's own terms. Against the multiclass control
`v4-multiclass.pt`, both checkpoints trained with `--permute-classes` show higher label
symmetry on every source, at both seeds. The probe is binary only, so it measures the
two-class case of a model trained mostly on multiclass tasks. No accuracy claim is made.

## Settings (MEASURED, CPU)

`scripts/label_symmetry_probe.py`, §165's settings: defaults (`--n-tasks 50` = 25 per prior,
512 rows, context fraction 0.5, `--real-draws 3`, 512 context + 512 query rows, seed 0) plus
`--allow-tabarena-for-reporting` (Polish and Taiwanese are TabArena-v0.1 datasets; reporting
only, nothing selected on them) and `--threads 2`. One task per prior was skipped at every
checkpoint because its context held one class (24 of 25 used), as in §165. Same machine and
session for all three runs, `nice -n 19`. Outputs: `runs/agents1104/AU5.5/`.

## Gaps first

- §165's "before" was on `runs/v4-cellattn-labels.pt`, not `v4-multiclass.pt`. The control
  here was re-measured on `v4-multiclass.pt`, so §165's numbers are not the "before".
- The after checkpoints differ from the control in more than the permutation: E1's training
  protocol (fp16, split attention, streaming loader, DDP), about five weeks of code, and a
  different training draw (§185). With `--permute-classes` off the prior was checked to match
  the control's batches (§185). So this is "recipe with permutation", not permutation alone.
- One probe seed (0), 24 tasks per synthetic source, 3 draws per real panel. No intervals.
- The probe is binary only. It does not measure symmetry over 3 to 10 classes.
- The control has no provenance record (trained before that field existed), so its
  permutation-off state rests on §185, not on the checkpoint.

## Provenance (checked on the checkpoints)

`dr74-mc-s0` and `dr74-mc-s1`: stored `prior_config.permute_classes` true, `scm_legacy` true,
`p_financial` 0.5, `max_classes` 10, 6,000 steps, 48,000 tasks seen; flags per §185 and §197
(kernels `scripts/kaggle/ts_dr74_mc_s{0,1}/`, `--seed 0` and `--seed 1`). Same ModelConfig
family as the control (§185).

## Probe statistics (MEASURED)

Pooled Pearson r between p and 1 - p_swapped (1 = symmetric), median per-task r, share of
tasks with r below 0, mean absolute change.

| source (tasks, queries) | checkpoint | pooled r | median r | share <0 | mean abs diff |
| --- | --- | --- | --- | --- | --- |
| synthetic financial (24, 6,144) | control `v4-multiclass` | 0.926 | 0.564 | 0.00 | 0.0187 |
| | `dr74-mc-s0` | 0.982 | 0.917 | 0.00 | 0.0095 |
| | `dr74-mc-s1` | 0.980 | 0.931 | 0.00 | 0.0091 |
| synthetic SCM (24, 6,144) | control | 0.976 | 0.757 | 0.12 | 0.0457 |
| | `dr74-mc-s0` | 0.993 | 0.960 | 0.00 | 0.0251 |
| | `dr74-mc-s1` | 0.989 | 0.955 | 0.00 | 0.0325 |
| Polish bankruptcy 3y (3 draws, 1,536) | control | 0.726 | 0.684 | 0.00 | 0.0148 |
| | `dr74-mc-s0` | 0.951 | 0.942 | 0.00 | 0.0063 |
| | `dr74-mc-s1` | 0.953 | 0.963 | 0.00 | 0.0055 |
| Taiwanese bankruptcy (3 draws, 1,536) | control | 0.887 | 0.880 | 0.00 | 0.0191 |
| | `dr74-mc-s0` | 0.981 | 0.978 | 0.00 | 0.0065 |
| | `dr74-mc-s1` | 0.964 | 0.968 | 0.00 | 0.0078 |

Cost (MEASURED): control 1,229 s, 1.85 GB peak RSS; s0 485 s, 1.32 GB; s1 661 s, 1.27 GB. A
2-task smoke run took 27 s, 0.98 GB (SMOKE-TEST). The control ran first, alongside other load.
Free memory read 12 GB, below the 14 GB gate, when s1 was launched; the process peaked at
1.27 GB.

## Reading

- Median per-task r on the financial prior rises from 0.564 to 0.917 and 0.931; on SCM from
  0.757 to 0.960 and 0.955. The control's 12% of SCM tasks below zero (3 of 24, derived) is 0
  at both seeds.
- Polish, the least symmetric source in §165, moves from 0.726 pooled r to 0.951 and 0.953.
- Same direction at both seeds on all four sources. The control's pooled r on the synthetic
  sources was already high (0.926, 0.976), so the pooled gain there is small; the per-task
  median is where it was weak.
- Not shown: that symmetry helps accuracy. §185 and §197 carry the proxy log-loss result.
