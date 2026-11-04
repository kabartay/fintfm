# DR3.2: 8 ensemble members against 1, and a class-shift arm (draft, 2026-11-04)

Task text (`openspec/changes/design-review-2026-10-16/tasks.md`): "V4: 8 ensemble members against
1 at 2,000 rows, and a second arm adding `ensemble_class_shift`. Same rule. Verify: per-fold AP
for each arm in FINDINGS." The preceding task, DR3.1, states "Rule: §163's (mean delta above the
bar, 4 of 5 folds the same sign)". §163's bar was 0.0073 and §173 (MEASURED, 2026-10-19) replaced
it with 0.0117 mean AP for arms compared on five folds, "the larger of 0.0073 and 0.0117".

## Gap first

- The class-shift arm has not been run. Nothing below about it is a measurement.
- DR3.2's first arm already exists as §196 and needs no new compute; it only needs the verdict
  under the stated rule written against it.

## Arm 1: already measured, §196 (MEASURED, 2026-10-26)

`runs/v4-cellattn-labels.pt`, `n_ensemble: 8` (`configs/arms/proxy/ens8.yaml`), `max_context`
2,000 (packaged default, confirmed in `runs/v4-lever-ens8/v4_protocol.json`), uniform context,
`fintfm-v4protocol --no-boosting --device cpu`, full panel (`max_rows: null`), five folds.
Control: the same checkpoint at the packaged defaults (1 member, 2,000 rows), `runs/lrsweep/eval-published`,
identical rows per fold (the comparison script refuses otherwise). It is the earlier published
score, not a same-day rerun (§196). Average precision first:

| fold | n_train | n_test | test positives | default AP | `ens8` AP | diff | 95% CI (companies) | Holm p (companies) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 600,120 | 200,361 | 741 | 0.2113 | 0.2091 | -0.0021 | [-0.0080, +0.0042] | 0.550 |
| 1 | 598,904 | 200,822 | 698 | 0.2050 | 0.1968 | -0.0083 | [-0.0172, -0.0002] | 0.088 |
| 2 | 599,873 | 199,392 | 700 | 0.1786 | 0.1892 | +0.0106 | [+0.0054, +0.0163] | 0.002 |
| 3 | 600,789 | 199,906 | 730 | 0.1810 | 0.1997 | +0.0187 | [+0.0120, +0.0252] | 0.002 |
| 4 | 600,575 | 199,606 | 718 | 0.2172 | 0.2271 | +0.0099 | [+0.0042, +0.0158] | 0.002 |
| mean | | | | 0.1986 | 0.2044 | +0.0058 | | |

Row counts are from `v4_protocol.json` (identical in both runs); the AP table is §196's, and the
paired company-clustered bootstrap is
`scripts/compare_v4_scores.py --cluster --control runs/lrsweep/eval-published ens8=runs/v4-lever-ens8`
(saved output `runs/v4-lever-ens8/cmp_vs_default.txt`; per-row predictions exist for both arms,
`predictions_fold{0..4}.npz`). Mean ROC-AUC 0.9890 against 0.9886 (§196).

**Verdict under the stated rule: null.** Mean +0.0058 AP is below the 0.0117 bar (half of it), and
3 of 5 folds are positive, below the 4 of 5 required. Three folds survive Holm upward and fold 1's
interval sits just below zero, so the effect is fold-dependent, which §196 recorded as a lead for
DR6.4. Default stays at 1 member. Cost (§196, ESTIMATED from wall times on a loaded shared Mac, not
a matched comparison): about 7 h per fold against about 1.1 h per fold for §194's one-member
retrieval run.

## Arm 2: `ensemble_class_shift`, not run

Config: `configs/arms/dr32-ens8-cshift.yaml` = arm 1's `inference` block plus
`ensemble_class_shift: true` (member i relabels the context by a cyclic shift of i, mapped back
before averaging; on a binary task, a label swap on every other member). Same checkpoint, same
protocol, so it differs from `ens8` in the shift alone and from the control in members and shift.
Compare it against both: the control (the task's rule) and `ens8` (isolates the shift).

Command (from the repository root; needs the `data/cache` link; do not launch without a go-ahead):

```bash
cd /Users/morganoko/proj/fintfm && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
nice -n 19 uv run --frozen --extra bench --extra hf fintfm-v4protocol \
  --model runs/v4-cellattn-labels.pt --config configs/arms/dr32-ens8-cshift.yaml \
  --no-boosting --device cpu --out runs/dr32/ens8-cshift > runs/dr32-ens8-cshift.log 2>&1; echo EXIT=$?

uv run --frozen --extra bench --extra hf python scripts/compare_v4_scores.py --cluster \
  --control runs/lrsweep/eval-published cshift=runs/dr32/ens8-cshift
uv run --frozen --extra bench --extra hf python scripts/compare_v4_scores.py --cluster \
  --control runs/v4-lever-ens8 cshift=runs/dr32/ens8-cshift   # shift-only contrast
```

**Cost, ESTIMATED:** the class shift changes only how each member's context is labelled, so the
forward-pass count matches `ens8`. Taking §196's about 7 h per fold under load, five folds are
about 35 h of wall time on 2 threads, at least 14 GB free memory, in competition with the 50-hour
DR3.4 job and the other queued work. The estimate has no measured basis for the class-shift arm
itself; a one-fold run (`--folds 0`) would give the first
real figure.

**Rule for arm 2 (as DR3.2 states it):** helps if the five-fold mean AP gain over the control
exceeds 0.0117 with at least 4 of 5 folds positive; hurts under the mirror; otherwise null.

## Related, not a substitute

`configs/arms/proxy/au512-ens8-cshift.yaml` ran the same config on the proxy suite (AU5.12, see
`docs/results/drafts/inference_tasks_2026-11.md`). The proxy is not validated against TabArena
(DR4.2), and CLAUDE.md puts V4FinBench first for credit claims, so it does not close DR3.2.
