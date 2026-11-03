# Inference-only open tasks, 2026-11-03 (draft for FINDINGS)

Draft, not yet merged into `docs/results/FINDINGS.md`. Covers three proxy-suite arms run for the
inference-only open tasks AU4.3 and AU5.12 (`openspec/changes/audit-2026-10-15/tasks.md`; AU5.12
also serves DR3.5's open `ensemble_class_shift` item), and the V4FinBench arms written for AU4.2
and AU5.8 but not run. The agent that started this (2026-11-02) stopped before writing it up;
the numbers below were read from its outputs afterwards.

**Setup.** Published checkpoint `runs/v4-cellattn-labels.pt` (48,000 tasks), zero-shot. Proxy
suite, 18 tasks, `--no-baselines --n-jobs 2 --device cpu`, one arm at a time under a 12 GB
free-memory guard; outputs in `runs/inf2611/`. 48 of 54 splits scored per arm: the 6 skipped
are the two tables wider than the checkpoint's 136 features, Internet-Advertisements (1,558) and
madelon (500). A proxy result advances a task; it does not decide a credit claim (CLAUDE.md, "Score on the benchmark this
project claims"). V4FinBench decides.

## Results, MEASURED

| arm | task | control | per task (W/L, p) | per split (W/L/T, p) | Elo, control 1,000 |
| --- | --- | --- | --- | --- | --- |
| `au43-ctx900-ens3`: `max_context` 900, `n_ensemble` 3 | AU4.3 | FinTFM-binary, 2,000-row single member (`runs/dr73/proxy_binary.csv`) | 11/5, 0.2101 | 25/21/2, 0.6587 | 1023 [964, 1091] |
| `au512-ens8-cshift`: 8 members, `ensemble_class_shift` | AU5.12, DR3.5 | `lever-ens8` (`runs/dr73/proxy_lever_ens8.csv`) | 11/5, 0.2101 | 30/17/1, 0.0789 | 1094 [979, 1216] |
| `au512-ens8-svd16`: 8 members, `ensemble_svd_components` 16 | AU5.12 | `lever-ens8` | 12/4, 0.0768 | 31/14/3, **0.0161** | 1126 [1021, 1259] |

Commands: `uv run python -m fintfm.evaluation.proxy_suite compare runs/inf2611/proxy_<arm>.csv
<control csv> --unit task|split --anchor <control>`.

- **AU4.3, null.** A context inside the trained range (at most ~921 rows), ensembled to about
  2,700 rows, does not beat the 2,000-row default on the proxy. Its V4FinBench arm is still the
  task's Verify clause; on this evidence it is low priority.
- **Class shift, null, leaning positive.** Neither leg reaches p < 0.05.
- **SVD members, a lead.** The only arm here with a significant leg (per split), and the
  task-level Elo interval excludes the control. Not a result until scored on V4FinBench against
  §196's `ens8` (+0.0058 mean AP, not adopted).

## Written, not run: V4FinBench arms (full panel, needs Kaggle or a quiet Mac)

| config | task | what it changes |
| --- | --- | --- |
| `configs/arms/au42-prototype.yaml` | AU4.2 | `context_strategy: prototype`, k-means prototype undersampling |
| `configs/arms/au58-retrieval-pc16.yaml` | AU5.8 | `context_strategy: retrieval`, `retrieval_per_class: 16` |
| `configs/arms/proxy/au512-ens8-svd16.yaml` | AU5.12 | the SVD lead above, at full panel |

Each is scored five-fold against zero-shot uniform (AU4.2), §163's arm (AU5.8) or §196's `ens8`
(SVD) under §159's rule. Full-panel scoring is not run on this Mac while memory is short
(the 2026-10-27 restart).

## Not touched by this pass

AU5.7, AU5.9, DR7.5, DR8.1 and DR8.3 have no new measurement here; their task lines are current.
