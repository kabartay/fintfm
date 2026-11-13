# Running FinTFM on TabArena

External evaluation, under someone else's protocol, against 94 other methods. This file is
the reproduction recipe and the list of things that cost a run.

**Result: 94th of 95.** §122 measured mean ROC-AUC **0.7823** over all 27 eligible datasets
with zero failures, against `fintfm==0.5.5` from PyPI and the published checkpoint. The
submission was reviewed by a TabArena collaborator, who confirmed the integration and the
numbers are correct and declined the entry because it does not meet the leaderboard's
competitiveness requirement.

Earlier runs reported 93rd [§225: read from a table numbered from 0; §98's own leaderboard is not stored]. §98 first measured mean ROC-AUC **0.7642** over 26 of 27 eligible
datasets; §100 located most of that deficit in categorical preprocessing rather than
architecture, and §101's fix lifted the mean to **0.7823** with the rank unchanged. Every
binary arm since has landed between 0.7746 and 0.7926, and none has moved the rank, except
§132's unpublished linear-probe variant: Elo 910, 93rd of 95 [§225: one place lower].

(2026-11-13, §224): the figures above are the submitted entry, zero-shot and binary only. The
current work is a different entry, "FinTFM with a trained linear readout", over all 51
datasets: **903 Elo (+102 / -110), 107th of 110** on TabArena-Lite (repeat 0, fold 0). It is
Lite-only, not a public-board number, and the PR is on hold (DR7.6). See "The 51-dataset Lite
entry" at the end of this file.

## Why bother, given the rank

Every accuracy number this project produced before §98 was measured against baselines this
project ran itself. That is not a criticism of those measurements — they use paired bootstrap,
Holm correction and tuned baselines — but it is a closed loop, and a closed loop cannot detect
a harness-level mistake that flatters every arm equally. TabArena breaks the loop: the split,
the metric, the baselines and the leaderboard all belong to someone else.

It also turned a vague worry into an ordered work queue. Before §98 "we should probably handle
categoricals" was an opinion; after §100 it is 82% of a measured gap.

## Where the integration lives

The wrapper is **not** in this repository — it is a model plugin inside a TabArena clone,
because TabArena discovers models from its own registry:

```text
<tabarena clone>/packages/tabarena/src/tabarena/models/fintfm/
    model.py   # FinTFMModel(AbstractTorchModel), ag_key = "TA-FINTFM", ag_name = "FinTFM"
    info.py    # ModelInfo(model_cls=..., search_space=..., method_metadata=...)
<tabarena clone>/run_fintfm_lite.py
```

TabArena's `fintfm` extra pins `fintfm==0.5.5` from PyPI (`packages/tabarena/pyproject.toml`).
To test an unreleased change, install the local checkout into the TabArena environment as an
editable install instead, as §132 did; a change here is then live there with no reinstall.

(2026-11-13): the pin above is the one PR #616 was submitted with. The fork's `fintfm` extra now
pins `fintfm==0.10.0` (fork commit 48c0101 on `fintfm-audit-fixes`, pushed 2026-11-13). §220
reproduced §215's configuration bit for bit on 27 of 27 binary datasets from the 0.9.0 PyPI wheel.

## Running it

```bash
export FINTFM_CHECKPOINT=/path/to/fintfm/runs/v4-cellattn-labels.pt
FINTFM_RUN_NAME=my_run python run_fintfm_lite.py --full
```

| variable | purpose |
| --- | --- |
| `FINTFM_CHECKPOINT` | optional override; by default the wrapper fetches the published weights from the Hugging Face Hub at a pinned commit |
| `FINTFM_RUN_NAME` | results directory. **See the caching trap below — this is not cosmetic.** |
| `FINTFM_DATASETS` | comma-separated subset, for smoke-testing a change before paying for the suite |
| `FINTFM_CATEGORICAL` | `target` (default) or `label`, to A/B the encoding on one checkpoint |

## Four things that cost a run, all of them silent

**Results are cached per (config, dataset, fold), and a preprocessing change does not
invalidate that cache.** Re-running in place after changing `_preprocess` returned scores
**identical to four decimal places**, which reads exactly like "the change had no effect". It
had not run at all. Always give a changed configuration a new `FINTFM_RUN_NAME`; the side
benefit is that the previous run's raw results survive for comparison.

**AutoGluon's GPU accounting is CUDA-only.** On Apple Silicon it reports zero GPUs, so the
first version of the wrapper ran on CPU and hit `TimeLimitExceeded` on a 45,211-row dataset.
Preferring MPS explicitly fixed it and the suite then completed in 1.5 h. **That timeout was an
integration defect and must never be cited as evidence about fintfm's deployability** — it is
the kind of number that gets quoted out of context precisely because it looks like a model
result.

**`seed_name = "random_state"` injects the seed into the params dict.** Passing it explicitly
as well raises a duplicate-argument error. Use `params.setdefault("random_state", ...)`.

**TabArena's task metadata does not populate its own categorical columns.** `has_categorical`,
`has_high_cardinality_categorical` and `num_high_cardinality_cats` are `None` for all 51
datasets in the current release. A split computed from them is vacuous and, worse, looks
plausible — every dataset lands in the "no categoricals" bucket. Read feature types from
OpenML instead (`openml.tasks.get_task(tid).get_dataset().get_data()`), as §100 does.

**More silent or costly traps found since** (2026-11-13; each is sourced where it is described):

- **A problem-type filter that leaves zero eligible datasets does not fail.** The script prints
  the existing leaderboard, which reads like a result (the DR7.4 multiclass run, "E1's
  480,000-task checkpoint" below).
- **Env-var options do not change the config name**, so the caching trap applies to
  `FINTFM_PARAMS` and `FINTFM_PROBE` as well (the fork's `fintfm_linprobe/model.py` says so in
  the docstring of `_probe_options`). Give each arm a new `FINTFM_RUN_NAME`.
- **MPS memory grows across fits in one process** (85 GB against 3.3 GB of live tensors), so
  Apple-Silicon arms ran one process per dataset ("Inference arms" below).
- **A readout that re-reads training rows through a private method can see a wider table than
  prediction does.** This one was loud: `feature width at predict time must match fit time`
  stopped Bioresponse, QSAR-TID-11 and the first multiclass run until fork commit f9ea61d
  (§220). No earlier Lite number was affected.
- **A dataset near the time limit passes by luck.** Amazon_employee_access hit AutoGluon's
  `TimeLimitExceeded` after 482 s under probe, squash and partition, and fitted in the next call
  (§215). Treat it as a risk for any submission run.
- **A merge can pass with the wrong count.** `merge_fintfm_lite.py` refuses a count other than
  `--expect`, so always pass `--expect 51` (§221).

## Coverage, which must be reported with any score

The suite is 30 binary, 13 regression and 8 multiclass. Coverage has moved twice:

| declared | eligible | why the rest are excluded |
| --- | --- | --- |
| `["binary"]` | **26 of 51 (51%)** | 13 regression, 8 multiclass, 3 wide, 1 over 10 classes |
| `+ multiclass` | **34 of 51 (67%)** | 13 regression, 4 wide |
| `+ regression` | **46 of 51 (90%)** | 5 wide — `max_features=136`, nothing else |
| **`["binary"]` again — current** | **27 of 51 (53%)** | **21 undeclared problem types, 3 wide** |

**The last row is a reversion, not a regression.** Multiclass and regression were declared,
then scored on real data for the first time, and both rank last — multiclass 93.4 of 95 over
7 datasets, regression 93.1 of 94 over 12 and **last on 5 of them** (§121). They remain
implemented and tested; declaring them would publish a capability the measurement says is not
there. **A higher coverage fraction is not worth a claim that does not hold.**

(2026-11-13, §221): the "5 wide" datasets now run. `max_features_strategy: select` admits them
without a new checkpoint, and the 51-dataset Lite entry scores 51 of 51 with no exclusion (all
three problem types declared, each with its own checkpoint). Its ranks on the wide tables are low
(§220, §221); see the last section. The last table row is the submitted entry, not the current one.

A mean over 27 datasets is not comparable with a mean over 51, and the leaderboard's Elo is
computed only where a method actually ran, so the fraction goes beside the number every time.

**Read the coverage number off `eligible_datasets()`, never off this table.** It is derived
from TabArena's own task metadata **and from the model's own `_supported_problem_types`** at run
time, which is what stops the documented fraction drifting from the executed one — the fraction
here is a record of what was measured when, not an input to anything.

That derivation was itself a bug fix. The runner used to restate the problem types in its own
tuple, so narrowing the adapter to binary left it reporting 90% coverage for a model that would
refuse 19 of those datasets. Mirrored values drift; the single source of truth already existed.

Both extensions are checkpoint-gated, not merely declaration changes: multiclass needs
`max_classes >= K`, and regression needs a checkpoint whose prior emitted continuous targets
at all. Declaring a problem type whose checkpoint is missing would score the wrong model,
so the adapter selects by problem type (`FINTFM_CHECKPOINT_MULTICLASS`,
`FINTFM_CHECKPOINT_REGRESSION`) and raises when the head is too narrow.

`FINTFM_PROBLEM_TYPES` narrows a run to a subset of them, which is how a binary score stays
comparable with §98/§101 after regression was added.

Capability limits are declared rather than worked around — a task the checkpoint cannot
represent raises, so the harness records a skip instead of a meaningless score. The one
exception is regression's bin count, which is *our* parameter rather than the task's: a
10-bin request against an 8-logit head is clamped to 8 and logged, because the coarser grid
is a real answer where a truncated class set would not be.

## The imbalanced subset, from results already held (2026-10-16)

**Result: Elo 914.1 (+127.3/-343.6), 87th of 95 (tied with `TABSTAR (default)`), over 7 of the
subset's 15 datasets.** MEASURED by re-aggregation on 2026-10-16 (design review task DR3.6): no
model was fitted. Below `RF (default)` (1000.0, 84th) and `LR (default)` (1021.2, 82nd) on the
same 7 datasets, so it does not change §122's reading. The interval is wide because 7 datasets
are few: its lower end, 570.5, is below the 764.6 of all 27.

| scope | datasets | Elo | position of 95 | win rate | mean rank |
| --- | --- | --- | --- | --- | --- |
| all eligible (§122, reproduced) | 27 | 764.6 (+124.9/-175.3) | 94 | 0.074 | 88.04 |
| **imbalanced, covered** | **7** | **914.1 (+127.3/-343.6)** | **87** | 0.147 | 81.14 |

**The subset.** TabArena-v0.1.9's `imbalanced` subset (added in v0.1.9, 2026-09-17): a
classification dataset whose largest class is at least 10 times its smallest, or a regression
dataset whose target has absolute skewness of at least 1. Read from the fork's own predicate
(`tabarena/benchmark/task/metadata/balance.py`, `IMBALANCE_RATIO_THRESHOLD = 10.0`,
`TARGET_SKEWNESS_THRESHOLD = 1.0`), not restated: 15 of 51 datasets, 9 binary, 3 multiclass, 3
regression.

**Covered: the 7 binary datasets FinTFM ran.** `Amazon_employee_access`, `Diabetes130US`,
`GiveMeSomeCredit`, `coil2000_insurance_policies`, `polish_companies_bankruptcy`,
`seismic-bumps`, `taiwanese_bankruptcy_prediction`. Not covered, 8: `APSFailure` (170 features)
and `kddcup09_appetency` (212) are over the 136-feature cap; `MIC`, `anneal` and `hiva_agnostic`
are multiclass and `diamonds`, `healthcare_insurance_expenses` and `miami_housing` are
regression, neither declared (§121). An Elo over all 15 would need FinTFM's 8 missing datasets
imputed from another method, which would score that method rather than FinTFM, so it was not
computed.

**Pool: §122's, unchanged.** The fork's `eval/fintfm_lite_tafintfm/results_per_split.csv`,
written by §122's `context.compare` on 2026-09-25: 95 methods by 27 datasets, split 0 (Lite).
Re-scored with the fork's `TabArenaContext.compare(ta_results=..., datasets=..., fillna=None)`,
Elo anchored at `RF (default)` = 1000. Re-scoring all 27 reproduces §122 exactly (764.6, 94 of
95), which is the check that the pool and the method are the same. Command, in the fork's
environment and read-only:

```bash
~/proj/tabarena/.venv/bin/python -P scripts/tabarena_subset_elo.py \
    ~/proj/tabarena/eval/fintfm_lite_tafintfm/results_per_split.csv --subset imbalanced
```

**Imputation.** None for FinTFM. In the 7-dataset frame, 12 of 665 rows are imputed, as §122
left them: `TABPFNV2_GPU` (default, tuned, tuned + ensemble) and `MITRA_GPU (default)` on
`Amazon_employee_access`, `Diabetes130US` and `GiveMeSomeCredit`, filled from `RF (default)`
(`MITRA_GPU`'s rows carry no `impute_method`).

**These Lite results were produced with NaN filled with 0.** They predate AU1.19: the adapter
then filled numeric NaN with 0 before the model saw it, so the model's missingness mask was
never used and a missing value was indistinguishable from a real zero. The fork's adapter now
passes NaN through; nothing here was produced by that version, and DR3.5's same-day A/B is
what will measure the difference. The fork's task metadata records no missing-value fraction for
these 7 datasets, so which of them the fill affected is not stated here.

**Per dataset**, ROC-AUC on split 0 (the only metric the cached results hold, so no average
precision is available for these low-base-rate sets):

| dataset | FinTFM | RF (default) | LR (default) | best of 95 | FinTFM rank of 95 |
| --- | --- | --- | --- | --- | --- |
| Amazon_employee_access | 0.8220 | 0.8415 | 0.8411 | 0.8855 | 81 |
| Diabetes130US | 0.6566 | 0.6372 | 0.6527 | 0.6950 | 69 |
| GiveMeSomeCredit | 0.8373 | 0.8532 | 0.8444 | 0.8735 | 91 |
| coil2000_insurance_policies | 0.7020 | 0.6828 | 0.7359 | 0.7758 | 90 |
| polish_companies_bankruptcy | 0.8436 | 0.9357 | 0.8727 | 0.9973 | 91 |
| seismic-bumps | 0.7862 | 0.7650 | 0.7788 | 0.8088 | 63 |
| taiwanese_bankruptcy_prediction | 0.9291 | 0.9458 | 0.9419 | 0.9598 | 83 |
| **mean** | **0.7967** | 0.8087 | 0.8096 | | 81.14 |

FinTFM is above `RF (default)` on 3 of 7 (`Diabetes130US`, `coil2000_insurance_policies`,
`seismic-bumps`) and above `LR (default)` on 2 of 7 (`Diabetes130US`, `seismic-bumps`). Ranks
use average ties over the 95 methods' split-0 errors.

**Not a public-board number.** The published imbalanced board scores all splits of all 15
datasets, and FinTFM is not on it. This is one split, 7 datasets, against §122's 2026-09-25
snapshot of the board.

## The imbalanced subset, every split (AU5.18, 2026-11-06)

**Result: Elo 1018.2 (+142.1/-332.8), 85th of 96, over the 7 covered datasets of the subset's
15, all 9 splits each (63 fits, none imputed).** MEASURED, 2026-11-05/06, the default config of
the published checkpoint through the fork's adapter (branch `fintfm-audit-fixes` at be201d7, NaN
passed through), on MPS. Above `RF (default)` (1000.0, 88th), below `LR (default)` (1086.7,
83rd). The interval is as wide as DR3.6's: 7 datasets are few.

| scope | splits | datasets | Elo | position | win rate | mean rank |
| --- | --- | --- | --- | --- | --- | --- |
| DR3.6, re-aggregated from §122 (above) | 1 | 7 | 914.1 (+127.3/-343.6) | 87 of 95 | 0.147 | 81.14 |
| **AU5.18, fitted** | **9** | **7** | **1018.2 (+142.1/-332.8)** | **85 of 96** | 0.176 | 79.30 |

**Not comparable to DR3.6 as a change.** Three things differ at once: nine splits against one,
the fork's current pool of 95 other methods against §122's 2026-09-25 snapshot (the top three
are now `TA-LIMIX-2`, `TA-TABPFN-3.5` and `MITRA_V2`), and NaN passed through against filled
with 0. The per-dataset standing did not move: above `RF (default)` on the same 3 of 7 and
above `LR (default)` on the same 2 of 7.

**Per dataset**, ROC-AUC averaged over the 9 splits (the only metric the results hold; no
average precision for these low-base-rate sets):

| dataset | FinTFM | RF (default) | LR (default) | best of 96 | FinTFM rank of 96 |
| --- | --- | --- | --- | --- | --- |
| Amazon_employee_access | 0.8302 | 0.8392 | 0.8482 | 0.8864 | 83 |
| Diabetes130US | 0.6524 | 0.6308 | 0.6480 | 0.6834 | 66 |
| GiveMeSomeCredit | 0.8251 | 0.8464 | 0.8408 | 0.8691 | 94 |
| coil2000_insurance_policies | 0.7200 | 0.6970 | 0.7368 | 0.7878 | 88 |
| polish_companies_bankruptcy | 0.8498 | 0.9274 | 0.8675 | 0.9966 | 92 |
| seismic-bumps | 0.7711 | 0.7475 | 0.7587 | 0.7940 | 45 |
| taiwanese_bankruptcy_prediction | 0.9259 | 0.9325 | 0.9359 | 0.9589 | 87 |
| **mean** | **0.7964** | 0.8030 | 0.8051 | 0.8537 | 79.29 |

Ranks are of each method's 9-split mean error, average ties. The two credit-default sets
(`GiveMeSomeCredit`, 94th) and the corporate-bankruptcy sets (`polish_companies_bankruptcy`
92nd, `taiwanese_bankruptcy_prediction` 87th) are among FinTFM's worst; its best is
`seismic-bumps` (45th).

**Imputation.** None for FinTFM. 108 of 6,048 rows are imputed in the frame, the same four
methods as DR3.6 (`TABPFNV2_GPU` default, tuned and tuned + ensemble, `MITRA_GPU (default)`) on
`Amazon_employee_access`, `Diabetes130US` and `GiveMeSomeCredit`.

**How it was run.** `scripts/tabarena_imbalanced_run.py --run-name fintfm_imbalanced_1105`, from
the fork's root in its environment; results in the fork's `experiments/fintfm_imbalanced_1105/`,
leaderboard in `eval/fintfm_imbalanced_1105/`; logs in `runs/au518/`. A single process grew
memory across fits on MPS and was stopped by a 10 GB-free watchdog after one dataset; the rest
ran one process per (dataset, repeat) of 3 splits (`runs/au518/queue.sh`), with results cached
per (config, dataset, split), and a final unnarrowed pass fitted the 4 `Diabetes130US` splits
the watchdog had stopped and printed the row. Wall time about 9 h, of which `GiveMeSomeCredit`
took 4.

**Not a public-board number.** The published imbalanced board scores all 15 datasets; FinTFM
covers 7 and is not on it. Imputing the other 8 from another method would score that method.

## NaN passed through against NaN filled with 0 (DR3.5, AU1.19; 2026-10-19)

**MEASURED, null.** TabArena-Lite, the 27 binary datasets, split 0, the published checkpoint
`v4-cellattn-labels.pt`, run back to back on 2026-10-19 from one fintfm source (editable install)
and one fork; only the adapter's missing-value handling differs (branch `fintfm-audit-fixes`,
93821e5, against `add-fintfm`). Logs: `runs/dr35/nan.log`, `runs/dr35/zerofill.log`; results under
the fork's `experiments/fintfm_lite_{nan,zerofill}_1019/`.

21 of 27 datasets give bit-identical ROC-AUC, as they should if they hold no missing values. The
6 that differ, ROC-AUC on split 0:

| dataset | NaN passed through | NaN filled with 0 | difference |
| --- | --- | --- | --- |
| polish_companies_bankruptcy | 0.8619 | 0.8436 | +0.0183 |
| jm1 | 0.7050 | 0.7038 | +0.0012 |
| customer_satisfaction_in_airline | 0.9043 | 0.9043 | -0.0000 |
| Fitness_Club | 0.7932 | 0.7978 | -0.0046 |
| Marketing_Campaign | 0.8262 | 0.8309 | -0.0047 |
| GiveMeSomeCredit | 0.8294 | 0.8373 | -0.0079 |

Passing NaN through is better on 2 of 6 and worse on 4; the mean difference over all 27 is
+0.00008 ROC-AUC, and the two-sided sign test over the 6 gives p = 0.69. Elo, which the task
says does not decide, is 756 (+126/-180) with NaN passed through and 763 (+123/-175) with zero
fill, 94th of 95 either way. The zero-fill arm reproduces this page's earlier numbers
(`polish_companies_bankruptcy` 0.8436, `GiveMeSomeCredit` 0.8373, `Diabetes130US` 0.6566), so the
two runs are comparable.

**Decision: keep NaN passed through.** It is the encoding the model was trained on (its
missingness mask), the A/B finds no cost at this resolution, and the one large move
(`polish_companies_bankruptcy`, +0.0183) is in its favour; that dataset is a TabArena panel and
may not choose an arm (DR4.1), so it is reported, not used to decide. One split cannot separate a
real effect of 0.005 from noise; the result is "no measurable difference", not "no difference".

## E1's 480,000-task checkpoint on TabArena-Lite (2026-10-24, reporting only)

**MEASURED, no measurable change.** TabArena-Lite, the 27 binary datasets (27 of 51, 53%
coverage), the default config, NaN passed through, checkpoint `e1-v480-s0` (`runs/kaggle-ts-e1-
v480-s0/e1-v480-s0/e1-v480-s0.pt`, fintfm `FINDINGS.md` §176), the adapter's default `max_context` of 1000 (the run name says 1024; no override was set), fork branch
`fintfm-audit-fixes`. Log `runs/lite_e1v480s0.log`; results under the fork's
`experiments/fintfm_lite_e1v480s0_1024/`.

Elo **842 (+139/-213), 94th of 95 [§225: one place lower]**, against 756 (+126/-180) for the published checkpoint with
NaN passed through in the 2026-10-19 run above: +86 Elo, inside either interval, same rank. Lite
reports and never selects (DR4.1), and E1's verdict was already taken on V4FinBench (§188: null
at both seeds), so this changes nothing; it records that ten times the pretraining tasks does not
move the TabArena standing either.

The DR7.4 multiclass checkpoint was queued for the same run and **scored nothing**: the adapter
declares binary only (`_supported_problem_types`, narrowed 2026-09-25), so `FINTFM_PROBLEM_TYPES=
multiclass` left 0 eligible datasets and the script printed the existing leaderboard. Measuring
it needs the adapter to accept multiclass first.

**Measured on 2026-10-26, after the adapter gained an opt-in** (`FINTFM_ALLOW_MULTICLASS=1`,
tabarena fork 5cb019c; unset, the declaration stays binary only). `dr74-mc-s0` on the 7 multiclass
datasets within the caps (MIC, SDSS17, anneal, maternal_health_risk, splice,
students_dropout_and_academic_success, website_phishing), default config, log
`runs/lite_dr74_mc.log`: **Elo 306 (+76/-2619), 94th of 95, mean rank 93.86**, against §121's
93.4 mean rank on the same 7 for the earlier multiclass checkpoint. No gain on TabArena; reporting
only, so it does not weigh against §186's proxy candidate, which DR7.4's seed-1 replicate decides.

## Context size: 500 and 750 rows against 1000 (AU5.14, 2026-10-26)

**MEASURED, reporting only.** TabArena-Lite, the 27 binary datasets, the published checkpoint,
NaN passed through, the default config with `FINTFM_MAX_CONTEXT` set (logs
`runs/au514_ctx{500,750}.log`; results under the fork's `experiments/fintfm_lite_ctx{500,750}_1026/`).
Pretraining contexts reach about 921 rows, so 500 and 750 sit inside the trained range and 1000
just beyond it.

| `max_context` | Elo | interval | rank |
| --- | --- | --- | --- |
| 500 | 702 | +113/-179 | 94 of 95 |
| 750 | 745 | +125/-183 | 94 of 95 |
| 1000 (2026-10-19, above) | 756 | +126/-180 | 94 of 95 |

Elo rises with context and every interval overlaps the others; the rank does not move. **Keep the
default of 1000.** Staying inside the trained range buys nothing here, and the shorter contexts
point the other way. The 1000 row comes from the 2026-10-19 run, not a same-day rerun.

**Paired per dataset (DR3.5), MEASURED 2026-11-03 from the results above, no new fit.** Each arm
against `experiments/fintfm_lite_nan_1019/` (the 1,000-row run), split 0, ROC-AUC, per-dataset
`results.pkl`:

| `max_context` | better / worse / tied (of 27) | sign test p | mean ROC-AUC change |
| --- | --- | --- | --- |
| 500 | 12 / 14 / 1 | 0.85 | -0.00048 |
| 750 | 14 / 8 / 5 | 0.29 | +0.00096 |

Largest moves at 500: Marketing_Campaign +0.0180, churn +0.0084; Is-this-a-good-customer -0.0118,
Diabetes130US -0.0092. At 750: in_vehicle_coupon_recommendation +0.0155, churn +0.0096;
seismic-bumps -0.0061. Null both ways, consistent with the Elo table; the default stays at 1000.
The 1000-row arm is a 2026-10-19 run, not same-day.

## Inference arms against the 8-member default (AU5.7, AU5.12, DR3.5; 2026-11-04/05)

**MEASURED. One arm is a general-tabular gain that decides nothing yet.** TabArena-Lite, the 27
binary datasets, split 0, the published checkpoint, NaN passed through, MPS. Each arm changes one
inference setting of `FinancialTFMClassifier` against `fintfm_lite_nan_1019` (the adapter's default
of 8 members, `max_context` 1000, 2026-10-19; **not a same-day run**), paired per dataset with
`scripts/compare_lite_arms.py` on ROC-AUC. Baseline Elo 756 (+126/-180), 94th of 95 [§225: one place lower].

| arm | change | better / worse / tied | sign test p | mean ROC-AUC change | Elo |
| --- | --- | --- | --- | --- | --- |
| `fintfm_lite_ens1_1104` (AU5.7) | `n_ensemble` 1 | 7 / 20 / 0 | 0.019 | -0.00445 | 696 (+115/-196) |
| `fintfm_lite_cshift_1104` (AU5.12) | `ensemble_class_shift` | 10 / 17 / 0 | 0.25 | -0.00611 | 716 (+150/-218) |
| `fintfm_lite_svd16_1104` (AU5.12) | `ensemble_svd_components` 16 | 17 / 10 / 0 | 0.25 | +0.00306 | 782 (+128/-196) |
| `fintfm_lite_tfsq_1104` (AU5.12, DR3.5) | `ensemble_transforms` `("rank", "squash")` | 20 / 6 / 1 | 0.009 | +0.01023 | 807 (+128/-195) |

Largest moves. One member: Diabetes130US -0.0267, coil2000 -0.0229, airline -0.0171, hazelnut
-0.0158, Is-this-a-good-customer -0.0150; E-CommereShippingData +0.0175, jm1 +0.0056,
GiveMeSomeCredit +0.0053. Class shift: Is-this-a-good-customer -0.0477, Diabetes130US -0.0451.
SVD: blood-transfusion +0.0127, hazelnut +0.0122, churn +0.0118. Transforms: credit-g +0.0497,
churn +0.0412, E-CommereShippingData +0.0383, in_vehicle_coupon_recommendation +0.0348.

**Readings.** One member is worse than eight, so the adapter's 8 stays (with FINDINGS §196's V4
`ens8` +0.0058 mean AP, both halves of AU5.7 are measured). Class shift is a null that reverses its
proxy lean. SVD is a null lean. Transforms `rank, squash` clear the sign test on Lite, which the
proxy's 12 / 4 lean (p 0.077, `rank, power, squash`) pointed to. **That gain is general-tabular
only.** Per CLAUDE.md ("Score on the benchmark this project claims"), it needs a five-fold
V4FinBench mean AP beside it before the default changes, and that run is held by AU5.12. All Elo
intervals overlap, the rank was not computed per arm here, and Lite reports and never selects
(DR4.1).

**How they were run.** The fork adapter gained `FINTFM_PARAMS` on 2026-11-04 (uncommitted at the
time): a JSON object of `FinancialTFMClassifier` arguments overriding the adapter's defaults, with
an unknown or reserved key raising and an explicit hyperparameter still winning. For example
`FINTFM_PARAMS='{"n_ensemble": 8, "ensemble_transforms": ["rank", "squash"]}'`, with
`FINTFM_PROBLEM_TYPES=binary` and a new `FINTFM_RUN_NAME` per arm (the caching trap above). On MPS
the Metal-side allocations grew across fits in one process (85 GB against 3.3 GB of live tensors)
until the first arm died after about 8 datasets, so every arm ran **one process per dataset**
(`runs/lite2611/queue_ds.sh`); that changes no number, since each dataset's fit is independent and
cached.

**Not run: `context_strategy=retrieval`.** It could not be run locally: 0 of 4 datasets attempted
completed (MPS out of memory, then TabArena's time limit on CPU, which is the integration defect
described above, not a model result). It needs a CUDA run.

## The leaderboard name is `FinTFM`, with no version

The leaderboard renders `ag_name`, and it lists **models**, not releases — TabSTAR, OrionMSP,
TabFlex and iLTM all appear unversioned. `0.3` is a version of one model, so it belongs in
`docs/CHANGELOG.md` next to the numbers it produced, not in the row a reader compares against
TabPFN. Registered as `TA-FINTFM-0.3` through §105; renamed after.

**The rename moves the results cache.** AutoGluon derives the stored config name from
`ag_name`, so §98 / §101 / §105 live under `TA-fintfm-0.3_c1_default_BAG_L1` and are *not*
picked up under the new name. Nothing is lost and nothing is silently mixed — but a
comparison spanning the rename has to read those directories directly, which is how §105's
paired deltas were computed, rather than trusting the harness to merge them.

## On submitting a PR

**The conditions are now met, and they were met by losing an argument with the measurement.**
(Since submitted as PR #616 and declined, §122: binary-only coverage was ruled ineligible, and
the model does not beat simple baselines. See `docs/roadmap/TABARENA_BAR.md`.)

The original position was: not yet, and the ranking is not the reason. At 51% coverage with a
knowingly degraded categorical path, the submitted number would have measured the workaround as
much as the model. The stated order was native categoricals (47.x), then multiclass and
regression (46.x), then a submission whose number means what it says.

- **Native categoricals: done** (§101). Out-of-fold target statistics, mean 0.7642 → 0.7823.
- **Multiclass and regression: implemented, scored, and withdrawn** (§121). The condition was
  never "make them runnable" — it was that a declaration must have a measurement behind it. The
  measurement now exists and says they rank last, so they are not declared.

**So the submission declares `binary` only**: 27 eligible datasets, 53% coverage, rank 93 of
95 as measured then (§122 measured 94 of 95 against the board at submission), with every
declared capability measured. That is a narrower entry than the 90% one, and a truer one.

**The weights are now published, which was the last blocker.** A reviewer could not reproduce
a number without them. `kabartay/fintfm-binary` is public and **Apache-2.0** as of 2026-09-25 —
the ~3.5 MB `v4-cellattn-labels.pt`, on which every published binary number was measured. The
remaining checkpoints stay private deliberately: multiclass and regression rank last (§121),
and the scale and tree arms lost (§114, §116). Publishing weights this project has measured as
worse than the one it published would mislead rather than inform.

**Read a weights licence separately from a code licence, including our own.** Four of the ten
peer projects surveyed in `docs/paper/RELATED_WORK.md` ship permissive code with
non-commercial weights, and one restricts commercial use of the model's *output* rather than
merely the weights. Apache-2.0 on this repository says nothing about a checkpoint published
from it.

**Running without error is not working.** Both arms cleared every integration gate and both are
last; an integration test and a capability claim are different things.

## One entry over every problem type: the merge tool and a 46-dataset assembled estimate (2026-11-10)

**Tool.** The tabarena fork's `merge_fintfm_lite.py` (fork 8aefd8d) reads the cached
`results.pkl` of several Lite runs, registers them as one new method and places it on the Lite
leaderboard without fitting anything. It exists because `run_fintfm_lite.py` runs one problem
type per checkpoint, so a 51-dataset entry is spread over three run directories. Runs of
different entrants merge only under `--framework`, which relabels them; `merged_from.json` beside
the leaderboard records which run supplied each dataset. Queued as `merge-all51` after the three
FinTFM-LinProbe runs (`lite-090`, `lite-mc-pp`, `lite-reg-pp`), with `--expect 51`.

**Assembled estimate, MEASURED components and an ESTIMATED entry.** The best existing result per
problem type, merged as `FinTFM-assembled-46` (`eval/fintfm_assembled46_1110/` in the fork, log
`runs/merge_assembled46.log`):

| problem type | datasets | run | entrant |
| --- | --- | --- | --- |
| binary | 27 | `fintfm_linprobe_squash_part_1108` (§215) | FinTFM-LinProbe, squash, partition, probe per leaf |
| multiclass | 7 | `fintfm_lite_dr74_mc` | zero-shot FinTFM, DR7.4 checkpoint |
| regression | 12 | `cov_regression` (§121) | zero-shot FinTFM, binned regressor |

(2026-11-13, §221: superseded as a planning figure by the real 51-dataset merge, 880 Elo, then
§224's 903. Kept below as written.)

It scores **862 Elo (+120/−133), 107th of 110 methods, over 46 datasets.** That is above LR
(default) at 827 and KNN (tuned) at 797, and below LR (tuned) at 923 and RF (default) at 1000.
The entry mixes two entrants, so it is a planning figure for the 51-dataset decision, not a
submittable number. The 5 missing datasets are the wide tables. The binary part sits at parity
with RF (default) on its own (§215), and the other two problem types pull the whole entry down
to 862. Those two are what `lite-mc-pp` and `lite-reg-pp` measure with the probe.

**A PR also needs the checkpoints public.** Only the binary checkpoint is published
(`kabartay/fintfm-binary`, Apache-2.0). A 51-dataset entry runs the DR7.4 multiclass
checkpoint and `v4-regression.pt`, which stay private (see above), so the owner would have to
publish both before a submission could be reproduced.

## The 51-dataset Lite entry: "FinTFM with a trained linear readout" (§220 to §224; 2026-11-13)

**Result: 903 Elo (+102 / -110), 107th of 110, over all 51 TabArena-Lite datasets with no
exclusion** (§224; repeat 0, fold 0, CPU; merged run `fintfm_linprobe_all51_1113`, log
`runs/merge_all51_1113.log`). MEASURED. This is breadth only: TabArena-Lite
is general-tabular, V4FinBench decides (§211), and it is not a public-board number (the public
board scores every split). Below `Linear (tuned)` 919, `Linear (tuned + ensembled)` 936 and
`RandomForest (default)` 1,000. The PR is **held** (owner decision DR7.6, 2026-11-11) until
multiclass and regression improve; nothing has been submitted.

| problem type | datasets | Elo | source |
| --- | --- | --- | --- |
| binary | 30 | 974 (+138 / -169) | §220 |
| multiclass | 8 | 865 (+120 / -293) | §224 |
| regression | 13 | 578 (+109 / -297) | §223 |
| overall (merge of the three) | 51 | 903 (+102 / -110) | §224 |

On the 27 narrow binary datasets alone, the §215 arm is 1,000.2 (+156 / -167), level with `RF
(default)` at 1,000.0 and 13 / 14 on paired datasets (§215). The three wide binary tables cost 26
Elo (§220): APSFailure (170 columns) ranks 102nd of 108, Bioresponse (1,776) 103rd and
kddcup09_appetency (212) 91st.

**Progression of the overall figure**, each step from its own finding: 880 (§221, 107th of 110),
888 (§222, multiclass 751 to 825), 898 (§223, regression 436 to 578), 903 (§224, multiclass 865
via four partition bags).

**What the entry is, and what it must declare.** Label: "FinTFM with a trained linear readout",
not zero-shot FinTFM. A logistic regression (binary and multiclass) or ridge regression
(regression) is fitted on the network's pre-head representation of the training split. For
multiclass and regression the readout also takes the row's own squashed features (`raw`, §222,
§223), so it is a linear model over the representation plus the features; the entry has to say
so. Column selection (`select`) changes what the model sees on wide tables and is declared too.
The squash transform is null on V4FinBench (§211), so this is a Lite-only configuration.

**Configuration**, all in the tabarena fork (`~/proj/tabarena`, branch `fintfm-audit-fixes`),
where each run is `run_fintfm_lite.py` with a distinct `FINTFM_RUN_NAME`:

| setting | value | source |
| --- | --- | --- |
| `max_features_strategy` | `select`, inside `FINTFM_PARAMS`; admits tables wider than the checkpoint | §220 |
| feature transform | `squash` | §212, §215 |
| `FINTFM_PARTITION` | JSON; `{}` is depth 2, `min_samples_leaf` 0.05, one bag; `{"n_bags": 4}` for multiclass; a probe is fitted per leaf. Not applied to regression | §212, §215, §224 |
| `FINTFM_PROBE` | JSON readout options: `scale`, `cv` (C from 10^-3 to 10^2 by 3-fold log loss), `raw`, `head`. Multiclass uses scale + cv + raw; regression uses `{"raw": true}`. `head` is available and not kept (§224) | §222, §223, §224 |
| `FINTFM_MAX_CONTEXT` | default 1,000; 4,000 changed nothing on multiclass | §224 |
| `FINTFM_ALLOW_MULTICLASS`, `FINTFM_ALLOW_REGRESSION` | opt-in problem types; each needs its own checkpoint | fork `run_fintfm_lite.py` |
| checkpoints | binary: published; multiclass: `dr74-mc-s0`; regression: `v4-regression.pt` | §221 |

The multiclass winner and the regression arm are the two with probe options set. The probe code
is `models/fintfm_linprobe/model.py` in the fork; the head option arrived at fork 7f0b062 and the
readout options at b149d96.

**The merge.** `merge_fintfm_lite.py` (fork root, introduced at 8aefd8d) registers the three
runs as one method and fits nothing. It refuses a dataset scored twice, a split other than
Lite's and a count other than `--expect` (pass `--expect 51`); `merged_from.json` records which
run scored each dataset. None of the 51 datasets is imputed for FinTFM (§221).

**What was tried on multiclass and did not help** (§222, §224): the 480,000-task checkpoint
(+11 Elo, 5 of 8 better, p 0.73), the head's log probabilities (825 to 825, 8 times slower), and
a 4,000-row context (820). Four partition bags give +40 Elo and are noisy: three datasets are
worse, website_phishing by 4.3%. `mc-bag8` is queued (§224); no result yet.

**Gaps to state first.** (1) Overall is 97 Elo below `RF (default)`. (2) Regression (578) is the
weakest type; the remaining lever is a regression checkpoint on the current recipe, which needs
a GPU (§223). (3) The readout options are untested on binary, and under the V4FinBench-first
rule they are scored there before any binary Lite run (§223). (4) The multiclass checkpoint
`dr74-mc-s0` and `v4-regression.pt` are private, so a PR would need both published, or the
entry could not be reproduced.
