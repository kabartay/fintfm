# Running FinTFM on TabArena

External evaluation, under someone else's protocol, against 94 other methods. This file is
the reproduction recipe and the list of things that cost a run.

**Result: 94th of 95.** §122 measured mean ROC-AUC **0.7823** over all 27 eligible datasets
with zero failures, against `fintfm==0.5.5` from PyPI and the published checkpoint. The
submission was reviewed by a TabArena collaborator, who confirmed the integration and the
numbers are correct and declined the entry because it does not meet the leaderboard's
competitiveness requirement.

Earlier runs reported 93rd. §98 first measured mean ROC-AUC **0.7642** over 26 of 27 eligible
datasets; §100 located most of that deficit in categorical preprocessing rather than
architecture, and §101's fix lifted the mean to **0.7823** with the rank unchanged. Every
binary arm since has landed between 0.7746 and 0.7926, and none has moved the rank, except
§132's unpublished linear-probe variant: Elo 910, 93rd of 95.

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
