# Running FinTFM on TabArena

External evaluation, under someone else's protocol, against 94 other methods. This file is
the reproduction recipe and the list of things that cost a run.

**Result: 93rd of 95.** §98 first measured mean ROC-AUC **0.7642** over 26 of 27 eligible
datasets; §100 located most of that deficit in categorical preprocessing rather than
architecture, and §101's fix lifted the mean to **0.7823** with the rank unchanged. Every
binary arm since has landed between 0.7746 and 0.7926, and none has moved the rank.

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
    model.py   # FinTFMModel(AbstractTorchModel), ag_key = "FinTFM"
    info.py    # ModelInfo(model_cls=..., search_space=..., method_metadata=...)
<tabarena clone>/run_fintfm_lite.py
```

`fintfm` itself is installed into the TabArena environment as an editable install, so a change
here is live there with no reinstall.

## Running it

```bash
export FINTFM_CHECKPOINT=/path/to/fintfm/runs/v4-cellattn-labels.pt
FINTFM_RUN_NAME=my_run python run_fintfm_lite.py --full
```

| variable | purpose |
| --- | --- |
| `FINTFM_CHECKPOINT` | required; weights are not distributed with the wrapper |
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

## The leaderboard name is `FinTFM`, with no version

The leaderboard renders `ag_key`, and it lists **models**, not releases — TabSTAR, OrionMSP,
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

The original position was: not yet, and the ranking is not the reason. At 51% coverage with a
knowingly degraded categorical path, the submitted number would have measured the workaround as
much as the model. The stated order was native categoricals (47.x), then multiclass and
regression (46.x), then a submission whose number means what it says.

- **Native categoricals: done** (§101). Out-of-fold target statistics, mean 0.7642 → 0.7823.
- **Multiclass and regression: implemented, scored, and withdrawn** (§121). The condition was
  never "make them runnable" — it was that a declaration must have a measurement behind it. The
  measurement now exists and says they rank last, so they are not declared.

**So the submission declares `binary` only**: 27 eligible datasets, 53% coverage, rank 93 of
95, with every declared capability measured. That is a narrower entry than the 90% one, and a
truer one.

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
