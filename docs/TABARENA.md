# Running fintfm on TabArena

External evaluation, under someone else's protocol, against 94 other methods. This file is
the reproduction recipe and the list of things that cost a run.

**Result so far: 93rd of 95** (§98), mean ROC-AUC 0.7642 over 26 of 27 eligible datasets.
§100 then located the deficit: it is mostly categorical preprocessing, not architecture.

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
    model.py   # FinTFMModel(AbstractTorchModel), ag_key = "TA-FINTFM-0.3"
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

`_supported_problem_types = ["binary"]` and `max_features=136` exclude 13 regression, 8
multiclass and 3 wide datasets. **26 of 51 ran: 51%.** A mean over 27 datasets is not
comparable with a mean over 51, and the leaderboard's Elo is computed only where a method
actually ran.

Capability limits are declared rather than worked around — a task the checkpoint cannot
represent raises, so the harness records a skip instead of a meaningless score.

## On submitting a PR

**Not yet, and the ranking is not the reason.** At 51% coverage with a knowingly degraded
categorical path, the submitted number would measure the workaround as much as the model. The
order is: native categoricals (47.x), then multiclass and regression (46.x), then a submission
whose number means what it says.
