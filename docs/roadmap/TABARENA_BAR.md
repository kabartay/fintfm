# What it would take to be listed on TabArena

[PR #616](https://github.com/autogluon/tabarena/pull/616) was reviewed, confirmed correct, and
declined. §122 records the outcome. This file records what would have to change before
resubmitting, and in what order.

Two requirements were stated, and only one of them is mechanical.

## Requirement 1: all 51 datasets

> "Excluding datasets or subsets of datasets because the model isn't performing well enough is
> not eligible. This would be cherry-picking the results." ... "We would like new submissions to
> cover all datasets."

Coverage today is 27 of 51. The suite is 30 binary, 13 regression, 8 multiclass.

| gap | datasets | what it needs |
| --- | --- | --- |
| multiclass not declared | 8 | publish the `max_classes=10` checkpoint measured in §121 |
| regression not declared | 13 | publish the binned-head checkpoint measured in §121 |
| over the 136-feature cap | 3 binary, 2 other | a checkpoint trained above **1776** features (`Bioresponse`) |

The first two are publication rather than research: both checkpoints exist and were scored. The
third is the expensive one. 1776 is 13x the current cap, and the cap is a property of the
pretrained weights.

**A cheaper route exists and should be costed before the retrain.** The wrapper could reduce
wide tables before they reach the model, by random projection or PCA, rather than the checkpoint
growing to fit them. Several entrants preprocess this way. It changes what the model is being
asked to do on those five datasets and that has to be declared, but it is days rather than a
pretraining run. Measure both against the five datasets in question before choosing.

## Requirement 2: competitiveness, which is the real work

> "Here, it seems the model does not beat simple baselines. Hence, I recommend withdrawing the
> submission until the model performance meets a reasonable minimum."

FinTFM sits at **Elo 765**. The gap, from the run in §122:

| to clear | Elo | gap |
| --- | --- | --- |
| KNN (tuned) | 859 | **+94** |
| Linear (default) | 936 | +171 |
| RandomForest (default) | 1000 | +235 |
| ExtraTrees (tuned + ensembled) | 1154 | **+389** |

**These Elo values are not the ones on the public leaderboard, and the two must not be mixed.**
Elo is relative to the set of methods and tasks it was computed over. This run covers **27
binary datasets at one split each**, TabArena-Lite restricted to what FinTFM can run; the public
board covers all 51 datasets across every split. The same method scores differently under each:
TabSTAR (default) reads 1102 here and 987 there, Linear (default) 936 here and 858 there. Both
are correct measurements of different things.

The table above is the right target anyway, because it is the comparison the submission was
judged on. It just cannot be quoted as a public-leaderboard number.

Beating *every* simple baseline means roughly **+389 Elo** in this comparison. The honest floor
for "a reasonable minimum" is somewhere between the two bolded rows, and the lower one is not a
serious target: a model that beats tuned KNN and nothing else is still last among real methods.

### What is already closed, and must not be re-run

The repository has spent real compute establishing what does not move this number. None of it
should be revisited without a new reason:

- **Scale.** §114 closed it on three independent lines: matched-task parameters (-0.0049),
  volume (null at 5x, +0.0028 at 2x), and Nori's published curve (+0.0049 R2 for 16.7x).
- **The binning head.** §121 showed rank is flat against the axis binning controls. Fixing it
  is ~1 day and $1.40 for an expected rank change of approximately zero.
- **Inference-time knobs.** Context size is nearly flat (S83/S84), retrieval harms at low
  prevalence (S70). This is why the search space is empty.
- **The tree prior for credit.** §118 measured -0.0221 AP, 0 of 5 folds.
- **Categorical preprocessing.** §101 already harvested it: +0.018 mean ROC-AUC, **rank
  unchanged**.

That last line is the one to keep in view. **A gain of +0.018 ROC-AUC moved the rank by zero.**
The standing deficit is roughly -0.035 uniform, and closing it is necessary but, on §101's
evidence, probably not sufficient. Any proposal worth running has to be sized against a
multiple of that, not a fraction.

### What has not been tried

Four axes remain, and they are the only places a step change can come from:

| axis | queued as | why it might matter |
| --- | --- | --- |
| **Architecture** | `factorized-attention` (6 tasks) | the attention factorisation itself has never been varied |
| **Objective** | `joint-objective-training`, p(x,y\|D) (4 tasks) | changes what the model is trained to represent, not how much |
| **Prior diversity** | `mechanism-diverse-prior` (9 tasks) | MITRA's three criteria; §112 showed the current prior's distinctiveness is weak |
| **Shape** | not queued | Nori-6M is **16 layers at width 128**. The scale-up here went wide and lost. Depth at constant width is the one design axis never varied. |

The fourth is the cheapest to test and has the clearest external evidence behind it. It should
go first.

## Ordering

**Competitiveness before coverage.** Coverage is a precondition for listing, not a contributor
to rank, and paying for a >1776-feature pretraining run before the model can clear a tuned KNN
spends compute on an entry that still cannot be listed. The reviewer's own recommendation was to
withdraw until performance meets a minimum, not to fix coverage.

So:

1. **Test depth at constant width**, against Nori's published shape. Cheapest, best-evidenced,
   and the one axis never varied here.
2. If it moves, **then** the architecture and objective axes, in that order.
3. Only once something clears `Linear (default)` at Elo 936 does the coverage work become worth
   its compute.
4. Resubmit with all 51 datasets from the start, as promised in the PR thread.

A resubmission that arrives with full coverage and the same rank would waste the reviewer's time
as well as ours.
