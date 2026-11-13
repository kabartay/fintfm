# What it would take to be listed on TabArena

> **Support status, 2026-11-13.** The project states it is no longer actively supported from
> mid-November 2026, so a resubmission is unlikely to be made by this project.

[PR #616](https://github.com/autogluon/tabarena/pull/616) was reviewed, confirmed correct, and
declined. §122 records the outcome. This file records what would have to change before
resubmitting, and in what order.

(2026-11-13, §224): standing now. The 51-dataset Lite entry "FinTFM with a trained linear
readout" scores 903 Elo (+102 / -110), 107th of 110, with no dataset excluded (binary 974 on 30,
multiclass 865 on 8, regression 578 on 13). Requirement 1 is met on Lite and requirement 2 is not.
The PR is held (owner decision DR7.6, 2026-11-11) until multiclass and regression improve. The
recipe is in the last section of [the TabArena recipe](../results/TABARENA.md); the dated notes below
update this file's statements and leave the originals in place.

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

(2026-11-13, §220, §221: the 5 wide datasets, multiclass and regression all run on Lite now.
`max_features_strategy: select` admits the wide tables with no new checkpoint, so the 1776-feature
retrain was not needed to score 51 of 51. The wide tables rank low: Bioresponse 103rd of 108,
APSFailure 102nd, kddcup09_appetency 91st (§220); hiva_agnostic 90th of 108 (§221). The cheaper
route named below is, in effect, what was measured. Publishing the multiclass and regression
checkpoints is still open, and the multiclass one is now `dr74-mc-s0`, not the §121 one.)

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

(2026-11-13, §224: FinTFM no longer sits at 765. The zero-shot binary entry is 756 and the
51-dataset linear-readout entry is 903; see the note under Requirement 2's table. The 765 and the
gaps below describe the submitted entry.)

FinTFM sits at **Elo 765** (763 on §150's refreshed baseline cache, same rank). The gap, from
the run in §122:

| to clear | Elo | gap |
| --- | --- | --- |
| KNN (tuned) | 859 | **+94** |
| Linear (default) | 936 | +171 |
| Linear (tuned), added 2026-10-16 | 966 | +201 |
| Linear (tuned + ensembled), added 2026-10-16 | 984 | +219 |
| RandomForest (default) | 1000 | +235 |
| KNN (tuned + ensembled), added 2026-10-16 | 1008 | +243 |
| ExtraTrees (tuned + ensembled) | 1154 | **+389** |

**The resubmission bar is about 1,010 Elo (corrected 2026-10-16, task DR7.1).** This file and
the roadmap set it at `Linear (default)`, 936, which would still leave `Linear (tuned)`,
`Linear (tuned + ensembled)`, `RF (default)` and `KNN (tuned + ensembled)` above. The
defensible bar is parity with `RF (default)`, at 1000.0 by construction, which with
`KNN (tuned + ensembled)` at 1008.5 means about 1,010. The added rows come from the run that
scored FinTFM at 764.6, the TabArena fork's `eval/fintfm_lite_tafintfm/tabarena_leaderboard.csv`
(one decimal there: 966.5, 983.6, 1008.5; Linear is `LR` in that file). Their Elo intervals
in that file (`elo+` plus `elo-`) run 154 to 262 Elo wide, so a pass is decided on paired per-dataset differences.

(2026-11-13, §215: the "about 1,010" figure does not carry over. It came from this fork file's
leaderboard, where `KNN (tuned + ensemble)` scored 1008.5. In the run that scored the
probe-plus-squash-plus-partition arm, `KNN (tuned + ensemble)` scores 999.8 and `LR (tuned)`
1016.0 (966.5 in the file above). The definition stands: parity with `RF (default)`, 1000.0 by
construction. Against it that arm is 1000.2 (+156 / -167) on the 27 binary datasets and 13 / 14
on paired datasets, a tie, not a pass; it beats neither bar method on paired differences. Every
Elo here depends on the pool it was computed in, so the neighbours move between runs. On the
51-dataset pool, §224 gives `Linear (tuned)` 919, `Linear (tuned + ensembled)` 936 and `RF
(default)` 1,000, against FinTFM's 903. The table above is the §122 pool and is kept as written.)

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
- **Inference-time knobs.** Context size is nearly flat (§83/§84), retrieval harms at low
  prevalence (§70). This is why the search space is empty.
- **The tree prior for credit.** §118 measured -0.0221 AP, 0 of 5 folds. (§166, 2026-10-16:
  the figure is §116's, and it is confounded with a financial share cut from 0.70 to 0.49; at a
  matched share the tree arm is +0.0119 mean AP, 4 of 5 folds, one seed. Not closed.)
- **Categorical preprocessing.** §101 already harvested it: +0.018 mean ROC-AUC, **rank
  unchanged**.

That last line is the one to keep in view. **A gain of +0.018 ROC-AUC moved the rank by zero.**
The standing deficit is roughly -0.035 uniform, and closing it is necessary but, on §101's
evidence, probably not sufficient. Any proposal worth running has to be sized against a
multiple of that, not a fraction.

### What has not been tried

Four axes remain, and they are the only places a step change can come from. (Since run: depth
at constant width is null, §124 and §126; `mechanism-diverse-prior` has 8 of its 9 tasks done,
and its combined prior measured negative on V4FinBench, §141.)

| axis | queued as | why it might matter |
| --- | --- | --- |
| **Architecture** | `factorized-attention` (6 tasks) | the attention factorisation itself has never been varied |
| **Objective** | `joint-objective-training`, p(x,y\|D) (4 tasks) | changes what the model is trained to represent, not how much |
| **Prior diversity** | `mechanism-diverse-prior` (9 tasks) | MITRA's three criteria; §112 showed the current prior's distinctiveness is weak |
| **Shape** | not queued | Nori-6M is **16 layers at width 128**. The scale-up here went wide and lost. Depth at constant width is the one design axis never varied. |

The fourth is the cheapest to test and has the clearest external evidence behind it. It should
go first.

## Ordering

**Superseded by [`ROADMAP.md`](ROADMAP.md)**, which is the current ordering. Kept for the
reasoning.

**Competitiveness before coverage.** Coverage is a precondition for listing, not a contributor
to rank, and paying for a >1776-feature pretraining run before the model can clear a tuned KNN
spends compute on an entry that still cannot be listed. The reviewer's own recommendation was to
withdraw until performance meets a minimum, not to fix coverage.

So:

1. **Test depth at constant width**, against Nori's published shape. Cheapest, best-evidenced,
   and the one axis never varied here.
2. If it moves, **then** the architecture and objective axes, in that order.
3. Only once something clears `Linear (default)` at Elo 936 does the coverage work become worth
   its compute. (Corrected 2026-10-16, DR7.1: the bar is about 1,010, `RF (default)` parity; see
   Requirement 2.)
4. Resubmit with all 51 datasets from the start, as promised in the PR thread.

A resubmission that arrives with full coverage and the same rank would waste the reviewer's time
as well as ours.
