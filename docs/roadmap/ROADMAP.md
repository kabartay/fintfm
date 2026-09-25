# Roadmap: from 94th of 95 to listable

The ordered execution list for one goal: **make FinTFM good enough to be listed on TabArena.**
[`TABARENA_BAR.md`](TABARENA_BAR.md) says what the bar is, [`NEXT.md`](NEXT.md) is the standing
tier queue, [`STRATEGY.md`](STRATEGY.md) is the thesis. This file is what to do next and in what
order, and it overrides `NEXT.md`'s Tier 0 ordering where they disagree, because Tier 0 predates
the external review.

Proposals live in [`openspec/changes/`](../../openspec/changes/); the IDs below point there.

## The target, stated once

| | |
| --- | --- |
| today | Elo **765**, rank **94 of 95**, mean ROC-AUC **0.7823** |
| clear a tuned KNN | +94 Elo |
| clear every simple baseline | **+389 Elo** |
| listable | full coverage of all 51 datasets **and** the above |

Elo here is over the 27 binary datasets at one split each that §122 ran, not the public board.
See `TABARENA_BAR.md` on why the two cannot be mixed.

**The calibration that matters:** §101 gained **+0.018 mean ROC-AUC and moved the rank by
zero.** The standing deficit is about −0.035 uniform. Anything proposed below has to be sized
against a multiple of that, not a fraction of it. A result that improves ROC-AUC and leaves the
rank is a null result for this goal, and should be written up as one.

## Do not re-run these

Measured closed. Reopening any of them needs a new reason, stated first.

- **Scale**, three independent lines (§114).
- **The binning head**, rank is flat against the axis it controls (§121).
- **Inference-time knobs**: context nearly flat (S83/S84), retrieval harmful at low prevalence
  (S70).
- **The tree prior for credit**: −0.0221 AP, 0 of 5 folds (§118).
- **Categorical preprocessing**: already harvested, +0.018 and no rank move (§101).
- **Larger query chunks**: 10.7× slower (§120).

---

## Phase A — cheap rank probes, in cost order

**`column_id_dim` is not on this list, and was on an earlier draft of it.** §104 already swept
{12,16,20,24,32} and found the curve **peaks at 16**, the value that had been chosen by
accident. `NEXT.md`'s Tier 0 ranked it second, and Tier 0 predates §104. Checking the task
status before scheduling work is cheaper than rediscovering a closed result.

Each is days not weeks, and each has external or internal evidence behind it. Run them before
committing to any architectural rewrite. **Exit condition for the phase: at least one arm moves
rank by 3 or more positions.** If none does, the deficit is structural and Phase B is the only
remaining move.

1. **Depth at constant width.** Nori-6M is 16 layers at width 128; every scale-up here went
   wide and lost. This is the one design axis never varied. Not currently queued; scope it.
2. **Learnability filter on the prior** (48.5). Nori filters tasks a simple learner cannot fit.
   §112 measured this prior's distinctiveness as weak, so the filter is aimed at a known gap.
3. **Cheap realism augmentations** (48.6): discretized features, noise, missingness. Published
   as worth a point or two, and no retrain of the architecture.
4. **Is-missing encoding** (48.10). Two peers disagree, so the answer is not knowable from
   reading and is cheap to measure.
5. **Schedule-free optimisation** (48.9). Decouples run length from a fixed schedule, which
   makes every later experiment cheaper even if it does not move rank itself.
6. **Random monotonic marginal augmentation** (40.8), motivated by a measured train/inference
   mismatch rather than by analogy.
7. **Second seed for the §91 ablation** (39.28). One run per arm is not a result, and the
   sub-chance inversion is load-bearing for the architecture argument.

## Phase B — the architecture and objective axes

Weeks, not days. Only worth starting if Phase A shows the deficit is structural, or if one
Phase A arm moves and suggests where.

8. **Write the factorized-attention cost model first** (44.1), and check it retrodicts linear
   CUDA and quadratic MPS growth (§95). §119 and §120 are both records of a cost model that was
   not checked before it was trusted.
9. **Establish what §91's mechanism requires** (44.2) before implementing anything.
10. **Implement the factorized encoder behind a flag** (44.3), defaulting off.
11. **Measure cost before accuracy** (44.4): peak training memory and step time.
12. **Then accuracy at the project's standard** (44.5): five-fold V4FinBench, paired bootstrap,
    Holm–Bonferroni.
13. **Re-test the levers it unblocks** (44.6): `max_context` past 2,000, model size past the
    current ceiling. These were closed under the current attention cost, not in principle.
14. **Joint objective, p(x,y|D)** (34.1): sum the objectives with a configurable weight.
15. **Pretrain at matched compute** (34.2) against `runs/v4-hazard-ldp.pt`.
16. **Score the joint checkpoint on the binary path** (34.3), Polish and Taiwan.
17. **Retire the two-checkpoint workflow if joint training wins** (34.4), and say so in the
    docs rather than leaving both paths alive.

## Phase C — prior diversity

The prior is the project's distinctive claim and §112 measured its distinctiveness as weak.
This phase is the one that could produce a genuinely different model rather than a better-tuned
one.

18. **Labelled task-family generators** (40.2): linear, threshold, XOR, and the rest.
19. **Interaction-order curriculum, measured before touched** (40.3).
20. **Compositional generalisation test on existing checkpoints first** (40.4), matching 40.3's
    protocol so the two are comparable.
21. **Correlation, confounding and collider families** (40.5) on the existing SCM.
22. **Missingness, shift and support-extrapolation axes, sampled independently** (40.6).
23. **Widen target mechanisms to a published list** (48.4), so the comparison is against
    someone else's taxonomy rather than our own.
24. **Scope the combined prior only after 40.2–40.6 each validate individually** (40.7).

## Phase D — coverage, and only once rank has moved

Nothing here improves rank. It is the admission ticket, and buying it early spends a
pretraining run on an entry that still cannot be listed.

25. **Publish the multiclass checkpoint** measured in §121, and declare `multiclass`.
26. **Publish the regression checkpoint** measured in §121, and declare `regression`.
27. **Cost the two routes past the 136-feature cap against each other**: a checkpoint trained
    above 1776 features, versus in-wrapper dimensionality reduction for wide tables. Decide on
    measurement, not preference, and declare whichever is chosen.
28. **Re-run TabArena-Lite across all 51 datasets** and confirm the aggregate, remembering that
    multiclass and regression rank worse than binary (93.4 and 93.1 against 86.3), so full
    coverage will lower the aggregate before it raises anything.

## Phase E — resubmit

29. **Resubmit with all 51 datasets from the start**, as promised in the PR thread, and only
    when something clears `Linear (default)`. A resubmission arriving with full coverage and the
    same rank wastes the reviewer's time as well as ours.

---

## The stop rule

This project has a habit worth keeping: it writes down what it measured, including when the
answer was no. Applied here, the rule is that **if Phases A, B and C all complete without
clearing `Linear (default)`, the honest conclusion is that this architecture and prior do not
produce a competitive general tabular model**, and the work should be written up as that rather
than continued.

That would not make the project worthless. The measurement log, the provenance claim and the
calibration result on real credit panels all stand on their own, and a maintainer has now
confirmed the integration and the numbers are correct (§122). But it would mean the leaderboard
is the wrong goal, and continuing to chase it would be the kind of thing §119, §120 and the two
retracted diagnoses are records of: committing to a direction the evidence had already closed.
