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

**Screen locally at batch 8, confirm on rented GPU.** §128 found the MPS ceiling is a `2^31`
indexing bound, `batch x features x heads x rows^2`, not a memory limit: batch 8 at `n_rows 512`
sits at 0.53x of it and runs at **1.74 s/step, about 2.9 hours per arm, free**. Every item below
needs pretraining, so this is what makes them affordable. The only remaining gap to the
reference recipe is that local runs never draw `n_rows 1024`. Confirm anything that moves at the
reference recipe before it is quoted.

**Screen downstream, not on held-out synthetic.** All four cells of §126's factorial score
0.768 to 0.771 pooled held-out AUC while their downstream AP spans 0.0952 to 0.1196. The cheap
signal is blind to an effect the credit protocol resolves on 5 of 5 folds, and it has now
mispredicted three times (§117, §124, §126) and called one correctly (§127). Score the arms
below on the credit protocol directly; at one for four the screening step is not worth the run
it costs, though it is not worthless either.

**The calibration that matters:** §101 gained **+0.018 mean ROC-AUC and moved the rank by
zero.** The standing deficit is about −0.035 uniform. Anything proposed below has to be sized
against a multiple of that, not a fraction of it. A result that improves ROC-AUC and leaves the
rank is a null result for this goal, and should be written up as one.

**A second population confirms this isn't specific to TabArena.** A private, unpublished
comparison against Neuralk-AI's TabBench (§134, not committed to this repo, no PR) found the
same standing on a differently-built 55-dataset overlap: mean rank 15.1 of 16, and checked
directly that it is not the categorical-encoding difference between fintfm and native-passthrough
TFM peers -- fintfm loses to matched-preprocessing GBTs by a mean AUC of −0.0815. Combined with
Neuralk-AI's own published account of why TFM scaling needs matched prior richness
(`docs/paper/RELATED_WORK.md`), read alongside §112's own weak-prior-distinctiveness measurement,
this is a second, independent reason scale and depth are closed and prior diversity (Phase C)
is where an untested lever actually is.

## Do not re-run these

Measured closed. Reopening any of them needs a new reason, stated first.

- **Scale**, three independent lines (§114), independently consistent with Neuralk-AI's own
  published account of TFM scaling requiring matched prior richness (`docs/paper/RELATED_WORK.md`).
- **The binning head**, rank is flat against the axis it controls (§121).
- **Inference-time knobs**: context nearly flat (S83/S84), retrieval harmful at low prevalence
  (S70).
- **The tree prior for credit**: −0.0221 AP, 0 of 5 folds (§118).
- **Categorical preprocessing**: already harvested, +0.018 and no rank move (§101).
- **Larger query chunks**: 10.7× slower (§120).

---

## Phase A' — a lever §130/§131 found, ahead of the queue because it needs no pretraining

§130 (48.22) measured that a plain logistic regression on fintfm's `encode_rows` representation
beats the model's own classification head, 5 of 5 folds, +0.039 mean AP -- but at a stage that
never reached `self.encoder`, the row-to-row transformer the head's real input passes through.
§131 re-ran the same comparison at the head's exact input (verified to reproduce its logits
exactly) and found the advantage **survives, roughly halved**: +0.0214 mean AP, 4 of 5 folds,
significant on 3 of 5. Item 1 below is therefore answered: depth explains part of the gap, not
all of it. §133 further found the decay is spread evenly through the stack rather than
concentrated at the end, answering item 2. §135 then tried the natural next hypothesis --
maybe the head just needs more nonlinearity -- and found it does not: an MLP on the same
frozen representation beats `own_head` but not the plain linear probe. **Phase A' has now
extracted what it has to give**: a real, external, reproduced gain (§132, +145 Elo), and no
further search inside the decision layer looks likely to add more. Item 4 (a second
checkpoint) is now closed too (§136); Phase A' is done and the next real lever is Phase C.

1. ~~**Distinguish label-conditioning/pooling from a stage-mismatch artifact.**~~ **Done (§131).**
   The `y_emb` + `self.encoder` + `norm` stages §130 skipped account for roughly 45% of the
   original margin (0.0390 to 0.0214), not all of it or none of it. A linear probe on the
   head's own input still beats the head's own final layer on 4 of 5 folds.
2. ~~**Locate where inside the remaining stack the advantage survives.**~~ **Done (§133):
   spread through the stack, not concentrated.** A probe at 2 of 4 layers (`encoder_mid`) scores
   +0.0326 mean AP, sitting between `encode_rows`'s +0.0390 and `pre_head`'s +0.0214 -- roughly
   even decay per increment of depth (36% of the total loss in the first half, 64% in the
   second), no single layer boundary where the advantage collapses. This rules out the
   cleanest version of "the last layer specifically is broken" and narrows, rather than
   answers, item 3's framing.
3. ~~**Prototype a differently-shaped final layer.**~~ **Done (§135): a nonlinear head beats
   `own_head` but not the plain linear probe.** An MLP on the same frozen `pre_head`
   representation scores +0.0185 mean AP over `own_head` (4/5 folds, 3/5 significant) but
   -0.0042 against the plain linear probe (1/5 folds, 0/5 significant) -- statistically
   indistinguishable from it. Whatever the head's own final layer does wrong, it is not
   "insufficient nonlinearity": a genuinely nonlinear alternative, same frozen input, does no
   better than a linear one at this training-data budget.
4. ~~**Repeat the full protocol on a second checkpoint.**~~ **Done (§136): replicates, slightly
   stronger.** `runs/lrsweep/lrsweep-3e4.pt`, same protocol: +0.0323 mean AP (against §131's
   +0.0214), 5/5 folds win (against 4/5), 3/5 significant after Holm. Not a quirk of one
   training run.
5. ~~**Check whether this transfers to TabArena's Lite protocol.**~~ **Done (§132): it does.**
   +145 Elo, 93rd of 95 against 94th, 26 of 27 datasets used the linear probe cleanly. Still
   below every real method -- 910 sits a whisker under `Linear (default)` at 936, the next
   milestone this file already named. Timed one dataset first (249s), confirming the discipline
   `docs/results/POSTMORTEM.md`'s second chapter records paying for in the other direction.

## Phase A — cheap rank probes, in cost order

**`column_id_dim` is not on this list, and was on an earlier draft of it.** §104 already swept
{12,16,20,24,32} and found the curve **peaks at 16**, the value that had been chosen by
accident. `NEXT.md`'s Tier 0 ranked it second, and Tier 0 predates §104. Checking the task
status before scheduling work is cheaper than rediscovering a closed result.

Each is days not weeks, and each has external or internal evidence behind it. Run them before
committing to any architectural rewrite. **Exit condition for the phase: at least one arm moves
rank by 3 or more positions.** If none does, the deficit is structural and Phase B is the only
remaining move.

1. ~~**Depth at constant width.**~~ **Done, null.** §123 priced it, §124 ran it, §126 confirmed
   it at a matched learning rate: depth contributes **+0.0010 AP** and the two tuned arms are
   identical to four decimals. Nori's shape does not transfer here.

   **It produced a better lead than itself.** Closing the learning-rate confound §124 recorded
   showed `--lr 1e-4` worth **+0.0243 AP on 5 of 5 folds** at batch 2, twelve times the depth
   effect (§126). That is not yet a reason to change a default, because optimal rate scales with
   batch size and the reference recipe is batch 8.

1b. ~~**Sweep the learning rate at the reference recipe.**~~ **Done, the default stands.** §127
   ran both arms at batch 8 for about $2.80: `3e-4` wins on 5 of 5 folds, so §126's result is a
   batch-size artifact. §114's scale arms were not measured on mis-tuned models and the
   published checkpoints are not under-optimised. Nothing changes.
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

18. ~~**Labelled task-family generators.**~~ **Done (40.2):** `src/fintfm/prior/task_families.py`,
    nine families, each difficulty-controlled to a requested Bayes AUC (five by exact closed
    form, four by calibrated bisection) and measured, not assumed, in `tests/
    test_task_families.py`. Not yet mixed into pretraining -- that is item 24 (40.7), gated on
    items 19-23 validating individually first.
19. ~~**Interaction-order curriculum, measured before touched.**~~ **Done (§137, 40.3):** both
    §74 checkpoints (capped and uncapped on the pure linear task) collapse to chance the
    instant any interaction is required (`k=2`, plain `xor`) and stay there through `k=5` --
    the linear-task capped/uncapped split has no bearing on interaction capability at all.
    Sets the floor items 20 and 24 must clear.
20. ~~**Compositional generalisation test on existing checkpoints first.**~~ **Done (§138,
    40.4):** both §74 checkpoints score ~0.86 on the AND-composition of two independently
    0.9-calibrated rules -- a real but modest drop, nothing like item 19's collapse to chance.
    Conjunction is not interaction for this architecture; the capped/uncapped split from §74
    has no bearing here either.
21. ~~**Correlation, confounding and collider families.**~~ **Done (§139, 40.5):** a model
    weighting a confound's proxy about as heavily as a true cause loses 0.16 AUC when the
    confound path is severed (`do(P)`) -- and lands *below* the no-confound-ever-existed
    ceiling, not at or above it, meaning it does not discard the now-uninformative proxy once
    broken. A fourth probe where the §74 capped/uncapped split has no bearing.
22. ~~**Missingness, shift and support-extrapolation axes, sampled independently.**~~ **Done
    (§140, 40.6):** additive shift and positive-scale extrapolation are exact Bayes-AUC no-ops
    (same rank-invariance argument as `threshold`); missingness leaves family identity and
    labels untouched while genuinely moving achieved AUC. All three compose freely with any of
    item 18's nine families, verified by test rather than by convention.
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
