# Postmortem: seven wrong diagnoses in three days, and the one question nobody asked

**Date:** 2026-09-09; the section on the seventh diagnosis was added 2026-09-10. Companion to
`docs/results/FINDINGS.md` §28-§32, which carry the numbers. This document exists for the
*pattern*, because the individual findings each read as an isolated slip and they were not
isolated.

On 2026-09-08 this project produced finding 26, its first out-of-time result on real corporate
data, and concluded that the synthetic prior could not generate the low-default regime the
strategy targets. A 6,000-step retrain followed. On 2026-09-09 that conclusion turned out to
be wrong, and so did the two explanations that replaced it, and so did the first fix for the
thing that was actually broken.

Every number below is measured; the commands are in the findings.

## The chain

| # | The claim | What was actually true | Cost |
| --- | --- | --- | --- |
| 1 | The prior cannot reach default rates below 1%, which is why out-of-time PD levels are 27× too high (§26) | The prior floor was real, but the level error came from the term-structure path **skipping the base-rate correction**. Context was resampled to 50% defaulters against a 1.545% population (§28) | A 6,000-step retrain, 6,372 s |
| 2 | Balanced context sampling is right, on published evidence (D5) | On this task uniform beats balanced by **10-12 AUC points**, and 12 in-context defaults outrank 1,122 (§29) | A default that every local measurement contradicted |
| 3 | The hazard head's decay across horizons is the signature of a context carrying no default *timing* (§30) | The baseline decays almost as fast (−0.2187 against −0.2429). **84% of the gap is horizon-independent** — a context-*size* limit, not a timing one (§31) | A proposal written and prioritised on a wrong premise |
| 4 | Retrieval needs the base-rate correction, like every other strategy | Retrieval selects on `x`, so label shift **fails by construction**. Applied per group it drove AUC to **0.3679, below chance** (§32) | Nearly discarded the one change that worked |
| 5 | Retrieval "rises monotonically" with context size, so more retrieved rows keep helping (§32) | On three seeds, 2,000 and 4,000 rows are **tied**. The single-seed 0.7934 was a favourable draw; retrieval raises the plateau's height, not where it starts (§33) | A recommendation to score at 4,000 rows for 1.7× the time and no gain |
| 6 | Tightening retrieval groups from 740 to 185 firms per context is worth +0.008 AUC | On three seeds, **0.8130 ± 0.0069 against 0.8126** — the effect is smaller than the seed spread and the single-draw 0.8209 was noise with a direction (§38) | Two experiments run to chase it, and a claim briefly made to the user |

And one defect that was not a diagnosis at all: a hazard checkpoint's **classification head is
never trained**, and `predict_proba` served its random initialisation as a 69% default
probability against a 4.7% base rate, ranking worse than chance, with no error (§34). Found
only because those numbers were too absurd to belong to the hypothesis being tested.

## What they had in common

**Not carelessness about the assumptions.** Each assumption was written down, correctly, in the
place it was implemented. The base-rate correction's docstring states its precondition
explicitly: "resampling selects on `y` alone, so it holds by construction here." That sentence
is true of the code it was written for and false of the two code paths that later reused it.

The common mechanism is that **a new code path inherits an assumption silently.** Failures 1
and 4 are the same failure in opposite directions — one bypassed the correction, the other
applied it where its precondition fails — and both happened because the correction was a
property of one method rather than of the object.

The second mechanism is **reading a pattern before decomposing it.** Failures 1 and 3 both
took a shape in a summary table and built a mechanism from it. A 27× level error and a
monotone decay each looked like they implied something about the prior or the labels; one was
a missing line of arithmetic and the other was a constant offset wearing a trend's clothes. A
monotone trend and a constant offset are indistinguishable in a table of four numbers.

## Why nothing caught them

Three guards were in place for failure 1 and all three were blind to it:

- **AUC could not see it.** Every prediction inflates alike, so ranking is untouched. This is
  the same blindness §5 found and D5 was written to fix.
- **The coherence check could not see it.** The curve was 100% monotone throughout. A
  perfectly coherent curve can state perfectly wrong levels.
- **The number that would have caught it was computed and never printed.** `mean_predicted`
  was in the JSON the whole time.

That last one is the most uncomfortable, because it was not a missing measurement. It was a
missing column.

## What changed

Mechanical changes, not resolutions:

- **The corrected path is the only public path.** `predict_term_structure` applies the
  correction; calling `FinancialTFM.term_structure` directly is a documented defect (D8).
- **The broken configuration is a permanent arm.** The out-of-time harness always scores an
  uncorrected context beside the corrected one, so the distortion is measured rather than
  assumed absent.
- **Levels print beside the truth.** Any harness reporting a probability reports its mean next
  to the observed rate.
- **Assumptions are asserted, not documented.** The shift's monotonicity and rank preservation
  are tests; so is retrieval's single-shift property; so is the invariant retrieval gives up.
- **Predictions print their context's base rate against the population's.** One line, and it
  would have replaced the retrain in failure 1.

## What this says about the project

Two things worth keeping separate.

**The failures were expensive but self-correcting, and the correction ran in one direction.**
Every one of the six was caught by measurement inside a day, and each correction made the
result *worse* for the project's story before it made it better: §28 retracted a finding, §29
reversed a decision taken from the literature, §31 demoted a proposal written an hour earlier.
Nothing here was caught by review or by reasoning about the code.

**The sixth has a different shape from the first five, and it is the more dangerous one.**
Failures 1 to 4 mis-attributed a *failure* — they were pessimistic about the wrong thing.
Failure 5 and especially 6 over-credited a *result*: a single-seed sweep produced a +0.008
"gain", and it was treated as a lever, reported as one, and used as the reason to re-run a
comparison. Three seeds erased it.

A single-seed sweep is not tuning; it is noise with a direction. And the direction is
seductive precisely when it agrees with what you hoped. §33 had already established that
hybrid at 1,000 rows spreads ±0.046 across seeds — the evidence that single draws were unsafe
was on file, in this repository, written by the same process that then ignored it.

**Two of the day's fixes came directly out of a failure.** §28's bypassed correction produced
D8 — the corrected path is the only public path — and §34's untrained head produced D11: a head
that was never trained is not reachable. Both are the same rule at different levels, and
neither would have been written from first principles.

## The seventh, and why the first six were all downstream of it

Findings 41 through 46 proposed and tested: a conjunction-representation limit (falsified), a
prior clamped in difficulty (no effect), insufficient training volume (no effect at 10×), and
insufficient prior diversity (+0.027, real but small). Each was a plausible mechanism, each was
measured honestly, and each was **downstream of a cause none of them named**.

§47 found it by asking whether the context mattered at all: shuffle the labels, see if
predictions move. They did not — rank correlation 0.977, and AUC *higher* with random labels
than true ones. The model had never been doing in-context learning, because the prior's
feature-to-label direction was fixed across every task it had ever generated.

**The lesson is not "we guessed wrong four times".** It is that four increasingly expensive
experiments were run before the cheapest possible question — *does the input matter?* — and
that question costs five minutes. When a system underperforms, establish that its central
mechanism operates at all before optimising the mechanism's parameters.

**The scorecard did move, in the end.** Mean AUC out of time went 0.5869 → 0.8118 across the
day and the gap to per-horizon logistic regression went 0.142 → 0.048 (both from §35), from
three inference-time changes and no retraining: the base-rate correction applied where it
belongs, retrieved contexts, and a rank transform on features whose standard deviation exceeds
ten times their interquartile range in 110 of 136 columns. **None of the three was on the
roadmap this morning**; all three came out of chasing a wrong diagnosis to its cause.

**But the story moved further than the position.** After all six corrections, the
measured position is that per-horizon logistic regression still leads on mean AUC 0.8616 to
0.8143 (§38's best configuration) and on calibration by roughly fourfold. The gap is a third of
what it was. It is not closed, and six fixed diagnoses do not close it.

**And the best configuration now uses a competitor's context construction.** §36 found that
our headline context mechanism had been published four months earlier; §38 found their method
beats ours on accuracy at the first horizon, calibration, cost and batch independence. Three
of the eight candidate claims in `docs/paper/CLAIMS.md` were superseded or retracted by
reading one paper and running one comparison — which is an argument for reading a benchmark's
own paper *before* scoring on its data, not after.

---

# Postmortem: a screening metric right once in four tries, and two cost estimates wrong before a third one held

**Date:** 2026-09-26. Companion to `docs/results/FINDINGS.md` §123-§130, which carry the numbers.
Different shape from the chain above — nothing here was a wrong diagnosis about the model.
Every claim shipped was eventually correct. What recurred was a **cheap proxy standing in for an
expensive measurement**, three times for cost and four times for direction, and the proxy was wrong
often enough that the pattern is worth naming rather than filed under each individual finding.

## The chain

| # | The cheap signal said | The expensive measurement said | Cost |
| --- | --- | --- | --- |
| 1 | A depth probe at toy size (`F=32`) priced 16 layers at **1.90x** the step time of 4 | At production width (`F=136`) the per-cell encoder dominates and depth costs **1.01x** per step — later corrected again to **1.16x** end to end, since a full run also varies `n_rows` and samples the prior, neither of which scales with depth (§123) | A wrong number stated with confidence before the real recipe was run |
| 2 | Held-out synthetic AUC ranked depth 16 **below** depth 4, cleanly, on two of three metrics | The five-fold credit protocol ranked depth 16 **above** depth 4, +0.0019 AP, though the folds disagreed in sign badly enough that the honest read was null either way (§124) | Would have shipped "depth hurts" in writing if the held-out number alone had been trusted |
| 3 | At batch 2, `--lr 1e-4` beat the shipped default by **+0.0243 AP, 5 of 5 folds**, all significant after Holm | At the reference recipe (batch 8) the shipped default **beat** `1e-4` by +0.0116, also 5 of 5 folds — the whole effect was a batch-size artifact the finding's own text had flagged as the reason not to trust it yet | About $2.80 and two HF Jobs runs to find out the flagged caveat was the entire story |
| 4 | Held-out pooled AUC at batch 8 ranked the shipped default above `1e-4` | The credit protocol agreed, for the first time this session | The one case where the cheap signal and the expensive one matched |
| 5 | A per-task ExtraTrees judge, timed on the wall clock of a probe that also sampled each task, implied filtering 48,000 pretraining tasks would cost **~20 hours** | Timed on judging alone, the real cost was **~0.03-0.04 s/task**, about 24 to 32 minutes for 48,000 tasks (the `p_learnability_filter` docstring in `prior/mixture.py`, citing §125) | Would have made a genuinely cheap experiment look prohibitive, and did briefly discourage running it |
| 6 | `PowerTransformer.fit` on real V4FinBench data raised on a literal `+-inf` from a near-zero-denominator ratio; clamping it fixed the crash | Clamping the input revealed a **second, independent** overflow inside Yeo-Johnson's own formula, on 5 of 78,015,600 cells, that the first fix did nothing about (§129) | Two separate debugging passes for what looked like one bug |
| 7 | A first design for testing whether fintfm's representation transfers extracted it over the full ~600,000-row training split, matching how `predict_proba` is normally costed | At this model's documented 8.6 s/1,000-row inference cost, that prices at roughly 14 hours; a bounded 20,000-row stratified subsample cut it to **under 15 minutes a fold** with no loss of signal (§130) | A run left going for over two hours before being killed on a projected cost rather than an observed completion |

## What they had in common

**A cheap proxy and the number it stands in for are not the same measurement, and nothing here
checked that they agreed before trusting the cheap one.** Rows 1, 5 and 7 are the same mistake at
three different distances: extrapolating a cost from a regime the real workload does not run in
— a smaller feature count, a probe that timed two things as one, a training-set size no one
asks the model to score in practice. Each was cheap to state and each was wrong in the direction
that would have blocked or mispriced real work.

Rows 2, 3 and 4 are a second, related pattern: **the held-out synthetic score and the downstream
credit-panel score measure different things**, and this session treated them as interchangeable
until the disagreement was counted. Combined with row 4 and the older §117 precedent (a
prior-side score that predicted the wrong sign for the tree prior, `docs/results/FINDINGS.md`
§118), the screening signal has now been checked against the real measurement four times across
this project's history and agreed once.

Row 6 is the odd one out and worth keeping distinct: not a wrong estimate but a **fix that
solved a symptom rather than a cause**, which is why sklearn's own input validation — refusing
to proceed on bad data rather than silently reshaping it — was what caught the second bug at
all. A metric that "looks reasonable" would not have.

## Why nothing caught them sooner

Every one of these was, in isolation, a defensible thing to trust. A toy-size probe is the
standard way to sanity-check a cost model before spending money on it. A held-out synthetic
score is this project's own cheapest sanity check, run before every downstream evaluation. A
wall-clock probe is a normal way to estimate a run's cost. None of the seven rows involved
skipping a check that was known to be necessary — each involved a check that looked sufficient
and was not.

What would have caught four of the seven earlier: running the cheap and expensive measurement
side by side on a small case *before* trusting the cheap one for a real decision, which is
exactly what happened after each failure and never before it. The single-fold timing probe that
found row 7's true cost (§130) is the pattern applied correctly, and it was only tried after row
7 had already cost two hours proving the opposite lesson.

## What changed

- **`docs/roadmap/ROADMAP.md`'s Phase A now says to score candidates on the credit protocol
  directly** rather than screening on held-out synthetic first, because the screening step has
  cost more (in wrong conclusions) than it has saved.
- **Every new cost estimate in this session's later findings names what regime it was measured
  in** — production feature count, not a toy size; judging alone, not judging-plus-sampling;
  a bounded training subsample, not the full split — after three estimates that did not.
- **A single-fold or single-probe timing check is now the default before committing to a full
  run**, the same discipline `docs/infra/COMPUTE.md`'s worst-case probing already asked for on
  memory and applied here to wall-clock cost.

## What this says about the project

The chain above did not retract a single claim about the model — every number in §123-§130
that survived to be written down held up under the expensive check. What it cost instead was
**time spent on the wrong side of each check**: a depth conclusion nearly shipped on a metric
later shown blind to the effect it was supposed to screen, a filter nearly deprioritised on a
cost that was never real, a run left going for hours on an estimate that a fifteen-minute probe
would have corrected immediately. None of these needed a smarter model or a better experiment.
They needed the cheap number checked against the expensive one before being acted on, which is
the same rule the first postmortem's failures 1 through 6 were already evidence for, applied
here to engineering cost rather than to scientific conclusions.

---

# Postmortem: two harness defects read as model results

**Date:** 2026-11-13. Companion to `docs/results/FINDINGS.md` §218-§220, which carry the numbers.
Nothing here was a wrong diagnosis about the architecture. In both cases the experiment harness,
not the model, produced the number, and the first reading of that number was a statement about
the model.

## The chain

| # | The first reading said | What was actually true | Cost |
| --- | --- | --- | --- |
| 1 | §218 (MEASURED 2026-11-09 to 11-10): a probe per partition leaf is harmful on V4FinBench, **−0.0570 mean AP against the global head, 0 of 5 folds** (Holm-significant on 5 of 5), so §215's best Lite configuration is "Lite-only, and harmful on credit". The mechanism offered was that pooled scorers sit on different scales | §219 recovered from the saved predictions, bit for bit, which scorer handled each row. In every fold 93 to 97% of the test defaults sit in one leaf of about 10,000 rows, and that leaf got no probe: `_stratified_subsample` kept every positive and filled the budget with negatives, so a leaf whose defaults outnumbered its 2,000-row budget produced an all-default sample, no probe could be fitted and the leaf's own head scored it. Inside its own leaf the leaf probe ranked better than the global probe in 5 of 5 folds (fold 0: 0.1260 against 0.0970). The loss is a **sampler defect in the harness**, not a model result (§219) | Five folds of CPU time (197, 320, 188, 60 and 78 min, §218) and a "harmful" verdict written into the ledgers, `docs/paper/NEGATIVE_RESULTS.md` and `openspec/specs/METHODS.md` before the cause was found |
| 2 | Bioresponse (TabArena-Lite, binary) stopped with `feature width at predict time must match fit time`, which reads as a model that cannot take wide tables | The fork's probe re-read its training rows through the private `_preprocess`, which keeps the columns AutoGluon dropped; prediction used the public `preprocess`. The classifier was fitted on one width and handed another. The same defect stopped the first multiclass run (`lite-mc-pp`) on its first dataset and the regression run on QSAR-TID-11 (§220) | A failed first run on three problem types. No earlier Lite number was silently affected, because a dropped column always crashed the run |

## What they had in common

**Row 1 is the same shape as the first postmortem's failure 1: a pattern read before it was
decomposed.** AUC stayed between 0.980 and 0.990 in every fold while AP fell by up to 0.13, which
is a ranking failure at the top of the score range. §218 recorded that, correctly, as a hypothesis
("not yet tested"), and then stated the verdict and the consequence ("must not be described as a
credit configuration") as findings. The test that separated the hypothesis from the cause was
cheap: no refit, only the saved predictions and a per-row record of which scorer produced each
score. It was written after the verdict, not before.

**The repaired arm is not the answer, either.** §219's post-hoc repairs are chosen after seeing
the data. Scoring the single-class leaf with the global probe gives 0.2260 mean AP (+0.0274
against the global head, 4 of 5 folds), but the global probe alone gives 0.2213, so the repair
adds little over the arm it falls back to. §219 states that the repaired AP is a post-hoc reading,
not a verdict. The five-fold rerun with the fixed sampler (`ppfix-v4-*`, `runs/pp-v4-fix/`) is the
verdict, and it was queued, not run, when §219 was written.

**Row 2 is the opposite case, and the useful one.** The defect failed loudly, on the first dataset
that exercised it, and a test now fails without the fix (fork commit f9ea61d). A harness that
crashes on a mismatch costs a run. A harness that silently scored the wider table would have cost
a number.

## Why nothing caught row 1 sooner

- **Both reproduced rows passed.** The partition head matched §169 to four decimals and the global
  probe matched the default run's arm. Those checks show the harness is sound for the arms they
  cover, and said nothing about the arm that combined them.
- **The sampler was correct below its budget.** The TabArena fork's `_probe_rows` already carried
  the fix, for `Amazon_employee_access`, so §215's Lite run did not have the defect. The V4FinBench
  harness had a second copy of the same logic, without it.
- **The per-fold scorer mix was recorded and not read as a defect.** §218 noted that each fold had
  one leaf scored by its own head, and attributed the loss to scale differences between scorers
  rather than asking why a leaf with both classes had been scored as single-class.

## What changed

- **`_stratified_subsample` caps both classes** when the positives alone fill the budget:
  negatives take up to half and positives the rest. Below the budget the draw is unchanged, so the
  default run's global arms stay bit for bit. Two tests in
  `tests/test_representation_transfer.py` pin both cases (§219).
- **`scripts/pp_v4_scorer_mix.py` reports which scorer handled each row**, so a pooled score can
  be decomposed by group before it is interpreted.
- **§218 stands as a record of the harness as it ran**, and its reading is replaced by §219's.
  The "harmful" consequence in §218 is superseded in cause and not yet in verdict, which waits on
  the fixed-sampler rerun. (2026-11-14: the rerun reverses the verdict, +0.0580 mean AP on 5 of
  5 folds, §230.)
- **The private/public preprocessing mismatch is fixed and tested** in the fork (§220).

The general rule, added to the earlier ones: **a pooled metric over rows scored by different
paths is not interpretable until the rows are split by path.** A mean AP can collapse because one
path is wrong, and nothing in the mean says which.
