# Postmortem: six wrong diagnoses in two days, and what they had in common

**Date:** 2026-09-09. Companion to `docs/FINDINGS.md` §28-§32, which carry the numbers. This
document exists for the *pattern*, because the individual findings each read as an isolated
slip and they were not isolated.

It was five by the end of the day. On 2026-09-08 this project produced finding 26, its first out-of-time result on real corporate
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
Every one of the four was caught by measurement inside a day, and each correction made the
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

**The scorecard did move, in the end.** Mean AUC out of time went 0.5869 → 0.8118 across the
day and the gap to per-horizon logistic regression went 0.142 → 0.048, from three inference-time
changes and no retraining: the base-rate correction applied where it belongs, retrieved
contexts, and a rank transform on features whose standard deviation exceeds ten times their
interquartile range in 110 of 136 columns. **None of the three was on the roadmap this
morning**; all three came out of chasing a wrong diagnosis to its cause.

**But the story moved further than the position.** After all four corrections, the
measured position is that per-horizon logistic regression still leads on mean AUC 0.8616 to
0.8143 and on calibration by roughly fourfold. The gap is a third of what it was. It is not
closed, and six fixed diagnoses do not close it.

**And the best configuration now uses a competitor's context construction.** §36 found that
our headline context mechanism had been published four months earlier; §38 found their method
beats ours on accuracy at the first horizon, calibration, cost and batch independence. Three
of the eight candidate claims in `docs/paper/CLAIMS.md` were superseded or retracted by
reading one paper and running one comparison — which is an argument for reading a benchmark's
own paper *before* scoring on its data, not after.
