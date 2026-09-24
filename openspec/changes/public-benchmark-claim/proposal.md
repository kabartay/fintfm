# Make one public claim, against V4FinBench's own published table

## Why

The question "are we ready to benchmark somewhere public" has a measured answer as of
2026-09-09, and it is **not on accuracy**. Best out-of-time configuration against the
incumbent it must displace (`docs/results/FINDINGS.md` §32):

| arm | mean AUC | mean ECE | coherence violations |
| --- | --- | --- | --- |
| fintfm, retrieval, 4,000 | 0.7934 | 0.0111 | **0.00%** |
| per-horizon logistic regression | **0.8616** | **~0.0018** | 39.06% |

Three of four horizon differences are significant against us after Holm correction. A public
accuracy comparison would show exactly that, and it should — but it is not a launch.

**What is publishable is the coherence result** (§26, §32): 0% cumulative-PD violations
against 39% of horizon steps for the per-horizon construction, with 98.6% of firms affected
and the portfolio aggregate hiding it. That is large, structural, and independent of accuracy.
It is also **cheap for a competitor to copy** — a hazard head is an afternoon's work — so its
value is in being first to state and measure it, not in being hard to reproduce.

**A leaderboard is the wrong venue for it.** Kaggle ranks a single accuracy metric, which is
the one dimension we lose on; the Home Credit submission already demonstrated that. The right
venue is the benchmark that published the dataset: Tomczak et al. compare tabular foundation
models, LLMs and standard methods on this exact data, and a comparison against **their
published table, on their protocol** is a credible public claim with no leaderboard involved.

## What

**Updated 2026-09-09 after task 33.1 (`docs/results/FINDINGS.md` §36).** The published protocol has now
been read, and it is further from ours than this proposal assumed: 5-fold company-grouped
stratified cross-validation rather than out-of-time, horizon tasks built on *different rows*,
a 10,000-row inference context, and a TabPFN **fine-tuned on their data**. Our numbers cannot
be placed against theirs at all, in either direction. Reproducing their protocol is therefore
the substance of this change rather than a validation step — and they release fold indices,
which makes it possible.

The other finding from 33.1 reshapes the claim itself: their best method, prototype
undersampling, **is §29's mechanism published in May 2026**. "Context construction matters, and
preserving majority-class structure is why" is theirs. What remains ours is that retrieval is
*query-conditioned* rather than global, measured against blind sampling on the same data. Any
public claim leads with that distinction or it is overclaiming.
- Reproduce their protocol exactly as a second arm, keeping our out-of-time arm beside it.
  Out-of-time is the harder split, so reporting only ours understates us; reporting only
  theirs would drop the property a model-risk reviewer cares about. Report both.
- Add the coherence measurement to their protocol, since their table does not contain it —
  that is the axis on which the claim is made.
- Three seeds on everything quoted publicly. §29, §31 and §32 are single-draw.

## Non-goals

- **No accuracy claim.** Not "competitive with", not "approaching". The measured gap is
  0.068 mean AUC and significant; any wording that blurs it is the failure mode this
  repository's evaluation-honesty spec exists to prevent.
- Not a Kaggle submission. Wrong metric, wrong protocol, and already tried.
- Not a paper. One reproducible benchmark report against a published table.

## Falsified by

If reproducing their protocol puts us far below their published baselines, the coherence claim
still stands but has to be published as "coherent and less accurate", which is a much weaker
position and would argue for delaying any public statement until the accuracy gap closes.

## Blocked by

- **Task 33.2**, reproducing their protocol, now blocks every comparative claim — see §36.
- ~~`retrieval-context` task 17.5 — three seeds~~. **Done** (§33): retrieval's worst seed beats
  uniform's best at every context size.
- ~~Reading the published protocol and table~~. **Done** (§36).
