# Make one public claim, against V4FinBench's own published table

## Why

The question "are we ready to benchmark somewhere public" has a measured answer as of
2026-09-09, and it is **not on accuracy**. Best out-of-time configuration against the
incumbent it must displace (`docs/FINDINGS.md` §32):

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

- Record V4FinBench's **published baseline table and protocol** in `docs/FINDINGS.md`. This is
  the blocking gap: every number this project has produced on V4FinBench uses an out-of-time
  split of our own design, and the published results have never been read into the repository.
  Until that is done, we cannot say where we would place.
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

- **`retrieval-context` task 17.5** — three seeds. A public number must not rest on one draw.
- Reading the published paper's protocol and table, which is task 33.1 and blocks everything
  else here.
