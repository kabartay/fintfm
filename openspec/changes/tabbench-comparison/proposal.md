# Compare against Seldon on TabBench v2, using published numbers only

## Why

Neuralk's Seldon is a direct point of comparison and, on published evidence, at the frontier:
on their TabBench v2 it beats TabPFN v3 on 18 of 22 problems and TabICL v2 on 15 of 22, while
TabPFN v3 edges it by 0.2 AUC — inside the noise floor — with more outright dataset wins
(27.5% against 14.8%). The honest reading is a three-way tie at the top. Seldon also ran
natively on 100% of the suite where two of eight tested models failed on 19% and 29% of
datasets.

**TabBench v2 is the licence-clean venue for this.** It is a public benchmark with published
results, so we can run our model on it and cite their numbers while touching no code, weights
or data of theirs — which is exactly the line `CLAUDE.md` draws: *"Evaluating against
published numbers is always fine; running someone's checkpoint inside anything commercial is
not."*

**What is explicitly excluded:** any third-party data received through a client engagement.
Such data belongs to the client, and reusing it to benchmark a different commercial project is
a permission question — see `CLAUDE.md`'s licensing boundary, which requires confirming the
terms, keeping the data outside this repository, and asking before publishing anything derived
from it.

## What

- Fetch TabBench v2 and confirm its licence permits our use before anything else.
- Run our checkpoint through it under the suite's own protocol, reporting the fraction of
  datasets we can run natively alongside accuracy — Seldon's 100% coverage is a real result
  and a model that silently skips hard datasets is flattered by the average.
- Report against their **published** figures. Never run their checkpoint.
- Include the untrained control from `fintfm-capability` in every table, so a reader can tell
  learning from architecture.

## Non-goals

- **Not a bid to win TabBench.** Decision D3 chose deliberately not to compete on general
  tabular accuracy against a $255M Series A; the claim is coherent PD term structures and
  validation evidence for regulated credit. A TabBench number is a *calibration of where we
  stand*, not a product claim, and the write-up must say so or it invites the comparison D3
  declined.
- Not a general-tabular pivot.

## Falsified by

Nothing to falsify — this is a measurement. The useful outcome is a defensible statement of
distance from the frontier, in either direction.

## Blocked by

**`docs/FINDINGS.md` §42.** The model scores 0.685 on a clean linear task that logistic
regression solves at 0.9997 and beats its own random weights by 0.015. Running TabBench today
would produce a number that measures §42 and nothing else. The gate is the capability probes
(`fintfm-capability`) showing the trained model clearing the untrained control by a wide
margin on `linear` and `conjunction`.
