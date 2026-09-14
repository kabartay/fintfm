# Cover data-generating mechanisms, not just marginal distributions

## Why

`openspec/changes/prior-width-and-fidelity` closes the gap between the prior's *shape* and
real credit panels — column count, inter-feature correlation. That is realism-matching: making
synthetic tables look statistically like the real ones. It is a different axis from what this
proposal is about.

`docs/FINDINGS.md` §76 bisected four content-side candidates for why training on the financial
prior caps basic signal extraction (§74) — base rate, raw signal-to-noise, feature cleanliness,
accounting-identity structure — and none of them closed the gap. What was common to every
capped variant and absent from the one that was not (`fin00`, pure generic SCM) was the
**label-generating function's form**: financial's label is always a signed linear combination
of a driver subset plus at most one two-way interaction, while SCM's label is read from an
arbitrary node of a random 1-4-layer computational graph with five candidate nonlinearities.

An externally-proposed review (relayed by the user, 2026-09-14, independently reviewing this
project's own documents) makes the same point from the literature side: current tabular
foundation models (TabICL, TabPFN-3) explicitly add *mechanism* diversity to their priors —
tree-structured relations, oscillatory (sinusoidal) functions, explicit extrapolation tasks —
not merely more marginal-distribution families. This project's SCM prior already has one such
mechanism (`sin` is in `prior/scm.py`'s `_ACTS`), acquired incidentally rather than by design.
The review's point, made explicitly: **"don't assume synthetic diversity solves everything...
optimise for mechanism coverage, not marginal realism."**

**§77 sharpens this further.** Even sharpened, the financial prior's realized task
difficulty never reaches the 0.99+ region the §74 probe tests -- zero of 92 sampled tasks
across both sharpness settings, against 17.5% for generic SCM. This is structural: a bounded
linear combination over driver weights is exactly the setting where extreme realizations are
rare (central-limit-like smoothing), and no rescaling of the whole score undoes that. A
mechanism-diverse prior needs label constructions that can concentrate variance sharply, not
merely a wider range on the current one's overall scale.

**Caveat on sourcing.** Two specific empirical claims relayed in the same review — a "2026
analysis" finding a commonly-used synthetic prior occupies a narrow region of real-table space
without the gap explaining downstream generalisation, and a shift-robustness evaluation of nine
TFMs reporting gaps up to 0.060 AUC — were not verified before this proposal was written and
should not be cited as established without checking the primary source first.

## What

1. **Task-family generators, deliberately varied**: linear `y=sigma(w^T x)`, threshold
   `y=1[x_j>c]`, interaction `y=1[x_1 x_2>c]`, XOR, max/min, piecewise, sparse (`|S|<<d`
   relevant features), dense, latent-factor (`z -> X`, `z -> y`), tree-structured — most of
   these do not exist in either current prior; the SCM prior's random-graph construction
   covers some implicitly but not by controllable, labelled family.
2. **A controlled difficulty axis independent of family**, reusing `docs/FINDINGS.md` §74's
   closed-form Bayes-AUC construction so difficulty is *known*, not estimated, for every family.
3. **Interaction-order curriculum**: `y=f(x_{i1},...,x_{ik})` for `k=1..5`, measuring the
   AUC-vs-k curve per family — the direct, controlled version of what §76's bisection could
   only gesture at.
4. **Compositional task generation**: train families separately, test on held-out compositions
   (`f1(x0)+f2(x1)+f3(x2,x3)`) never seen exactly assembled — the sharper test of whether the
   prior is foundational rather than a lookup over seen shapes.
5. **Correlation/confounding/SCM diversity**: redundant predictors, proxies, confounders,
   collider structures, building on the SCM machinery already in this repo.
6. **Missingness, shift and support-extrapolation axes**, sampled independently of the above
   rather than bundled into "financial" or "SCM" as fixed packages.

## UPDATE 2026-09-14: premise substantially weakened by §78

`cell-attention-and-task-inference` task 39.4 closed §74's capacity cap with an architecture
change alone -- regret at Bayes AUC 0.90-0.999 fell from 0.234-0.277 to 0.001-0.005 on a
checkpoint trained *exclusively* on the financial prior, no change to its content at all. The
cap this proposal was written to explain from the prior side was, per §78, architectural.

This does not make the label-functional-form candidate (§77: financial tasks, even sharpened,
essentially never reach realized difficulty above 0.99 AUC) uninteresting on its own terms --
it remains a real, measured, structural property of the current financial generator. It does
remove the urgency this proposal was written under. **Do not resume this proposal's pretraining
tasks (40.7) without first checking whether the architecture fix alone is sufficient** for
whatever the next real-data question is; building mechanism diversity to fix a cap that no
longer exists would be solving an already-solved problem.

## Non-goals

- **Not matching real-panel marginal statistics.** That is `prior-width-and-fidelity`'s job;
  duplicating it here would blur two genuinely different objectives.
- **Not started before `cell-attention-and-task-inference` reports a result.** Per the user's
  explicit priority (2026-09-14) and this proposal's own evidence (§76): if the architecture
  experiment closes §74's gap, this proposal's urgency drops sharply, since the cap would have
  been architectural rather than a prior-coverage gap. If it does not, this becomes the leading
  hypothesis.

## Blocked by

`cell-attention-and-task-inference` task 39.4 (the architecture-vs-content bisection's
result). Task 39.6 there explicitly hands off to this proposal if the architecture change does
not close the gap.
