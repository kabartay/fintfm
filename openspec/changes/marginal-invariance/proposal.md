# Make the model invariant to marginal shape, or prove it already is

## Why

`docs/FINDINGS.md` §87 measured something nobody had checked: **the model is fitted on
marginals of kurtosis 40.70 and served marginals of kurtosis 1.81.** `modeling/train.py`
applies no feature transform at all, while `FinancialTFMClassifier` defaults to
`feature_transform="rank"` — and every real-data number this project has ever reported was
produced with that default on. The network has never seen a rank-transformed feature during
pretraining, and has never made a real-data prediction on anything else.

This is a different axis from `mechanism-diverse-prior`. That proposal is about which
*data-generating mechanisms* the prior covers. This one is about a **preprocessing mismatch
between two code paths**, which no amount of mechanism coverage fixes.

**It reframes §35 rather than contradicting it.** §35 measured the rank transform as worth
+0.086 AUC and attributed it to financial ratios being pathologically heavy-tailed. That
attribution stands. What it could not see is that the transform delivers that gain *while
simultaneously moving inputs off the training distribution*. Two real effects, opposed, and
+0.086 is their net. The achievable gain from conditioning may be larger than measured.

**Three independent sources converged on the remedy.** An external review listed "random
monotonic marginal augmentation" in its priority ordering; a researcher, in conversation,
raised small perturbations of causal-graph priors as a training improvement; and then the
measurement above turned up in this codebase. Convergence is a reason to *test*, not to
believe — this proposal is built so a null result is as reportable as a positive one.

## What changes

A random strictly-monotone map is applied per column, per task, during training, so the model
learns that marginal **shape** carries no signal. Rank information and therefore all
label-relevant content survive by construction — which is assertable in a test rather than
argued in prose.

Preferred over the simpler alternative of rank-transforming during training: that removes the
mismatch but ties the weights to one preprocessing choice and turns `feature_transform="none"`
into a mismatch in the other direction. Matching-the-transform is the degenerate case of this
proposal with the augmentation distribution collapsed to a point, so the general version
subsumes it.

## Gate

**No GPU spend until the cheap test reports.** A strictly-monotone warp preserves every
within-column ordering, so the Bayes-ceiling task's optimum is *unchanged* under it — any fall
in achieved AUC is architectural non-invariance with nothing confounded. That test runs on
existing checkpoints in minutes and has three outcomes implying three different projects, one
of which is "do nothing". Committing pretraining before reading it would repeat the mistake
§76 was written to prevent.

## Non-goals

- **Not matching real-panel marginal statistics.** That is `prior-width-and-fidelity`'s job.
  This proposal is the opposite instinct: teach the model that marginal shape is *not* signal,
  rather than making synthetic marginals resemble real ones.
- **Not mechanism coverage.** `mechanism-diverse-prior` changes which data-generating
  processes the prior spans. This changes only the marginal presentation of whatever it
  already spans, and the two can be tested independently.
- **Not a claim that the mismatch costs accuracy.** §87 measured the shift and explicitly did
  not measure its cost. `normalize_features` already removes location and scale per task,
  though not shape, so the network may be robust to it. Task 43.1 exists to find out, and
  "no effect" closes this proposal rather than embarrassing it.
- **Not a retraction of §35.** The rank transform's measured +0.086 AUC stands. This proposal
  asks whether that number is a net of two opposing effects, not whether it is real.

## Blocked by

- **Task 43.1, the cheap invariance gate**, which runs on existing checkpoints in minutes.
  Nothing here may consume GPU before it reports, per §76's discipline of cheap bisection
  before spend, and per task 43.2's pre-registered routing.
- **The `cell_labels` ablation** (`cell-attention-and-task-inference` runs A/B, launched
  2026-09-17). It is the control for task 43.5 and fixes a variable that must not move at the
  same time as the augmentation (task 43.6).

## Outcome, 2026-09-17

**Closed by the gate, as designed.** Task 43.1 (§88) found outcome (c): both architectures are
genuinely non-invariant to marginal shape, and the rank transform — already the inference
default — pins performance flat across every warp. The augmentation's remaining value is the
gap between rank-clamped and best-raw performance, measured at ~0.001 on the cell-attention
checkpoint. No pretraining run is justified.

The gate cost ten minutes of CPU on existing checkpoints and prevented a GPU run. It also
produced a result nobody was looking for: cell attention is **3.4x more robust to marginal
shape** than the architecture it replaced.

Task 43.7 (the ±10 clip) remains open and is independent of this conclusion.
