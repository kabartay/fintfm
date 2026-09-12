# One checkpoint that works across target domains, not one per domain

## Why

`docs/FINDINGS.md` §63 controlled the two explanations that §61 left confounded, on the same
V4FinBench fold with a paired bootstrap:

| contrast | dAP | 95% CI | Holm p |
| --- | --- | --- | --- |
| **domain content**, density matched | **+0.0982** | [+0.0333, +0.1747] | **0.002** |
| **density**, domain matched | +0.0047 | [-0.0675, +0.0750] | 0.895 |

**The financial prior's contribution is its structure — the accounting identities and driver
relationships — and its base rate is irrelevant.** A twenty-fold change in prior density moves
nothing measurable on the target.

That is good news for the premise and bad news for portability, because the same structure
that helps on credit hurts elsewhere. A checkpoint trained without the financial prior is
substantially better on a non-credit asset-failure panel, by a margin comparable to the one the
financial prior wins by on credit. Each prior is best where its structure matches, and we
currently have no checkpoint that is good at both.

**This is the gap between a domain model and a foundation model.** Picking the right checkpoint
requires knowing the domain's structure in advance, which is a judgement the customer is paying
us to make. Two narrow models plus a manual routing decision is a defensible product; it is not
the thing this repository says it is building.

The obvious fix has already been tried and does not work. `p_financial=0.7` *is* a mixture of
the two generators, and it does not deliver best-of-both: it matches the financial-only arm on
credit (AP 0.1853 against 0.1900, indistinguishable) while being clearly beaten by the SCM-only
arm off-domain. Mixing the two sources at the task level buys the credit performance and not
the transfer.

## What

Establish what actually governs cross-domain transfer, then build for it — rather than assuming
a wider mixture suffices, which is the assumption §63 just falsified in the density direction
and which `p_financial=0.7` falsifies in the mixture direction.

Scope:

1. **Characterise the axis.** The financial and SCM generators differ measurably in
   missingness (8.9% against 0.0%), per-column separability (best single-column AUC 0.812
   against 0.911) and correlation structure. Measure which of these predicts transfer, using
   the checkpoints that already exist rather than new pretraining runs.
2. **Test structure-preserving variants.** Strip one property at a time from the financial
   generator — missingness first, as the cheapest — and measure both what the prior *teaches*
   (symmetry probes) and what it *transfers* (protocol AP). §63 established these are
   different properties, so both must be reported.
3. **Evaluate on at least two domains every time.** A single-domain score cannot detect the
   failure this proposal exists to fix, and every checkpoint comparison in this repository so
   far has been single-domain.

## Non-goals

- **Not dropping the financial prior.** §63 is the controlled evidence that it works, and an
  earlier reading of §58 that pointed the other way was wrong.
- **Not sampling the base rate as a curriculum.** That was this proposal's original content and
  §63 removed its justification: density does not affect transfer. The envelope stays
  configurable for the reasons §26 gave, and is no longer a research direction.
- **Not claiming competitiveness.** Both fintfm arms are indistinguishable from logistic
  regression, LightGBM and XGBoost, and both lose to CatBoost (§60). This proposal is about
  which prior to build, not about closing that gap.

## Blocked by

Nothing. Task 38.1 is closed by §62 and §63, which is what redefined this proposal's subject.
Task 38.2 is the cheapest next measurement and gates the pretraining tasks below it.
