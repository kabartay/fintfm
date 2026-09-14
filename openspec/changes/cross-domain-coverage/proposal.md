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

## UPDATE 2026-09-14: §74-§75 substantially revise this proposal's orientation

§63 was controlled evidence that the financial prior's structure helps **on V4FinBench**.
§75 now shows the opposite ordering on two independent real credit panels (Polish, Taiwan):
pure SCM beats the 70%-financial checkpoint on both, by a wide margin on Taiwan. §63's finding
is not wrong -- it is narrower than it read at the time. The financial prior's one demonstrated
real-data benefit is specific to V4FinBench and does not generalise to "credit risk" as a
domain.

Worse, §74 -- a task with an *exactly known* Bayes-optimal AUC, proposed externally as the
decisive test of capacity versus prior-content -- found that training predominantly or
exclusively on the financial prior induces a severe, reproducible cap on basic signal
extraction (~0.73 AUC regardless of true difficulty, even on a single-dimension task with no
column-identity structure at all), while the identical architecture trained on the generic SCM
prior tracks the true Bayes curve almost exactly across three orders of magnitude of
difficulty. This rules out architecture capacity as the cause and locates the defect in what
the financial prior specifically teaches.

**This changes the Non-goals below.** "Not dropping the financial prior" is no longer
defensible as stated; the correct position is narrower: the financial prior has one proven,
narrow benefit (V4FinBench) and one severe, unexplained cost (§74's cap, which shows up
everywhere including outside finance). Until the cap's mechanism is understood, the financial
prior cannot be recommended as a production default, and the generic SCM prior is the safer
choice for anything not specifically targeting V4FinBench's own feature construction.

## Non-goals

- **Not asserting the financial prior should be dropped entirely** -- §63's controlled result
  is real evidence it helps in at least one setting, and dropping it would be an overcorrection
  without understanding *why* it caps signal extraction elsewhere. The right non-goal is
  narrower: not treating it as a safe default until task 38.15 below explains the cap.
- **Not sampling the base rate as a curriculum.** §63 removed this justification: density does
  not affect transfer. The envelope stays configurable for the reasons §26 gave, and is no
  longer a research direction.
- **Not claiming competitiveness.** Every fintfm arm measured so far is well behind tuned
  gradient boosting (§60, §69). This proposal is about which prior to build and whether the
  current one is safe to build on, not about closing that gap.

## Blocked by

Task 38.15 (explain §74's capacity cap) now gates any recommendation about the financial
prior's role in a production mixture. Task 38.1 is closed by §62/§63; task 38.2 by §64-§67.
