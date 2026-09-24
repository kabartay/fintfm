# Outline

The structure a paper would take, with the claim carrying each section (see
[CLAIMS.md](CLAIMS.md)) and the honest state of each. **Sections are ordered by what a
reviewer checks first, not by narrative appeal.**

**Rewritten 2026-09-20**, replacing a version that had been stale since §78. The previous
draft left a framing question open — a credit-domain paper or an architecture-diagnosis paper
— and said it should be decided once real-data validation reported. It has reported, along
with four other things that bear on the choice, and the question is now settled by evidence
rather than preference. What follows is the argument for that, then the structure.

## The framing question, and why the evidence closed it

The previous draft offered two candidate papers:

- **A.** *A synthetic-only prior transfers to corporate credit risk.*
- **B.** *A symmetric row encoder cannot learn column-specific rules: diagnosis and a two-way
  cell-attention fix.*

Four measurements since then make **B** the only defensible one:

| finding | effect on framing |
| --- | --- |
| §80/§84 — the fix transfers to real data, +0.042 AP best-vs-best, 5/5 folds | B's central result holds outside synthetic probes |
| §96 — the accuracy deficit is **general, not credit-specific**; behind on 14 of 15 public tasks | A's premise is weakened at its core |
| §98 — 93rd of 95 on TabArena | A cannot be written at all: there is no transfer story to tell against the field |
| §100 — 82% of that gap is categorical preprocessing, −0.668 with log cardinality | B's residual is small and *characterised*, which is what a diagnosis paper needs |

**A is not currently writable.** Its claim is comparative and the comparison is lost, on our
own benchmark and on the field's. **B is writable now**, and its strength is unusual: it has a
defect with a *provable* ceiling, an instrument with a closed-form known answer, seven
eliminated alternative explanations, a controlled fix, and independent real-data confirmation.

Claim 1 (term-structure coherence) remains true and strong, but it is a **separate, smaller
paper** — a measurement of a live defect in a standard construction — and folding it into B
would dilute both. Keep them apart.

## Title, provisionally

*"A Symmetric Row Encoder Cannot Learn Column-Specific Rules: Diagnosis and a Two-Way
Cell-Attention Fix"*

Promises a measurement, not a method. Anything containing "state of the art", "outperforms" or
"competitive" is **false** and Claim 11 says so with someone else's leaderboard.

## Abstract — must contain the placement, not only the fix

Three sentences of result and one of deficit, deficit not buried: a pooled symmetric row
encoder is provably incapable of representing a rule as simple as `x₀ − x₁`, and a probe with
an exactly-known Bayes-optimal AUC shows the resulting cap is severe and independent of task
difficulty; seven content-side explanations are eliminated one at a time; two-way cell
attention with per-cell label injection closes the regret to 0.001–0.005 and transfers to real
credit panels at +0.042 AP on 5 of 5 folds; **and the fixed model still places 93rd of 95 on
TabArena, where 82% of the residual gap is categorical preprocessing rather than
architecture.**

That last clause is not self-flagellation — it is what makes the diagnosis credible. A paper
that fixes a defect and then claims the field is solved invites exactly the scrutiny it cannot
survive.

## 1. Introduction

Carries Claim 7 (low-default portfolios are underserved) as **motivation for the setting**,
not as a result. The regulatory wedge (decision D3) explains why the accuracy bar is low and
the validation bar is high, and why a term structure is the object that matters.

## 2. Related work

See [RELATED_WORK.md](RELATED_WORK.md). Unusually load-bearing: §36 established that our
context-construction mechanism was published first by Kostrzewa et al., and the best
configuration in this project uses **a competitor's** context construction. Saying so plainly
is cheaper than being caught.

## 3. Method

- **3.1 The defect.** The row encoder as a symmetric function of the row's multiset;
  `x₀ − x₁` has a symmetric ceiling of exactly 0.5 and the pre-fix model scored 0.5097.
- **3.2 The instrument.** The Bayes-ceiling probe (§74): closed-form `Φ(μ/√2)`, verified
  numerically against the empirical AUC of the true Bayes-optimal statistic. **This is the
  paper's methodological contribution as much as the fix is** — a probe whose correct answer
  is known in advance cannot flatter a model the way a benchmark can.
- **3.3 The fix.** Row-attention-within-feature plus per-cell label injection. §91 shows both
  halves are jointly necessary: without labels the variant scores **below chance**.
- **3.4 What it costs.** Column-order invariance becomes distributional rather than exact
  (D12), recovered by ensembling over identity draws.

## 4. Experimental protocol

**Write this before any results.** Paired bootstrap with Holm correction on every quoted
difference; average precision read first at low prevalence (D13); three seeds minimum; every
metric labelled by how it was produced.

**State plainly** that Fisher combination is sign-blind and detects heterogeneity rather than
direction (§93/§97) — this project misread it once and the correction belongs in the open.

## 5. Results

- **5.1 The cap, and its independence from the prior.** §74, plus the seven eliminations
  (§58–§77). The eliminations are the evidence that the diagnosis is right, so they are
  results and not an appendix.
- **5.2 The fix, synthetic.** §78, both `p_financial` arms, two independent instruments.
- **5.3 The fix, real.** §80/§84: +0.0486 matched-context, +0.0417 best-vs-best, 5/5 folds.
- **5.4 Where it still loses, externally.** §98 and §100 together — the placement *and* the
  decomposition. Presented as one result, because the placement without the decomposition
  overstates the architectural deficit by 2.8×.
- **5.5 What does not work.** Five measured nulls: training volume −0.0012 (§93), the whole
  context axis 0.0069 (§83), marginal augmentation ~0.001 (§88), ±10 z-clip (§89), prior
  domain +0.010 (§96). Plus retrieval, which **actively hurts** at V4FinBench's base rate
  (§70), and the one non-null lever, `column_id_dim` (§97).

  A results section with no negative results is not a measurement paper. This one has the
  unusual property that its *most actionable* recent result is a free hyperparameter nobody
  had ever tuned.

## 6. Limitations

See [LIMITATIONS.md](LIMITATIONS.md). Written first, cited here. The TabArena placement and
the missing regression head both belong in the paper's own text, not only in this repository.

## 7. Conclusion

Resist the summary that upgrades the claims. It says: a standard pooled encoder has a provable
representational limit; a probe with a known answer measures it; the fix is controlled and
transfers; and the resulting model is still far from competitive, in a way that is now
*located* rather than merely acknowledged.

## Appendices

- Reproduction: `configs/`, the entry points, the packaged default config, and
  `config_sources` recorded in every run's JSON.
- The prior's generative process in full.
- **The failure log.** [`../results/POSTMORTEM.md`](../results/POSTMORTEM.md) — wrong diagnoses, each caught by
  measurement rather than review. **Include it.** It is unusual, it is the strongest available
  evidence that the numbers were adversarially checked, and it costs nothing but candour.

## What a second paper would be

Claim 1, alone: per-horizon PD construction is incoherent for 39% of real firms while the
portfolio aggregate hides it; a discrete-time hazard parameterisation makes that impossible by
construction at zero accuracy cost, holding at 0.00% violations out of time. Structural rather
than statistical, decades-old mathematics, and a live defect in something IFRS 9 consumes
directly. Small, clean, and independent of whether B's residual ever closes.
