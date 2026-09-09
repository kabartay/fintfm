# Outline

The structure a paper would take, with the claim carrying each section (see
[CLAIMS.md](CLAIMS.md)) and the honest state of each. **Sections are ordered by what a
reviewer checks first, not by narrative appeal.**

## Title, provisionally

Something that promises a measurement, not a method. *"Coherent PD Term Structures from a
Synthetic-Only Tabular Foundation Model"* claims the two things that survive. Anything
containing "state of the art" or "outperforms" is currently false (Claim 6).

## Abstract — must contain the accuracy gap

Three sentences of result and one of deficit, in that order, with the deficit not buried:
per-horizon models are incoherent on 39% of real firms while portfolio aggregates hide it; a
survival parameterisation removes that by construction and holds at 0.00% out of time; a
synthetic-only prior transfers well enough to be within 0.041 mean AUC of a fitted
per-horizon logistic regression **while remaining behind it**, and cannot have memorised the
benchmark.

## 1. Introduction

Carries Claim 7. The wedge is regulatory rather than statistical: credit scorecards are
logistic regression *because* supervisors demand interpretability, so the accuracy bar is low
and the validation bar is high (decision D3). IFRS 9 lifetime ECL consumes a *term structure*,
which is exactly the object the field's standard construction breaks.

## 2. Related work

See [RELATED_WORK.md](RELATED_WORK.md). This section is unusually load-bearing here because
§36 established that our context-construction mechanism was published first. Getting this
wrong is the fastest way to lose a reviewer.

## 3. Method

- **3.1 The prior.** Synthetic corporate panels obeying accounting identities, sampled macro
  regime, ratio families, a default-rate envelope reaching 0.195%. Claim 2 and decision D2.
- **3.2 Architecture.** Cell embedding → column attention → pooling → row attention. The
  column stage exists to buy permutation and padding invariance, both asserted in tests
  (decision D4); it is not presented as novel.
- **3.3 The hazard head.** Claim 1. Cumulative PD as `1 - Π(1 - h_k)`, monotone by
  construction, trained on the discrete-time survival likelihood with **per-row** censoring.
- **3.4 Inference-time conditioning.** Claim 3 (rank transform), and context construction
  **citing Kostrzewa et al. for the method we use** (§36, §38). The base-rate correction and
  its precondition stated explicitly, including that it fails under retrieval because
  retrieval selects on features (§32).

## 4. Experimental protocol

**Write this before any results.** Out-of-time split on V4FinBench, train ≤ 2016 and test ≥
2017, refusing overlap. Every metric labelled by how it was produced. Paired bootstrap with
Holm correction on every difference quoted. Three seeds minimum.

**Must state plainly** that this protocol differs from V4FinBench's published one in four
ways (§36) and that our numbers are therefore not comparable with their table until task 33.2
lands. Out-of-time is the harder split, so the difference does not flatter us — which is a
reason to be precise, not a reason to skip it.

## 5. Results

- **5.1 Coherence.** Claim 1. The headline, and the only dimension where we lead.
- **5.2 Accuracy and calibration, including the gap.** Claim 6. Table with logistic
  regression, the gradient boosters (LightGBM, CatBoost, XGBoost, run out-of-process — §25),
  and us. The gap is reported in the same table, not in a later paragraph.
- **5.3 Ablations.** Claim 3 (feature conditioning: +0.086), Claim 4 and 5 (context
  construction, including the published prototype method as an arm), Claim 2 (prior ablation
  against a generic SCM prior and an untrained control).
- **5.4 What does not work.** Balanced context on low-default portfolios; enlarging the
  context past 2,000 rows; the base-rate correction under retrieval; tightening retrieval
  groups; and **query-conditioned retrieval itself**, which loses to global prototype
  selection on accuracy at the first horizon, calibration, cost and batch independence (§38).
  A results section with no negative results is not a measurement paper — and this one has the
  unusual property that its best configuration uses a *competitor's* context construction.

## 6. Limitations

See [LIMITATIONS.md](LIMITATIONS.md). Written first, cited here.

## 7. Conclusion

Resist the summary that upgrades the claims. The conclusion says: the field's standard term
structure is broken, a survival head fixes it for free, a synthetic-only prior gets within
0.041 AUC of the incumbent without ever seeing real data, and closing that gap is open work.

## Appendices

- Reproduction: `configs/`, the entry points, the packaged default config, and
  `config_sources` recorded in every run's JSON.
- The prior's generative process in full.
- Per-horizon tables for every arm.
- The failure log. `docs/POSTMORTEM.md` — five wrong diagnoses in a day, each caught by
  measurement. **Consider including this.** It is unusual, it is the strongest available
  evidence that the numbers were adversarially checked, and it costs nothing but candour.
