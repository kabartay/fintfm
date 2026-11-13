# Outline

The structure a paper would take, with the claim carrying each section (see
[CLAIMS.md](CLAIMS.md)) and the honest state of each. **Sections are ordered by what a
reviewer checks first, not by narrative appeal.**

**Rewritten 2026-09-20**, replacing a version that had been stale since §78. The previous
draft left a framing question open — a credit-domain paper or an architecture-diagnosis paper
— and said it should be decided once real-data validation reported. It has reported, along
with four other things that bear on the choice, and the question is now settled by evidence
rather than preference. What follows is the argument for that, then the structure. Rank,
categorical and regression figures were corrected against `docs/results/FINDINGS.md` on
2026-10-15.

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
| §98 — 93rd of 95 [§225: read from a table numbered from 0; §98's own leaderboard is not stored] on TabArena (94th of 95 on the later board snapshot, §122) | A cannot be written at all: there is no transfer story to tell against the field |
| §100 — up to 82% of that gap sits on datasets with categorical columns (a ceiling; §101 recovered 34%), −0.668 correlation with log cardinality | B's residual is small and *characterised*, which is what a diagnosis paper needs |

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
attention with per-cell label injection closes the regret to 0.001–0.005 and transfers to one
real credit panel, V4FinBench, at +0.042 AP best-vs-best on 5 of 5 folds; **and the fixed model
still places 94th of 95 on TabArena (§122; 2026-11-13: the readout entry on all 51 datasets is 107th of 110, §224). Native categorical handling recovered 34% of the gap
to tuned logistic regression (§101), leaving a uniform deficit of about 0.035 ROC-AUC.**

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
- **3.5 The priors, both of them.** Sections 3.1 to 3.4 describe the encoder. What it is
  trained on is two generators, `src/fintfm/prior/financial.py` and `src/fintfm/prior/scm.py`,
  drawn per task by `src/fintfm/prior/mixture.py`. Both are ours, nothing in either is fitted
  to real data, and neither is the paper's subject; the paper needs them described because the
  evidence about which one to train on is mixed and depends on the panel.

  *The financial prior.* A structural story: a macro regime, a sector mix, company scale, then
  accounting quantities that obey identities (assets = debt + equity, interest = debt times
  rate). Features are a derived numerator/denominator ratio family, which is how real panels
  such as the Polish one are wide. A latent distress score (a signed linear combination of
  about nine drivers, one optional two-way interaction, a sigmoid) gives the label, at a
  sampled base rate of 0.1% to 30%, with log or rank transforms, MCAR and MNAR missingness and
  shuffled column order applied afterwards. Its realised difficulty is bounded: of 92 sampled
  financial tasks (45 default, 47 sharpened) none reached a realised AUC above 0.99, against
  17.5% of 57 generic-SCM tasks (§77, MEASURED, gradient boosting on an in-task split).

  *The generic SCM prior.* A TabPFN-style random layered graph: 1 to 4 layers, a randomly
  chosen activation per layer (seven, including per-column rank and row softmax), Cauchy-
  weighted sparse edges, and features and target read from random nodes, with up to 10 classes.
  It encodes no domain. It is the reference for tracking a known Bayes ceiling: on the exact
  Bayes-AUC probe (§74, MEASURED, 10 seeds by 8 column-identity draws) the old-architecture
  pure-SCM checkpoint reached 0.997 against a true 0.999, where the pure-financial one
  reached 0.728. §78 later showed that cap was architectural: with two-way cell attention the
  pure-financial checkpoint reaches 0.998 (regret 0.001), so §74 is no longer evidence
  against the financial prior's content. The mixture also holds a tree prior, task families
  and a gated source (§115, §141, §180), covered in the negative-results record.

  *Which one the real-data evidence favours, by panel.* This is the current state; it is
  not one verdict.

  | panel | favours | evidence | status |
  | --- | --- | --- | --- |
  | V4FinBench, horizon 0 | financial | §63: `finbal` over `fin00` +0.0982 AP, Holm p 0.002, fold 0 only, 79 positives, old architecture, MEASURED. §180/§191, five folds, zero-shot mean AP: financial-only `e1-v48-s0` 0.1975; general trunk (SCM 0.4286, tree, task families, gated; no financial share) 0.0713; MEASURED, one seed each | holds with cell attention. Giving up financial share costs AP: `scm50` (0.5 financial, 0.5 SCM) 0.1342 against 0.1681 at 0.7 (§118, five folds, MEASURED, caveated by §166) |
  | Polish bankruptcy (1, 3 and 5 years) | SCM | §75: `fin00` over `fin07` in AP, 0.1363 against 0.1185 (1y), 0.1191 against 0.1071 (3y), 0.3021 against 0.2763 (5y); one 70/30 split each, MEASURED | **not re-measured since §78.** Old architecture; §75 itself is marked superseded in part, and the "SCM is the stronger default" reading is weakened |
  | Taiwanese bankruptcy | SCM | §75: AP 0.3223 against 0.2569 (+0.065), ROC-AUC 0.9220 against 0.8858; one 70/30 split, MEASURED | same: not re-measured with cell attention |
  | 15 OpenML tasks (general tabular) | SCM, slightly | §96: SCM better on 11 of 15, +0.0100 mean over the financial prior, both behind the best baseline on 14 of 15 (deficit 0.1442 financial, 0.1342 SCM); one split per task, untuned baselines, metric not stated in the section; MEASURED | the SCM arm carried a handicap (`column_id_dim` 12 against 16), so the gap may be near +0.025 |
  | Proxy suite, 16 binary tasks | general trunk | §191: Elo 764 (trunk then financial v2), 729 (trunk alone) against 548 for `e1-v48-s0`; sign tests 13 to 3 (p 0.021) and 14 to 2 (p 0.0042); MEASURED, one seed per arm | the trunk is a mixture, not pure SCM |

  Read together: the financial prior is the better choice on V4FinBench and the generic side
  is the better choice on general tabular data and, in the one measurement taken, on Polish
  and Taiwanese bankruptcy. The Polish and Taiwanese result predates the cell-attention fix
  and has never been repeated with it, so "SCM wins two of three credit panels" is a
  statement about the old encoder. One checkpoint cannot currently serve both V4FinBench and
  the proxy suite (§191); whether to name a credit checkpoint and a general checkpoint is an
  open decision for the owner. FinTFM-R's real-data sources are a separate family and are not
  part of either prior described here.

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
- **5.3 The fix, real.** §80/§84: +0.0486 AP matched-context, +0.0417 AP best-vs-best, 5/5
  folds, V4FinBench (the gain confounded with the prior mix, 100% against 70% financial
  tasks; confirmed 2026-10-16, audit task AU2.8). Resolved 2026-10-18 (§172): at a matched
  prior and protocol the block and its labels are worth +0.0520 mean AP (0.1947 against
  0.1428), 5/5 folds, so 5.3 can quote that figure as the architecture effect.
- **5.4 Where it still loses, externally.** §98 and §100 together — the placement *and* the
  decomposition. Presented as one result, because the placement without the decomposition
  overstates the architectural deficit by 2.8×.

  (2026-11-13: the placement has moved. On TabArena-Lite, all 51 datasets with none excluded,
  FinTFM with a trained linear readout scores 903 Elo, 107th of 110 (§224; binary 974 on 30,
  §220; multiclass 865, §224; regression 578, §223), against RandomForest (default) at 1,000.
  Wide tables rank near the bottom under column selection (§220). On four credit datasets
  FinTFM is below untuned logistic regression on mean AP (§216). The 94 of 95 in the abstract
  is the 2026-10 zero-shot rank. A paper written now would quote §224 and say that the readout
  is a trained component, not zero-shot. TabArena-Lite is breadth only; V4FinBench decides
  (§211).)
- **5.5 What does not work.** Five measured nulls: training volume −0.0012 (§93), the whole
  context axis 0.0069 (§83), marginal augmentation ~0.001 (§88), ±10 z-clip (§89), prior
  domain +0.010 (§96). Plus retrieval, which **actively hurts** at V4FinBench's base rate
  (§70), and the one non-null lever, `column_id_dim` (§97).

  The full, sourced record (harmful levers, nulls, one-benchmark wins, refuted predictions,
  deferrals and their reasons, through §224) is `docs/paper/NEGATIVE_RESULTS.md`; this list is
  the 2026-09 subset.

  (2026-11-13, §211, §222, §224: the 2026-09 subset above is not the current list. Since then
  no V4FinBench lever has cleared the §173 bar (§211), muon is harmful (§211), 480,000 tasks are
  null on V4FinBench (§188) and on TabArena-Lite multiclass (§222), and the head's probabilities
  and a 4,000-row context are null on multiclass (§224). Use `NEGATIVE_RESULTS.md`.)

  A results section with no negative results is not a measurement paper. This one has the
  unusual property that its *most actionable* recent result is a free hyperparameter nobody
  had ever tuned.

## 6. Limitations

See [LIMITATIONS.md](LIMITATIONS.md). Written first, cited here. The TabArena placement and
the absence of any LGD or EAD panel both belong in the paper's own text (a regression head now
exists, §106, and ranks near last on TabArena regression, mean rank 93.1, §121; with a ridge readout it scores 578 Elo, §223), not only in this repository. (2026-11-13: the limitations
list also needs the wide-table result, §220, and the credit gap, §216.)

## 7. Conclusion

Resist the summary that upgrades the claims. It says: a standard pooled encoder has a provable
representational limit; a probe with a known answer measures it; the fix is controlled and
transfers; and the resulting model is still far from competitive, in a way that is now
*located* rather than merely acknowledged.

## Appendices

- Reproduction: `configs/`, the entry points, the packaged default config, and
  `config_sources` recorded in every run's JSON.
- The priors' generative processes in full (both; summary in 3.5).
- **The failure log.** [`../results/POSTMORTEM.md`](../results/POSTMORTEM.md) — wrong diagnoses, each caught by
  measurement rather than review. **Include it.** It is unusual, it is the strongest available
  evidence that the numbers were adversarially checked, and it costs nothing but candour.

## What a second paper would be

Claim 1, alone: per-horizon PD construction is incoherent for 98.7% of firms out of time on V4FinBench (§229) while the
portfolio aggregate hides it; a discrete-time hazard parameterisation makes that impossible by
construction at zero accuracy cost, holding at 0.00% violations out of time. Structural rather
than statistical, decades-old mathematics, and a live defect in something IFRS 9 consumes
directly. Small, clean, and independent of whether B's residual ever closes.
