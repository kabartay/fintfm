# Related work

Positioning, and what is actually left for us. Cited from primary sources; where a claim comes
from an abstract rather than the full text, that is stated. **§36 is the reason this file
exists in this much detail** — our context-construction mechanism turned out to be published
four months earlier, and that was discovered only after scoring on the discoverer's own data.

## The benchmark we evaluate on

**Kostrzewa, Tomczak, R. Furman, Poberezhna, Furgała, Farganus, O. Furman, Zięba (2026).**
*V4FinBench*, arXiv:2605.10896v2. Read in full (§36).

1,106,879 company-years, 203,900 companies, Visegrád Group, 2006-2021, 131 features, six
horizons, positive rates 0.19-0.36%. Composite distress label requiring *simultaneous*
deterioration in solvency (equity/assets < 0), profitability (EBITDA/assets < 0) and liquidity
(current ratio < 0.6). Data CC BY 4.0; label-construction code MIT.

**Their protocol, which is not ours:** 5-fold stratified cross-validation with company-level
grouping within country, ~60/20/20 per fold, released fold indices, thresholds calibrated on
the validation fold by maximising F₁. Primary metrics F₁ and ROC-AUC. **Not out-of-time** —
grouping is by company, not by date. Their horizon-*h* task removes a distressed company's
final *h* years and labels the resulting final observation positive, so their horizon *h* and
ours are **not the same prediction**.

**Their results:** prototype-undersampled, V4FinBench-fine-tuned TabPFN matches or exceeds
gradient boosting on ROC-AUC at every horizon and on F₁ from *h*=2. QLoRA-fine-tuned
Llama-3-8B trails XGBoost on ROC-AUC at every horizon (Table 2: 0.825 vs 0.995 at *h*=0,
0.553 vs 0.811 at *h*=5). Fine-tuning transfers to the American Bankruptcy Dataset (ROC-AUC
0.818 → 0.842).

**What this means for us.** Their TabPFN is *fine-tuned on the data*; ours never touches real
data, which is decision D2's whole point. Those answer different questions — "can a TFM be
adapted to this task" versus "can a synthetic-only TFM transfer to it" — and the paper must
say so rather than implying a like-for-like comparison.

## The method that pre-empts our context finding

Same paper, §5.1. Three context constructions for a bounded 10,000-row context: no
resampling; random undersampling to minority/majority = 0.3; and **prototype undersampling**,
same budget but the majority subset chosen by MiniBatchKMeans, keeping per cluster the real
observation nearest the centroid. Prototype wins, and their conclusion is that *preserving
majority-class structure matters beyond simply increasing minority exposure*.

That is §29's mechanism. **Cite them for it.** Implemented here from the description for
comparison (`prototype_context`); no code or data from that work is used, per `CLAUDE.md`'s
licensing boundary.

## Prior-fitted networks

**Hollmann, Müller, Purucker, Krishnakumar, Körfer, Hoo, Schirrmeister, Hutter (2025).**
*Accurate predictions on small data with a tabular foundation model*, Nature 637:319-326.
TabPFN. The architecture family this project reimplements independently; the in-context
mechanism, the synthetic prior idea and the small-data regime are theirs.

**Qu et al. (2025).** TabICL, ICML. Scaling in-context tabular learning to larger tables.

**Our position:** the architecture is not a contribution and should not be presented as one.
It is reproducible from this literature in days, which is also why decision D6 chose Apache-2.0
— the moat is the prior and the weights, not the code. **The weights licence must be checked
separately from the code licence, every time** (§24): Google's TabFM and TimesFM 3.0 both ship
Apache-2.0 code with non-commercial weights.

## Credit-risk benchmarking

**Baesens et al. (arXiv:2605.18147).** The authoritative benchmark. Found statistical
significance in only 22 of 406 pairwise comparisons — the reason every difference here carries
a paired bootstrap with Holm correction. Names the low-default case as promising and **does not
test it**; their datasets average a 22% default rate (§9). That absence is our motivation.

**Tanna et al. (2026), arXiv:2605.18635.** *Data Presentation Over Architecture.* Seven
context-construction strategies for credit-risk TFMs; balanced and hybrid worth 3-4 AUC points
over uniform. **We measured the reverse** on the survival path, by three times the margin
(§29), and §35 explains why both can hold: "balanced" is not one operation, and the harm
scales with how extreme the rebalancing actually is.

**Meyer et al.** Leakage and contamination in tabular benchmarking, up to 32 points of MAPE
(§1). The basis of the synthetic-only provenance argument.

## Commercial context, cited as context only

Neuralk (Seldon), Fundamental (NEXUS), Kumo (KumoRFM), Google TabFM, Feedzai RiskFM, Prior
Labs, The Forecasting Company. Public papers and posts are legitimate to read and cite; none
is a source of code, weights or data (`CLAUDE.md`). Fundamental's published oil-and-gas result
beats *linear regression* rather than gradient boosting (§2), which is the evidence that the
accuracy bar in regulated domains is lower than the validation bar.

## The 2026-09-21 peer sweep, and what it costs this project's positioning

Seven TabArena entrants opened on 2026-09-21 (full entries in `docs/REFERENCES.md`). Two of
them change what this file can claim.

### Nori is what we are, executed further — and it is public

**Synthefy (2025).** *Nori.* `github.com/Synthefy/synthefy-nori`. **Code and weights both
Apache-2.0**, checked separately on 2026-09-21 as the rule requires.

Trained **entirely on synthetic data**, in-context, single forward pass, no fine-tuning.
Alternating **feature attention and sample attention** — independently the same two-way
structure as this project's cell attention (§54), which is mild external validation of that
design and simultaneously removes it from the contribution list. Hierarchical-DAG SCM prior
with 8 edge-function types, 9 regression target families, an ExtraTrees learnability filter,
and a **999-quantile pinball head**.

**"Synthetic-only" is no longer a differentiator.** Claim 2 rests on a synthetic-only prior
with an auditable provenance argument; the *synthetic-only* half is now published, permissively
licensed, and better executed by someone else. What survives is the **financial** prior, the
provenance *argument* (not the technique), and the calibration result — a narrower claim than
the one currently written, and the paper must make it in that narrower form.

**Their scaling curve is the most useful number anyone has published for us:**

| variant | parameters | TabArena R² | overall R² (95 tasks) |
| --- | --- | --- | --- |
| nori-6m | 6M | 0.8069 | 0.7567 |
| nori-30m | 30M | 0.8099 | 0.7588 |
| nori-100m | 100M | 0.8118 | 0.7601 |

16.7× parameters returns **+0.0049** on TabArena and **+0.0034** overall. The metric is not
ours and the number does not transfer; the order of magnitude does. Our residual is **0.035**
(§101). **A perfectly executed scaling programme would not have closed it** — which is worth
stating in the paper explicitly, because "the model is small" is the explanation a reader will
reach for and it is quantitatively insufficient.

Nori is also **regression-only** (`_supported_problem_types = ["regression"]`), which is the
one axis where our declared coverage is broader.

### The counter-thesis is now the field's direction, and it is aimed at our foundation

**Spinaci, Polewczyk, Hoffart, Kohler, Thelin, Klein (2025).** *ConTextTab: A Semantics-Aware
Tabular In-Context Learner*, arXiv:2506.10707. Since renamed **SAP-RPT-1-OSS**
(`github.com/SAP-samples/sap-rpt-1-oss`); the checkpoint is unchanged.

Their abstract states the position directly: table-native ICL architectures' "**exclusive
training on synthetic data limits their ability to fully leverage the rich semantics and world
knowledge contained in real-world tabular data**". They train on large-scale real tables and
set a new standard on CARTE.

**Bonet, Comajoan Cara, Calafell, Mas Montserrat, Ioannidis (2025).** *iLTM*,
arXiv:2511.15941, KDD '26. Pretrained on **more than 1,800 heterogeneous classification
datasets**; tree-derived embeddings, a meta-trained hypernetwork, MLPs and retrieval in one
architecture. Reports beating well-tuned GBDTs and leading deep tabular models.

**TabDPT** (already recorded below as a lead) makes the same argument from the other side.

**This is three independent groups arguing that synthetic-only is a ceiling, not a virtue.**
This project cannot follow them, and the reason is not licensing: `docs/STRATEGY.md`'s
differentiator is *auditable* freedom from benchmark contamination, which real-table
pretraining destroys by construction. That trade must be argued in the paper rather than
assumed, with the accuracy cost named — and the honest framing is that we are choosing a
provenance property over accuracy, in a regulated domain where the provenance property is
worth paying for. Task 48.7 records it as a reversible decision.

### The three remaining entrants, as design references only

- **TabSTAR** (arXiv:2505.18125) — semantically target-aware representations of text and
  categorical fields. The principled version of §100/§101's out-of-fold target statistics, and
  `native-categoricals` task 47.4's open question. Second-largest rank/harmonic-rank spread on
  TabArena, for the structural reason §107 showed ours was spurious: it is genuinely a
  specialist.
- **TabFlex** (arXiv:2506.05584) — linear attention, in-context learning at millions of rows.
  Aimed at the constraint that caps `max_context` at 1000 here.
- **Orion-MSP** (arXiv:2511.02818) — multi-scale **sparse** attention. Prior art on exactly the
  axis `factorized-attention` (44.x) proposes, and 44.x should cite it rather than presenting
  the idea as new.

**Nothing from any of these enters the repository.** `CLAUDE.md`'s boundary is independent of
licence, and Apache-2.0 on Nori's weights changes only whether we may *evaluate* it.

## What is left for us

After the above, the defensible list is short and should be stated as such:

1. **Measuring that the field's standard PD term structure is incoherent on 39% of real
   firms**, invisibly at portfolio level, and removing it by construction (Claim 1).
2. **A ~~synthetic-only prior~~ *financial* prior with an auditable provenance argument**
   that transfers to real corporate default data (Claim 2). **Narrowed 2026-09-21:** Nori is
   synthetic-only, in-context, Apache-2.0 in both code and weights, and further along. The
   technique is not ours to claim; the domain-specific generative story and the provenance
   *argument* are what remain.
3. **Rank conditioning for financial ratios** — a preprocessing default inherited from general
   tabular work that is wrong here, quantified at +0.086 AUC (Claim 3).
4. ~~Query-conditioned retrieval over global prototype selection~~ — **resolved against us**
   (§38). Their prototype context is better on accuracy at the first horizon, calibration,
   cost and batch independence. Our best configuration uses their construction, and the only
   residue is the negative result itself.

Not on the list: the architecture, in-context learning, hazard models, conformal prediction,
"context construction matters", or retrieval.

**Four of the eight candidate claims have now been superseded, narrowed or retracted by
reading papers and running comparisons** — the fourth by a single afternoon's reading of the
TabArena field, which cost nothing and removed a differentiator that would otherwise have been
written into a draft.

**Three of the original eight were superseded or retracted by reading one
paper and running one comparison.** That is the value of this file existing before a draft
does, and it is an argument for reading the benchmark's own paper *before* scoring on its
data rather than after.

## Leads surfaced 2026-09-14, not yet verified from primary sources

An externally-relayed review cited several specific results while discussing architecture
priorities. **Read before citing** — this project has already been burned once by a WebFetch
summariser fabricating a results table (`docs/POSTMORTEM.md`), and the discipline that caught
it applies here too: an abstract or a relayed summary is a lead, not a citation.

- **TabDPT** (Ma, Thomas et al., NeurIPS 2025, arXiv link supplied and the abstract read in
  full this session) — combines retrieval with self-supervised learning on real tables,
  reports real data speeds training and improves downstream generalisation over synthetic-only,
  and reports scaling laws in both model and data size. Directly relevant to Claim 2's
  synthetic-only positioning (decision D2) as the standing counter-example; **their code and
  weights may not enter this repository** (standing project rule), evaluation-only comparison
  is gated on the same weight-licence check every other candidate needs.
- **TabPFN-3** (Grinsztajn et al., cited via a third-party summary, not read directly) —
  reportedly adds sinusoidal activations to its prior for oscillatory data and explicit
  extrapolation tasks. Worth noting this project's own SCM prior already includes `sin` in its
  activation set (`prior/scm.py::_ACTS`), acquired incidentally rather than by design — a
  coincidence worth mentioning if TabPFN-3 is ever cited, not a claim of having anticipated it.
- **TabICL's tree-structured prior mechanism** (cited via the same third-party summary) —
  reportedly 30% of pretraining tasks. Not verified against the primary paper.
- Two specific empirical claims relayed in the same conversation — a "2026 analysis" finding a
  synthetic prior occupies a narrow region of real-table space without the gap explaining
  downstream generalisation, and a shift-robustness evaluation of nine TFMs reporting gaps up
  to 0.060 AUC — were explicitly **not verified** when relayed and must not be cited without
  locating and reading the primary source first.
