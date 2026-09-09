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

## What is left for us

After the above, the defensible list is short and should be stated as such:

1. **Measuring that the field's standard PD term structure is incoherent on 39% of real
   firms**, invisibly at portfolio level, and removing it by construction (Claim 1).
2. **A synthetic-only prior with an auditable provenance argument** that transfers to real
   corporate default data (Claim 2).
3. **Rank conditioning for financial ratios** — a preprocessing default inherited from general
   tabular work that is wrong here, quantified at +0.086 AUC (Claim 3).
4. ~~Query-conditioned retrieval over global prototype selection~~ — **resolved against us**
   (§38). Their prototype context is better on accuracy at the first horizon, calibration,
   cost and batch independence. Our best configuration uses their construction, and the only
   residue is the negative result itself.

Not on the list: the architecture, in-context learning, hazard models, conformal prediction,
"context construction matters", or retrieval.

**Three of the eight candidate claims have now been superseded or retracted by reading one
paper and running one comparison.** That is the value of this file existing before a draft
does, and it is an argument for reading the benchmark's own paper *before* scoring on its
data rather than after.
