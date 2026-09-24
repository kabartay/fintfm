# References

Literature this project stands on. **Every entry below was verified against the source**
(title, authors, venue confirmed 2026-09-08) rather than cited from memory — the sibling
repos have shipped a fabricated cross-reference before, and a citation reads as authority
whether or not anyone checked it. Add nothing here without opening it first.

## The architectural lineage

These describe the idea this repository implements independently. **Read them; do not copy
from their implementations** — see the licensing boundary in `CLAUDE.md`.

**Müller, Hollmann, Pineda Arango, Grabocka & Hutter (2022).** *Transformers Can Do Bayesian
Inference.* ICLR 2022. [arXiv:2112.10510](https://arxiv.org/abs/2112.10510)
The Prior-Data Fitted Network (PFN). A transformer trained on tasks sampled from a prior
approximates the posterior for a new supervised task in a single forward pass. This is the
mechanism in `model.py`: context rows carry labels, query rows attend only to context, and
training is one gradient step per freshly sampled synthetic task.

**Hollmann, Müller, Eggensperger & Hutter (2022).** *TabPFN: A Transformer That Solves Small
Tabular Classification Problems in a Second.* [arXiv:2207.01848](https://arxiv.org/abs/2207.01848)
PFNs applied to tabular classification, with a prior built from **structural causal models**.
`prior/scm.py` is an independent implementation of that general idea (random layered MLP
graphs, random activations, target read from a hidden node); the financial prior in
`prior/financial.py` is this project's own contribution and has no analogue there.

## Framing and risk

**Bommasani et al. (2021).** *On the Opportunities and Risks of Foundation Models.*
[arXiv:2108.07258](https://arxiv.org/abs/2108.07258)
The report that named the category. Its **homogenization** warning is the one that matters
here and it is rarely quoted by vendors: *"the defects of the foundation model are inherited
by all the adapted models downstream."*

That argument lands unusually hard in credit. If many lenders score borrowers with the same
tabular foundation model, their failures correlate, and correlated failure across lenders is
systemic risk — precisely what banking supervision exists to prevent. This cuts two ways for
this project and both should be stated honestly to any regulator or buyer: it is a **real
risk** of the product category, and it is a **reason** supervisors will demand exactly the
per-deployment validation evidence this project intends to sell. Do not pitch the category
without acknowledging it; a vendor who has clearly thought about supervisory concerns is more
credible, not less.

## Evaluation and leakage

**Meyer, Kaltenpoth, Zalipski & Müller (2025).** *Rethinking Evaluation in the Era of Time
Series Foundation Models: (Un)known Information Leakage Challenges.*
[arXiv:2510.13654](https://arxiv.org/abs/2510.13654)
Data lineage of 15 prominent TSFMs; two leakage modes (test-set contamination via
multi-purpose dataset reuse, and memorisation of global patterns from external shocks);
measured effect sizes; 11 requirements for fair benchmarking. Fully worked through in
`docs/results/FINDINGS.md` §1, including the caveat that limits how strongly this project may claim
the advantage.

**Makridakis, Spiliotis & Assimakopoulos (2022).** *The M5 competition: Background,
organization, and implementation.* International Journal of Forecasting, 38(4):1325–1336.
doi:[10.1016/j.ijforecast.2021.07.007](https://doi.org/10.1016/j.ijforecast.2021.07.007)

**Makridakis, Spiliotis, Hollyman, Petropoulos, Swanson & Gaba (2024).** *The M6 forecasting
competition: Bridging the gap between forecasting and investment decisions.* International
Journal of Forecasting.
doi:[10.1016/j.ijforecast.2024.11.002](https://doi.org/10.1016/j.ijforecast.2024.11.002)

Cited within Meyer et al. above as the leakage-resistant evaluation design: forecasts are
registered *before* the outcome exists. **M6 went furthest** — live data from 100 financial
assets, predictions registered into the real future. The critique noted there is
administrative cost and long waiting periods between competitions. See `docs/results/FINDINGS.md` §3
for why this design matters more to this project than to a forecasting vendor.

## Domain benchmarks and datasets

**Tanna, Solanki, Bouadi, Bouarour, Seth & Sankarapu (2026).** *Data Presentation Over
Architecture: Resampling Strategies for Credit Risk Prediction with Tabular Foundation
Models.* [arXiv:2605.18635](https://arxiv.org/abs/2605.18635)
Seven context-construction strategies across four classical models and five TFMs on Home
Credit and Lending Club. Balanced and hybrid sampling add 3-4 AUC points over uniform, a gap
wider than the spread between TFM families. Acted on in `docs/results/FINDINGS.md` §5, and its
AUC-only framing corrected by §6.

**Tomczak et al. (2026).** *V4FinBench: Benchmarking Tabular Foundation Models, LLMs, and
Standard Methods on Corporate Bankruptcy Prediction.*
[arXiv:2605.10896](https://arxiv.org/abs/2605.10896)
1,106,879 company-year observations, Visegrád Group economies, 2006-2021, 131 features, six
horizons, 0.19-0.36% positive rate. Code [MIT](https://github.com/genwro-ai/V4FinBench);
**data CC BY 4.0** on Kaggle per the repository's separate `DATA_LICENSE.md`. The dataset
this project needs — see `docs/results/FINDINGS.md` §8.

**Zieba, Tomczak & Tomczak.** *Polish companies bankruptcy data.* UCI Machine Learning
Repository, CC BY 4.0. [doi:10.24432/C5F600](https://doi.org/10.24432/C5F600)
In use via `evaluation/datasets.py`. 64 anonymous ratios, no dates, no identifiers (§7).

**Liang, Lu, Tsai & Shih.** *Taiwanese bankruptcy prediction.* UCI Machine Learning
Repository, CC BY 4.0. 6,819 firms, 95 features, 3.23% positive rate, 1999-2009, no missing
values, and — verified by inspection — no dates or identifiers (§8).

**Baesens, Goethals, Lessmann, De Vos, Bravo, Martens, Medina-Olivares, Mues, Oskarsdóttir,
vanden Broucke, Van Gestel, Verdonck & Verbeke (2026).** *Foundation Models for Credit Risk
Prediction: A Game Changer?* [arXiv:2605.18147](https://arxiv.org/abs/2605.18147)
**The single most important reference for this project.** Five TFMs (TabPFN, TabPFNv2,
TabPFN-Real, MITRA, TabICL) against 29 PD and 22 LGD methods over 14 PD datasets
(1,000-532,428 rows) and 7 LGD datasets. TFMs win more often and their advantage grows as
data shrinks, with the LGD crossover near 8,000 observations. Read the caveats in
`docs/results/FINDINGS.md` §9 before quoting it: margins are small, only 22 of 406 pairwise PD
comparisons were significant, and the low-default-portfolio benefit is conjectured rather
than tested.

**Purucker, Tschalzev, Erickson et al. (2026).** *Beyond IID: How General Are Tabular
Foundation Models, Really?* [arXiv:2606.30410](https://arxiv.org/abs/2606.30410)
Introduces BeyondArena over IID, temporal and grouped tasks. TFMs excel on tiny-to-medium
IID data; trees and deep learning still dominate on non-IID, large and high-dimensional
data. Defines which half of credit risk is winnable (`docs/results/FINDINGS.md` §9).

**Hollmann, Müller, Purucker, Krishnakumar, Körfer, Hoo, Schirrmeister & Hutter (2025).**
*Accurate predictions on small data with a tabular foundation model.* Nature 637:319-326.
[doi:10.1038/s41586-024-08328-6](https://doi.org/10.1038/s41586-024-08328-6)
TabPFN v2. The canonical citation for the small-data strength this project is betting on.

**Erickson, Purucker, Tschalzev, Holzmüller, Mutalik Desai, Salinas & Hutter (2025).**
*TabArena: A Living Benchmark for Machine Learning on Tabular Data.* NeurIPS Datasets and
Benchmarks Track.
[proceedings](https://papers.neurips.cc/paper_files/paper/2025/hash/1697e3fb412da11dc9488249f9e7bbc9-Abstract-Datasets_and_Benchmarks_Track.html)
51 datasets (38 classification, 13 regression). The standard leaderboard, and where Google's
TabFM reports its results.

**Fonseca & Stoyanovich (2026).** *ExplainerPFN: Towards tabular foundation models for
model-free zero-shot feature importance estimations.*
[arXiv:2601.23068](https://arxiv.org/abs/2601.23068)
A TabPFN-based model pretrained on synthetic datasets with Shapley-value labels, predicting
attributions with no model access, gradients or example explanations. Competitive with
few-shot surrogates using 2-10 SHAP examples. Honest about the core limitation: several
models can share predictions yet differ in Shapley decomposition, so attributions are "true
to the data" rather than "true to the model". Underpins
`openspec/changes/zero-shot-attribution`.

**Rahimikia, Ni & Wang (2025).** *Re(Visiting) Time Series Foundation Models in Finance.*
SSRN working paper, 138 pp.
[record](https://research.manchester.ac.uk/en/publications/revisiting-time-series-foundation-models-in-finance/)
Daily excess returns across global markets. **Off-the-shelf TSFMs performed poorly zero-shot
and fine-tuned, while models pretrained from scratch on financial data improved
substantially**, with synthetic augmentation helping further. Time-series rather than
tabular, so suggestive rather than direct evidence — but it is the same hypothesis Phase 1
tests, confirmed in an adjacent modality.

**Lu, Juntong (2025).** *Time-Series Foundation Models in Finance: Pretraining Corpora,
Architectures, Financial Benchmarks, and Risk-Aware Evaluation.* SSRN 5570099.
[record](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5570099)
Read for its **evaluation discipline**, not its models: probabilistic scoring (CRPS, pinball
loss) paired with supervisory backtests — Kupiec unconditional coverage, Christoffersen
conditional coverage, Acerbi-Székely — plus data-snooping control via White's Reality Check
and Hansen's SPA with family-wise error control. Directly motivated the paired bootstrap and
Holm correction now in `evaluation/metrics.py`, and the coverage tests belong in
`openspec/changes/conformal-pd-certificate`.

**Qu, Holzmüller, Varoquaux & Le Morvan (2025).** *TabICL: A Tabular Foundation Model for
In-Context Learning on Large Data.* ICML, PMLR 267:50817-50847.
[proceedings](https://proceedings.mlr.press/v267/qu25d.html) · code
[soda-inria/tabicl](https://github.com/soda-inria/tabicl)
**Architecturally the closest published work to this project.** A two-stage design —
column-then-row attention producing fixed-dimensional row embeddings, then a transformer for
ICL — which is the same decomposition `modeling/model.py` arrived at independently. Their
result matters for our scaling question: pretrained on synthetic sets up to 60K samples,
handling 500K, and on 53 datasets above 10K samples it **beats both TabPFNv2 and CatBoost**.
That qualifies the Beyond IID reading (§9): ICL can scale, given the right architecture.
Note also that TabPFNv2 is reported as excelling only up to ~10K samples because alternating
column/row attention becomes prohibitive — the cost our row-pooling stage also avoids.

**Zhu, Chen, Qu & Chung (2025).** *FinCast: A Foundation Model for Financial Time-Series
Forecasting.* [arXiv:2508.19609](https://arxiv.org/abs/2508.19609)
**The clearest published statement of this project's core bet, in the adjacent modality.**
Their motivation is ours verbatim: general foundation models (TimesFM, Chronos, TimesMoE)
"do not specifically address the idiosyncrasies of financial data, such as volatility, noise,
and pattern shift". A 1B-parameter decoder trained on 20B+ financial time points reports
20-23% error reductions over the general models — i.e. **domain-specific pretraining beats
general pretraining in finance**, which is exactly what `docs/results/FINDINGS.md` §14 measured for
tabular credit (+0.049 AUC, financial prior over generic).

Two ideas worth borrowing as *concepts*, not code. Their **Point-Quantile loss** jointly fits
a point forecast and quantiles, claimed to prevent forecast collapse under non-stationarity —
relevant to `conformal-pd-certificate`, since our hazard head currently emits point
probabilities and the certificate needs intervals. And their framing of the three shift
sources (temporal non-stationarity, multi-domain diversity, varying resolution) maps cleanly
onto credit: economic regimes, country and sector heterogeneity, and reporting frequency.

**The caveat is the usual one:** 1B parameters on 20B time points against our 850K. The
principle transfers; the scale does not, and their result is not evidence that a small model
inherits the benefit.

**Wu et al. (2023).** *BloombergGPT: A Large Language Model for Finance.*
[arXiv:2303.17564](https://arxiv.org/abs/2303.17564)
50B parameters, 363B financial tokens plus 345B general-purpose. The landmark
domain-specific financial model, and the third independent data point on pretraining
mixture (see `docs/results/FINDINGS.md` §14): its **51% domain / 49% general** split was chosen to
avoid sacrificing general capability. Also the cost benchmark the FinGPT line defines itself
against — ~$2.67M of compute, which is the asymmetry `docs/roadmap/STRATEGY.md` argues not to fight
head-on.

### TabArena's nearer neighbours — the four methods immediately above us

Opened 2026-09-21 because §107 needs the competitive context, and because these four sit in
the band fintfm is trying to leave rather than at the top of the board. **Ideas only.**
`CLAUDE.md`'s prohibition on ingesting code or weights from tabular-foundation-model products
applies to every one of them, and a weights licence is checked separately from a code licence,
every time — none of these has had its weights licence checked, because nothing from them is
being ingested.

**Arazi, Shapira & Reichart (2025).** *TabSTAR: A Tabular Foundation Model for Tabular Data
with Text Fields.* [arXiv:2505.18125](https://arxiv.org/abs/2505.18125). Paper CC-BY-4.0;
code at `github.com/alanarazi7/TabSTAR`. **The nearest structural analogue.** Learns
*semantically target-aware* representations of text and categorical fields rather than
encoding them by a fixed rule — the principled version of what §100/§101 does with out-of-fold
target statistics, which is `openspec/changes/native-categoricals` task 47.4's open question.
On TabArena it has the **second-largest rank/harmonic-rank spread** (67.67 vs 16.10) for the
structural reason fintfm's turned out to be spurious: it is genuinely a text-field specialist.

**Zeng, Dinh, Kang & Mueller (2025).** *TabFlex: Scaling Tabular Learning to Millions with
Linear Attention.* [arXiv:2506.05584](https://arxiv.org/abs/2506.05584). Replaces quadratic
attention with a linear variant so in-context learning reaches millions of rows. Directly
relevant to our binding constraint: attention cost is why `max_context` is 1000 and why §83/§84
had to measure whether context even helps. Note the trade this implies — linear attention is
not free, and §80's finding that context is nearly flat on a 1M-row panel means more context is
not obviously the lever for us.

**Bouadi, Seth, Tanna & Sankarapu (2025).** *Orion-MSP: Multi-Scale Sparse Attention for
Tabular In-Context Learning.* [arXiv:2511.02818](https://arxiv.org/abs/2511.02818). Code at
`github.com/Lexsi-Labs/Orion-MSP`. Multi-scale **sparse** attention over tabular structure.
Our two-way cell attention (§54) is dense in both directions, and `openspec/changes/
factorized-attention` (44.x) is the remaining architectural candidate for the uniform residual
— this is prior art on the same axis.

**Bonet, Comajoan Cara, Calafell, Mas Montserrat & Ioannidis (2025).** *iLTM: Integrated Large
Tabular Model.* [arXiv:2511.15941](https://arxiv.org/abs/2511.15941), KDD '26. Hypernetworks
plus retrieval plus boosted-tree integration, **pretrained on real tabular datasets**. The
retrieval component is an axis this project already measured and rejected in its
query-conditioned form (§38, Claim 5). The real-data pretraining is the half that is
off-limits here — not for licence reasons but for the contamination-auditability argument in
`docs/roadmap/STRATEGY.md`, which is a positioning asset rather than a constraint.

**Amazon Science (2025).** *MITRA: Mixed Synthetic Priors for Enhancing Tabular Foundation
Models.* [arXiv:2510.21204](https://arxiv.org/abs/2510.21204). **Rank 8 on TabArena, Elo 1729.**
**The most directly relevant paper in this list.** Synthetic-only and state of the art, beating
TabPFNv2 and TabICL on classification and regression with better sample efficiency — the
counterweight to TabDPT/ConTextTab/iLTM, and evidence that the field's real division is prior
quality rather than synthetic-versus-real. Contributes three criteria for prior selection
(**performance, diversity, distinctiveness**) and a mixture of SCM plus **tree-based priors**
(gradient boosting, random forest, decision tree, extra tree), the latter chosen because
SCM-pretrained models generalise poorly to tree-generated structure. Reported **model-agnostic**:
the same mixture improves 1D row attention and 2D element attention alike. See tasks 48.13/48.14.

**Ma, Thomas et al. (2024).** *TabDPT: Scaling Tabular Foundation Models on Real Data.*
[arXiv:2410.18164](https://arxiv.org/abs/2410.18164). Opened in full 2026-09-21, promoted from
the unverified leads below. ICL retrieval plus self-supervised learning, pretrained on **real**
OpenML tables; reports real data giving faster convergence and better generalisation than
"purely open-source synthetic data generators", and **power-law scaling in both model and data**.
Open weights, training code and pretraining dataset list. Two parts matter here beyond the
headline: **Table B.1** enumerates the training corpus, which §109 intersects with TabArena; and
**Appendix A, "Bitter Lessons"**, lists what did not work, including that cell-token
architectures with vertical+horizontal attention "proved more memory intensive" than the simpler
`(B, N, d)` form — this project's architecture and its measured wall. Scored against our own
choices in `docs/paper/RELATED_WORK.md`.

**Defazio, Mehta, Mishchenko, Khaled & Cutkosky (2024).** *The Road Less Scheduled.* NeurIPS
2024. Schedule-free optimisation, cited by TabDPT. Relevant here for an operational reason
rather than an accuracy one: this project's cosine schedule couples the learning rate to a fixed
step budget, which is why §108's matched-task rerun had to start fresh rather than extend. See
task 48.9.

**LAMDA-Tabular (2025).** *TabSwift: An Efficient Tabular Foundation Model with Row-Wise
Attention.* `github.com/LAMDA-Tabular/TabSwift`. Synthetic-only, in-context, one checkpoint for
classification and regression. Independently uses **this project's exact attention mask** —
context rows self-attend, query rows attend to context but not to each other — which is what
makes our query chunking exact. Contributes **register tokens** (learnable dataset-level slots,
discarded before decoding; a published instance of `explicit-task-representation` 41.2) and
**gated attention** (`sigmoid(W·x)` per head, head-wise or element-wise). Row-wise `(B, N, d)`
attention rather than element-level 2D — the third peer to choose the cheap form.

**Prior Labs (2025).** *TabPFN-2.5: Advancing the State of the Art in Tabular Foundation
Models.* [arXiv:2511.08667](https://arxiv.org/abs/2511.08667). Purely synthetic-pretrained
(§110's frontier); a separately released real-data variant (Real-TabPFN-2.5) is fine-tuned on
43 OpenML/Kaggle datasets **deduplicated against the full TabArena suite** — the contamination
discipline §109 argues for. Depth 12→18/24 layers, feature group size 2→3, 64 learned "thinking"
rows acting partly as attention sinks (third independent relative of `explicit-task-
representation`'s task token, after TabSwift's register tokens), and a surrogate-model
hyperparameter search using TabPFNv2 itself to score 10,000 configurations from 100 real runs.
Ships a distillation engine converting a fitted model to a dataset-specific MLP or tree
ensemble for low-latency deployment. See `docs/paper/RELATED_WORK.md`.

**Soda-Inria (2026).** *TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation
Model.* [arXiv:2602.11139](https://arxiv.org/abs/2602.11139). Code BSD-3-Clause (a subdirectory
derived from TabPFN-TS separately Apache-2.0); weights BSD-3-Clause, checked independently via
the HF API. **Untuned TabICLv2 beats hyperparameter-tuned, ensembled, real-data-fine-tuned
RealTabPFN-2.5** on TabArena and TALENT — the sharpest single data point for §110's synthetic-
top-14 finding. New synthetic-prior engine for pretraining diversity, a "scalable softmax" for
generalising to larger datasets without long-sequence pretraining, Muon replacing AdamW, and
million-row datasets under 50 GB GPU memory. Its own related-work section is a current,
citable taxonomy of prior families across TabPFN/TabICL/MITRA/TabPFNv2/LimiX/Drift-Resilient
TabPFN. See `docs/paper/RELATED_WORK.md`.

**LG AI Research (2026).** *EXAONE Tabular*, `github.com/LGAI-Research/EXAONE-Tabular`,
arXiv:2608.25774. Rank 7, Elo 1741 on this project's own TabArena run. **Code and weights
licences diverge, checked separately** — code BSD-3-Clause-LG AI Research License; weights
under "EXAONE AI Model License Agreement 1.2 - NC", whose §3.1 prohibits commercial use of the
model's **output**, not only the weights. The third confirmed instance of the trap §24 names
for Google's TabFM/TimesFM. May be evaluated via TabArena's own published numbers; its weights
may never be loaded here. Cross-axis Summary Transformer (~21M params) with item- and
feature-summary tokens — a fourth published cross-axis summary mechanism, after TabSwift's
register tokens and TabPFN-2.5's thinking rows. See `docs/paper/RELATED_WORK.md`.

**Nums AI (2026).** *Causilo*, `github.com/nums-ai/causilo`. **Rank 6, Elo 1751 on this
project's own TabArena run** — the README's self-reported "Elo position 1, 1792.9" is a
different, higher figure not reconciled here; the discrepancy is recorded, not resolved.
**Fourth confirmed licence-split**: code Apache-2.0 (verified from the repo's `LICENSE` file),
weights under a separate non-commercial "Causilo License v1.0" (verified from the README).
No architectural detail obtained. See `docs/paper/RELATED_WORK.md` for the full note,
including why a surface resemblance to Neuralk's Seldon is noted as unverified speculation
and does not change `CLAUDE.md`'s Neuralk boundary either way.

**What none of these licenses.** Reading a paper licenses an *idea*. Nothing in this section
authorises copying an implementation, a weight file, or a pretraining corpus, and a repository
under a permissive code licence proves nothing about the weights inside it.

### Still unverified — do not cite

- Credit-risk TFM evaluations reporting that TFMs are strongest in small-data PD/LGD
  settings. Referred to in conversation; not opened.

Confirm title, authors, venue, licence and commercial-use terms before citing or ingesting.
**Check the data licence separately from the code licence** — Google's TabFM ships
Apache-2.0 code with non-commercial weights, so a permissive repository proves nothing about
what is inside it.
