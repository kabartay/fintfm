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
`docs/FINDINGS.md` §1, including the caveat that limits how strongly this project may claim
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
administrative cost and long waiting periods between competitions. See `docs/FINDINGS.md` §3
for why this design matters more to this project than to a forecasting vendor.

## Domain benchmarks and datasets

**Tanna, Solanki, Bouadi, Bouarour, Seth & Sankarapu (2026).** *Data Presentation Over
Architecture: Resampling Strategies for Credit Risk Prediction with Tabular Foundation
Models.* [arXiv:2605.18635](https://arxiv.org/abs/2605.18635)
Seven context-construction strategies across four classical models and five TFMs on Home
Credit and Lending Club. Balanced and hybrid sampling add 3-4 AUC points over uniform, a gap
wider than the spread between TFM families. Acted on in `docs/FINDINGS.md` §5, and its
AUC-only framing corrected by §6.

**Tomczak et al. (2026).** *V4FinBench: Benchmarking Tabular Foundation Models, LLMs, and
Standard Methods on Corporate Bankruptcy Prediction.*
[arXiv:2605.10896](https://arxiv.org/abs/2605.10896)
1,106,879 company-year observations, Visegrád Group economies, 2006-2021, 131 features, six
horizons, 0.19-0.36% positive rate. Code [MIT](https://github.com/genwro-ai/V4FinBench);
**data CC BY 4.0** on Kaggle per the repository's separate `DATA_LICENSE.md`. The dataset
this project needs — see `docs/FINDINGS.md` §8.

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
`docs/FINDINGS.md` §9 before quoting it: margins are small, only 22 of 406 pairwise PD
comparisons were significant, and the low-default-portfolio benefit is conjectured rather
than tested.

**Purucker, Tschalzev, Erickson et al. (2026).** *Beyond IID: How General Are Tabular
Foundation Models, Really?* [arXiv:2606.30410](https://arxiv.org/abs/2606.30410)
Introduces BeyondArena over IID, temporal and grouped tasks. TFMs excel on tiny-to-medium
IID data; trees and deep learning still dominate on non-IID, large and high-dimensional
data. Defines which half of credit risk is winnable (`docs/FINDINGS.md` §9).

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

### Still unverified — do not cite

- Credit-risk TFM evaluations reporting that TFMs are strongest in small-data PD/LGD
  settings. Referred to in conversation; not opened.

Confirm title, authors, venue, licence and commercial-use terms before citing or ingesting.
**Check the data licence separately from the code licence** — Google's TabFM ships
Apache-2.0 code with non-commercial weights, so a permissive repository proves nothing about
what is inside it.
