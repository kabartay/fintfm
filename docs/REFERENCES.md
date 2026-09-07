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

## Domain benchmarks (NOT yet verified — check before citing)

Named in conversation but **not opened, so not usable as citations yet**:

- **V4FinBench** — a corporate-default panel described as ~1M company-year observations, 131
  features, multiple horizons, severe class imbalance, with TabPFN baselines.
- Credit-risk TFM evaluations reporting that context construction materially affects
  performance, and that TFMs are strongest in small-data PD/LGD settings.

Open these and confirm title, authors, venue, licence and commercial-use terms before either
citing them or ingesting the data. Licence matters as much as content — see `CLAUDE.md`.
