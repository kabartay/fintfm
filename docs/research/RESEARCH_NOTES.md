# Research notes — context, syllabus and open threads

Ideas, references and framings gathered during the founding sessions, kept so they are not
lost when a conversation compacts. **Nothing here is a measurement.** Findings live in
`docs/results/FINDINGS.md`; verified citations in `docs/research/REFERENCES.md`; this file is the map.

## The four fields, and how they differ

The most useful mental model, and the reason "financial foundation model" is ambiguous:

| field | learns | object |
| --- | --- | --- |
| **Tabular FM** | how to infer relations between columns and rows | a label for a row |
| **Time-series FM** | reusable temporal dynamics | future values of a sequence |
| **Financial FM** | one or both, specialised, under severe non-stationarity | varies |
| **panel hazard** (ours) | event probability over time per entity | a hazard path |

"Financial FM" splits further into a **language** branch (BloombergGPT, FinGPT, PIXIU,
FinBen) and a **numerical/time-series** branch (FinCast), plus multimodal ambitions combining
them. This project is in none of those: it is tabular, and specifically the panel-hazard
object no camp predicts (`docs/competition/LANDSCAPE.md`).

## The concept that ties it together

Not transformers — **learning a prior over data-generating processes**, then doing fast
amortized inference on a new task:

```
tabular      p(y | x, D_context)
time series  p(x_{t+1:t+H} | x_{1:t})
finance      p(R_future | market_past, information_past)
```

Pretraining approximates `p(possible worlds)`. Inference asks which worlds are compatible
with the evidence in front of it. That is why Bayesian reasoning, meta-learning and
in-context learning are the right lenses, and it is the framing behind `docs/results/FINDINGS.md`
§13's explanation of where our calibration comes from.

## Syllabus — concepts, audited against what this repo implements

| concept | status here |
| --- | --- |
| transformer attention | implemented |
| row/column attention | implemented (`modeling/model.py`, three stages) |
| permutation invariance/equivariance | asserted in tests (spec M1-M2): exact at `column_id_dim=0`, distributional with column identities on (D12) |
| in-context learning | implemented |
| meta-learning | implemented (PFN training is meta-learning) |
| Bayesian / amortized inference | the mechanism in §13 |
| synthetic priors and SCMs | implemented (`prior/financial.py`, `prior/scm.py`) |
| calibration and uncertainty | metrics + base-rate correction; regression intervals via `predict_interval`; **PD intervals missing** |
| leakage-safe evaluation | provenance invariant (FinTFM family, D17); V4FinBench out-of-time split and published protocol in use (§26, §39) |
| concept drift / distribution shift | measured on V4FinBench's out-of-time split (§26, §31) |
| quantile regression / CRPS | binned and pinball quantile heads implemented (§106, §148); CRPS not reported; PD coverage still needed by `conformal-pd-certificate` |
| retrieval-augmented models | implemented, opt-in (`inference/retrieval.py`, §32); beaten by `prototype` (§38); (2026-11-13: retrieval context collapses V4FinBench to 0.0126 mean AP, §194; k-means prototype context is null at +0.0002, §211) |
| domain adaptation / LoRA | full fine-tuning opt-in (`finetune_steps`, §163); LoRA not done (`changes/lender-adaptation`, tasks 16.1-16.3 open on 2026-11-13); (2026-11-13: fine-tuning per partition leaf is null, +0.0049 mean AP, §208) |
| masked/self-supervised pretraining | not done (TabDPT's approach: real tables + retrieval) |
| scaling laws across datasets/tasks | `changes/scaling-curve`, deliberately demoted (tasks 9.1-9.3 open on 2026-11-13); (2026-11-13: ten times the tasks is null on V4FinBench at two seeds, §188; depth and width at 480,000 tasks null, §195, §202) |
| patching, tokenization/quantization | time-series concepts, not applicable to tabular |
| decoder-only vs encoder-decoder | not applicable; rows are not a sequence |
| covariates / multivariate forecasting | not applicable |

## Baselines a tabular FM must be compared against

A common failure is demonstrating `FM > poorly-tuned NN` when the question is
`FM > tuned CatBoost/LightGBM/XGBoost`. The families:

| family | methods | our status |
| --- | --- | --- |
| gradient boosting | CatBoost, LightGBM, XGBoost | running, out of process to avoid the OpenMP conflict (§25, `docs/infra/COMPUTE.md`) |
| classical | logistic regression, RF, ExtraTrees | logreg, RF, sklearn GBM running |
| tabular neural | FT-Transformer, ResNet, RealMLP, TabM | none |
| tabular FM | TabPFN, TabFM, TabICL, TabDPT | none (licence care needed: §24) |
| AutoML | AutoGluon, H2O, auto-sklearn | none |

(2026-11-13: the table above is the founding-session audit. TabArena-Lite Elo is scored against the full TabArena field, which includes tuned neural, boosted and foundation-model entries (§212, §224), but those entries are not run by this project. FinTFM with a trained linear readout is 903 Elo, 106th of 110, on all 51 TabArena-Lite datasets (§224).)

## Reading order

**Tabular:** PFNs conceptually (Müller et al.) → TabPFN → TabICL → TabDPT → Google TabFM.
**Time series:** PatchTST as background → TimesFM → Chronos → Lag-Llama → Moirai / GIFT-Eval.
**Finance:** BloombergGPT → FinGPT / PIXIU / FinBen for the language branch, then FinCast for
the time-series branch, kept separate.

## Finance-specific traps, none of which are optional

- **Leakage is the dominant failure mode.** Every feature must satisfy `x_t ∈ F_t`. Revised
  macro statistics, current index constituents applied to the past (survivorship), and
  analyst estimates all violate it silently.
- **Never split financial panels at random.** Walk-forward or at minimum a date cut. This is
  what `changes/time-based-evaluation` exists for, and §7 is why it was blocked so long
  (2026-11-13: the splitter is implemented, tasks 3.1 and 3.2 done 2026-11-08; 3.3 still open).
- **Returns, not prices**, and expect predictability to be far lower than for ordinary series.
  Spectacular financial forecasting results should trigger a leakage hunt, not celebration.
- **Volatility is more predictable than returns.** GARCH-style conditional variance is a more
  honest target than price level.
- **Economic metrics, not only ML metrics.** Sharpe, max drawdown, information coefficient —
  a lower RMSE is not automatically worth more money. Our analogue is that a lender provisions
  against the *level* of PD, which is why §17's skill-score work matters.

## Open threads not yet proposals

- **Asymmetric cost.** A missed default costs far more than a false alarm, and every metric
  here is symmetric. Cost-sensitive calibration is unexplored.
- **Sector/country mixture-of-experts.** FinCast uses token-level sparse MoE for domain
  specialisation. Premature at 885,650 parameters (§207); revisit if scale grows.
- **Point-Quantile loss.** FinCast fits point and quantile estimates jointly, claimed to
  prevent collapse under non-stationarity. Relevant to the certificate.
- **LGD and EAD.** Expected credit loss needs all three of PD, LGD, EAD; we model one.
