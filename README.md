# FinTFM

[![PyPI](https://img.shields.io/pypi/v/fintfm?color=blue)](https://pypi.org/project/fintfm/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22949759.svg)](https://doi.org/10.5281/zenodo.22949759)
[![CI](https://github.com/kabartay/fintfm/actions/workflows/ci.yml/badge.svg)](https://github.com/kabartay/fintfm/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![Checkpoint](https://img.shields.io/badge/%F0%9F%A4%97%20weights-fintfm--binary-yellow)](https://huggingface.co/kabartay/fintfm-binary)
[![TabArena](https://img.shields.io/badge/TabArena-94th%20of%2095-critical)](docs/results/TABARENA.md)

**A tabular foundation model for corporate credit risk, built from scratch and measured in
public.**

Given a table of labelled rows, FinTFM predicts new rows in a **single forward pass**, with your
data supplied as context rather than trained on. There are no gradient steps at fit time and no
per-dataset tuning — the model is pretrained once, on **synthetic data only**, and never sees a
real table during training.

That last property is the point. A model that provably never saw your benchmark cannot have
memorised it, and in a regulated domain the ability to *demonstrate* that is worth more than a
few points of accuracy.

```python
from huggingface_hub import hf_hub_download
from fintfm.inference import FinancialTFMClassifier

ckpt = hf_hub_download("kabartay/fintfm-binary", "v4-cellattn-labels.pt")
clf = FinancialTFMClassifier(ckpt, device="cpu")
clf.fit(X_train, y_train)            # stores the table as context; no training happens
pd_estimates = clf.predict_proba(X_test)[:, 1]
```

The checkpoint is on Hugging Face at
[**kabartay/fintfm-binary**](https://huggingface.co/kabartay/fintfm-binary) — 885K parameters,
binary, up to 136 features, **Apache-2.0**. It is the checkpoint every published binary number
below was measured on, so those results are reproducible against this file rather than a
variant of it.

## Status, stated plainly

**FinTFM is a research codebase with a public claims ledger, not a product.** It exists to find
out whether in-context tabular learning can be made competitive with gradient-boosted trees on
financial tables, and to say honestly where it currently is not.

| | |
| --- | --- |
| **External benchmark** | [TabArena](https://tabarena.ai), 27 binary datasets against 94 other methods: **rank 94 of 95**, integration confirmed correct by a maintainer and the entry declined on competitiveness (§122) |
| **On real credit panels** | calibration consistently among the best measured; discrimination consistently loses to tuned gradient boosting — both, on every panel tried |
| **Licence** | Apache-2.0, **code and weights** — chosen separately, not inherited (see [Licensing](#licensing--provenance)) |
| **Checkpoint** | [kabartay/fintfm-binary](https://huggingface.co/kabartay/fintfm-binary) — 885K, binary, ≤136 features |
| **Tests** | 246, plus `ruff`, the openspec validator, a dependency-licence check and a documentation-link check — all in CI |
| **Measurement log** | 121 numbered findings, each declaring how its numbers were produced |
| **Problem types** | binary declared; multiclass and regression implemented but **not** declared (§121) |

**Read [`docs/paper/CLAIMS.md`](docs/paper/CLAIMS.md) before anything else.** Every claim this
project could make is tagged SURVIVES / SINGLE DRAW / SUPERSEDED / RETRACTED / OPEN against its
evidence. That file, not this one, is the source of truth — and it records what stopped being
true as prominently as what holds.

> Throughout this repository, **§N** refers to entry N in
> [`docs/results/FINDINGS.md`](docs/results/FINDINGS.md), a measurement log in which every number states how it
> was produced (MEASURED, SMOKE-TEST, SIMULATED or ESTIMATED) and negative results are kept
> deliberately, because they are what stops the same wrong conclusion being reached twice.

## Results, as measured

TabArena, 27 binary datasets, one fold each, against 94 other methods:

| change | mean ROC-AUC | Elo | rank |
| --- | --- | --- | --- |
| label encoding (§98) | 0.7642 | 662 | 93 / 95 |
| **+ out-of-fold target statistics** (§101) | **0.7823** | 765 | 93 / 95 |
| + multiclass-capable prior (§105) | 0.7817 | **813** | 93 / 95 |
| + 5.0M parameters, matched task count (§114) | 0.7774 | 751 | 93 / 95 |
| + tree-structured prior (§115), two seeds | 0.7926 / 0.7913 | 842 / 826 | 93 / 95 |

**One change has moved the number on the benchmark that matters, and it was preprocessing
rather than architecture.** The tree prior adds a further +0.0094 here and replicates across two
seeds — but costs **−0.0221 average precision on credit data** (§116), negative on 5 of 5 folds
with three surviving multiple-comparison correction at p < 0.001, so it ships off by default.
Everything else tried — parameter scale at matched task volume, training volume, a widened
structural-causal prior — has been null or negative. **The rank has never moved.**

## What this project does that others do not

Three things, stated at their true weight:

**Incoherent PD term structures are impossible by construction.** The field's standard
per-horizon construction produces non-monotone cumulative default curves for **39% of firms** on
real data — a firm whose 3-year default probability is below its 2-year. A hazard head makes
that unrepresentable, at zero measured accuracy cost.

**A severe capacity defect was found by measurement, not review.** The original architecture had
capped discrimination regardless of true task difficulty, caught against an exactly-known
Bayes-optimal AUC. Its cause was isolated to the architecture rather than the prior after
eliminating seven other candidates one at a time, then closed by a two-way cell-attention change
worth +0.042 average precision on real data.

**The record corrects itself in public.** This README has been wrong about its own results at
least twice (§107, §112). Both are retracted in place, with the reasoning kept. No peer project
publishes a document that tags its own claims RETRACTED as prominently as its wins, and in a
domain where a model must arrive with its own validation evidence, that is the differentiator.

## How it works

1. **Priors** (`src/fintfm/prior/`) — synthetic task generators, sampled and mixed per batch
   (`mixture.py`).
   - `financial.py` — a structural generative story for company balance sheets, P&L and
     default labels: accounting identities, sector and macro effects, a wide derived-ratio
     family, missingness, a sharpness (signal-to-noise) knob, and a default-rate envelope that
     can reach real low-default-portfolio rates (~0.2%).
   - `scm.py` — a generic random-graph structural-causal-model prior (the TabPFN/TabICL-style
     idea): random layered functions with several nonlinearities, for general nonlinear and
     multiclass structure the financial prior does not cover.
   - `tree.py` — ensembles of oblivious decision trees, generating the **axis-aligned,
     piecewise-constant** structure real tabular data is full of and the other priors do not.
     Added on *distinctiveness* grounds after `fintfm-priorscore` measured it as the only
     member of the mixture with a positive tree-versus-linear gap (§112). Off by default
     (`p_tree=0.0`).
   - `trivial.py`, `crossed.py` — diagnostic-only priors used to isolate specific hypotheses
     (whether the architecture can learn at all; whether a prior's features or its label
     mechanism carries a measured effect). Not part of the default training mixture.
2. **Model** (`src/fintfm/modeling/model.py`) — a from-scratch Transformer. Cells are embedded
   individually and carry a random per-task column identity (so the model can tell columns
   apart without a positional embedding, which would break invariance to column order).
   Columns attend to each other within a row; optionally — `ModelConfig.n_cell_blocks`,
   currently experimental — cells also attend across rows *within one feature* before that,
   giving each column a distribution-derived identity rather than only a random tag. Rows are
   then pooled and attend to context rows to perform in-context learning. An optional hazard
   head (`modeling/hazard.py`) produces a cumulative PD term structure that is monotone by
   construction. Independent implementation of ideas described in the public TabPFN / TabICL
   literature — no code or weights from any existing project (see Licensing below).
3. **Training** (`src/fintfm/modeling/train.py`, `fintfm-train`) — infinite synthetic data, one
   gradient step per fresh batch, cosine LR schedule, checkpointing, a held-out quality metric
   that reports per-task AUC (not the pooled-across-tasks number, which is inflated whenever
   task base rates differ) and Brier skill against a base-rate-only predictor.
4. **Inference** (`src/fintfm/inference/`) — an sklearn-compatible `FinancialTFMClassifier`:
   `fit()` stores the table as context, `predict_proba()` runs the frozen network. Several
   context-construction strategies (uniform, balanced, hybrid, retrieval, prototype) and a
   base-rate correction, plus ensembling over column-identity draws to recover the
   distributional column-order invariance a random identity trades for expressiveness.
   `inference/categorical.py` encodes categorical columns as **out-of-fold** smoothed target
   statistics, because the model reads every cell as an ordered scalar and label encoding's
   arbitrary order is measurably worse than no order at all (§100). The out-of-fold step is
   not a refinement: the naive form puts a row's own label into its own encoding, which makes
   the context self-predictive and the feature absent at query time — it harms the model
   rather than flattering the score, so it survives careless validation.
   `inference/regressor.py` adds a `FinancialTFMRegressor` on the same frozen network and no
   new architecture: a continuous target cut into K quantile bins is an integer index over K
   outcomes, so the existing classification head regresses as-is, and the output is natively
   **distributional** — quantiles and prediction intervals come free, and the predicted
   density can be bimodal, which is what loss given default actually is and what a Gaussian
   head cannot represent.
5. **Evaluation** (`src/fintfm/evaluation/`, `src/fintfm/experiments/`) — real corporate-default
   panels (V4FinBench via its published protocol, `fintfm-v4protocol`; Polish and Taiwan
   bankruptcy, `fintfm-bench`), an out-of-time harness, a synthetic capability suite
   (`fintfm-capability`, including a class-count sweep for multiclass, §99) with probes of a
   *known* ceiling — some with a closed-form
   Bayes-optimal AUC — specifically built to catch a model that looks fine on average while
   capped in a way an aggregate score cannot see. Average precision is reported alongside
   ROC-AUC everywhere, and read first at low base rates (ROC-AUC's chance floor is 0.5
   regardless of prevalence; AP's floor is the prevalence itself, so it stays legible at the
   base rates this project's target segment actually has).
   `fintfm-priorscore` scores a *prior* rather than a model, on the three criteria the
   literature converged on — performance, diversity, distinctiveness — using **fitted**
   baselines only, so it cannot confuse "the prior lacks this structure" with "our model cannot
   learn it". Its first version could, and §112 is the record of what that cost.

## Quickstart

```bash
pip install fintfm
```

The published checkpoint is Apache-2.0 and ungated, so nothing else is needed to predict:

```python
from huggingface_hub import hf_hub_download

ckpt = hf_hub_download(
    "kabartay/fintfm-binary",
    "v4-cellattn-labels.pt",
    revision="f116bfd43a2b15c65ed3551ea8c38e3364629ddc",  # pin it; weights behind a number should not move
)
```

To work on the model rather than use it, clone the repository instead:

```bash
uv sync --extra bench --extra hf   # naming one extra uninstalls the others
uv run pytest                      # 246 tests; 1 skip is expected, more means look
```

### Using a checkpoint

The estimators are scikit-learn compatible. `fit()` **stores** the table as context — it takes
no gradient steps — and `predict_proba()` runs the frozen network once.

```python
from fintfm.inference import FinancialTFMClassifier, FinancialTFMRegressor

clf = FinancialTFMClassifier(ckpt, device="mps")   # ckpt from hf_hub_download, above
clf.fit(X_train, y_train)          # stores context; no training happens
proba = clf.predict_proba(X_test)[:, 1]

reg = FinancialTFMRegressor("runs/v4-regression.pt", n_bins=10)
reg.fit(X_train, y_cont)
point = reg.predict(X_test)                     # distribution mean
lo, hi = reg.predict_interval(X_test, 0.8)      # 80% interval, free from the same head
```

Categorical columns need encoding before they reach the model — it reads every cell as an
ordered scalar, and label encoding is measurably worse than no order at all (§100). Use
`fintfm.inference.categorical.CategoricalTargetEncoder`, which is out-of-fold on the context
rows for reasons that are **not** optional; see [How it works](#how-it-works).

### Training a checkpoint

```bash
uv run fintfm-train --steps 300 --d-model 32 --d-cell 16 --n-layers 2 --max-features 16 \
    --max-classes 2 --device cpu --out runs/v0-smoke.pt   # pipeline check, a couple of minutes
uv run fintfm-bench --model runs/v0-smoke.pt --credit      # real corporate-default panels
```

That smoke config exists to check the pipeline runs, not to produce a usable checkpoint — see
[`docs/infra/COMPUTE.md`](docs/infra/COMPUTE.md) for measured step costs at real scale. A real pretraining
run (thousands of steps, `--d-model 128`+, `--max-features` matching your data) takes
hours-to-a-day and needs a GPU: `--device mps` on Apple Silicon, `--device cuda`, or see
[`docs/infra/HF_JOBS.md`](docs/infra/HF_JOBS.md) for the Hugging Face Jobs recipe this project actually
uses for training runs. **Check `uptime` before running anything heavy locally** — see
[`CLAUDE.md`](CLAUDE.md) for why.

Both `fintfm-bench --credit` (Polish/Taiwan need 64/95 features) and `fintfm-v4protocol`
(V4FinBench needs 136) will print `SKIPPED for fintfm: model takes 16 features, data has N` —
correct, expected behaviour for the smoke checkpoint above, not a bug. Baselines still run and
score normally; only the FinTFM arm needs a checkpoint pretrained with a matching
`--max-features` to be evaluated.

## Configuration

The numbers experiments use — split years, context sizes, strategies, seeds, scoring
thresholds, the prior's default-rate and sharpness envelope — live in
[`src/fintfm/configs/default.yaml`](src/fintfm/configs/default.yaml), not scattered through
the code. Every entry point takes `--config` with a file that is **deep-merged** over that
default, so an override carries only what it changes:

```bash
uv run fintfm-ctxsweep --model runs/m.pt --config configs/context-sweep-3seed.yaml
uv run fintfm-v4oot    --model runs/m.pt --config configs/retrieval-best.yaml
```

Resolution order is packaged default → `--config` (or `$FINTFM_CONFIG`) → explicit CLI flag.
Runs print the layers they used and record them as `config_sources` in their output JSON.
Unknown keys are refused by name rather than ignored, and an overlapping train/test split is
rejected before the run starts. [`configs/README.md`](configs/README.md) explains what
belongs in configuration and what deliberately stays in code; several files under
[`configs/`](configs/) are **deliberately diagnostic, not production** (their own headers say
so), used to isolate a single variable while chasing a specific finding.

## Licensing / provenance

Everything here — the prior, the architecture, the training loop — is original code written
for this project. **No weights, datasets, or source from Neuralk (Seldon), Fundamental
(NEXUS), Google TabFM, TabPFN, TabICL, TabDPT, or any other tabular-foundation-model product
are used or may be added.** Those are cited in project discussion purely as public
research/product context. Every third-party dataset or dependency's licence is checked before
use, and a *weight* licence is checked separately from its *code* licence, every time — never
assumed from a prior check. Direct dependencies are all permissive — numpy, pandas and
scikit-learn BSD-3, scipy BSD-3, torch BSD-3, PyYAML MIT, and the optional benchmark extras
lightgbm MIT, xgboost Apache-2.0, catboost Apache-2.0, pyarrow Apache-2.0 — and
`scripts/check_licences.py` enforces that in CI rather than leaving it to this paragraph.

**One honest exception, which CI found and this paragraph previously did not mention.** On
**Linux**, `torch` pulls in around fifteen NVIDIA CUDA runtime packages that are **NVIDIA
Proprietary**, not permissive. They are a transitive runtime dependency rather than a choice
made here, this project does **not redistribute them** — `uv build` produces a pure-Python
wheel and pip fetches them from PyPI under NVIDIA's own terms — and they are absent on macOS,
which is why a check run only on a developer's Mac reported everything permissive while CI did
not. **Anyone shipping this in a product must read NVIDIA's EULA themselves**; it is not
something this repository can assert on their behalf.

## What would change the picture

Stated so the project is falsifiable rather than open-ended. The ~0.035 deficit to the field
has no measured axis of variation, and these are the remaining candidates:

| still open | why it is a candidate | status |
| --- | --- | --- |
| **factorized attention** | the encoder is memory-bound at every turn, and three peers independently chose the cheaper form | proposed (44.x), prior art recorded |
| **a joint objective** | `p(x, y \| D)` rather than `p(y \| x, D)` makes every column a training signal, not just the target | proposed (48.15), scoped as a measurement before a rewrite |
| **prior design** | the only lever a peer's own ablations identify as decisive | instrument built and partly falsified — see below |

### Closed by measurement

Not abandoned — **tested and ruled out**, which is the more useful half of the record and the
reason the list above is short:

| candidate | what the measurement said |
| --- | --- |
| parameter scale | **−0.0049** at 5.7× the parameters on *matched* task volume (§114). A peer's published curve returns +0.0049 R² for 16.7×, against a 0.035 deficit. |
| training volume | null at 5× (§93); **+0.0028** at 2× (§114). |
| depth over width | **could not be scored at all** — the harness's per-dataset time limit, after 8 of 27 datasets (§114). Depth costs inference time and this model is already slow. |
| `column_id_dim` | peaks at the value chosen by accident; all four non-peak arms below it on all five folds (§104). |
| a widened structural-causal prior | **p = 1.000** against its own matched control. |
| a tree-structured prior | +0.0094 on general tabular data, replicated across two seeds (§115) — and **−0.0221 average precision on credit** (§116). Ships off by default. |

**The prior-scoring instrument is half-confirmed and half-falsified.** It predicted the
general-tabular gain above before it was measured (§112 → §115). A second arm, added to ask the
credit question, then predicted the wrong direction on its first real test — raising the
structural-causal prior's share cost **−0.0339 AP**, 0 of 5 folds, all five significant (§118).
So it predicts **breadth, not fit**, and no claim rests on its credit arm.

## Reproducing the measurements

Every number in [`docs/results/FINDINGS.md`](docs/results/FINDINGS.md) names the command that produced it.
The entry points:

| command | what it measures |
| --- | --- |
| `fintfm-v4protocol` | V4FinBench under its **published** five-fold protocol — the benchmark this project's claims rest on |
| `fintfm-bench --credit` | Polish and Taiwan bankruptcy panels |
| `fintfm-v4oot` | out-of-time split, which the published protocol is not |
| `fintfm-capability` | synthetic probes with a **known** ceiling, some with a closed-form Bayes-optimal AUC |
| `fintfm-priorscore` | scores a *prior*, not a model, on performance / diversity / distinctiveness |
| `fintfm-ctxsweep` | context construction, which explains more variance than model family (§5) |

Two conventions worth knowing before reading any of it. **Every number declares how it was
produced** — SMOKE-TEST, MEASURED, SIMULATED or ESTIMATED — because in an ML repository a wrong
number does not crash, it looks like a result. And **negative and superseded results are kept**,
because they are what stops the same wrong conclusion being reached twice.

## Where this is going

[`docs/`](docs/) holds the project's reasoning, indexed in [`docs/README.md`](docs/README.md).

- [`docs/paper/CLAIMS.md`](docs/paper/CLAIMS.md) — **start here.** Every claim this project
  could make, tagged by status, newest evidence wins.
- [`docs/results/FINDINGS.md`](docs/results/FINDINGS.md) — the full measurement log, numbered sequentially
  (121 entries and counting), each declaring how its numbers were produced.
- [`docs/design/DECISIONS.md`](docs/design/DECISIONS.md) — why the project is built the way it is, and what
  would reverse each choice.
- [`docs/roadmap/STRATEGY.md`](docs/roadmap/STRATEGY.md) — the plan of record.
- [`docs/design/ARCHITECTURE.md`](docs/design/ARCHITECTURE.md) — how the model works, in more depth than
  this file.
- [`docs/results/TABARENA.md`](docs/results/TABARENA.md) — the external-evaluation recipe, the coverage
  fraction any score must carry, and four silent failure modes including a results cache that
  returns stale numbers after a preprocessing change.
- [`docs/paper/RELATED_WORK.md`](docs/paper/RELATED_WORK.md) — the competing models, read from
  primary sources, with what each costs this project's positioning. Its digest is the shortest
  useful summary of why the remaining lever is the prior.
- [`docs/results/POSTMORTEM.md`](docs/results/POSTMORTEM.md) — wrong diagnoses, each caught by measurement
  rather than review, kept on the record deliberately.
- [`openspec/changes/`](openspec/changes/) — the live roadmap as structured proposals with
  verifiable tasks, not prose. `uv run python openspec/tools/validate.py` checks every
  proposal is well-formed and every finding declares its provenance.

Short version: the mechanism claims ("no training on your data", "no feature engineering") are
already owned by better-funded competitors, so the thesis is not the mechanism. It is a credit
model that arrives with its own validation evidence — calibrated, auditably free of benchmark
contamination, and eventually backed by a pre-registered forward track record that cannot be
bought — plus a public, self-correcting record of what has and has not been shown to be true.

## Project structure

```
src/fintfm/
  prior/         the synthetic data-generating processes: a structural model of company
                 financials, a random-graph SCM, and the mixture that samples them
  modeling/      the network and the pretraining loop
  inference/     scikit-learn compatible estimators, plus the categorical target encoder
  evaluation/    benchmark harnesses and baselines
  experiments/   one module per measured question; each writes a numbered finding
  configs/       packaged defaults, read at runtime by path

docs/
  results/       FINDINGS.md, the numbered measurement log this project is organised around,
                 plus TABARENA.md and POSTMORTEM.md
  paper/         CLAIMS.md (what holds, what was retracted), RELATED_WORK.md, LIMITATIONS.md
  design/        ARCHITECTURE.md, DECISIONS.md, GLOSSARY.md
  roadmap/       ROADMAP.md (what to do next), STRATEGY.md, NEXT.md, TABARENA_BAR.md
  competition/   LANDSCAPE.md, SUBMISSION.md
  infra/         COMPUTE.md, HF_JOBS.md
  research/      REFERENCES.md, RESEARCH_NOTES.md

openspec/changes/   one proposal per change, with its task list; the queue lives here
scripts/            the repository's own gates: licence, documentation links, Zenodo metadata
```

**Read `docs/results/FINDINGS.md` first.** Every number quoted anywhere in this repository
points at a numbered entry there stating how it was produced, and entries that were later
retracted stay in place rather than being edited away.

## Acknowledgements

**No code, weights, or training data from any tabular foundation model was used here**; see
[Licensing / provenance](#licensing--provenance). What follows is credit for what *was* used.

- **[V4FinBench](https://github.com/genwro-ai/V4FinBench)** (Kostrzewa et al.,
  [arXiv:2605.10896](https://arxiv.org/abs/2605.10896), CC BY 4.0) for the corporate-default
  panel and its evaluation protocol, which this project reimplements from the published
  specification rather than vendoring.
- **[TabArena](https://tabarena.ai)** for the external protocol, and its maintainers for
  reviewing the submission and confirming the integration and numbers are correct while
  declining the entry on competitiveness (§122). That review is the only independent check this
  project has had.
- **The published work of TabPFN, TabICL, TabDPT, LimiX, Nori, MITRA, TabSTAR, TabSwift,
  OrionMSP, iLTM, ConTextTab and EXAONE-Tabular** — for **ideas**, which this project took
  freely and then measured:

  | idea | from | what happened |
  | --- | --- | --- |
  | three criteria for scoring a prior: performance, diversity, distinctiveness | MITRA | built into `fintfm-priorscore`; §112 then measured this prior's distinctiveness as weak |
  | a tree-structured prior | TabICLv2, reimplemented from its appendix | **+0.0094** on general tabular data, **−0.0221 AP** on credit; ships off (§118) |
  | published scaling curve | Nori | capped a scale hypothesis GPU budget was being spent on, before more was spent (§114) |
  | rejecting unlearnable synthetic tasks | Nori | measured: would reject 35% of this prior, and discards signal the model uses (§125) |
  | narrow-and-deep shape, 16 layers at width 128 | Nori | measured: null here, +0.0010 AP (§124, §126) |

  Four of those five ideas were measured and **not** adopted. That is the point of taking them:
  a published result from a different prior and architecture is a hypothesis about this one, not
  a conclusion about it.

Reading a paper is not ingesting a codebase, and the distinction is enforced in `CLAUDE.md`.

## Citing

A `CITATION.cff` is in the repository root, so GitHub's **"Cite this repository"** button
renders BibTeX and APA directly. The same entry, to copy:

```bibtex
@software{Organokov_FinTFM_2026,
  author    = {Organokov, Mukharbek},
  title     = {{FinTFM: a tabular foundation model for corporate credit risk,
               pretrained only on synthetic data}},
  year      = {2026},
  version   = {0.5.5},
  doi       = {10.5281/zenodo.22949759},
  url       = {https://github.com/kabartay/fintfm},
  publisher = {Zenodo},
}
```

There is **no paper**; the numbered measurement log is this project's public record, so the
software entry above is the citable artifact rather than a stand-in for one.

Cite the **repository** for the method, the measurement log or any finding; cite the
**checkpoint** ([kabartay/fintfm-binary](https://huggingface.co/kabartay/fintfm-binary)) when
the specific weights matter to what you are reporting. They are different artifacts and a
reader can only check the one you name.

Every release is archived on Zenodo with its own DOI. The **concept DOI**
[10.5281/zenodo.22949759](https://doi.org/10.5281/zenodo.22949759) always resolves to the
newest version; each release also has one of its own. That distinction is load-bearing here,
because this repository retracts and supersedes results: §107 and §112 both withdrew earlier
claims. Cite a **version** DOI to pin what you actually read, and the **concept** DOI to point
at current state.

## Contributing, and what cannot come in

Issues and discussion are welcome. Two hard boundaries, both of which protect the only claim
this project has that competitors do not:

**No code, weights, or training data from any tabular-foundation-model product** — TabPFN,
TabICL, TabDPT, LimiX, Nori, MITRA, or any other — may enter this repository. Their published
papers are legitimate to read and cite, and [`docs/paper/RELATED_WORK.md`](docs/paper/RELATED_WORK.md)
does exactly that for ten of them. Ingesting any of it would destroy the provenance argument
above, and nothing would warn us.

**Check a weights licence separately from its code licence, every time.** Four of the ten peer
projects read for this release ship permissive code with **non-commercial weights**, a split
invisible from a repository's headline licence badge. One of them restricts commercial use of
the model's *output*, not merely the weights.
