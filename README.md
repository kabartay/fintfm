<p align="center">
  <img src="https://raw.githubusercontent.com/kabartay/fintfm/main/images/logo.png" width="112" alt="FinTFM logo: labelled table rows read in one forward pass into a new row's predicted cell">
</p>

<h1 align="center">FinTFM</h1>

<p align="center">
  <strong>A tabular foundation model for corporate credit risk, built from scratch and measured in public.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/fintfm/"><img src="https://img.shields.io/pypi/v/fintfm?color=blue" alt="PyPI"></a>
  <a href="https://doi.org/10.5281/zenodo.22949759"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.22949759.svg" alt="DOI"></a>
  <a href="https://github.com/kabartay/fintfm/actions/workflows/ci.yml"><img src="https://github.com/kabartay/fintfm/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green.svg" alt="License"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg" alt="Python"></a>
  <a href="https://huggingface.co/kabartay/fintfm-binary"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20weights-fintfm--binary-yellow" alt="Checkpoint"></a>
  <a href="#status-stated-plainly"><img src="https://img.shields.io/badge/TabArena--Lite-107th%20of%20110-critical" alt="TabArena-Lite"></a>
  <a href="https://kabartay.github.io/fintfm/leaderboard/"><img src="https://img.shields.io/badge/leaderboard-live-8957e5" alt="Leaderboard"></a>
</p>


FinTFM predicts new rows of a labelled table in a **single forward pass**: your data is the
context, not training data. By default `fit()` takes no gradient steps and does no per-dataset
tuning; fine-tuning is opt-in (`finetune_steps`). The model is pretrained once, on **synthetic
data only**, and never sees a real table.

That is the point. A model that never saw your benchmark cannot have memorised it, and in a
regulated domain being able to *show* that is worth more than a few points of accuracy.

```python
from huggingface_hub import hf_hub_download
from fintfm.inference import FinancialTFMClassifier

ckpt = hf_hub_download(
    "kabartay/fintfm-binary",
    "v4-cellattn-labels.pt",
    revision="f116bfd43a2b15c65ed3551ea8c38e3364629ddc",  # the weights every number below used
)
clf = FinancialTFMClassifier(ckpt, device="cpu")
clf.fit(X_train, y_train)            # stores the table as context; no training happens
pd_estimates = clf.predict_proba(X_test)[:, 1]
```

The checkpoint, [**kabartay/fintfm-binary**](https://huggingface.co/kabartay/fintfm-binary) on
Hugging Face, has 885K parameters, takes binary targets with up to 136 features, and is
**Apache-2.0**. Every published binary number below was measured on this exact file.

**Contents:** [Status](#status-stated-plainly) ·
[Results](#results-as-measured) ·
[What this project does](#what-this-project-does) ·
[How it works](#how-it-works) ·
[Quickstart](#quickstart) ·
[Configuration](#configuration) ·
[Two model families](#two-model-families) ·
[Licensing](#licensing--provenance) ·
[What would change the picture](#what-would-change-the-picture) ·
[Reproducing](#reproducing-the-measurements) ·
[Citing](#citing) ·
[Contributing](#contributing-and-what-cannot-come-in)

## Status, stated plainly

**FinTFM is a research codebase with a public claims ledger, not a product.** It asks whether
in-context tabular learning can compete with gradient-boosted trees on financial tables, and
says plainly where it does not yet.

| | |
| --- | --- |
| **TabArena-Lite (own run)** | **905 Elo, 107th of 110, all 51 datasets, none excluded** (§227). One fold, run in this project's TabArena fork, not on [tabarena.ai](https://tabarena.ai). The entry is "FinTFM with a trained linear readout": a linear model on the network's representation, which for multiclass and regression also takes the row's own features. By type: binary 974 on 30 datasets (§220), multiclass 874 on 8, regression 582 on 13 (§227). On the same board RandomForest (default), the resubmission bar, scores 1,000 and Linear (tuned) 919. General-tabular results, not credit results |
| **TabArena submission** | The zero-shot model on 27 binary datasets ranked **94th of 95**; a maintainer confirmed the integration and declined the entry on competitiveness (§122). On today's field it scores 750 Elo, 107th of 108 (§225). No resubmission until the readout entry reaches 1,000 overall Elo (design review DR7.6) |
| **Real credit panels** | Gradient boosting wins. On V4FinBench's full panel FinTFM scores 0.20 mean AP against 0.40 for untuned CatBoost (§163); LightGBM at 1,000 trees reaches 0.3908 (§167), so the gap is boosting rounds, not one library. On four more credit datasets (SBA 7(a), ULB, FDIC, Estonia) FinTFM is below untuned logistic regression on mean AP (§216). Calibration was best or near-best on four panels for the pre-cell-attention architecture, one split each (Claim 9, a single draw, §73), and is not claimed for the published checkpoint: it over-predicts the default rate 1.2 to 1.7 times on ULB (§174), and 1.9 and 3.2 times on two FDIC splits while under-predicting the third (0.396% against 0.735%; run record behind §216, `docs/results/drafts/dr4.3-runs_2026-11-07.md`) |
| **Levers on V4FinBench** | Opt-in fine-tuning (`finetune_steps`, 0.7.0): mean AP 0.1986 to 0.2062, better on 5 of 5 folds, 3.8% of the gap to CatBoost (§163). Zero-shot partitioned contexts (`fintfm-v4protocol --large-n partition`): 0.1986 to 0.2362, +0.0375, 5 of 5 folds (§169), and ahead of fine-tuning on 5 of 5 (§205) |
| **Licence** | Apache-2.0 for **code and weights**, each chosen separately (see [Licensing](#licensing--provenance)) |
| **Checkpoint** | [kabartay/fintfm-binary](https://huggingface.co/kabartay/fintfm-binary): 885K parameters, binary, up to 136 features |
| **Tests** | about 1,350 (2026-11-13), plus `ruff`, the openspec validator, a dependency-licence check and a documentation-link check, all in CI |
| **Measurement log** | 228 numbered entries, each stating how its numbers were produced; pre-registrations are written before their arms run |
| **Problem types** | Binary is declared and is the only published checkpoint. Multiclass and regression run on TabArena-Lite with unpublished checkpoints (§221-§224) and are **not** declared (§121) |

**Read [`docs/paper/CLAIMS.md`](docs/paper/CLAIMS.md) before anything else.** It tags every
claim the project could make as SURVIVES, SINGLE DRAW, SUPERSEDED, RETRACTED or OPEN against
its evidence. It, not this README, is the source of truth, and it records what stopped being
true as prominently as what holds.

> **§N** means entry N in [`docs/results/FINDINGS.md`](docs/results/FINDINGS.md), the
> measurement log. Every number there is labelled MEASURED, SMOKE-TEST, SIMULATED or
> ESTIMATED, and negative results stay in, so the same wrong conclusion is not reached twice.

## Results, as measured

TabArena-Lite, one fold per dataset, scored against TabArena's cached results for every other
method. Positions come from each run's stored leaderboard (§225). TabArena's printed table
numbers its rows from 0, so ranks quoted from it before 2026-11-13 were one place too high.

**Standing on 2026-11-13:**

| entry | datasets | Elo | position | source |
| --- | --- | --- | --- | --- |
| **FinTFM with a trained linear readout** | all 51, none excluded | **905** (+104 / −110) | **107th of 110** | §227 |
| the published model, zero-shot, own head | 27 binary | 750 (+134 / −186) | 107th of 108 | §225 |
| RandomForest (default), the resubmission bar | all 51 | 1,000 | 100th of 110 | §224, §225 |

The zero-shot row re-scores §204's stored predictions against today's field, with no refit.
**No change so far has moved FinTFM's position on the field it was scored against:** Elo rose
from 880 to 905 over §221 to §227, and the position stayed 107th of 110.

### TabArena-Lite, with a trained linear readout

From §204 on, the work moved to the readout and to the adapter in this project's TabArena
fork. Every row below except the first is **"FinTFM with a trained linear readout"**: a
logistic regression (ridge for regression) fitted on the network's pre-head representation of
each training split, so not the zero-shot model. Each Elo comes from that run's own
leaderboard and carries an interval of roughly ±100 to ±180, so a step of a few Elo is not
significant on its own; the cited entries give the paired per-dataset tests.

| change | datasets | Elo | position | source |
| --- | --- | --- | --- | --- |
| zero-shot default (missing values passed through) | 27 binary | 756 | 95th of 96 | §210, §225 |
| linear readout + `feature_transform="squash"` | 27 binary | 969 | 104th of 108 | §212, §225 |
| + partitioned contexts, one readout per leaf | 27 binary | 1,000.2, level with RandomForest (default) at 1,000.0: 13 datasets better, 14 worse | 101st of 108 | §215, §225 |
| + the three wide binary tables, via `max_features_strategy: select` | 30 binary | 974 | 104th of 108 | §220, §225 |
| all 51 datasets, none excluded | 51 | 880 | 107th of 110 | §221 |
| multiclass readout scaled, C cross-validated, row features appended | 51 | 888 | 107th of 110 | §222, §225 |
| regression readout with row features appended | 51 | 898 | 107th of 110 | §223, §225 |
| four partition bags for multiclass | 51 | 903 | 107th of 110 | §224, §225 |
| eight bags for multiclass, and a pure-regression checkpoint (`--p-regression 1.0`) | 51 | **905** | **107th of 110** | §227 |

**These are breadth results, not credit results.** On V4FinBench a readout per partition leaf
scored −0.0570 mean AP, better on 0 of 5 folds (§218). §219 traced that to a sampling defect
in the harness, fixed in 0.10.0; the rerun is pending, so no credit claim rests on this table.

### The first series: the zero-shot model on 27 binary datasets

September 2026, one fold each, against the 95-method field of the time. Every arm placed 94th
of 95, whatever its Elo (§225; first recorded as 93rd from the printed index):

| change | mean ROC-AUC | Elo | position | source |
| --- | --- | --- | --- | --- |
| label encoding, 26 of 27 datasets | 0.7642 | 662 | not stored | §98 |
| **+ out-of-fold target statistics** | **0.7823** | 765 | 94th of 95 | §101, §225 |
| + multiclass-capable prior | 0.7817 | **813** | 94th of 95 | §105, §225 |
| + 5.0M parameters, matched task count | 0.7774 | 751 | 94th of 95 | §114, §225 |
| + tree-structured prior, two seeds | 0.7926 / 0.7913 | 842 / 826 | 94th of 95 | §115, §225 |

The §101 model is the entry submitted to TabArena and declined on competitiveness (§122).

**The largest step in this series came from preprocessing (§101), not architecture.** The tree
prior adds +0.0094 mean ROC-AUC and replicates across two seeds, but it cost **−0.0221 average
precision on credit data** (§116: worse on 5 of 5 folds, three surviving multiple-comparison
correction at p < 0.001), so it ships off. §166 later found that comparison confounded: drawing
the tree source first also cut the financial share from 0.70 to 0.49, and at a matched share
the tree arm is +0.0119 mean AP (4 of 5 folds, one seed per arm). It still ships off until a
matched re-test at two seeds; DR2.4's mixed-prior arms (§180, §191, all four harmful on
V4FinBench) are not that test.

Everything else tried on the pretrained model (parameter scale at matched task volume, training
volume, a widened structural-causal prior) was null or negative.

The full current leaderboard, with every method's Elo and interval, is at
**[kabartay.github.io/fintfm/leaderboard](https://kabartay.github.io/fintfm/leaderboard/)**,
generated from the stored run by `scripts/build_leaderboard_page.py`. It is this project's own
comparison, not [tabarena.ai](https://tabarena.ai), and its Elo is comparable only within the
methods it was computed over.

## What this project does

Three things, at their true weight.

**Incoherent PD term structures are impossible by construction.** The field's standard
per-horizon construction gives non-monotone cumulative default curves for **most firms** on
real data: a 3-year default probability below the 2-year one. Re-measured on 2026-11-13 with the
query rows kept out of the context, 47.8% of horizon steps fall and none of 3,151 firms gets a
monotone curve (§228); the earlier 39% of firms (§11) does not reproduce. A hazard head makes
that unrepresentable, at no measured accuracy cost.

**A capacity defect was found by measurement, not review.** The original architecture capped
discrimination whatever the task's true difficulty, which showed against an exactly known
Bayes-optimal AUC. Seven other candidate causes were eliminated one at a time before the cause
was traced to the architecture rather than the prior. A two-way cell-attention change closed
it. Its first estimate, +0.042 average precision on real data, also changed the prior mix (100%
against 70% financial tasks; audit AU2.8). Retrained at a matched prior and protocol (seed 0,
48,000 tasks, `p_financial` 1.0), the cell-attention block and its labels are worth +0.0520 mean
AP on V4FinBench (0.1947 against 0.1428), higher on 5 of 5 folds, each significant on a
company-level bootstrap (§172).

**The record corrects itself in public.** This README has been wrong about its own results at
least three times (§107, §112, §225). Each is corrected in place with the reasoning kept, and
`docs/paper/CLAIMS.md` shows retracted claims as prominently as the ones that hold. When a model
must arrive with its own validation evidence, that record is part of what is offered.

## How it works

1. **Priors** (`src/fintfm/prior/`): synthetic task generators, sampled and mixed per batch
   (`mixture.py`).
   - `financial.py`: a structural story for company balance sheets, P&L and default labels,
     with accounting identities, sector and macro effects, a wide family of derived ratios,
     missingness, a sharpness (signal-to-noise) knob and a default-rate envelope that reaches
     real low-default-portfolio rates (about 0.2%).
   - `scm.py`: a random-graph structural-causal-model prior (the TabPFN/TabICL idea), random
     layered functions with several nonlinearities, for general nonlinear and multiclass
     structure the financial prior does not cover.
   - `tree.py`: ensembles of oblivious decision trees, for the **axis-aligned,
     piecewise-constant** structure common in real tables and absent from the other priors.
     Added because `fintfm-priorscore` measured it as the only mixture member with a positive
     tree-versus-linear gap (§112). Off by default (`p_tree=0.0`).
   - `trivial.py`, `crossed.py`: diagnostic priors that isolate one hypothesis each (can the
     architecture learn at all; does a prior's effect come from its features or its labels).
     Not in the default mixture.
2. **Model** (`src/fintfm/modeling/model.py`): a Transformer written from scratch. Each cell is
   embedded with a random per-task column identity, so the model can tell columns apart without
   a positional embedding, which would break invariance to column order. Optionally
   (`ModelConfig.n_cell_blocks`, 0 by default but used by the published checkpoint), cells
   first attend across rows *within one feature*, which gives each column an identity derived
   from its distribution. Columns then attend to each other within a row; rows are pooled and
   attend to the context rows, which is the in-context learning step. An optional hazard head
   (`modeling/hazard.py`) outputs a cumulative PD term structure that is monotone by
   construction. The ideas come from the public TabPFN and TabICL papers; no code or weights
   from any project are used (see [Licensing](#licensing--provenance)).
3. **Training** (`src/fintfm/modeling/train.py`, `fintfm-train`): unlimited synthetic data, one
   gradient step per fresh batch, a cosine learning-rate schedule and checkpointing. The
   held-out quality metric reports per-task AUC (the AUC pooled across tasks is inflated
   whenever base rates differ) and Brier skill against a base-rate-only predictor.
4. **Inference** (`src/fintfm/inference/`): `FinancialTFMClassifier` follows scikit-learn.
   `fit()` stores the table as context and `predict_proba()` runs the frozen network. It offers
   several ways to build the context (uniform, balanced, hybrid, retrieval, prototype), a
   base-rate correction, and ensembling over column-identity draws (`n_ensemble`, default 1; the
   TabArena adapter uses 8, kept after §196 and the Lite arms in `docs/results/TABARENA.md`),
   which recovers the column-order invariance a random identity gives up.
   - `inference/categorical.py` encodes categorical columns as **out-of-fold** smoothed target
     statistics. The model reads every cell as an ordered number, and label encoding's arbitrary
     order is measurably worse than no order at all (§100). The out-of-fold step is required:
     the naive form puts a row's own label into its encoding, which makes the context
     self-predictive and the feature useless at query time. It harms the model rather than
     flattering the score, so careless validation does not catch it.
   - `inference/regressor.py` adds `FinancialTFMRegressor` on the same frozen network, with no
     new architecture. Cutting a continuous target into K quantile bins turns it into a K-way
     label, so the classification head regresses as it is. The output is a distribution:
     quantiles and prediction intervals come with it, and the predicted density can be bimodal,
     as loss given default is and a Gaussian head cannot be.
5. **Evaluation** (`src/fintfm/evaluation/`, `src/fintfm/experiments/`):
   - real corporate-default panels: V4FinBench under its published protocol
     (`fintfm-v4protocol`), Polish and Taiwan bankruptcy (`fintfm-bench`), and an out-of-time
     harness;
   - `credit_suite.py`, a second credit suite outside V4FinBench, used for evaluation only,
     never training: ULB card fraud, SBA, Estonia and FDIC (§174 and §177 for the first two,
     §216 for all four);
   - `fintfm-capability`, synthetic probes with a *known* ceiling, some with a closed-form
     Bayes-optimal AUC, built to catch a model that looks fine on average while capped in a way
     an aggregate score hides (with a class-count sweep for multiclass, §99);
   - `fintfm-priorscore`, which scores a *prior* rather than a model on performance, diversity
     and distinctiveness, using **fitted** baselines only, so it cannot mistake "the prior lacks
     this structure" for "the model cannot learn it". Its first version could, and §112 records
     what that cost.

   Average precision is reported beside ROC-AUC everywhere and read first at low base rates:
   ROC-AUC's chance level is 0.5 at any prevalence, while AP's is the prevalence itself, so AP
   stays readable at the base rates this project targets.

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

To work on the model rather than use it, clone the repository:

```bash
uv sync --extra bench --extra hf   # naming one extra uninstalls the others
uv run pytest                      # about 1,350 tests
```

| extra | adds | needed for |
| --- | --- | --- |
| `bench` | CatBoost, LightGBM, XGBoost, pyarrow | the baselines in every benchmark harness |
| `hf` | huggingface_hub | downloading the checkpoint, Hugging Face Jobs |
| `overlap` | openml, huggingface_hub, pyarrow | checking a suite for overlap with TabArena and BeyondArena |
| `real` | pyarrow, requests | building FinTFM-R's real-data panels and training on them (`--p-real-edgar`) |
| `kaggle` | kaggle, kagglehub, pyarrow | Kaggle kernels and their datasets |
| `schedulefree` | schedulefree | the schedule-free optimiser option in training |

### Using a checkpoint

The estimators follow scikit-learn. By default `fit()` **stores** the table as context with no
gradient steps, and `predict_proba()` runs the frozen network once. Fine-tuning is opt-in
(`finetune_steps`); a fine-tuned model has seen your training rows and is no longer the
zero-shot model (§163).

```python
from fintfm.inference import FinancialTFMClassifier, FinancialTFMRegressor

clf = FinancialTFMClassifier(ckpt, device="mps")   # ckpt from hf_hub_download, above
clf.fit(X_train, y_train)          # stores context; no training happens
proba = clf.predict_proba(X_test)[:, 1]

# Regression needs a checkpoint trained for it (--max-classes >= n_bins). None is published:
# the only public checkpoint is binary, so this line needs one you trained yourself.
reg = FinancialTFMRegressor("runs/v4-regression.pt", n_bins=10)
reg.fit(X_train, y_cont)
point = reg.predict(X_test)                     # distribution mean
lo, hi = reg.predict_interval(X_test, 0.8)      # 80% interval, from the same head
```

Encode categorical columns before they reach the model: it reads every cell as an ordered
number, and label encoding is measurably worse than no order at all (§100). Use
`fintfm.inference.categorical.CategoricalTargetEncoder`, which is out-of-fold on the context
rows for reasons that are **not** optional (see [How it works](#how-it-works)).

### Training a checkpoint

```bash
uv run fintfm-train --steps 300 --d-model 32 --d-cell 16 --n-layers 2 --max-features 16 \
    --max-classes 2 --device cpu --out runs/v0-smoke.pt   # pipeline check, a couple of minutes
uv run fintfm-bench --model runs/v0-smoke.pt --credit      # real corporate-default panels
```

This smoke configuration checks that the pipeline runs; it does not produce a usable
checkpoint. Measured step costs at real scale are in
[`docs/infra/COMPUTE.md`](docs/infra/COMPUTE.md). A real pretraining run (thousands of steps,
`--d-model` 128 or more, `--max-features` matching your data) takes hours to a day and needs a
GPU: `--device mps` on Apple Silicon, `--device cuda`, or the Hugging Face Jobs recipe this
project uses, in [`docs/infra/HF_JOBS.md`](docs/infra/HF_JOBS.md). **Check `uptime` before
running anything heavy locally**; [`CLAUDE.md`](CLAUDE.md) says why.

With the smoke checkpoint, `fintfm-bench --credit` (Polish and Taiwan need 64 and 95 features)
and `fintfm-v4protocol` (V4FinBench needs 136) print `SKIPPED for fintfm: model takes 16
features, data has N`. That is expected, not a bug: the baselines still run, and only the FinTFM
arm needs a checkpoint pretrained with a matching `--max-features`.

## Configuration

The numbers experiments use (split years, context sizes, strategies, seeds, scoring
thresholds, the prior's default-rate and sharpness envelope) live in
[`src/fintfm/configs/default.yaml`](src/fintfm/configs/default.yaml), not in the code. Every
entry point takes `--config` with a file **deep-merged** over that default, so an override holds
only what it changes:

```bash
uv run fintfm-ctxsweep --model runs/m.pt --config configs/context-sweep-3seed.yaml
uv run fintfm-v4oot    --model runs/m.pt --config configs/retrieval-best.yaml
```

Precedence runs packaged default, then `--config` (or `$FINTFM_CONFIG`), then an explicit CLI
flag. Runs print the layers they used and record them as `config_sources` in their output JSON.
Unknown keys are refused by name, and an overlapping train/test split is rejected before the run
starts. [`configs/README.md`](configs/README.md) explains what belongs in configuration and what
stays in code. Several files under [`configs/`](configs/) are **diagnostic, not production**, as
their headers say: each isolates one variable for one finding.

## Two model families

**FinTFM** is the synthetic-only model this repository has shipped from the start. Every
result, release and benchmark number here refers to it unless a page says otherwise. Its
provenance argument (it cannot have memorised a benchmark because it never saw real data)
depends on that, and is the project's differentiator (decisions D2, D14).

**FinTFM-R** (**R** for *real data*) is a second family, opened by decision **D17** on
2026-10-06. It mixes real tables into pretraining: SEC EDGAR's public-domain Financial Statement
Data Sets with 8-K Item 1.03 bankruptcy disclosures as labels (§151-§152), loaders for Freddie
Mac's loan-level sample (§154) and LendingClub (§155), and a generic real-panel sampler that
SBA loans go through (§157). It tests the one lever D16 showed had never been pulled, after
seven architecture and prior changes returned seven nulls or losses.

**So far it has not helped.** At a 30% real share, 6,000 steps from scratch and two seeds each,
neither real source improved V4FinBench mean AP over five folds: EDGAR is a stable null (−0.0019
and −0.0029 at seeds 0 and 1), and SBA a null whose seeds disagree (−0.0355 and +0.0024)
(§160-§161). No FinTFM-R checkpoint is published.

The two are not interchangeable: **the provenance claim holds for FinTFM and not for
FinTFM-R**, so every statement names which one it is about. FinTFM keeps its name, package, DOI
and TabArena entry; "FinTFM (synthetic-only)" is wording for when the contrast matters, not a
new name.

## Licensing / provenance

The prior, the architecture and the training loop are original code written for this project.
**No weights, datasets or source from Google TabFM, TabPFN, TabICL, TabDPT or any other
tabular-foundation-model product are used or may be added**; they are cited only as public
research context. Every third-party dataset and dependency licence is checked before use, and a
*weights* licence is checked separately from its *code* licence, every time.

Direct dependencies are all permissive: numpy, pandas, scikit-learn, scipy and torch (BSD-3),
PyYAML (MIT); the optional benchmark extras lightgbm (MIT), xgboost, catboost and pyarrow
(Apache-2.0); and the `overlap` extra's openml (BSD-3) and huggingface_hub (Apache-2.0), used to
fetch TabArena-v0.1 and BeyondArena for the overlap report, for evaluation hashing only.
`scripts/check_licences.py` enforces this in CI.

**FinTFM-R's real-data sources, checked and dated.**
- *SEC EDGAR* Financial Statement Data Sets and 8-K full-text search (`efts.sec.gov`): a US
  federal government work, public domain, commercially usable, verified 2026-10-06 by direct
  access. (`curl` returns 200 with a descriptive `User-Agent`; without one, SEC's Akamai front
  end returns 403, a bot filter, not a licensing gate.)
- *Freddie Mac's Single-Family Loan-Level Dataset*, registered and used under its **free
  tier**, whose terms were read in full (§153). They allow internal use (training and
  evaluating, including against public benchmarks) and academic or research use, including
  publishing results and derived products, **provided the distribution is itself
  noncommercial**, as fintfm's is (Apache-2.0, free checkpoints, no paid product). A separate
  fee-based agreement ($27,562.50) covers reselling or commercially redistributing the data or
  derived products; it would apply only if fintfm's status changed. One obligation binds on any
  tier: **never correlate the data to individuals**, since this is consumer loan-level data,
  unlike EDGAR's corporate filings.

**One exception, which CI found.** On **Linux**, `torch` pulls in about fifteen NVIDIA CUDA
runtime packages under **NVIDIA's proprietary licence**. They are a transitive runtime
dependency, not a choice made here, and this project does **not redistribute them**: `uv build`
produces a pure-Python wheel, and pip fetches them from PyPI under NVIDIA's terms. They are
absent on macOS, which is why a licence check on a developer's Mac passed while CI did not.
**Anyone shipping this in a product must read NVIDIA's EULA themselves**; this repository cannot
answer for them.

## What would change the picture

Stated so the project can be proven wrong, not left open-ended. Two gaps are measured
separately:

- **TabArena:** zero-shot FinTFM trailed tuned logistic regression by about 0.035 mean ROC-AUC
  over 27 datasets (§101). With a trained linear readout, squash and partitioned contexts it is
  level with it: +0.0036 mean ROC-AUC, 13 of 27 datasets better (§215).
- **V4FinBench, full panel:** untuned CatBoost (library defaults, 1,000 iterations) leads by
  about 0.19 mean AP over five folds (0.4005 against FinTFM's 0.2062 fine-tuned, §163). Both
  use V4FinBench's published features, which keep `operational_status`. On those features
  LightGBM at 1,000 trees and learning rate 0.05 scores 0.3908, so a booster given enough rounds
  reaches about 0.39 to 0.40; the 100-tree LightGBM (0.2666) and XGBoost (0.2229) defaults are
  under-trained and are not the field. Without `operational_status`, CatBoost's defaults score
  0.3662 (§167).

Fine-tuning on each fold's training rows was the first lever to move V4FinBench: +0.0076 mean
AP, 5 of 5 folds (0.1986 to 0.2062, §163), 3.8% of the gap to CatBoost, at the cost of no
longer being the zero-shot model. Partitioned contexts (a depth-2 tree, each leaf its own
2,000-row context) moved it more with no training: 0.1986 to 0.2362, +0.0375, 5 of 5 folds
(§169), ahead of fine-tuning on 5 of 5 (§205). Real-data pretraining (FinTFM-R) was null at GPU
scale (§160-§161). The remaining candidates:

| still open | why it is a candidate | status |
| --- | --- | --- |
| **factorized attention** | the encoder is memory-bound at every turn, and three peers independently chose the cheaper form | cost model written and checked against measured runs (44.1, [`docs/design/attention_cost_model.md`](docs/design/attention_cost_model.md)); the encoder is not built |
| **a joint objective** | `p(x, y \| D)` rather than `p(y \| x, D)` makes every column a training signal, not just the target | masked-column objective implemented (`--masked-column-prob`, 0.9.0); its 48,000-task arm is queued, not measured (48.15) |
| **prior design** | the only lever a peer's own ablations identify as decisive | instrument built and partly falsified (below) |

### Closed by measurement

**Tested and ruled out**, not abandoned. This is the more useful half of the record, and the
reason the list above is short:

| candidate | what the measurement said |
| --- | --- |
| parameter scale | **−0.0049** at 5.7× the parameters on *matched* task volume (§114). A peer's published curve gives +0.0049 R² for 16.7×, against a 0.035 deficit. |
| training volume | null at 5× (§93); **+0.0028** at 2× (§114); **−0.0106** mean AP at 10× over two seeds on V4FinBench, 9 of 10 folds negative (§188). For multiclass, 480,000 tasks is null on the proxy (§198) and on TabArena-Lite (+11 Elo, 5 of 8 datasets, §222). |
| the Muon optimiser | **−0.0125** mean AP at 48,000 tasks, 0 of 5 folds positive: harmful (§211). |
| longer training contexts | up to 2,000 rows per training task: −0.0042 mean AP, 1 of 5 folds positive, null (§211). |
| depth over width | on TabArena it could not be scored: the per-dataset time limit stopped it after 8 of 27 datasets (§114). On V4FinBench, 16 layers against 4 at width 128 is null, +0.0010 mean AP over five folds at a matched learning rate (§124, §126); at 480,000 tasks depth is null again, −0.0047 mean AP, 3 of 5 folds positive (§195). |
| `column_id_dim` | peaks at the value chosen by accident; all four other values score below it on all five folds (§104). |
| a widened structural-causal prior | **p = 1.000** against its own matched control. |
| a tree-structured prior | +0.0094 on general tabular data, replicated across two seeds (§115), and **−0.0221 average precision on credit** (§116), so it ships off. (§166, 2026-10-16: §116's arms differed in financial share, 0.49 against 0.70; at a matched share the tree arm is +0.0119 mean AP, 4 of 5 folds, one seed, so the credit loss is not the tree prior's.) |

**The prior-scoring instrument is half confirmed, half falsified.** It predicted the
general-tabular gain above before it was measured (§112, then §115). A second arm, added for the
credit question, predicted the wrong direction on its first real test: raising the
structural-causal prior's share cost **−0.0339 AP**, worse on 5 of 5 folds, all significant
(§118). So it predicts **breadth, not fit**, and no claim rests on its credit arm.

## Reproducing the measurements

Every number in [`docs/results/FINDINGS.md`](docs/results/FINDINGS.md) names the command that
produced it. The entry points:

| command | what it measures |
| --- | --- |
| `fintfm-v4protocol` | V4FinBench under its **published** five-fold protocol, the benchmark this project's claims rest on |
| `fintfm-bench --credit` | Polish and Taiwan bankruptcy panels |
| `fintfm-v4oot` | an out-of-time split, which the published protocol is not |
| `fintfm-capability` | synthetic probes with a **known** ceiling, some with a closed-form Bayes-optimal AUC |
| `fintfm-priorscore` | a *prior*, not a model, on performance, diversity and distinctiveness |
| `fintfm-ctxsweep` | context strategy against context size, out of time. Strategy mattered at smoke-test scale (§5); on V4FinBench's full panel, context size is not what limits average precision (§162) |
| TabArena-Lite, in the fork | the recipe, the fork's environment variables and the 51-dataset merge are in [`docs/results/TABARENA.md`](docs/results/TABARENA.md); `scripts/lite_positions.py` derives positions from the stored leaderboards (§225), and `scripts/build_leaderboard_page.py` regenerates the [leaderboard page](https://kabartay.github.io/fintfm/leaderboard/) |

Two conventions to know first. **Every number states how it was produced** (SMOKE-TEST,
MEASURED, SIMULATED or ESTIMATED), because in an ML repository a wrong number does not crash; it
looks like a result. And **negative and superseded results are kept**, so the same wrong
conclusion is not reached twice.

## Where this is going

[`docs/`](docs/) holds the project's reasoning, indexed in [`docs/README.md`](docs/README.md).

- [`docs/paper/CLAIMS.md`](docs/paper/CLAIMS.md): **start here.** Every claim the project could
  make, tagged by status; the newest evidence wins.
- [`docs/results/FINDINGS.md`](docs/results/FINDINGS.md): the measurement log, numbered in
  order (228 entries on 2026-11-13), each stating how its numbers were produced.
- [`docs/paper/NEGATIVE_RESULTS.md`](docs/paper/NEGATIVE_RESULTS.md): every null and harmful
  result in one place, so a lever already ruled out is not tried again by accident.
- [`docs/paper/LIMITATIONS.md`](docs/paper/LIMITATIONS.md): what the model and the evidence
  cannot support today.
- [`docs/design/DECISIONS.md`](docs/design/DECISIONS.md): why the project is built as it is,
  and what would reverse each choice.
- [`docs/design/ARCHITECTURE.md`](docs/design/ARCHITECTURE.md): how the model works, in more
  depth than this file.
- [`docs/roadmap/STRATEGY.md`](docs/roadmap/STRATEGY.md): the plan of record.
- [`docs/results/TABARENA.md`](docs/results/TABARENA.md): the external-evaluation recipe, the
  coverage any score must state, and the silent failure modes found so far, including a results
  cache that returns stale numbers after a preprocessing change.
- [`docs/paper/RELATED_WORK.md`](docs/paper/RELATED_WORK.md): the competing models, read from
  primary sources, and what each means for this project's position. Its digest is the shortest
  summary of why the remaining lever is the prior.
- [`docs/results/POSTMORTEM.md`](docs/results/POSTMORTEM.md): wrong diagnoses, each caught by
  measurement rather than review, kept on the record.
- [`openspec/changes/`](openspec/changes/): the live roadmap, as proposals with verifiable
  tasks. `uv run python openspec/tools/validate.py` checks that every proposal is well formed
  and every finding states its provenance.

In short: the mechanism claims ("no training on your data", "no feature engineering") already
belong to better-funded competitors, so the mechanism is not the thesis. The thesis is a credit
model that arrives with its own validation evidence (its calibration measured, demonstrably free
of benchmark contamination, and eventually backed by a pre-registered forward track record that
cannot be bought), plus a public, self-correcting record of what has and has not been shown.

## Project structure

```
src/fintfm/
  prior/         the synthetic data-generating processes: a structural model of company
                 financials, a random-graph SCM, a tree prior, and the mixture that samples
                 them; plus FinTFM-R's real-panel samplers (real_*.py)
  modeling/      the network and the pretraining loop
  inference/     scikit-learn compatible estimators, plus the categorical target encoder
  evaluation/    benchmark harnesses and baselines
  experiments/   one module per measured question; each writes a numbered finding
  configs/       packaged defaults, read at runtime by path

docs/
  results/       FINDINGS.md, the numbered measurement log this project is organised around,
                 plus TABARENA.md and POSTMORTEM.md
  paper/         CLAIMS.md (what holds, what was retracted), NEGATIVE_RESULTS.md,
                 RELATED_WORK.md, LIMITATIONS.md, OUTLINE.md, FIGURES.md
  design/        ARCHITECTURE.md, DECISIONS.md, GLOSSARY.md, DESIGN_REVIEW_2026-10-16.md,
                 attention_cost_model.md
  roadmap/       ROADMAP.md (what to do next), STRATEGY.md, NEXT.md, TABARENA_BAR.md
  competition/   LANDSCAPE.md, SUBMISSION.md
  infra/         COMPUTE.md, HF_JOBS.md
  research/      REFERENCES.md, RESEARCH_NOTES.md, IMPROVEMENT_PLAN_2026-11.md
  leaderboard/   the generated Pages leaderboard, its template and data

openspec/changes/   one proposal per change, with its task list; the queue lives here
scripts/            the repository's own checks (licences, documentation links, Zenodo metadata)
                    and run tooling: real-data panel builders, Kaggle kernels, score
                    comparison, pre-registered verdicts and the leaderboard page
```

Every number quoted in this repository points at a numbered entry in
`docs/results/FINDINGS.md` that states how it was produced, and retracted entries stay in place
rather than being edited away.

## Acknowledgements

**No code, weights or training data from any tabular foundation model was used here** (see
[Licensing / provenance](#licensing--provenance)). This is credit for what *was* used.

- **[V4FinBench](https://github.com/genwro-ai/V4FinBench)** (Kostrzewa et al.,
  [arXiv:2605.10896](https://arxiv.org/abs/2605.10896), CC BY 4.0), for the corporate-default
  panel and its evaluation protocol, which this project reimplements from the published
  specification rather than copying.
- **[TabArena](https://tabarena.ai)**, for the external protocol, and its maintainers, for
  reviewing the submission and confirming the integration and numbers correct while declining
  the entry on competitiveness (§122). That review is the only independent check this project
  has had.
- **The published work of TabPFN, TabICL, TabDPT, LimiX, Nori, MITRA, TabSTAR, TabSwift,
  OrionMSP, iLTM, ConTextTab and EXAONE-Tabular**, for **ideas**, which this project took freely
  and then measured:

  | idea | from | what happened |
  | --- | --- | --- |
  | three criteria for scoring a prior: performance, diversity, distinctiveness | MITRA | built into `fintfm-priorscore`; §112 then measured this prior's distinctiveness as weak |
  | a tree-structured prior | TabICLv2, reimplemented from its appendix | **+0.0094** on general tabular data, **−0.0221 AP** on credit; ships off (§116). (§166, 2026-10-16: the credit loss is confounded with a lower financial share; matched, the tree arm is +0.0119 mean AP, one seed.) |
  | published scaling curve | Nori | capped a scale hypothesis that GPU budget was being spent on, before more was spent (§114) |
  | rejecting unlearnable synthetic tasks | Nori | measured: it would reject 35% of this prior and discard signal the model uses (§125) |
  | narrow-and-deep shape, 16 layers at width 128 | Nori | measured: null here, +0.0010 AP (§124, §126) |

  Three of those five ideas were measured and **not** adopted. That is why they were taken: a
  published result from a different prior and architecture is a hypothesis about this one, not
  a conclusion about it.

Reading a paper is not ingesting a codebase; `CLAUDE.md` enforces the difference.

## Citing

`CITATION.cff` in the repository root makes GitHub's **"Cite this repository"** button render
BibTeX and APA. The same entry, to copy:

```bibtex
@software{Organokov_FinTFM_2026,
  author    = {Organokov, Mukharbek},
  title     = {{FinTFM: a tabular foundation model for corporate credit risk,
               pretrained only on synthetic data}},
  year      = {2026},
  version   = {0.10.1},
  doi       = {10.5281/zenodo.22949759},
  url       = {https://github.com/kabartay/fintfm},
  publisher = {Zenodo},
}
```

There is **no paper**: the numbered measurement log is the public record, so the software entry
is the citable artifact, not a stand-in for one.

Cite the **repository** for the method, the measurement log or a finding, and the
**checkpoint** ([kabartay/fintfm-binary](https://huggingface.co/kabartay/fintfm-binary)) when the
specific weights matter. They are different artifacts, and a reader can only check the one you
name.

Every release is archived on Zenodo with its own DOI. The **concept DOI**
[10.5281/zenodo.22949759](https://doi.org/10.5281/zenodo.22949759) always resolves to the newest
version. The difference matters here because this repository retracts and supersedes results
(§107 and §112 both withdrew earlier claims): cite a **version** DOI to pin what you read, and the
**concept** DOI to point at the current state.

## Contributing, and what cannot come in

**From mid-November 2026 this project is no longer actively supported.** The code, the
checkpoint and the measurement log stay public, but issues and pull requests may go unanswered.

Two hard limits protect the one claim this project has that competitors do not:

- **No code, weights or training data from any tabular-foundation-model product** (TabPFN,
  TabICL, TabDPT, LimiX, Nori, MITRA or any other) may enter this repository. Ingesting any of
  it would destroy the provenance argument above, and nothing would warn us.
- **Check a weights licence separately from its code licence, every time.** Reading and citing
  these projects' papers is fine, and [`docs/paper/RELATED_WORK.md`](docs/paper/RELATED_WORK.md)
  does so for ten of them. Four of the ten ship permissive code with **non-commercial
  weights**, a split invisible from a repository's licence badge, and one restricts commercial
  use of the model's *output*, not only the weights.
