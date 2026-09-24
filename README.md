# fintfm

**A from-scratch tabular foundation model for corporate credit risk.** An in-context classifier
pretrained only on synthetic tasks — it never sees real data during training, and makes
predictions in a single forward pass with your table supplied as context. No gradient steps on
customer data, no per-dataset training.

| | |
| --- | --- |
| **Status** | research codebase, actively developed — not a product |
| **Licence** | Apache-2.0, code **and** weights (see [Licensing](#licensing--provenance)) |
| **Tests** | 242 (`uv run pytest`) |
| **Measurement log** | 117 numbered findings, each declaring how it was produced |
| **External benchmark** | [TabArena](docs/TABARENA.md), 90% coverage, **rank 93 of 95** |
| **Problem types** | binary, multiclass, regression |

**Read [`docs/paper/CLAIMS.md`](docs/paper/CLAIMS.md) first** — every claim this project could
make, tagged SURVIVES / SINGLE DRAW / SUPERSEDED / RETRACTED / OPEN against its evidence. That
file, not this one, is the source of truth. This README is an orientation map.

## Results, as measured

Real corporate-default panels, five-fold published protocol:

| | discrimination | calibration |
| --- | --- | --- |
| **fintfm** | loses to tuned gradient boosting, on every panel tried | **consistently among the best measured** |

TabArena, 27 binary datasets, single fold each, against 94 other methods:

| checkpoint / change | mean ROC-AUC | Elo | rank |
| --- | --- | --- | --- |
| label encoding (§98) | 0.7642 | 662 | 93 / 95 |
| **+ out-of-fold target statistics** (§101) | **0.7823** | 765 | 93 / 95 |
| + multiclass-capable prior (§105) | 0.7817 | **813** | 93 / 95 |
| + 5.0M parameters, confounded (§108) | 0.7746 | 751 | 93 / 95 |
| + 5.0M parameters, **matched tasks** (§114) | 0.7774 | — | 93 / 95 |
| + tree-structured prior (§115), two seeds | 0.7926 / 0.7913 | 842 / 826 | 93 / 95 |

**The tree prior does not transfer to credit data.** On V4FinBench's five-fold protocol — 1M
rows at a 0.359% default rate, which is the regime this project exists for — it scores
**−0.0221 average precision** against the same control, dropping *below* untuned logistic
regression where the control clears it (§116). Both results are real: it helps on general
tabular data and harms the low-default case. The second is the one that decides whether it
ships.

**One change has moved the number on the benchmark that matters.** Out-of-fold categorical
encoding lifted TabArena's mean from 0.7642 to 0.7823 (§101). A tree-structured prior,
selected on measured distinctiveness, adds a further +0.0094 there and replicates across two
seeds (§112, §115) — but costs **−0.0221 average precision on credit data** (§116), so it is
not shipped. Everything else tried, including parameter scale at matched tasks, has been null
or negative. **The rank has never moved.**

## What is currently true

**On real credit panels, calibration is consistently among the best measured and discrimination
consistently loses to tuned gradient boosting** — both facts together, on every panel tried.
That is the project in one line, and everything below is detail.

**Two things it does that nothing else does.** A hazard head makes incoherent PD term
structures impossible by construction — a real, measured defect in the field's standard
per-horizon construction, affecting 39% of firms on real data — at zero accuracy cost. And a
severe capacity defect in the original architecture (capped discrimination regardless of true
task difficulty, caught against an exactly-known Bayes-optimal AUC) was isolated to the
architecture rather than the prior after eliminating seven other candidates one at a time, then
closed by two-way cell attention and confirmed on real data at +0.042 average precision.

**Measured externally on [TabArena](docs/TABARENA.md), it places 93rd of 95** (§98), and the
diagnosis of that deficit is the most useful work here. Most of it was *preprocessing*, not
architecture: the per-dataset gap correlated **−0.668** with log categorical cardinality, and
replacing label encoding with out-of-fold target statistics lifted the mean from 0.7642 to
**0.7823** while collapsing that correlation to −0.025 (§100, §101). **The rank did not move.**
What remains is a uniform ~0.035 ROC-AUC deficit with no measured axis of variation — a sharper
target than before, and still a losing one.

**What has since been ruled out, which is most of what looked promising.** `column_id_dim`, the
only untuned lever that had ever moved real-data accuracy, peaks at the value chosen by accident
(§104). Parameter scale lost (§108). Training volume at 5× was inert (§93). Class balance did
not survive testing (§102, §103). A widened SCM prior was a flat null against its own control
(p = 1.000). Reading the field explains why: a peer's published scaling curve returns +0.005 R²
for 16× the parameters, so **no scaling programme was ever going to close 0.035** — and every
one of TabArena's top fourteen ranks is synthetic-pretrained (§110), which says the constraint
this project chose costs nothing in rank and the lever is prior *design*.

**Coverage, which any score must carry.** The 93rd place was measured at **51% coverage**;
multiclass and regression have since taken it to **90% (46 of 51)**, with `max_features` the
only remaining exclusion. Those arms are *runnable*, not yet *scored* on real data, and a
coverage fraction is not a result.

**The honest one-line summary: this is not a competitive general tabular model, and on TabArena
its median rank is 94 of 95 — including on the credit panels.** An earlier version of this line
claimed its best public results were the corporate-credit panels it was designed for. §107
retracts that: those panels rank 83–95, and the high absolute AUC on them (0.9287 on Taiwanese
bankruptcy) is what everyone scores there, not an edge. Whether a credit specialism exists is a
question for V4FinBench's five-fold protocol, not for a single-fold leaderboard.

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
uv sync --extra bench --extra hf   # naming one extra uninstalls the others
uv run pytest                      # 242 tests; 1 skip is expected, more means look
```

### Using a checkpoint

The estimators are scikit-learn compatible. `fit()` **stores** the table as context — it takes
no gradient steps — and `predict_proba()` runs the frozen network once.

```python
from fintfm.inference import FinancialTFMClassifier, FinancialTFMRegressor

clf = FinancialTFMClassifier("runs/v4-cellattn-labels.pt", device="mps")
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
[`docs/COMPUTE.md`](docs/COMPUTE.md) for measured step costs at real scale. A real pretraining
run (thousands of steps, `--d-model 128`+, `--max-features` matching your data) takes
hours-to-a-day and needs a GPU: `--device mps` on Apple Silicon, `--device cuda`, or see
[`docs/HF_JOBS.md`](docs/HF_JOBS.md) for the Hugging Face Jobs recipe this project actually
uses for training runs. **Check `uptime` before running anything heavy locally** — see
[`CLAUDE.md`](CLAUDE.md) for why.

Both `fintfm-bench --credit` (Polish/Taiwan need 64/95 features) and `fintfm-v4protocol`
(V4FinBench needs 136) will print `SKIPPED for fintfm: model takes 16 features, data has N` —
correct, expected behaviour for the smoke checkpoint above, not a bug. Baselines still run and
score normally; only the fintfm arm needs a checkpoint pretrained with a matching
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
assumed from a prior check. Current dependencies are all permissive: numpy, pandas and
scikit-learn BSD-3, scipy BSD-3, torch Apache-2.0, PyYAML MIT, and the optional benchmark
extras lightgbm MIT, xgboost Apache-2.0, catboost Apache-2.0, pyarrow Apache-2.0.

## What would change the picture

Stated so the project is falsifiable rather than open-ended. The uniform ~0.035 deficit has no
measured axis; the levers that remain, in the order the evidence ranks them:

| lever | status |
| --- | --- |
| **prior design** — distinctiveness is measurable and predicts general-tabular gains | +0.0094 on TabArena, replicated (§115); **−0.0221 AP on credit** (§116), because at a 0.4% base rate the tree prior's own tasks are the least learnable of the three (§117). The instrument now asks the credit question it previously could not |
| **factorized attention** — the current encoder is memory-bound at every turn, and three peers independently chose the cheaper form | proposed (44.x), prior art recorded |
| **objective** — `p(x, y \| D)` rather than `p(y \| x, D)`, which makes every column a training signal | proposed (48.15), scoped as a measurement before a rewrite |
| ~~parameter scale~~ | closed (§114): −0.0049 at 5.7× **matched tasks**, and a peer's curve returns +0.005 R² for 16× |
| ~~`column_id_dim`~~ | closed (§104) |
| ~~training volume~~ | null at 5× (§93) and +0.0028 at 2× (§114) |
| ~~depth over width~~ | cannot be scored (§114): `TimeLimitExceeded` after 8 of 27 datasets |

## Reproducing the measurements

Every number in [`docs/FINDINGS.md`](docs/FINDINGS.md) names the command that produced it.
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
- [`docs/FINDINGS.md`](docs/FINDINGS.md) — the full measurement log, numbered sequentially
  (117 entries and counting), each declaring how its numbers were produced.
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — why the project is built the way it is, and what
  would reverse each choice.
- [`docs/STRATEGY.md`](docs/STRATEGY.md) — the plan of record.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the model works, in more depth than
  this file.
- [`docs/TABARENA.md`](docs/TABARENA.md) — the external-evaluation recipe, the coverage
  fraction any score must carry, and four silent failure modes including a results cache that
  returns stale numbers after a preprocessing change.
- [`docs/paper/RELATED_WORK.md`](docs/paper/RELATED_WORK.md) — the competing models, read from
  primary sources, with what each costs this project's positioning. Its digest is the shortest
  useful summary of why the remaining lever is the prior.
- [`docs/POSTMORTEM.md`](docs/POSTMORTEM.md) — wrong diagnoses, each caught by measurement
  rather than review, kept on the record deliberately.
- [`openspec/changes/`](openspec/changes/) — the live roadmap as structured proposals with
  verifiable tasks, not prose. `uv run python openspec/tools/validate.py` checks every
  proposal is well-formed and every finding declares its provenance.

Short version: the mechanism claims ("no training on your data", "no feature engineering") are
already owned by better-funded competitors, so the thesis is not the mechanism. It is a credit
model that arrives with its own validation evidence — calibrated, auditably free of benchmark
contamination, and eventually backed by a pre-registered forward track record that cannot be
bought — plus a public, self-correcting record of what has and has not been shown to be true.

## Status

Actively developed research codebase, not a PoC skeleton: a config-driven experiment harness,
real GPU pretraining infrastructure ([`docs/HF_JOBS.md`](docs/HF_JOBS.md)), an external
benchmark integration at 90% coverage, and 117 numbered, provenance-tagged findings.

**What is proven, open and retracted is tracked in
[`docs/paper/CLAIMS.md`](docs/paper/CLAIMS.md), not here** — the honest state changes faster
than this file gets edited, which is exactly the failure the claims ledger exists to prevent.
This README has been wrong about its own results at least twice (§107, §112); the ledger is
where that gets caught.
