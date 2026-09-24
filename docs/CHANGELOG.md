# Changelog

Hard-wrapped, because it is read in an editor and a diff. Release bodies on GitHub are
**not** wrapped — they are read in a browser at full width. Same words, different shape; do
not paste one into the other. See `CLAUDE.md`.

## [Unreleased]

Measuring the things this release declared, and finding that two of them do not hold.

### Measured

- **Multiclass and regression run correctly and rank last** (§121). Scored on real data for the
  first time: multiclass **93.4 of 95** over 7 datasets, regression **93.1 of 94** over 12 and
  **last on 5 of them**. Both completed with no crashes and no timeouts, every prediction in the
  right units. **Running without error is not working**, and §106 established only the first
  while reading as though it established the second.
- **The binning limit is real and is not the binding constraint** (§121). Error ratios track it
  exactly — 6.5× on wide smooth targets, 1.2× on near-discrete ones — while **rank is flat
  against that axis**. On `wine_quality`, where binning costs least, the model ranks last. So the
  quantile head proposed to fix it would recover the ratio and leave the rank.
- **The tree prior harms credit** (§116, §117): −0.0221 AP, with the mechanism measured — at a
  0.4% base rate the tree prior's own tasks are the **least learnable** of the three priors,
  because axis-aligned splits need positives on both sides of a threshold.
- **A prior-selection prediction was falsified** (§118). Raising the structural-causal prior's
  share, which the diagnostic ranked most learnable at credit's base rate, cost **−0.0339 AP** —
  0/5 folds, all five significant. The instrument's ranking is *inverted* on the question it was
  built for.
- **KV caching would save 17%, not the order of magnitude claimed** (§119), and **larger query
  chunks are 10.7× slower, not faster** (§120) — a cost model fitted over 256–2,048 queries and
  extrapolated to 8,192, in a system whose dominant cost is quadratic.

### Changed

- **TabArena declares `binary` only.** Multiclass and regression remain implemented, tested and
  documented but undeclared, on §121's measurement. Coverage is reported as **27 of 51 (53%)**,
  derived from the model's own declaration rather than a duplicated list that had already
  drifted.

### Fixed

- **A `ruff` failure shipped red on the 0.4.0 release commit**, and that release's notes claimed
  this repository had no CI — a stale sentence in `CLAUDE.md`, quoted as though checked. Both
  corrected; `CLAUDE.md` now names `ruff check` and `gh run list` in the release procedure.

## [0.4.0] — 2026-09-23

First external measurement, and the first decomposition of a deficit into a part that belongs
to the model and a part that does not.

### Added

- **Native categorical handling** (`inference/categorical.py`): out-of-fold smoothed target
  statistics, replacing label encoding. No retraining — it runs against released checkpoints,
  which is why it is sequenced before any architectural work. On one checkpoint,
  `Amazon_employee_access` moved from **0.5455 to 0.8220**, within 0.022 of tuned logistic
  regression. The out-of-fold step is load-bearing rather than a refinement: the naive form
  puts a row's own label into its own encoding, which makes the model worse while leaving
  aggregate scores looking reasonable.
- **Regression, by de-binning** (`inference/regressor.py`). A continuous target cut into K
  quantile bins *is* an index over K outcomes, so `FinancialTFMRegressor` needed no change to
  the head, the loss or the label injection — it bins on fit and takes the predicted
  distribution's mean on predict. The output is natively distributional: `predict_quantile`
  and `predict_interval` expose a shape that a Gaussian head could not represent, which is
  the property loss given default actually needs.
- **TabArena coverage 51% -> 90%** (46 of 51). Declaring `multiclass` and `regression` leaves
  `max_features=136` as the only remaining exclusion. The fraction is derived from TabArena's
  task metadata at run time, not written down, so it cannot drift from what was executed.
  `FINTFM_PROBLEM_TYPES` narrows a run to one type, which is what keeps a binary score
  comparable with §98/§101 after the other two were added.
- **A tree-structured prior** (`prior/tree.py`, `--p-tree`): ensembles of oblivious decision
  trees generating the axis-aligned, piecewise-constant structure real tabular data is full of
  and the other priors are not. Added on measured *distinctiveness* grounds, not intuition —
  it is the only member of the mixture with a positive tree-versus-linear gap (§112). Off by
  default.
- **`fintfm-priorscore`**: scores a *prior* rather than a model, on the three criteria the
  literature converged on (performance, diversity, distinctiveness), using **fitted** baselines
  only so it cannot confuse "the prior lacks this structure" with "our model cannot learn it".
- **SCM graph re-targeting** (`--scm-reuse-graph`): several tasks per graph, each using a
  different node as the target. Siblings are genuinely distinct problems — a model fitted on
  one scores **0.4775, chance**, on another (§113).
- **A multiclass instrument** (`fintfm-capability --class-sweep`). The capability suite was
  binary-only, so "trained at `--max-classes 10`" was a statement about a command line. The
  trained checkpoint clears its own untrained control at K = 3, 5 and 10 (§99). The accuracy
  floor is the *measured* majority-class rate, not `1/K`, which would have understated it by
  0.06 at K = 10.
- **`docs/results/TABARENA.md`** — the external-evaluation recipe, including four silent failure
  modes. The costliest: TabArena caches results per config name, so a preprocessing change
  re-run in place returns the previous numbers to four decimal places and reads as "no
  effect".

### Measured

- **TabArena: rank 93 of 95**, mean ROC-AUC 0.7642 over 26 datasets, 51% coverage (§98). The
  first accuracy measurement in this project whose baselines and protocol belong to someone
  else.
- **`column_id_dim` peaks at 16** (§104), the value that was chosen by accident. Five arms
  (12/16/20/24/32) at matched everything else, scored on the same five V4FinBench folds: all
  four non-peak arms are below 16 on all five folds, 20 of 20, with parameter count spanning
  0.11%. It was the only untuned lever that had ever moved real-data accuracy, and it is now
  closed — which promotes **scale** from one hypothesis to the leading untested explanation
  for the uniform residual §101 left.
- **The tree-structured prior works on general tabular data, and replicates** (§115).
  **Superseded in scope by §116, after this release**: it costs **−0.0221 average precision on
  credit data**, negative on 5 of 5 folds with three surviving Holm correction at p < 0.001,
  and ships **off by default** (`p_tree=0.0`). Both results hold; the credit one decides. The
  entry below stands as written and describes the general-tabular half only. Two seeds, identical
  architecture and task count, differing only in whether the mixture contains it:
  **+0.0083** (17/27, p = 0.248) and **+0.0105** (20/27, p = 0.019), mean **+0.0094** with
  signs agreeing. The two controls differ from each other by −0.0035, so the seed noise is
  measured rather than assumed and the effect is ~2.7x it. Means of 0.7926 and 0.7913 are the
  highest recorded, against a previous best of 0.7869; Elo 842 and 826. **The rank does not
  move** — 93 of 95 — and nothing has been measured on credit data. Predicted before it was
  measured, by §112's prior-scoring instrument.
- **Scale is closed, on three independent lines** (§114). 5.7x the parameters at *matched*
  task volume scores **-0.0049** (12/27, p = 0.701) — §108's earlier -0.0077 was confounded by
  half the task count, and removing the confound recovered +0.0028 of it and no more. Training
  volume is null at 5x (§93) and +0.0028 at 2x. A peer's published curve returns **+0.0049 R²
  for 16.7x** the parameters, against our 0.035 deficit. `docs/roadmap/STRATEGY.md`'s 10-50M Phase 1
  target is withdrawn rather than questioned.
- **The deep-narrow arm could not be scored at all** (§114), raising `TimeLimitExceeded` after
  8 of 27 datasets. Depth costs inference time and this project is already at 8.6 s/1K against
  a field norm near 0.1, so a configuration that cannot finish the benchmark is not a candidate
  regardless of its accuracy. The question was posed as accuracy; the feasibility answer
  arrived first.
- **Every top-14 TabArena rank is synthetic-pretrained** (§110). The three groups arguing that
  real-table pretraining is the axis that pays do not place above the synthetic frontier, so
  decision D2's synthetic-only constraint costs nothing measurable in rank.
- **82% of that gap is categorical preprocessing** (§100). The per-dataset gap correlates
  **−0.668** with log cardinality; the numeric-only residual is **−0.0320**, 2.8× smaller than
  the headline. §98's attribution was one dataset deep and is corrected.

### Retracted

- **§111's claim that the SCM prior favours linear models** (-0.0233) rested on a linear
  baseline fitted on raw heavy-tailed features that failed to converge. Corrected, the gap is
  **-0.0021** — no effect (§112). The conclusion it supported survives and is better supported:
  `tree` is the only prior with positive distinctiveness. **The first version of the instrument
  reported the financial prior as the most tree-favourable member**, which would have argued
  against building the tree prior at all.
- **§107 retracts "its best public results are the corporate-credit panels it was designed
  for."** By rank those panels are 83rd to 95th of 95; the high absolute AUC on them is what
  every method scores there. The apparent specialist profile (mean rank 86.26 against harmonic
  rank 20.72, the largest spread on the board) decomposed to **one dataset, one fold, and a
  0.0026 AUC margin**.

### Fixed

- **Target statistics were gated to binary targets**, so every regression dataset would have
  silently fallen through to frequency encoding on the very run that declared regression
  supported — discarding the label information on a quarter of the suite. A rate is a mean of
  an indicator, so the out-of-fold arithmetic already worked for continuous targets; the gate
  was narrower than the maths. The line that actually matters is quantity versus name:
  integer multiclass codes stay on frequency encoding, because averaging against "class 3" is
  §100's mistake moved to the target side.

- `hf jobs run` attaches and streams logs unless given `--detach`, so a loop launching three
  jobs silently launched one. Recorded in `docs/infra/HF_JOBS.md`.

## [0.3.0] — 2026-09-19

The architecture defect found in 0.2.x is fixed and the fix holds on real data. Five separate
explanations for the remaining accuracy gap were tested and ruled out, which is most of what
this release contains.

### Added

- **Two-way cell attention** (`ModelConfig.n_cell_blocks`, `cell_labels`). Cells attend across
  rows *within* a feature before pooling, and context cells carry their row's label. On a probe
  with a closed-form Bayes-optimal AUC, regret falls from 0.234–0.277 to 0.001–0.005 on the
  identical prior that produced the cap (§78). On V4FinBench, five folds of five, each
  architecture at its own best measured context: **+0.0417 average precision** (§84).
  The two changes are **jointly necessary** — without per-cell labels the same architecture
  scores *below chance*, a 0.514 AUC swing (§91).
- **Resumable training** (`--run-steps`, `--resume`). A `.state` sidecar carries optimiser
  moments, schedule position, both RNG streams and the step counter, so a run longer than one
  job's wall-clock can be chained. Two 6-step chunks produce weights matching one 12-step run
  to 1e-6, asserted in `tests/test_train.py` — a broken implementation would still train and
  still print a plausible loss curve (§92).
- **Feature chunking** (`FinancialTFM.feature_chunk`), identity-preserving and asserted
  byte-for-byte. Worth 6.2× memory at inference on CPU/MPS; worth nothing in training or on
  CUDA, for reasons measured in §94/§95.
- **A breadth diagnostic** (`experiments/openml_breadth.py`): 15 public OpenML binary tasks
  spanning 2.3–44.5% prevalence, so the accuracy deficit can be read against dataset properties
  rather than quoted as one number measured only on credit panels.

### Measured and ruled out

Each of these was, at some point, the leading explanation for the deficit to gradient boosting.

- **Training volume.** 5× the tasks (240,000 against 48,000), everything else identical:
  **−0.0012 AP**, 3 of 5 folds nominally positive, and the only individually-significant fold
  favouring the *smaller* run (§93). Retires a ~$130 full-scale run.
- **Context size.** The whole axis spans **0.0069 AP** on the full 1M-row panel, peaking at
  2,000 and declining at 4,000 (§83). The architecture effect is 6–7× larger (§84).
- **Marginal mismatch.** The model is fitted on kurtosis-41 marginals and served kurtosis-1.8
  ones (§87) — but the rank transform, already the inference default, pins performance flat
  across every monotone warp, leaving ~0.001 for augmentation to recover (§88).
- **Prior domain.** Swapping the *entire* prior from financial to generic structural-causal
  moves the mean deficit from −0.144 to −0.134 — about a tenth of the effect (§96).
- **The accuracy deficit is not credit-specific.** It reproduces on ecology, speech,
  software-defect, medical and astronomical data: behind the best baseline on **14 of 15**
  public tasks under either prior (§96).

### Fixed

- `_eval_quality` inherits the training batch size instead of a hardcoded 16, which is what
  actually closed §78's protocol deviation (§86, attribution corrected in §94).
- `v4_protocol` passes `cfg.inference.query_chunk` to the classifier; the key had been read
  from configuration, printed in `config_sources`, and then ignored.
- A `--device` flag on `fintfm-v4protocol`; nothing in that path could previously select MPS,
  where a cell-attention checkpoint is ~100× faster than on CPU.

### Corrected

Kept in the record rather than quietly edited, because each was propagated before it was caught.

- **§82**: §69–§71 ran on a **10× subsample** (105,900 test rows against 1,000,087), and
  "0.2116" was a single fold of it rather than §71's five-fold mean of 0.1676. A caveat built on
  that comparison was withdrawn, and those three findings now carry a subsample header.
- **§94**: feature chunking does not reduce *training* memory — measured flat to 0.2% across a
  34× range — so §86's attribution of the protocol fix to chunking is withdrawn.
- **§95**: attention memory is linear in N on CUDA and quadratic on CPU/MPS. §79's "~63 GB" and
  its 92× cliff are Mac measurements and were being quoted as though general.
- **§90**: two complete 3-hour GPU runs were lost to a `pip install` line missing
  `huggingface_hub` while ending in `hf upload`. The recipe now installs it and fails fast.

## [0.2.0] — 2026-09-09

### Fixed

- **The term-structure path skipped the base-rate correction**, so out-of-time evaluation
  stated a 12.8% default probability against a 0.47% observed rate. The out-of-time harness
  built its own forward pass through `FinancialTFM.term_structure` and reached into the
  fitted estimator's private context, bypassing `predict_proba` — the only place decision
  D5's correction was ever applied. Fourth-horizon calibration error falls **32×** (0.3677 →
  0.0114) on the same checkpoint and the same split, and mean AUC is unchanged to four
  decimals, as the rank-preservation property requires. `docs/results/FINDINGS.md` §28.
- `pyarrow` was declared only in the `kaggle` extra, so `uv sync --extra bench` broke the
  V4FinBench loader that every out-of-time finding depends on. CI never caught it because CI
  has no data and the test skips.

### Added

- **`.env` / `.env.example` for credentials**, mirroring `finkele-axiom`. Nothing sources
  `.env` automatically; commands that need it source it for that call only. First key is a
  read-only fine-grained Hugging Face token for gated public datasets, with the exact scopes
  documented — and with the note that scope **cannot** enforce this repository's licensing
  boundary, since HF grants every token read access to all public repo contents.
- **`prototype` context strategy** — Kostrzewa et al.'s prototype undersampling
  (arXiv:2605.10896 §5.1), implemented from their description for comparison. It is now the
  **recommended** construction: mean AUC 0.8143 ± 0.0038 and ECE 0.0072 out of time, against
  our query-conditioned retrieval's 0.8130 ± 0.0069 and 0.0119, while being 4× cheaper and
  keeping batch independence. `configs/best.yaml`. `docs/results/FINDINGS.md` §38.
- **`docs/paper/`** — a workspace for a potential paper: a claims ledger mapping every
  candidate claim to its evidence and status, an outline, related work, limitations, and the
  figure list with the command behind each. Nothing enters without a `FINDINGS.md` section
  number. Three of eight candidate claims are already recorded as superseded or retracted.
- **`fintfm-retrgroup`** — measures the retrieval grouping approximation against exact
  per-query retrieval, which §32 and §35 had relied on without quantifying. About 0.01 AUC at
  the portfolio level, but **one firm's cumulative PD moved 0.64** depending on its scoring
  batch (§37).
- **Configuration in one declared place.** Split years, row caps, context sizes, strategies,
  seeds, scoring thresholds, the scored arms and the prior's default-rate envelope now live in
  `src/fintfm/configs/default.yaml` instead of as literals across the experiment modules.
  Entry points take `--config`, deep-merged over the packaged default, with explicit flags
  still winning; runs print their config layers and record them as `config_sources` in their
  output JSON. Unknown keys are refused **by name** rather than ignored, and an overlapping
  train/test split is rejected before a run starts. Example overrides in `configs/`.
  - Not everything moved, on purpose: the `CENSORED` sentinel, the float32 numerical guards
    and the prior's accounting identities stay in code, and `configs/README.md` records why.
    The prior's rate constants stay declared beside their measured reasoning and are
    overridable by argument.
  - The `inference` section deliberately **mirrors** the estimator's literal defaults, so the
    library behaves identically without reading a file; a test fails if the two ever diverge.
  - CI now installs the built wheel into a clean environment and loads the packaged config
    from it, because presence in the archive is not the same as usable once installed.
- **A feature conditioner** (`rank` and `winsor`), applied before the model's normalisation.
  Worth **+0.086 mean AUC** to uniform context and +0.023 to retrieval on the V4FinBench
  out-of-time split, and improving AUC in seven of eight configurations across two further
  panels. 110 of 136 features here have a standard deviation more than ten times their
  interquartile range, which is what the model's mean/standard-deviation normalisation could
  not survive. `docs/results/FINDINGS.md` §35.
- **Checkpoints record which objectives they were trained on**, and a head that was never
  trained can no longer be served. The training loop optimises one objective per step, so a
  hazard checkpoint's classification head sat at random initialisation and `predict_proba`
  reported AUC 0.3745 with a stated 69% default rate against a 4.7% base — no error raised.
  §34, decision D11.
- **A `retrieval` context strategy**, building each query group's context from its nearest
  training rows instead of a blind sample. **+0.066 to +0.095 AUC over the best blind
  strategy** on the V4FinBench out-of-time split, Holm-significant at three of four horizons
  — the first accuracy gain in this project to survive a family-wise correction. Costs about
  2.5× the scoring time and gives up batch independence: a query's prediction depends on its
  group-mates. `docs/results/FINDINGS.md` §32.
- `holm_adjusted_p`, beside `holm_bonferroni`, which returns booleans. Reading those booleans
  as p-values inverted every verdict in the first write-up of §32.
- `FinancialTFMClassifier.predict_term_structure`, the corrected public path for a PD term
  structure, chunked exactly and coherence-preserving. Calling
  `FinancialTFM.term_structure` directly is now a documented defect.
- `fintfm.modeling.hazard.base_rate_shift` and `shift_cumulative_pd`, with monotonicity and
  cross-row rank preservation asserted rather than argued.
- `fintfm-ctxsweep`, sweeping context strategy against context size out of time. It exists
  because §29 reverses a decision, and a decision reversal has to be re-runnable.
- The out-of-time harness scores three hazard arms — corrected, **uncorrected**, and
  uniform-context — and prints mean predicted PD beside the observed rate. The uncorrected
  arm is permanent: the distortion is measured beside the fix rather than assumed absent.

### Changed

- **Query-conditioned retrieval is retired as a contribution** (§38). Against blind uniform
  sampling its +0.066 to +0.095 AUC gain is real and replicated; against the actual state of
  the art on this benchmark it is not a gain at all. Our best configuration uses their
  context construction. The residue is a negative result: on a low-default portfolio,
  conditioning the context on the query does not improve on selecting a structurally
  representative context once.
- **Retrieval group count does not matter**, correcting a single-seed sweep that showed
  +0.008 from tightening groups. Three seeds gave 0.8130 ± 0.0069 against 0.8126 — smaller
  than the seed spread (§38).
- **Defaults now follow measurement rather than the literature** (decision D10):
  `feature_transform="rank"` and `context_strategy="uniform"`, replacing `"balanced"`.
  Retrieval is the best strategy on all three panels and both prediction paths but stays
  opt-in, because it costs ~2.5× the scoring time and makes a prediction depend on its query
  group-mates. Every number recorded before 2026-09-09 used `balanced` and no transform.
- **Best out-of-time result: mean AUC 0.5869 → 0.8118** over the day, closing the gap to
  per-horizon logistic regression from 0.142 to 0.048, from three inference-time changes and
  **no retraining**. Still behind on discrimination and calibration.
- **Uniform context sampling beats balanced by 10-12 mean AUC points** on the V4FinBench
  out-of-time survival split, at every context size tested, reversing the ordering this
  project adopted from the literature. Twelve in-context defaults outrank 1,122. The class
  default is unchanged pending re-measurement on the binary path (decision D9).
- The base-rate correction is **not applied per query group** under retrieval. It is exact
  under label shift, which holds by construction only when context selection looks at `y`
  alone; retrieval selects on features. Applied per group it pushed the riskiest clusters
  down hardest and drove mean AUC to **0.3679, below chance**. A single pooled shift replaces
  it, which cannot reorder anything.
- §31's "more context does not help" is **corrected twice**: retrieved rows do help, but on
  three seeds retrieval is tied at 2,000 and 4,000 rows, so it raises the plateau's height
  rather than moving where it starts. 2,000 is the operating point; §32's monotone rise was a
  single favourable draw (§33).
- §26's headline is **retracted**: the synthetic prior's 1% base-rate floor was real and is
  now 0.195%, but it was never what caused the out-of-time failure. The retrain it prompted
  is worth +0.023 mean AUC and 2.7× better calibration, which the broken evaluation
  configuration had hidden. `docs/results/FINDINGS.md` §30.

## [0.1.0] — 2026-09-08

First release. A working prior-fitted tabular model for corporate credit risk, evaluated on
real corporate defaults, plus the strategic and evidentiary scaffolding around it.

### Model

- Three-stage architecture: cells are embedded individually, attend across columns within a
  row, then pool into one vector per row before rows attend to labelled context rows.
- **Column-order invariance and padding-width invariance**, both asserted in tests. The
  first architecture used a flat `Linear(2·max_features → d_model)`, which gave every
  feature a fixed weight slot and broke under permutation. That was the main limit on
  transfer and the reason for the rewrite.
- Query rows attend to context rows and to themselves, never to each other, so a prediction
  never depends on which rows share its batch.
- Normalisation uses context-row statistics only, so a query cannot influence its own
  normalisation.

### Priors

- A financial prior generating company balance sheets, P&L, a sampled macro regime, sector
  effects, distress-correlated missingness and default labels at sampled base rates from 1%
  to 30%.
- A generic structural-causal-model prior for nonlinear and multiclass structure.
- **Pretraining is entirely synthetic.** No real data can reach it, which is what makes a
  benchmark number auditable rather than merely asserted.

### Evaluation

- Real corporate-default evaluation on the UCI Polish companies bankruptcy panels, at 1, 3
  and 5 year horizons, CC-BY-4.0, evaluation only.
- **Calibration is a first-class metric.** AUC is rank-only and cannot see a stated 2%
  probability of default that occurs 15% of the time. Every result also carries Brier score,
  expected calibration error, reliability bins and recall at a base-rate operating point.
- Benchmark harness against logistic regression, random forest, gradient boosting and
  LightGBM.
- Phase 1 ablation harness that enforces matched compute across prior variants and refuses
  to report if parameter counts diverge.

### Findings

Six numbered findings in `docs/results/FINDINGS.md`, of which two changed the code:

- Balanced context sampling improves ranking but **inflated predicted default rates roughly
  threefold** (14.9% against a 4.7% actual rate), because an in-context model reads the base
  rate out of its context. AUC cannot see this, since every prediction inflates alike.
  Corrected by an exact label-shift adjustment to the logits, which leaves ranking untouched
  and reduced expected calibration error from 0.102 to 0.007 at the 3-year horizon. The
  1-year horizon, where it slightly overcorrects, is recorded rather than omitted.
- Firm-level financial data is absent from every major foundation model's pretraining corpus,
  because it sits behind commercial licences. Energy got foundation models because its data
  was free. That makes synthetic generation the only licence-clean route into this domain
  rather than a budget substitute.

### Documentation

- `docs/roadmap/STRATEGY.md`, the plan of record, with a falsifiable exit condition per phase.
- `docs/design/ARCHITECTURE.md`, `docs/design/DECISIONS.md` with seven decisions and what would reverse
  each, `docs/infra/COMPUTE.md` with measured throughput, `docs/competition/LANDSCAPE.md`, and
  `docs/research/REFERENCES.md` where every entry was verified against its source.
- `openspec/` with nine proposals and 42 tasks, and `docs/roadmap/NEXT.md` as the tiered queue.

### Known gaps

- **No CI.** Tests pass locally, on a tree that carries a cached download the repository does
  not. Covered by `openspec/changes/ci-and-release-gates`, including a retrospective check of
  this release.
- **All splits are random, not time-based.** Optimistic for credit data in exactly the way a
  validation committee looks for. No headline accuracy number should be published before
  `time-based-evaluation` lands.
- **One dataset.** Every real-data number rests on Polish firms in 2000-2013.
- **No result yet on the central question.** Whether the financial prior beats a generic one
  at matched compute is unmeasured; the run was in flight when this was tagged. Every number
  in the repository so far is smoke-test scale.
