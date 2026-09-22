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

**Executive digest, for anyone who reads one section of this file.** Ten TabArena entrants
opened from primary sources in one afternoon. The three findings worth carrying into any
future draft or conversation:

1. **The scale hypothesis is capped by evidence we didn't have to generate.** Nori's published
   curve (6M→100M params, +0.0049 R²) and TabPFN-2.5's own numbers (see below) both show
   parameter scale returning single-digit thousandths at this point in the curve, against our
   0.035 residual. §108's own scaling attempt (−0.0077, confounded) is consistent with this,
   not contradicted by it.
2. **Every top-14 TabArena rank is synthetic-pretrained (§110).** TabPFN-2.5 and its lineage,
   LimiX-2, Mitra-v2, TabICLv2 — all synthetic. The best real-data model (TabDPT-Turbo) is
   21st. Decision D2's synthetic-only constraint costs nothing measurable in rank; the frontier
   is *there*, not despite it.
3. **The lever nobody here has pulled is the prior, not the architecture.** §104 closed the one
   architectural knob that ever moved real-data accuracy; MITRA's entire contribution — rank 8,
   synthetic-only — is prior design (performance/diversity/distinctiveness) plus a tree-based
   prior this project's mixture lacks. `learn-from-peers` (48.x) is the resulting task list,
   led by 48.13 (tree prior) and 48.14 (score our own mixture against MITRA's criteria).

Also recorded as a process lesson: the synthetic-vs-real conclusion above was **rewritten three
times in one session**, once per arriving abstract, before anyone checked the rank ordering
that actually settles it (§110). Read the measurement before writing the claim.

Ten TabArena entrants opened on 2026-09-21 (full entries in `docs/REFERENCES.md`). Two of
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

### MITRA is the paper this project should have written, and it names our gap

**Amazon Science (2025).** *MITRA: Mixed Synthetic Priors for Enhancing Tabular Foundation
Models*, arXiv:2510.21204. **Rank 8 on TabArena, Elo 1729** (Mitra-v2 default).

**Synthetic-only, and state of the art.** It beats TabPFNv2 and TabICL on classification and
regression with better sample efficiency. That single fact is the counterweight to TabDPT,
ConTextTab and iLTM: the field's real division is not synthetic-versus-real, it is
**well-designed prior versus poorly-designed prior**.

Their framing is the one this project has been operating without: *"This shifts the focus in
tabular machine learning from model architecture design to the design of synthetic datasets, or,
more precisely, to the prior distributions that generate them. Yet the guiding principles for
prior design remain poorly understood."*

**Three criteria for a prior**, which is the contribution and is directly reusable:

| criterion | what it asks |
| --- | --- |
| **performance** | does a TFM pretrained on this prior alone do well on real data? |
| **diversity** | does it cover a wide region of task space? |
| **distinctiveness** | does it generate structure the *other* priors do not? |

They select SCMs for performance and diversity, then add **tree-based priors** — gradient
boosting, random forest, decision tree, extra tree — purely on distinctiveness: SCM-pretrained
TFMs "do not always generalize well to all types of data generated from TBPs".

**This project has no tree prior.** `PriorConfig` is `p_financial=0.7`, SCM for the remainder,
with `p_trivial`/`p_crossed` diagnostic-only and `p_regression` new. Every baseline we lose to
is a tree ensemble, real tabular data is full of threshold structure, and our prior generates
none of it. `mechanism-diverse-prior` asked this question; MITRA answers it with a measurement.

**And their priors are model-agnostic** — the same mixture improves both 1D row attention and 2D
element-based attention. So this is a prior-side gain that needs no change to the architecture
this project is memory-bound by (§79, §94, §108), which makes it cheaper than anything in
`factorized-attention` and testable on the existing checkpoint pipeline.

**The uncomfortable read.** §104 closed an architectural lever, §102/§103 found no axis, §108
lost on parameters, §93 lost on volume. MITRA's thesis is that the lever is the prior, and this
project's one genuinely distinctive asset *is* a prior — which has never been evaluated against
performance, diversity or distinctiveness, and is missing the one family MITRA singles out.

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

### TabDPT's "Bitter Lessons" appendix, scored against what we do

**Ma, Thomas et al. (2024).** *TabDPT: Scaling Tabular Foundation Models on Real Data*,
arXiv:2410.18164, Appendix A. They list what **did not** work while building an ICL TFM. Read
against this project's choices, one entry is a direct hit and three others contradict a peer
or a finding here — which is more useful than agreement.

| their negative result | our status |
| --- | --- |
| **Cell-token architectures with vertical+horizontal attention "proved more memory intensive"**; the simpler `(B, N, d)` form "permits a higher embedding dimension `d`" | **This is our architecture.** §79's 16 GB at N=2024, §84's unreachable `max_context=4000`, §94's OOM for 5M parameters at `n_rows=1024`, and §108's forced batch 4 are all the same wall. They chose the cheaper form and got a wider `d`. |
| Robust Scaler / Power Transform: "no improvement", "slowing the training process" | **Contradicted here.** §35's rank conditioning is worth **+0.086 AUC**, six of six configurations. Their tables are general OpenML; ours have 110 of 136 features with sd > 10× IQR. Domain, not technique. |
| NaN tokens no better than mean imputation; binary is-missing features "also failed" | **Agrees with our default** (`fillna(0)`, which is the mean post-normalisation). Note **Nori disagrees** — it uses learned mask embeddings. Two peers, opposite conclusions, so this is unsettled and cheap to test. |
| Class embeddings and proto-network query-class similarity "hurt the performance, especially on real data" | Relevant to `regression-and-multiclass`; we use a plain `Linear(d_model, max_classes)` head and should keep it. |
| Alternative `y_ctx` embeddings "did not lead to performance improvements **in large models with sufficient data**" | **Do not transfer this one.** §91 found removing per-cell label injection drops this model **below chance**. Their qualifier is the whole sentence — we are neither large nor data-sufficient. |

**The entry that matters most is the first, and their own explanation is the reason not to
panic.** They write: "While Hollmann et al. is able to make this architecture work, we suspect
that the **differences between synthetic and real data** are enough to change which
architectures are performant." TabPFN is synthetic-only and makes cell attention work; TabDPT
is real-data and does not. **This project is synthetic-only**, so their negative result may
simply not apply — and §78/§80/§91 measured cell attention as load-bearing here (+0.0486 AP,
and below chance without labels).

What does transfer is the **cost**: they got a wider `d` for the same memory, and this project
is memory-bound at every turn. That is `factorized-attention` (44.x)'s thesis with an
independent measurement behind it, and 44.x should cite this appendix.

They also frame the whole list as an endorsement of the Bitter Lesson — "efficient use of
computation and access to high-quality data are much more important for driving performance
than architectural manipulations". Against §104 (a closed architectural lever), §102/§103 (no
axis found), §93 (5× data, null) and §108 (parameters, negative), this project has now spent
most of its effort on the half they say matters least — while being locked out of the half they
say matters most, since "high-quality data" means real tables.

### TabSwift, which independently arrived at our attention mask and has two things we lack

**LAMDA-Tabular (2025).** *TabSwift: An Efficient Tabular Foundation Model with Row-Wise
Attention.* `github.com/LAMDA-Tabular/TabSwift`.

**Their split attention pattern is our row mask, arrived at independently.** "Training rows
(with their label embeddings added) attend to each other via self-attention. Test rows attend to
all training rows but not to each other." That is exactly the mask that makes this project's
query chunking *exact rather than approximate* — a property asserted in `tests/` and relied on
for every score above 2,048 queries. Independent convergence on a load-bearing design choice is
worth citing; it is not a contribution either of us can claim.

**They are the third data point against our 2D architecture.** TabSwift treats each row as a
single token — the `(B, N, d)` form — rather than embedding cells individually. With TabDPT's
Bitter Lessons and TabFlex's linear attention, three independent groups have chosen the cheap
form. TabPFN, MITRA and this project use element-level 2D attention. The split is not random:
TabDPT's own hypothesis is that synthetic-versus-real changes which architecture wins, and the
2D camp is the synthetic camp.

**Two mechanisms this project does not have:**

- **Register tokens** — learnable tokens prepended to the ICL sequence, providing "additional
  capacity for storing dataset-level information without interfering with the data tokens", and
  discarded before decoding. `explicit-task-representation` (41.x) asks whether this model forms
  a representation of *which task it is looking at*; register tokens are a published
  implementation of exactly that, and 41.2 should be designed against it rather than from
  scratch. Note this is a **dataset-level** slot — orthogonal to `column_id_dim`, which §104
  closed and which carries *column* identity.
- **Gated attention** — a learned `sigmoid(W·x)` gate on each attention head's output,
  head-wise or element-wise. Cheap, architecture-local, and untested here.

**One place this project is ahead, and the draft should say so.** TabSwift's regression head is
`Linear → GELU → Linear(1)`: a **scalar**. Ours reuses the classification head over quantile
bins (§106), so the output is a *distribution* — prediction intervals and quantiles come free,
and the predicted density can be bimodal. For loss given default, which is bounded and piles up
at both ends, a scalar head cannot express the shape at all. Their single checkpoint serving
both tasks is the better engineering; our head is the better statistics, and they are
independent choices.

### TabICL's repository, read for inference cost — the liability we measured today

Implementation detail from `github.com/soda-inria/tabicl`'s README, absent from the paper.
Recorded because **inference speed stopped being a footnote on 2026-09-22**: the deep-narrow
architecture arm (48.1) failed to score on TabArena at all, hitting `TimeLimitExceeded` after
8 of 27 datasets, and this project's median predict time is **8.6 s/1K against a field norm
near 0.1**.

| their mechanism | our position |
| --- | --- |
| **KV caching** — "cached key-value projections" for repeated inference | not implemented. Our context is fixed after `fit()` and re-projected on every `predict_proba` chunk, which is exactly the case a cache serves. The most concrete speed lever available. |
| **O(n² + nm²)** complexity, "10× faster than TabPFN-2.5", 50K×100 in under 10 s on an H100 | the cost model `factorized-attention` (44.x) predicts, now with a peer's measured number beside it |
| **class permutation** as an ensemble axis (`class_shuffle_method="shift"`) | independent support for `ensemble_label_swap`, which exists here to cancel a *measured defect* — §45 found predictions correlated **−0.62** with their label-swapped twin. They use the same operation as routine diversity; we use it as a repair, and the difference is worth stating rather than eliding. |
| **feature shuffling** for diversity (`feat_shuffle_method="latin"`) | deliberately **not** an axis here: predictions are column-order invariant by construction (decision D4, asserted at 1.19e-07 maximum change), so permuting would average identical members. Their needing it is evidence their architecture is not invariant. |
| **outlier removal** by z-score threshold before scaling | we rank-condition instead (§35, +0.086 AUC). Same problem, and ours is measured on financial tails specifically. |
| `support_many_classes=True` beyond 10 | our ceiling is also 10 (§99); EXAONE's is too. Three independent projects at the same limit. |

**The actionable item is KV caching**, and it is not an optimisation in the discretionary sense:
a configuration that cannot finish the benchmark is not a candidate regardless of its accuracy,
which is what 48.1's failure demonstrated. Filed as task 48.20.

### TabICLv2's prior appendix, read against `prior/scm.py` line by line

**Soda-Inria (2026).** Appendix E of the same paper. Kept separate from the main entry above
because it is implementation detail, not a claim, and because it is the single most directly
actionable thing in the whole sweep: their random-graph SCM prior is architecturally the same
idea as `sample_scm_task`, described in enough detail to diff against.

**What `prior/scm.py` has:** a fixed-depth layered graph (1–5 dense layers, not a general DAG),
5 fixed activations (`tanh`, `sin`, ReLU, identity, signed-sqrt), independent sparse Gaussian
edge weights, quantile or random-cutpoint discretisation for the label, and — as of §106/46.4 —
four target shapes for regression. That is the whole generative vocabulary.

**What theirs adds, concretely, ranked by expected value per line of code:**

- **Random graph connectivity via `sigmoid(A + Bᵢ + Cⱼ)` with Cauchy `A, Bᵢ, Cⱼ`** (Appendix
  E.4), rather than our fixed dense layers. Their stated reason — Cauchy's heavy tails yield
  "higher probabilities of exceptions to the rule" — is exactly the kind of task-difficulty
  diversity §42 already established this project needs (a prior of only-easy tasks teaches
  nothing). This is a **five-line change** to `sample_scm_task`'s edge sampling and the highest
  ratio of expected gain to implementation cost in this entire sweep.
- **`RandomTreeFunction`**: ensembles of oblivious (CatBoost-style) trees as a node function,
  split dimension chosen proportional to feature standard deviation. This is a from-scratch,
  citable implementation route for **task 48.13's tree-based prior** — MITRA names tree priors
  as necessary on distinctiveness grounds; TabICLv2 publishes exactly how to build one without
  needing an actual gradient-boosting library in the generation loop.
  **They note explicitly that TabPFNv2 uses single trees; they moved to ensembles because
  single trees "facilitate efficient computations" only up to a point** — worth reading before
  choosing between single-tree and ensemble for 48.13, rather than guessing.
- **21 fixed activations plus 4 parametric ones** (Appendix E.9), against our 5. Their
  activation set includes `rank`, `softmax`, `one-hot argmax` and `argsort` — order-statistics
  functions with no analogue in ours, which is notable given §35's own finding that this
  project's *inference-time* feature conditioning (`feature_transform="rank"`) is worth +0.086
  AUC. If rank-like transforms help at inference, a prior whose *label mechanism* never uses
  one may be under-representing exactly the structure real financial ratios have.
- **Correlated scalar sampling** (Appendix E.2) — hyperparameters like categorical cardinality
  drawn from a shared per-name distribution rather than independently, so different aspects of
  one synthetic dataset covary the way real datasets' properties do. Cheap and orthogonal to
  everything else here.
- **A filtering mechanism** (Appendix E.14, "similar to Dong et al. 2025") that rejects sampled
  graphs. Independent confirmation of the MITRA/Nori learnability-filter idea already filed at
  task 48.5 — a third group has now converged on filtering the prior rather than only widening
  it.

**What is deliberately not adopted from this section, and why:** the GP/random-Fourier-feature
function class (Appendix E.8, `RandomGPFunction`) requires a real spectral-density
construction to sample from correctly, and a rough approximation would silently produce a
different task distribution than intended — the exact failure mode `CLAUDE.md`'s "prior must
span difficulty" rule exists to catch. Not worth the implementation risk before the cheaper
items above are tried and measured.

**Their own admitted bug is worth recording, because it undercuts nothing.** Appendix E.2
states correlated categorical sampling "was not used due to a bug" — left in the paper rather
than silently fixed and omitted. A useful norm to hold this project to as well.

### TabICLv2 explicitly beats a real-data-adapted model with a synthetic-only one

**Soda-Inria (2026).** *TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation
Model.* [arXiv:2602.11139](https://arxiv.org/abs/2602.11139). Code **BSD-3-Clause**
(`src/tabicl/forecast` separately Apache-2.0, derived from TabPFN-TS); weights also
**BSD-3-Clause**, checked independently via the HF API on 2026-09-21 rather than trusting the
README, which states code licensing but not the weights' — the checked-separately habit paying
for itself again.

**The headline claim is the sharpest data point for §110 yet.** TabICLv2, **untuned**, beats
**RealTabPFN-2.5 — hyperparameter-tuned, ensembled, *and* fine-tuned on real data** — on TabArena
and TALENT. This is not "synthetic ties real"; it is synthetic-only-and-untuned beating
real-adapted-and-tuned outright. §110's rank ordering already showed the top 14 are
synthetic-pretrained; this is the same fact stated as a head-to-head.

**Three engineering pillars, useful independent of the accuracy result:**

- **A new synthetic-prior engine "designed for high pretraining diversity."** Cited by the
  paper's own related-work section alongside MITRA's prior-mixture work and LimiX's
  hierarchical SCMs, so this is now the third independent group treating prior *diversity*, not
  scale, as the lever — directly reinforcing task 48.14, which asks whether this project's
  prior has ever been measured on that axis at all.
- **A "scalable softmax" claimed to improve generalisation to larger datasets "without
  prohibitive long-sequence pretraining."** Mechanism not available from the abstract; worth a
  follow-up read once the primary claim (below) is scoped, since "generalise to more rows
  without training at that length" is close to this project's own `max_context` ceiling (§79,
  §84, §94, §108's batch-4 wall).
- **Muon replacing AdamW** as the pretraining optimiser. This project has never varied its
  optimiser as an axis; worth a citation the next time a training-loop change is considered,
  alongside the schedule-free optimiser already filed at task 48.9.
- **Million-row datasets under 50 GB GPU memory.** A concrete number to benchmark this
  project's own `n_cell_blocks=1` memory wall against (§79's 16 GB at N=2024) — not a fair
  comparison as stated, since the architectures differ, but a number worth having when
  `factorized-attention` (44.x) needs a target to beat.

**Their own related-work paragraph is a ready-made citation for this project's prior-family
taxonomy**, more current than anything already in `RELATED_WORK.md`: TabPFN uses SCMs;
TabICL/TabForestPFN mix in tree priors; MITRA studies prior-design principles and proposes
mixed SCM+tree priors; TabPFNv2 enriches DAG construction; **LimiX introduces hierarchical SCMs
with controllable difficulty** — a mechanism this project's own prior does not have and
`prior-width-and-fidelity` should read against; Drift-Resilient TabPFN modulates SCM parameters
over time for distribution shift, which is directly relevant to `temporal-financial-prior`.

### Causilo (Nums AI): a self-reported rank that does not match this project's own leaderboard run

**Nums AI (2026).** *Causilo.* `github.com/nums-ai/causilo`. **On this project's own live
leaderboard run: rank 6, Elo 1751** (`Causilo (default)`, 1751±115). Their README claims
"Elo position 1, 1792.9 (Full) / 1817.4 (Lite)" — a materially different, and higher, number.
**Neither figure is corrected here**; the discrepancy is recorded rather than resolved,
following the same rule that caught EXAONE's stale self-report a session prior: quote from a
re-derivation of the measurement in hand, never from a vendor's own framing, and when the two
disagree say so rather than picking one silently. Possible causes not distinguished — a
different leaderboard snapshot, a different subset, a self-selected framing of "Elo position"
— and this project has no basis to adjudicate which.

**Fourth confirmed instance of the licence-split trap** (after Google TabFM/TimesFM at §24,
and EXAONE above): code Apache-2.0, verified from the repository's own `LICENSE` file. Weights
under a **separate "Causilo License v1.0"**, verified from the README's explicit statement
rather than assumed from the permissive code licence — "Non-commercial research and free
research redistribution are permitted... Commercial or production use, and hosted/API/SaaS
services whether paid or free, require separate licenses." TabArena's own published numbers
may be cited; the checkpoint may never be loaded here.

**No architectural detail was available from what was read** — the page returned no
description of method, training data, or whether the prior is causal despite the name. Not
enough to compare against this project's own choices; noted as an open item rather than
guessed at.

**The user's observation that this resembles Neuralk's Seldon is plausible on the surface —
a causally-named proprietary tabular foundation model — but unverified and not investigatable
further here.** `docs/LANDSCAPE.md` records Neuralk (Seldon) as proprietary with no public
technical report; nothing about Causilo's method has been read that would confirm or refute a
relationship, and `CLAUDE.md`'s boundary against Neuralk's code/weights/training data is
unaffected either way — Causilo is a separate legal entity (Nums AI Inc.) under its own
licence, evaluated on its own terms.

### EXAONE Tabular: the third confirmed instance of the licence trap, and above us at rank 7

**LG AI Research (2026).** *EXAONE Tabular.* `github.com/LGAI-Research/EXAONE-Tabular`,
arXiv:2608.25774. **On this project's own live leaderboard run: rank 7, Elo 1741** (its own
README's self-reported 1755/2nd is a different, presumably earlier, snapshot — re-derived from
this session's own data rather than quoted from the source, per the standing rule on re-reading
a number before quoting it).

**Code and weights checked separately, and they diverge — a third confirmed instance of the
exact trap `docs/FINDINGS.md` §24 already names for Google's TabFM/TimesFM.** Code is
BSD-3-Clause-LG AI Research License (permissive, commercial use fine). Weights are
**"EXAONE AI Model License Agreement 1.2 - NC"**, verified from the licence file itself rather
than the model card summary: §3.1 prohibits commercial use of the Model **and of its Output**
— not only redeploying the weights, but *using predictions from them* commercially. `tabarena`
may evaluate it because TabArena's own published leaderboard numbers are being read, not the
checkpoint run; this project may never load `LG-AI-Research/EXAONE-Tabular` weights itself.

**Architecture: Cross-axis Summary Transformer (CAST), ~20.8–21.1M parameters** — item-summary
tokens (shape `[batch, items, 3, width]`, concatenated along the feature axis) and
feature-summary tokens (`[batch, 32, features, width]`, concatenated along the item axis). This
is a **fourth** independent group building a compressed cross-axis summary mechanism, after
TabSwift's register tokens and TabPFN-2.5's "thinking" rows — `explicit-task-representation`
(41.2) now has four published implementations to design against rather than one.

**"SSMax" attention normalisation** is named but not documented in what was read; a primary-
source detail worth chasing only if 41.2 is actually scoped, not before.

**Native support for only 10 classes**, with larger label spaces needing ECOC decomposition —
the same ceiling our own multiclass checkpoint has (`--max-classes 10`, §99), independently
arrived at, which is at minimum a sign the ceiling is not an obviously wrong design choice.

### TabPFN-2.5, the line at the top of the board, read for engineering rather than headline

**Prior Labs (2025).** *TabPFN-2.5: Advancing the State of the Art in Tabular Foundation
Models.* [arXiv:2511.08667](https://arxiv.org/abs/2511.08667). TabPFN-3.5 (the checkpoint on
today's leaderboard, rank 1, Elo 1812) is the next generation past this report; the report
itself documents TabPFNv2 → 2.5, which is instructive regardless of which exact checkpoint
currently sits on top. **Purely synthetic** — "Like the original TabPFNv2, TabPFN-2.5 is
trained purely on synthetically generated data" — with a *separately released* real-data
variant, Real-TabPFN-2.5, fine-tuned on 43 OpenML/Kaggle datasets **deduplicated against the
full TabArena suite**. That dedup step is the discipline §109 argues this project needs and
most peers do not disclose; naming it here is a citation worth having on hand.

**What changed from v2, concretely — six items, none of them "bigger":**

| change | detail | relevance here |
| --- | --- | --- |
| depth | 12 → 18 layers (regression), 24 (classification) | depth, not width — §108 scaled width and lost; task 48.1 already tests the opposite corner |
| feature grouping | group size 2 → 3, embedding several features together | a cheap lever this project has never had a knob for |
| regression encoder | linear → 2-layer MLP | trivial to test against our binned head (§106) |
| **"thinking" rows** | 64 additional learned rows appended to the input, present only at pretraining, acting partly as **attention sinks** so the model can learn to ignore rows | closest published relative of `explicit-task-representation`'s task token and TabSwift's register tokens — a third independent group solving the same problem a different way |
| preprocessing | robust scaling + soft clipping + quantile transforms + standard scaling, combined and diversified across ensemble members | our `feature_transform="rank"` (§35, +0.086 AUC) is one point in this space; theirs argues for combining several rather than picking one |
| **surrogate hyperparameter search** | used TabPFNv2 itself as a regression surrogate over ~50 hyperparameters, from 100 real runs to 10,000 evaluated configurations | "TabPFN tunes TabPFN" — a general pattern for expensive prior/architecture sweeps that this project has never used and every `openspec` sweep task pays for by brute force |

**Distillation is the one idea here aimed at deployment rather than accuracy**, and it is the
answer to a question this project has not yet had to answer: their engine converts a fitted
TabPFN-2.5 into a dataset-specific MLP or tree ensemble — no in-context learning, single-row
inference, "orders-of-magnitude lower latency". Worth returning to once `FinancialTFMClassifier`
has a customer whose deployment constraints (latency, interpretability, a regulator's model
inventory) rule out shipping the foundation model itself.

**Threshold tuning and temperature scaling are named as separate, off-by-default steps** — "all
classification results in this report are computed using uncalibrated, default scores". That
discipline is one `docs/paper/LIMITATIONS.md`'s multiclass entry already asks for (temperature
scaling, currently unimplemented, with multiclass 94th of 94 on log loss despite decent AUC).

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
