# Improvement plan, November 2026

Written 2026-11-02, against `docs/results/FINDINGS.md` up to §202 and commit 82238f6. Reading and
planning only: no number below was produced for this document. Every FinTFM figure is quoted
from the finding named beside it (MEASURED there); every cost is ESTIMATED unless it says
otherwise. Peer facts were read from primary sources (arXiv PDFs converted with `pdftotext`,
official repository files and model cards fetched raw), never from a summariser
(`docs/results/POSTMORTEM.md`); `docs/paper/RELATED_WORK.md` carries the full peer notes with
page and section locations.

**Support status, 2026-11-13.** The project states it is no longer actively supported from mid-November 2026; experiments not yet run will not be run by this project.

**Status note, 2026-11-13.** The plan below is kept as written on 2026-11-02. Each of the eight
experiments carries a dated status line, read on 2026-11-13 from `docs/results/FINDINGS.md`
(§203-§224) and the openspec task files, and the lever tables in section 1 are not updated: for
levers tried since, read the status lines. Summary: experiments 1 and 3 (partial) and 4 (per-leaf
half) have been run and 1 and 4 closed by their own kill rules; 8 is half run (Muon harmful, RBF
value embedding harmful, length scaling queued); 2 was overtaken by a different stack on
TabArena-Lite; 5, 6 and 7 have no result.

## The question, and the answer this file arrives at

**Question.** What is most likely to make FinTFM (synthetic-only) (a) beat simple baselines,
untuned logistic regression and random forest, on TabArena, and (b) close the gap to CatBoost on
V4FinBench?

**Hypothesis tested here, not assumed:** the gain lies in the prior or in how the head reads the
representation, not in scale.

**Answer, in order of confidence:**

1. **Scale is closed at this project's budget**, by five of its own measurements and by one
   peer's negative result. That half of the hypothesis holds.
2. **The largest measured gains are at inference, not in the prior**: partitioned contexts
   +0.0375 mean AP (§169), the linear probe +145 Elo on TabArena-Lite (§132, same cache). The cheapest next steps
   are therefore inference-only and can run today.
3. **"The head under-reads the representation" is not yet established.** Every probe result is
   confounded with label count: the probe was fitted on about 2,150 positives, the head
   conditions on about 7 (`docs/design/DESIGN_REVIEW_2026-10-16.md` §1.6). The matched-label
   test (DR3.4) is built and has never been run. It is experiment 1 because it decides which
   half of the hypothesis to fund. (2026-11-13: run as DR3.4 and closed. At equal labels the
   probe ties the head, −0.0005 mean AP cross-fitted, 3 of 5 folds positive (§209); on the
   prior's own tasks the head beats the probe, −0.0046 AP, 75 of 200 tasks (§203). The "head
   under-reads" half is not supported; §130-§136 measured label count.)
4. **The prior points in opposite directions on the two goals.** The general trunk lifts the
   proxy suite and harms V4FinBench (§191); ten times the tasks does the same (§188). Goals (a)
   and (b) probably need different checkpoints, and the plan treats them separately.

## Corrections to the brief, re-read from the source

| brief said | source says | where |
| --- | --- | --- |
| linear probe beats the head by +0.039 AP | +0.0390 at `encode_rows`, which never reaches `self.encoder`; **+0.0214 at the head's own input**, 4 of 5 folds; both fitted on a 20,000-row stratified sample, so confounded with label count | §130, §131, design review §1.6 |
| fine-tuning 0.2063 | 0.2062 in §163 (Mac); 0.2063 in §202's Kaggle rerun of the same 200-step recipe | §163, §202 |
| linear-probe variant 910 Elo | 910 against the own head's 765 on §132's cache; the current own-head figure, 756, is a later run (NaN passed through) | §132, `docs/results/TABARENA.md` |

The other figures in the brief match their sources: 0.1986 (§130's own-head mean), 0.2362
(§169), 0.4005 (§167), 756 Elo and 94th of 95 [§225: one place lower] (`docs/results/TABARENA.md`, 2026-10-19 run), 936 for
`Linear (default)` on §132's cache.

Two facts the plan depends on, also from the source:

- **On V4FinBench the gap is to boosted trees, not to logistic regression.** Full panel, fold 0:
  FinTFM 0.2113, logistic regression 0.1671 (§162); five-fold logistic regression 0.1614
  (§167). Goal (b) is CatBoost (0.4005) and LightGBM at 1,000 trees (0.3908, §167).
- **The V4FinBench label is gated.** 3,586 of 3,587 positives have `Insolvency_flag = 1`; a
  uniform 2,000-row context carries about 7 positives (§168, §169). A tree finds the gate with
  one split. The 0.034 AP that `operational_status` gives CatBoost is worth about 0.002 to FinTFM
  (§167, §202).

## 1. What has and has not been tried

V4FinBench figures are five-fold mean AP on the full panel unless marked. The pretraining noise
yardstick is 0.0117 mean AP (§173). "Proxy" is the licence-clean proxy suite, not yet validated
against TabArena (DR4.2).

### Scale and volume

| lever | tried? | result | § |
| --- | --- | --- | --- |
| 5x task volume (240,000) | yes | −0.0012 AP | §93 |
| 5.7x parameters at matched tasks | yes | −0.0049 | §114 |
| 10x task volume (48,000 → 480,000), two seeds | yes | −0.0109 and −0.0103: null on V4; proxy +168 and +186 Elo | §176, §188 |
| depth 8 at 480,000 tasks | yes | −0.0047: null on V4 and proxy | §193, §195 |
| width 192 at 480,000 tasks | yes | +0.0003: null on V4 and proxy | §202 |
| multiclass volume, 480,000 tasks | yes | null on the multiclass proxy | §202 |
| 2M+ tasks | no | not funded: E1's "helps" branch closed | §176 |

### Prior

| lever | tried? | result | § |
| --- | --- | --- | --- |
| financial prior against SCM only | yes | +0.049 AUC, credit only; pure SCM wins on Polish and Taiwan | §14, §75 |
| tree-structured prior | yes | +0.0094 TabArena, −0.0221 AP V4; confounded with the financial share | §115, §116, §166 |
| nine task families at weight 0.3 | yes | worse on V4 | §141 |
| realism augmentations | yes | small net negative | §144 |
| learnability filter | yes | small net negative | §145 |
| widened SCM (Cauchy-style difficulty spread) | yes, statistics only | difficulty spread +45%, never scored alone on V4 | §143 |
| prior v2: gate + sign flip + feature bands (arm a) | yes | 0.1762 against 0.1975: harmful | §180, §191 |
| general trunk, alone or before financial (arms b, c, d) | yes | V4 0.0713 to 0.1616: harmful; proxy 721 to 764 Elo against 548 | §191 |
| gated label alone, without sign flip or bands | **no** | arm a bundles three changes | §180 |
| class permutation in pretraining | yes, multiclass only | candidate, adopted for the multiclass recipe | §186, §197 |
| per-task feature transforms in pretraining (AU5.6) | **no** | built, `feature_transforms` | config |
| pretraining contexts above 921 rows | **no** | every checkpoint trained on 256/512/1,024-row tasks, contexts 76-921 | §162 |

### Architecture and optimiser

| lever | tried? | result | § |
| --- | --- | --- | --- |
| two-way cell attention with cell labels | yes | +0.0520 at matched prior | §172 |
| column identity vectors | yes | needed for column-specific rules; +0.0009 AUC on V4 | §56, §59 |
| attention pooling | yes | null | §52 |
| mask embedding | yes | uniform loss | §147 |
| induced cell rows; 4 readout tokens + MLP head | yes | −0.0186; −0.0261: harmful | §178 |
| schedule-free optimiser | yes | clear loss | §146 |
| Muon (AU5.17) | **no** | built behind `--optimizer muon` | audit tasks |
| length-scaled attention (QASSMax-like, AU5.17) | **no** | built behind `--attn-length-scale` | audit tasks |
| Fourier/RBF value basis and ECDF cell feature (AU5.16) | **no** | built behind `--value-embedding`, `--value-ecdf` | audit tasks |
| classification readout by attention over context labels | **no** | not built | |

### Head and representation

| lever | tried? | result | § |
| --- | --- | --- | --- |
| linear probe at `encode_rows` | yes | +0.0390, 5 of 5 folds (label-count confounded) | §130 |
| linear probe at the head's input | yes | +0.0214, 4 of 5 (label-count confounded) | §131 |
| same probe on TabArena-Lite | yes | 910 Elo against 765 | §132 |
| MLP probe | yes | beats own head, not the linear probe | §135 |
| replicate on a second checkpoint | yes | replicates | §136 |
| probe fitted on the context rows only (DR3.4) | **no** | built behind `--probe-rows context`, cross-fitted | design review DR3.4 |
| probe against head on the prior's own tasks | **no** | `scripts/head_probe_prior.py`, committed 82238f6, not run | |

### Inference

| lever | tried? | result | § |
| --- | --- | --- | --- |
| rank transform | yes | +0.086 AUC; Yeo-Johnson loses to rank on every fold | §35, §129 |
| context size, full panel | yes | nearly flat; in-range 500 rows does not beat 2,000 | §83, §162 |
| context size, TabArena-Lite | yes | 500: 702, 750: 745, 1,000: 756 Elo | `docs/results/TABARENA.md` |
| 8 members | yes | V4 +0.0058, 3 of 5; proxy candidate 13/2 | §187, §196 |
| 8 members with label swap; with transform cycling | yes, proxy | 11/5, not candidates | §187 |
| retrieval context | yes | V4 collapses to 0.0126 | §194 |
| fixed positive count in the context (DR3.3) | yes | +0.0024 and +0.0055: null | §200 |
| partitioned contexts, depth 1/2/3 | yes | 0.2347 / 0.2361 / 0.2357; depth 2 +0.0375, 5 of 5 | §169, DR3.11 |
| partition with 8 members, bagged trees, per-leaf fine-tuning | **no** | arm configs committed 0e27592 | DR3.11 |
| partition on TabArena-Lite | **no** | | DR3.11 |
| fine-tuning 200 steps | yes | V4 +0.0076 (0.2062); proxy 16 of 16, +307 Elo on one split | §163, §192 |
| fine-tuning on TabArena-Lite | **no** | | |
| SVD members, class shift on binary tasks | **no** | built, `ensemble_svd_components`, `ensemble_class_shift` | config |
| FinTFM inside CatBoost | yes | +0.0039: null | §201 |
| FinTFM inside LightGBM, credit suite | yes | +0.011 AP on SBA (2026-11-13: flat on SBA once the booster is tuned, §213; lifts a tuned booster on ULB, 3 of 3) | §189, §213 |
| TabArena imbalanced subset (AU5.18) | **no** | | audit tasks |

## 2. The field, read for prior, volume and inference

Full notes with quotes and page locations are in `docs/paper/RELATED_WORK.md`, "Prior row
counts, task volume and inference defaults, read 2026-11-02". The comparison that matters:

| | FinTFM | field (primary source) |
| --- | --- | --- |
| rows per pretraining task | 256, 512 or 1,024; contexts 76-921 (§162) | TabPFN v2 uniform up to 2,048; TabICL 1,024 then 1K-40K then 40K-60K; TabICLv2 1,024 then 400-10,240 then 400-60K; Mitra-v2 160-5,120 support plus 1,280 query; TabFM 2,048 to 16,384 |
| rows at inference | 2,000 (V4), 1,000 (TabArena adapter) | Mitra-v2 caps prediction-time support at 32,768 rows |
| tasks seen | 48,000 (published), 480,000 (E1) | TabPFN v2 about 130M; TabICL about 82M; TabICLv2 about 35M; Mitra about 45M; Mitra-v2 about 28.7M nominal; LightPFN 7.68M |
| prior families | financial (70%) + MLP-SCM; tree, task-family and gated sources built | every leader mixes MLP-SCM with trees (TabICL 30% XGBoost-SCM; Mitra-v2 0.30 across five tree generators; TabICLv2 eight function types including CatBoost-style tree ensembles); TabICLv2 and Nori filter out tasks an ExtraTrees cannot learn |
| label into the model | `y_emb` at row level and `cell_labels` into every context cell | TabICLv2 target-aware embedding (early target inclusion, about 100 Elo in its leave-one-out ablation); TabFM, Mitra-v2 add a target embedding to labelled rows |
| classification readout | `nn.Linear(d_model, max_classes)` | TabPFN-3: attention-based retrieval decoder, "soft nearest-neighbor retrieval over the in-context training set"; TabPFN-3.5: "read out by attention over the one-hot training labels"; TabICL, TabFM: two-layer MLP |
| default ensemble | 1 member | 8 in TabPFN (current code), TabPFN-3/3.5, TabICLv2 and TabDPT; 4 in LimiX and TabPFN-3.5-Fast; TabICL's paper used 32 |
| ensemble axes | column-identity draw | feature shuffle and class permutation in all of them; preprocessing diversity in TabPFN v2/2.5/3, TabICL(v2), LimiX; TabPFN-3.5 drops extra transforms and keeps permutations only |
| preprocessing | rank (§35) | TabICLv2 `["none", "power"]` with z-score outlier clipping at 4.0; TabPFN v2 quantile, SVD, Yeo-Johnson, outlier removal at 12 SD; TabPFN-3.5 none beyond its in-model ECDF and Fourier encodings |
| fine-tuning | 200 steps, off by default | Mitra-v2 on TabArena: 50 fine-tuning steps per fold inside 8-fold bagging, and its "default" means fine-tuned, not zero-shot |

Three readings follow, each stated with its limit.

**Volume.** FinTFM trains on two to three orders of magnitude fewer tasks than any leader. But
this project's own 10x step was null on credit at two seeds, Mitra-v2 reports that longer or
continued training "left the board flat" (Appendix C), and LightPFN's gain from 256,000 to 7.68M
draws was +1.66 AUC points on 55 non-TabArena datasets. Volume plausibly matters for goal (a),
where 480,000 tasks moved the proxy, and has shown nothing for goal (b). The plan does not fund
volume first.

**Row counts.** This is the largest structural difference and it has never been varied here:
every leader trains on contexts of at least 2,048 rows in some stage, and FinTFM never past 921
while being scored at 1,000 or 2,000. §162's full-panel result (in-range 500 rows does no better
than 2,000) argues that inference past the trained length is not itself the harm. It does not
say whether a model trained at 2,000-4,000 rows would use them better. Experiment 6 tests that.

**Readout.** The newest Prior Labs models replaced the fixed output head with retrieval over the
context's own labels. That is the architectural form of what the linear probe does post hoc:
fit the decision rule to this task's labelled rows instead of reading fixed class slots. It is
the strongest peer support for the "head" half of the hypothesis, and it is untested here.

## 3. Ranked experiments

Costs: one 48,000-task screening arm is 6,000 steps at batch 8. At ~0.53 s/step on Kaggle T4x2
that is about 0.9 GPU-hours (ESTIMATED; §197 MEASURED 3,285 s for 6,000 steps). One 480,000-task
arm is about 9.3 hours (§202's multiclass arm MEASURED 33,505 s for 60,000 steps). Inference-only
arms run on this Mac's CPU or MPS and cost no GPU-hours; their wall time is not estimated here.
Every V4FinBench verdict uses the company-level paired bootstrap and, for pretraining arms, the
0.0117 yardstick (§173). TabArena-Lite reports and never selects (DR4.1); goal (a) selects on the
proxy suite first.

**Inference-only and runnable today: 1, 2, 3.** Experiment 4 is inference-time but needs a GPU.

### 1. Matched-label probe: does the head under-read, or did the probe just see more labels?

**Status, 2026-11-13: done, kill rule met (§209, §203).** V4FinBench, five folds, published checkpoint: head 0.1986, head cross-fitted 0.1981, probe on the 2,000 context rows 0.1976 mean AP, so the context-only probe minus the cross-fitted head is −0.0005 (3 of 5 folds positive; point values, no bootstrap in §209). The 20,000-row stratified probe is 0.2213, so label count explains the earlier gap. Prior-task arm: −0.0046 AP for the probe against the head, 75 of 200 (§203). Experiment 5 is not built, as this experiment's own rule required (openspec DR3.4 closed 2026-11-05).

- **Hypothesis.** The head extracts less than a linear readout of the same representation would,
  at the same labelled rows.
- **Change.** None to weights. Run `representation_transfer` with `--probe-rows context`
  (DR3.4, cross-fitted: each fifth of the context is represented against the other four-fifths,
  and the head is scored the same way as `own_head_crossfit`), five folds, published checkpoint.
  Beside it, `scripts/head_probe_prior.py --model runs/v4-cellattn-labels.pt --tasks 200` on the
  prior's own tasks.
- **Cheapest falsification.** This is it: CPU, no training.
- **Cost.** 0 GPU-hours.
- **Kills it.** The context-only probe does not beat `own_head_crossfit` (mean Δ ≤ 0, or fewer
  than 4 of 5 folds positive). Then §130-§136 measured label count, the head half of the
  hypothesis falls, and experiment 5 is not built. If the probe wins on V4 but not on prior
  tasks, the head is fitted to the prior and the gap is a prior-to-real shift; if it wins on
  both, the head is the defect.

### 2. Goal (a): a general-tabular checkpoint with a linear readout

**Status, 2026-11-13: not run as specified; superseded in practice by the readout stack on the published checkpoint.** All figures are TabArena-Lite, label "FinTFM with a trained linear readout", and breadth only. Probe plus squash 969 Elo (§212); plus partition 1,000.2 Elo, level with RF (default), not above it (§215); binary 974 on 30 datasets (§220); multiclass 865 (§224); regression 578 (§223); all 51 datasets 903 Elo, 107th of 110 (§224), with Linear (tuned) at 919 and RF (default) at 1,000. The 480,000-task checkpoint is null on multiclass (§222). The proxy suite that this experiment selects on failed its validation bar, rho +0.717 against 0.8 (§216, DR4.2), so the plan's selection step no longer stands. Success criterion stated above (above `Linear (default)` on §132's cache) is superseded by the bar in `docs/roadmap/TABARENA_BAR.md`, not met on the 51-dataset run.

- **Hypothesis.** For TabArena the right checkpoint is a general-trunk one, not the credit one,
  and the probe's gain stacks with it. The two strongest general-tabular signals measured here,
  the trunk prior (proxy 764 against 548, §191) and the probe (+145 Elo on Lite, §132), have never been
  combined.
- **Change.** Score the existing `dr24-c-s0.pt` (trunk then financial v2) and `dr24-b-s0.pt`
  (trunk alone) on the proxy's 16 binary tasks with the default head, with `n_ensemble 8`, and
  with a linear probe on the pre-head representation (a probe arm has to be added to
  `proxy_suite`, mirroring the fork's `fintfm_linprobe` adapter). The winner, and the published
  checkpoint with the probe, then go to TabArena-Lite for reporting.
- **Cheapest falsification.** Proxy, CPU, existing checkpoints.
- **Cost.** 0 GPU-hours.
- **Kills it.** On the proxy, neither trunk checkpoint with the probe beats the published
  checkpoint with the probe (per-task sign test p ≥ 0.05). Success for goal (a) is a Lite Elo
  above `Linear (default)` on the same cache, same day; 910 is 26 Elo short of 936 on §132's
  cache, inside every interval recorded.
- **Caveat stated now.** A probe fitted on all training rows is a different method from
  in-context prediction, and FinTFM must carry the multiclass and regression datasets before a
  TabArena submission counts (`docs/roadmap/TABARENA_BAR.md`).

### 3. Goal (b): stack the partition with members and bagging

**Status, 2026-11-13: mostly not run.** Narrowed by the owner on 2026-11-08: the 8-member and 8-bag arms are dropped; only `task_dr311_bag4_f23` stays queued as a futility test, and openspec DR3.11 (2026-11-11 note) still lists it as not run, waiting on Kaggle GPU quota. Related results: partition on TabArena-Lite 888 Elo against 751 for the same-GPU default, 23 of 27 datasets better (§212); the probe per leaf on V4FinBench is −0.0570 mean AP on 0 of 5 folds (§218), traced to a sampler defect in the harness (§219), rerun pending, so no V4FinBench verdict exists for it. The partition's own V4FinBench result stays 0.2362 (§169, §205).

- **Hypothesis.** The gap to CatBoost is mainly context composition around the gate (§168), and
  §169's partition has headroom: one context draw per leaf is noisy (§162 showed one 500-row
  draw collapse a fold to 0.0684).
- **Change.** The committed DR3.11 arms: `configs/arms/dr311-ens8.yaml` (8 members per leaf),
  `dr311-bag4.yaml` and `dr311-bag8.yaml` (4 or 8 depth-2 trees on 60% bags, leaf scores
  averaged), each against §169's `runs/dr31/partition`.
- **Cheapest falsification.** Five-fold V4, CPU.
- **Cost.** 0 GPU-hours.
- **Kills it.** No arm beats 0.2362 by more than 0.0073 (§163's inference bar) with 4 of 5 folds
  positive. Then the partition is a one-off gain and goal (b) needs the prior (experiment 7).
  Report the partition on TabArena-Lite in the same pass; DR3.11 requires it not to lose on
  numeric-only datasets.

### 4. Fine-tuning where it has not been scored: TabArena-Lite, and per leaf on V4

**Status, 2026-11-13: per-leaf half closed, Lite half not found.** Fine-tuning per leaf of §169's partition is null: +0.0049 mean AP, 4 of 5 folds, below the +0.0073 bar (§208). I found no FINDINGS entry scoring `finetune_steps: 200` on TabArena-Lite; treat it as not run. §205 reports the zero-shot partition beating §163's fine-tuned arm on 5 of 5 folds.

- **Hypothesis.** Fine-tuning is the only lever positive on both benchmarks (§163, §192), and
  Mitra-v2's TabArena "default" is itself fine-tuned and bagged. Scored zero-shot, FinTFM is
  being compared under a weaker protocol than at least one leader uses.
- **Change.** `finetune_steps: 200` in the TabArena adapter, published checkpoint, scored on 3
  proxy-selected Lite datasets first to time it, then all 27. On V4, `dr311-ft200.yaml`
  (partition plus 200 steps per leaf), on Kaggle, since full-panel fine-tuning exhausts this
  Mac's memory (§202).
- **Cheapest falsification.** The 3-dataset timing run.
- **Cost.** ESTIMATED 5 to 20 A10G-hours for Lite (§192 MEASURED 62 to 638 s per fit; 27
  datasets times the adapter's fit count, which is not recorded here and must be read first);
  ESTIMATED 2 to 4 T4-hours for the V4 per-leaf arm (§202's fine-tuned pair ran inside a 25.3 h
  five-session batch; the per-arm share is not separated there).
- **Kills it.** Lite: no gain over the default beyond its interval and a per-dataset sign test p
  ≥ 0.05. V4: per-leaf fine-tuning does not beat §169 by more than 0.0073 on 4 of 5 folds.

### 5. A readout that attends over the context's labels (only if experiment 1 survives)

**Status, 2026-11-13: not built, condition failed.** Experiment 1 did not survive (§209), so by this plan's own rule it is not funded.

- **Hypothesis.** A fixed linear head maps the representation to fixed class slots; a readout
  that scores each query against the context rows' representations and labels (TabPFN-3's
  retrieval decoder, TabPFN-3.5's attention over one-hot training labels) fits the rule to the
  task, which is what the probe does post hoc.
- **Change.** A new `head_type`: one attention layer whose queries are the query rows' final
  states, keys the context rows' final states, values the context labels' one-hot vectors;
  logits from the attended label mass. Off by default, so every checkpoint is unchanged. Not the
  4-token CLS plus MLP head, which §178 measured as harmful.
- **Cheapest falsification.** One 48,000-task arm against `e1-v48-s0` and `e1-v48-s1`, then
  experiment 1's probe on the new checkpoint: if the gap between probe and head does not shrink,
  the readout did not do what it was for, whatever the AP.
- **Cost.** About 0.9 GPU-hours per seed, two seeds: about 1.8 (ESTIMATED).
- **Kills it.** V4 Δ inside ±0.0117 at both seeds, and the proxy null.

### 6. Train on the row counts the model is scored at

**Status, 2026-11-13: the specified resume stage has no result.** A related from-scratch arm, `--n-rows-choices 256,512,1024,2223` with contexts to 2,000 rows (AU5.4), is null: 0.1933 against 0.1975, −0.0042 mean AP, 1 of 5 folds positive (§211). A 4,000-row inference context on TabArena-Lite multiclass is also null, 820 against 825 Elo (§224).

- **Hypothesis.** A model that never saw more than 921 context rows cannot use 2,000 well, and a
  short final stage at longer contexts, as TabICL, TabICLv2, TabPFN v2 and TabFM all use, lets
  it hold more of the gate's positives.
- **Change.** Resume the published recipe for a final stage with `--n-rows-choices 2048,4096`,
  `max_query_rows` capped (AU5.4, built) so the cost is the context's, and a smaller batch.
- **Cheapest falsification.** A 200-step timing run first: per-step cost at these row counts has
  not been measured on the current architecture (the only figure, "2,048-row tasks cost 32× per
  step", predates cell attention and is not used). Then one arm of 2,000 steps.
- **Cost.** ESTIMATED 0.5 GPU-hours for timing; the stage itself is priced from the timing, and
  is not funded if it exceeds about 5 GPU-hours per seed.
- **Kills it.** The stage does not raise V4 AP at `max_context 2000` beyond 0.0117 at two seeds,
  or its V4 context curve (1,000 / 2,000 / 4,000) stays as flat as §83's.

### 7. The gate alone, in the prior

**Status, 2026-11-13: no result found.** FINDINGS §195-§224 contain no 48,000-task arm with `--financial-gate-prob 0.5` alone; the gated prior appears only inside the bundled prior v2 arms (§180, §191). Treat as not run.

- **Hypothesis.** V4's label is a hard gate (§168) that trees learn in one split, and the prior
  never taught one in isolation. Prior v2 arm (a) bundled the gate with a sign flip and feature
  bands, so the gate's own effect is unmeasured.
- **Change.** The published 48,000-task recipe plus `--financial-gate-prob 0.5` only.
- **Cheapest falsification.** One 48,000-task arm against both 48k controls, scored zero-shot and
  with the depth-2 partition: the partition already supplies the gate at inference, so a gated
  prior that helps zero-shot but not partitioned is not worth carrying.
- **Cost.** About 0.9 GPU-hours per seed (ESTIMATED).
- **Kills it.** Zero-shot V4 Δ inside ±0.0117, or below the control by more than 0.0117 (then the
  sign flip and bands were not what hurt arm (a)).

### 8. The peers' encoding and optimiser levers, already built

**Status, 2026-11-13: two of three arms run, both harmful.** Muon: 0.1849 against 0.1975, −0.0125 mean AP, 0 of 5 folds positive (§211). RBF value embedding (AU5.16, without the ECDF channel): 0.1654 against 0.1975, −0.0321, 0 of 5 folds positive (§213); the Fourier basis and the ECDF channel were not trained. Length-scaled attention (`au517-lenscale-s0`) was queued and not trained as of the openspec note of 2026-11-11.

- **Hypothesis.** TabICLv2's leave-one-out ablation gives about 100 Elo each to early target
  inclusion, Muon and QASSMax; FinTFM has the first (`cell_labels`, +0.0520, §172) and has never
  run the other two, nor the ECDF/Fourier cell encoding TabPFN-3.5 uses in place of
  preprocessing.
- **Change.** Three 48,000-task arms: `--optimizer muon`; `--attn-length-scale`; `--value-ecdf`
  with `--value-embedding`.
- **Cheapest falsification.** One seed each, V4 first.
- **Cost.** About 2.7 GPU-hours for three arms, one seed (ESTIMATED).
- **Kills it.** Each arm inside ±0.0117 on V4 and null on the proxy. Ranked last because
  Mitra-v2 reports that length-scaled logit temperature "gave no reliable gain", and because the
  rank transform already supplies much of what an ECDF feature would.

## What this plan does not fund, and why

- **2M-task or larger runs.** §176 closed E1's "helps" branch for credit; the proxy-positive
  branch waits on DR4.2's validation (§188).
- **Depth or width.** DR8.2 closed (§195, §202).
- **Retrieval contexts.** §194.
- **Real-data training.** Outside the FinTFM family by definition (D17); FinTFM-R is separate.

## Ordering

Run 1, 2 and 3 together this week: no GPU, and between them they decide whether experiment 5 is
built, which checkpoint goal (a) uses, and whether goal (b) still has an inference route. Run 4's
timing step alongside. Experiments 5 to 8 total about 7 GPU-hours at one seed (ESTIMATED, before
6's timing) and are ordered by what 1 to 3 return.

**8 experiments, 3 of them inference-only and runnable today.** (2026-11-13: see the status lines above; the ordering paragraph is the 2026-11-02 plan and is not revised.)
