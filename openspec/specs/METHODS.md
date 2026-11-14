# Methods register, 2026-10-10 to 2026-11-10

This file lists every method, option and harness built, tested or evaluated in the month up to
0.9.0, with its current status. It is a register, not a spec. The guarantees are stated in the
capability specs (`inference-options`, `experiment-decisions`, `credit-evaluation-data` and
the four older ones). Each number here is quoted from the `docs/results/FINDINGS.md` section
named beside it, and that section is the source. Nulls and harmful results are also listed in
`docs/paper/NEGATIVE_RESULTS.md`.

**Status vocabulary.**

- **default**: on in the shipped configuration.
- **adopted**: used in a recipe or adapter.
- **lead**: positive but below the bar, worth a further run.
- **Lite-only**: helps on TabArena-Lite, null or unmeasured on V4FinBench.
- **null**: within the bar.
- **harmful**: the mirror of a candidate.
- **prepared**: code and kernel exist, no result yet.
- **deferred**: stopped on cost, licence or time.
- **infrastructure**: not an arm.

Unless stated otherwise:

- Verdicts are on V4FinBench under `experiment-decisions` D1.
- "Lite" is TabArena-Lite, 27 binary datasets, where RandomForest at defaults anchors 1,000 Elo.
- Every option is off by default (`inference-options` I1).

## Inference

| method | switch | status | evidence |
| --- | --- | --- | --- |
| Partitioned contexts | `large_n.method: partition` | **lead**, best zero-shot lever on V4FinBench; adopted in the Lite entry | +0.0375 AP, 5 of 5 folds (§169); beats §163's fine-tuning by +0.0299, 5 of 5 (§205); Lite 888 Elo (§212) |
| Linear probe + `feature_transform: squash` + partition, probe per leaf | fork `FinTFM-LinProbe`, `FINTFM_PARTITION`; `representation_transfer --partition` | Best Lite configuration; **candidate** on V4FinBench since 2026-11-14 (§230; §218's "harmful" was a harness defect) | Lite 1,000.2 Elo, a tie with RandomForest at defaults (§215); V4FinBench −0.0570 mean AP, 0 of 5 folds (§218), traced to a sampler defect in the harness (§219); with the fix +0.0580, 5 of 5, and 0.3015 mean AP with the scale/cv/raw readout options (§230) |
| `feature_transform: squash` | `feature_transform` (default `rank`) | **Lite-only** | Lite +0.0100 mean ROC-AUC, better on 18 of 27 datasets (§210); V4FinBench null with rank, +0.0095 (§211) |
| `n_ensemble` 8 | `n_ensemble` (default 1) | **null** on V4FinBench, **adopted** in the Lite adapter | +0.0058, 3 up and 2 down (§196); on Lite one member is worse than eight on 20 of 27 datasets, p 0.019 (`docs/results/TABARENA.md`, AU5.7) |
| `ensemble_svd_components` | default 0 | **lead** (proxy only) | proxy 12/4/0 (§206); Lite 782 Elo, p 0.25; not run on V4FinBench |
| `ensemble_transforms` rank + squash | default none | **null** | +0.0038 against ens8, +0.0095 against the default (§211) |
| `ensemble_class_shift` | default false | **null**, stopped for futility | fold 0: 0.2033 against 0.2113 default (§217) |
| `context_positive_floor` | default none | **null** | +0.0024 exact, +0.0055 floor 16 (§200); 48.24's majority-downsample arms queued |
| `context_positive_cap` | default none | **null** on Lite | Lite +0.0016 mean ROC-AUC, 9 better, 2 worse, p 0.065 (§210) |
| `max_context` 900, 3 members | `max_context` (default 2,000; Lite adapter 1,000) | **null** | +0.0093, 4 of 5 (§211); Lite 500/750/1,000 give 702/745/756 Elo (§206) |
| k-means prototype context | `context_strategy: prototype` | **null** | +0.0002, 2 of 5 (§211) |
| Retrieval context | `context_strategy: retrieval` | **harmful** on V4FinBench, infeasible on Lite | AP 0.0126 (§194); exceeds TabArena's time limit (§210) |
| Per-class retrieval | `retrieval_per_class` | **harmful** | −0.1934, 0 of 5 (§213) |
| Boosted contexts | `large_n.method: boost` | **deferred** (about 18 GPU-h, owner 2026-11-08) | never scored |
| Column selection for wide tables | `max_features_strategy: select` | **null** | does not rescue wide tasks (§190) |
| Fitted calibration | `calibration_val_frac` (default 0) | **prepared** | SMOKE-TEST only, one panel and one seed (openspec `fitted-calibration` 4.3) |

## Fine-tuning and probes

| method | switch | status | evidence |
| --- | --- | --- | --- |
| Dataset-specific fine-tuning | `finetune_steps` (default 0) | **lead** | +0.0076, 5 of 5 (§163); second seed +0.0071, 0.0002 short of the bar (§164) |
| Enriched fine-tuning queries | `finetune_query_pos_frac` | **null** | +0.0009 (§164) |
| Early stopping on inner-validation companies | `finetune_val_frac` and five companions | **prepared**, AU5.1 interim | lr 3e-5 folds 0 and 1: +0.0023, −0.0042; folds 2-4 wait for Kaggle quota |
| Per-leaf fine-tuning | DR3.11 arm | **null** | +0.0049, 4 of 5 (§208) |
| Representation probe at equal labels | `--probe-rows context` | **null**, a tie with the head | −0.0005, 3 of 5 (§209); §130-§136 measured label count |
| Head against probe on prior tasks | `scripts/head_probe_prior.py` | **infrastructure** | the head wins on the prior's own tasks (§203) |

## Hybrids (a booster with the FinTFM logit)

| method | switch | status | evidence |
| --- | --- | --- | --- |
| CatBoost hybrid | `hybrid_catboost --booster catboost` | **null** | +0.0039 baseline, +0.0018 feature (§201) |
| Tuned LightGBM hybrid, V4FinBench | `--booster lightgbm_tuned` | **lead** as a system | init score +0.0304, 5 of 5; the capped control stays below uncapped CatBoost, 0.3967 against 0.4005 (§214) |
| Credit-suite hybrid | `credit_suite --hybrid --save-oof`, `scripts/credit_lightgbm_tuned.py` | mixed | ULB init score +0.0211, 3 of 3; SBA nothing once the booster is tuned (§213) |

## Pretraining and prior (48,000-task arms against the same-seed control)

| method | switch | status | evidence |
| --- | --- | --- | --- |
| Muon optimiser | `--optimizer muon` | **harmful** | −0.0125, 0 of 5 (§211) |
| RBF value embedding | `--value-embedding rbf` | **harmful** | −0.0321, 0 of 5 (§213) |
| Training contexts to 2,000 rows | `--n-rows-choices`, `--max-query-rows` | **null** | −0.0042, 1 of 5 (§211) |
| Induced cell rows; readout tokens | `--cell-row-attn induced`; `--readout cls --head-mlp` | **harmful** | −0.0186; −0.0261 (§178) |
| Prior v2 curriculum (DR2.4, four arms) | `--new-prior-phase`, prior v2 knobs | **harmful** | −0.0212 to −0.1262 (§180, §191) |
| Depth and width at 480,000 tasks | DR8.2 | **null** | §195, §202 |
| Multiclass volume, class permutation | DR7.4, `--permute-classes` | **adopted** in the multiclass recipe (permutation); volume **null** | §197, §202 |
| Masked-column objective | `--masked-column-prob` | **prepared** (48.15) | arm at P = 0.25 |
| Row and column latents | `cell_mixing: latent` | **prepared** (48.27) | arm written |
| Head-only training | `--train-only head`, `--reinit-trained` | **prepared** (48.25) | arm written |
| Diverse-prior flags | `heavy_tail_column_prob`, `tree_split_by`, `missing_label` | **prepared** (AU5.11) | two arms written |
| Hazard period context; float32 hazard loss under autocast | `--hazard-period-context` | **prepared** (AU5.9) | arms pinned at e89def2 |
| 512 columns | `--max-features 512`, ratio-family top-up | **prepared** (DR8.3, DR7.5) | the prior could not emit more than about 370 columns before the top-up |
| Attention length scale; second cell pass | `--attn-length-scale`; `--n-cell-blocks 2` | **prepared** (AU5.17, AU5.3) | kernels written, no result |
| Streaming loader, DDP, fp16, split attention | `--loader stream`, `--ddp`, `--amp fp16`, `--attn-impl split` | **infrastructure** | split 1.21×, DDP about 1.9×, fp16 2.38× (§170) |

## Evaluation and decision tooling

| method | where | status | evidence |
| --- | --- | --- | --- |
| Company-level bootstrap with Holm | `compare_v4_scores.py --cluster` | **adopted** for every verdict from §169 on | `experiment-decisions` D2 |
| Arm verdict scripts | `scripts/c5_verdict.py` (+0.0073), `scripts/dr64_arm_verdict.py` (+0.0117) | **adopted** | `experiment-decisions` D1 |
| Proxy suite | `fintfm.evaluation.proxy_suite`, `scripts/dr42_spearman.py` | **null** as a stand-in for TabArena | rho +0.717, under 0.8 (§216) |
| Credit suite, four admitted datasets | `fintfm.evaluation.credit_suite` | **adopted** for evaluation | FinTFM below logistic regression on all four (§216) |
| Time-based split, random against time | `splits.time_based_split`, `fintfm-bench --time-split` | **infrastructure** | SMOKE-TEST recorded in `time-based-evaluation` |
| Overlap guard and report | `evaluation/overlap.py` | **adopted** | 0 of 44 suite datasets overlap 193 references |
| Golden outputs, provenance | `tests/test_golden_outputs.py`, `fintfm/provenance.py` | **adopted** | `inference-options` I1 |
| Kaggle CPU and TPU sessions for scoring | `scripts/kaggle/make_cpu_kernels.py`, host probes | **deferred** | a CPU session is about 10× slower than the development machine (MEASURED 2026-11-10) |

## Data

| source | status | evidence |
| --- | --- | --- |
| ULB card fraud, SBA 7(a), FDIC bank failures, Estonian register | admitted, evaluation only | `credit-evaluation-data` L1-L7 |
| PPDai, Bondora | excluded on licence | `credit-evaluation-data` L2 |
| LendingClub | retired as a training source (D19) | `credit-evaluation-data` L4 |
| FICO HELOC label-shift check | **deferred** (48.28) | FICO's data-use agreement page is gone |
