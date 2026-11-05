Merged into FINDINGS §204 (2026-11-04).

# Where FinTFM loses on TabArena-Lite (draft, 2026-11-02)

**MEASURED by re-reading cached results; no model was fitted or scored.** TabArena-Lite, the 27
binary datasets (27 of 51, 53% coverage), split 0, ROC-AUC. FinTFM's head is the 2026-10-19 run
with NaN passed through (`experiments/fintfm_lite_nan_1019/`, Elo 756 (+126/-180), 94th of 95,
`docs/results/TABARENA.md`). The linear probe is §132's run (`fintfm_linprobe_lite/`, Elo 910,
93rd of 95), which predates AU1.19 and so filled numeric NaN with 0. Baselines are the board's
cached `LR (default)`, `RF (default)`, `CAT (default)` (CatBoost) and the best non-imputed default
config of nine in-context foundation models. `LR (default)` is 932.0 Elo in the 1019 snapshot and
935.5 in the probe's; `RF (default)` is the 1000 anchor in both.

Reproduce (fork read-only, local OpenML parquet cache):

```bash
.venv/bin/python -I scripts/tabarena_loss_analysis.py --markdown table.md
```

## Result

**FinTFM behaves like a weaker logistic regression.** It is below `LR (default)` on 23 of 27
datasets by a roughly constant amount (mean -0.0297 ROC-AUC), below CatBoost and below the best
default TFM on 27 of 27. Its gap to CatBoost and to the TFMs is predicted by how much a dataset
rewards a nonlinear model (RF minus LR: Spearman ρ = -0.69 against the CatBoost gap and -0.65
against the TFM gap, both p < 0.001), while its gap to LR is not (ρ = -0.12, p = 0.54). FinTFM
captures none of the nonlinear headroom that trees and TFMs take; that is where the Elo goes.

Row count, width, categorical share and missingness explain nothing measurable (|ρ| ≤ 0.17
against the LR gap, every p > 0.38). Class balance is the only secondary axis and is not
significant at n = 27 after counting the covariates tried (gap to LR against the coded positive
rate: ρ = -0.37, p = 0.054, unadjusted). The linear probe recovers 0.027 of the 0.030 LR gap on
average, and recovers most where the task is nonlinear (ρ = +0.47, p = 0.014), so the head throws
away signal the representation holds. Even the probe stays 0.035 below CatBoost on average.

What this points at, in order: the prior and model (no interaction structure, §137; a base-rate
range capped at 30%; no tied or discrete columns), then the head. Adapter and inference settings
can at best close the gap to LR; they cannot reach CatBoost.

## 1. Per-dataset table

ROC-AUC on split 0 (error = 1 - value), worst gap to LR first. "Train rows" is TabArena's split-0
training size; under the 8x1 bagged protocol each fold model sees 7/8 of it. "Coded positive
rate" is the share of the label AutoGluon codes as 1 (`LabelCleanerBinary`'s rule, recomputed
here, not read from the run). "Rank of 96" ranks FinTFM among every entry in the 1019 snapshot's
per-split file, average ties; "rank of 5" is among FinTFM, LR, RF, CatBoost and the best TFM.

| dataset | train rows | features | cat share | missing share | coded positive rate | FinTFM | probe | LR | RF | CatBoost | best default TFM | FinTFM rank of 96 | rank of 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Marketing_Campaign | 1,493 | 25 | 0.32 | 0.000 | 0.15 | 0.8262 | 0.8925 | 0.9235 | 0.9042 | 0.9335 | 0.9455 (TA-TABPFN-3.5) | 96 | 5 |
| hazelnut-spread-contaminant-detection | 1,600 | 30 | 0.00 | 0.000 | 0.50 | 0.8533 | 0.8972 | 0.9427 | 0.9529 | 0.9683 | 0.9945 (TA-TABPFN-3.5) | 96 | 5 |
| credit-g | 666 | 20 | 0.65 | 0.000 | 0.70 | 0.6789 | 0.7442 | 0.7664 | 0.7783 | 0.7915 | 0.8082 (TA-LIMIX) | 96 | 5 |
| in_vehicle_coupon_recommendation | 8,456 | 24 | 0.88 | 0.000 | 0.57 | 0.6827 | 0.7317 | 0.7457 | 0.8206 | 0.8478 | 0.8682 (TA-TABPFN-3.5) | 96 | 5 |
| customer_satisfaction_in_airline | 86,586 | 21 | 0.76 | 0.000 | 0.55 | 0.9043 | 0.9043\* | 0.9649 | 0.9923 | 0.9948 | 0.9960 (TA-TABPFN-3.5) | 96 | 5 |
| qsar-biodeg | 702 | 41 | 0.12 | 0.000 | 0.34 | 0.8504 | 0.8770 | 0.8970 | 0.9211 | 0.9250 | 0.9387 (TABICL_GPU) | 96 | 5 |
| bank-marketing | 30,140 | 13 | 0.54 | 0.000 | 0.12 | 0.7088 | 0.7445 | 0.7553 | 0.7339 | 0.7746 | 0.7768 (TABICLV2) | 96 | 5 |
| HR_Analytics_Job_Change_of_Data_Scientists | 12,772 | 12 | 0.75 | 0.000 | 0.25 | 0.7636 | 0.8013 | 0.8055 | 0.7973 | 0.8124 | 0.8158 (TABICL_GPU) | 95 | 5 |
| blood-transfusion-service-center | 498 | 4 | 0.00 | 0.000 | 0.24 | 0.6955 | 0.7133 | 0.7311 | 0.6651 | 0.7059 | 0.7452 (TABICLV2) | 77 | 4 |
| coil2000_insurance_policies | 6,548 | 85 | 0.04 | 0.000 | 0.06 | 0.7020 | 0.6955 | 0.7359 | 0.6828 | 0.7431 | 0.7682 (TA-LIMIX) | 91 | 4 |
| churn | 3,333 | 19 | 0.21 | 0.000 | 0.14 | 0.7316 | 0.8354 | 0.7634 | 0.9252 | 0.9363 | 0.9449 (TABICLV2) | 96 | 5 |
| jm1 | 7,256 | 21 | 0.00 | 0.000 | 0.19 | 0.7050 | 0.7264 | 0.7342 | 0.7630 | 0.7537 | 0.7939 (TA-TABPFN-3.5) | 96 | 5 |
| E-CommereShippingData | 7,332 | 10 | 0.40 | 0.000 | 0.60 | 0.6744 | 0.7279 | 0.6995 | 0.7447 | 0.7409 | 0.7435 (REALTABPFN-V2.5) | 96 | 5 |
| online_shoppers_intention | 8,220 | 17 | 0.41 | 0.000 | 0.15 | 0.8796 | 0.9137 | 0.9018 | 0.9184 | 0.9266 | 0.9346 (TA-TABPFN-3.5) | 95 | 5 |
| Amazon_employee_access | 21,846 | 9 | 1.00 | 0.000 | 0.94 | 0.8220 | 0.8220\* | 0.8411 | 0.8415 | 0.8709 | 0.8775 (TA-TABPFN-3.5) | 82 | 5 |
| credit_card_clients_default | 20,000 | 23 | 0.13 | 0.000 | 0.22 | 0.7295 | 0.7812 | 0.7467 | 0.7727 | 0.7898 | 0.7961 (TA-TABPFN-3.5) | 96 | 5 |
| GiveMeSomeCredit | 100,000 | 10 | 0.00 | 0.022 | 0.07 | 0.8294 | 0.8637 | 0.8445 | 0.8532 | 0.8699 | 0.8735 (TA-TABPFN-3.5) | 94 | 5 |
| taiwanese_bankruptcy_prediction | 4,546 | 94 | 0.00 | 0.000 | 0.03 | 0.9291 | 0.9320 | 0.9419 | 0.9458 | 0.9457 | 0.9557 (TA-TABPFN-3.5) | 84 | 5 |
| heloc | 6,972 | 23 | 0.00 | 0.000 | 0.48 | 0.7731 | 0.7867 | 0.7850 | 0.7929 | 0.7981 | 0.8022 (TA-TABPFN-3.5) | 94 | 5 |
| NATICUSdroid | 4,994 | 86 | 1.00 | 0.000 | 0.35 | 0.9703 | 0.9783 | 0.9813 | 0.9799 | 0.9860 | 0.9877 (TA-TABPFN-3.5) | 95 | 5 |
| polish_companies_bankruptcy | 3,940 | 64 | 0.00 | 0.012 | 0.07 | 0.8619 | 0.8871 | 0.8726 | 0.9357 | 0.9704 | 0.9943 (TA-TABPFN-3.5) | 89 | 5 |
| Fitness_Club | 1,000 | 6 | 0.50 | 0.002 | 0.30 | 0.7932 | 0.8003 | 0.8032 | 0.7797 | 0.8051 | 0.8101 (TABPFNV2_GPU) | 89 | 4 |
| diabetes | 512 | 8 | 0.00 | 0.000 | 0.35 | 0.8188 | 0.8365 | 0.8223 | 0.8255 | 0.8340 | 0.8537 (TA-TABPFN-3.5) | 86 | 5 |
| Is-this-a-good-customer | 1,148 | 13 | 0.62 | 0.000 | 0.11 | 0.7263 | 0.7381 | 0.7235 | 0.6695 | 0.7413 | 0.7450 (TA-TABPFN-3.5) | 56 | 3 |
| Diabetes130US | 47,678 | 47 | 0.83 | 0.041 | 0.09 | 0.6566 | 0.6433 | 0.6528 | 0.6372 | 0.6800 | 0.6950 (TA-TABPFN-3.5) | 70 | 3 |
| Bank_Customer_Churn | 6,666 | 10 | 0.40 | 0.000 | 0.20 | 0.7717 | 0.8003 | 0.7648 | 0.8510 | 0.8767 | 0.8824 (TA-TABPFN-3.5) | 93 | 4 |
| seismic-bumps | 1,722 | 15 | 0.20 | 0.000 | 0.07 | 0.7862 | 0.7855 | 0.7788 | 0.7650 | 0.7908 | 0.7965 (MITRA_GPU) | 66 | 3 |
| **mean** | | | | | | **0.7824** | 0.8096 | 0.8120 | 0.8241 | 0.8449 | 0.8572 | 89.3 | 4.63 |

\* **The probe never ran on these two datasets.** Its score is the own head's to 16 digits (the
own head from the zero-fill era, `fintfm_lite_tafintfm`, 0.0956796806130708 error on the
airline set). §132 reported one fallback dataset (Amazon); there are two, and the cause is a
defect, not AutoGluon bookkeeping (section 3, D1).

FinTFM is first of the five on no dataset. It is best only against RF or LR on the small
low-base-rate sets where every method is close (`seismic-bumps`, `Is-this-a-good-customer`,
`Diabetes130US`).

## 2. The pattern of losses

Spearman ρ of FinTFM's ROC-AUC gap against dataset properties, 27 datasets, two-sided p
unadjusted. A negative ρ means FinTFM falls further behind as the property grows.

| property | gap to LR | gap to CatBoost | gap to best TFM | probe minus head |
| --- | --- | --- | --- | --- |
| log training rows | -0.00 (p 0.99) | -0.15 (0.45) | -0.06 (0.78) | -0.06 (0.77) |
| features (4 to 94) | -0.17 (0.39) | -0.26 (0.20) | -0.20 (0.31) | -0.10 (0.63) |
| categorical share | -0.07 (0.74) | -0.13 (0.53) | +0.01 (0.97) | -0.06 (0.76) |
| missing share | +0.07 (0.73) | +0.02 (0.93) | +0.01 (0.95) | -0.10 (0.61) |
| tie-heavy numeric share | +0.03 (0.87) | -0.01 (0.97) | -0.02 (0.93) | +0.07 (0.74) |
| minority share | -0.34 (0.085) | -0.24 (0.23) | -0.21 (0.29) | +0.32 (0.11) |
| coded positive rate | -0.37 (0.054) | -0.32 (0.11) | -0.26 (0.20) | +0.24 (0.22) |
| **RF minus LR** | -0.12 (0.54) | **-0.69 (<0.001)** | **-0.65 (<0.001)** | **+0.47 (0.014)** |
| **best TFM minus LR** | -0.17 (0.38) | **-0.73 (<0.001)** | **-0.79 (<0.001)** | +0.42 (0.029) |

Threshold splits, Mann-Whitney two-sided:

| split | n | median gap to LR | median gap to CatBoost | p (LR, CatBoost) |
| --- | --- | --- | --- | --- |
| whole training split fits one 1,000-row context | 5 / 22 | -0.0356 / -0.0206 | -0.0152 / -0.0489 | 0.69, 0.31 |
| over 10,000 training rows | 7 / 20 | -0.0191 / -0.0236 | -0.0489 / -0.0478 | 0.81, 0.94 |
| over 40 features | 6 / 21 | -0.0119 / -0.0251 | -0.0323 / -0.0489 | 0.38, 0.59 |
| categorical share ≥ 0.5 | 10 / 17 | -0.0305 / -0.0222 | -0.0489 / -0.0487 | 0.74, 0.90 |
| coded positive rate > 0.30 | 11 / 16 | -0.0251 / -0.0197 | -0.0665 / -0.0478 | 0.29, 0.44 |
| any missing value | 8 / 19 | -0.0222 / -0.0222 | -0.0488 / -0.0489 | 0.98, 0.98 |
| RF beats LR by > 0.03 | 5 / 22 | -0.0251 / -0.0206 | **-0.1085 / -0.0441** | 0.79, **0.003** |

Reading, axis by axis:

- **Size: no.** The five datasets whose whole training split fits one context
  (`blood-transfusion`, `diabetes`, `credit-g`, `qsar-biodeg`, `Fitness_Club`) lose to LR by as
  much as the rest. `credit-g` is the plainest case: all of each bag fold's 582-583 rows are in
  context, no subsampling happens, and FinTFM is 0.0874 below LR. The existing context arms
  agree: 500 to 1000 rows moved the 22 tables that exceed the context by +0.0008 mean ROC-AUC,
  positive on 13 of 22 (`fintfm_lite_ctx500_1026` against the 1019 run). This repeats §102's
  size null on the newer run.
- **Width: no.** No eligible dataset exceeds 94 features, and pretraining drew widths up to the
  136-feature cap (`sample_financial_task` at the checkpoint's commit d32b125 draws
  `n_expose` uniformly up to `max_features`). The six tables over 40 features lose less to LR
  than the narrow ones. The three over-cap tasks are excluded from Lite coverage, not lost on.
- **Categorical share: no** at this resolution, after §101's target encoding. Fully categorical
  `NATICUSdroid` (binary flags) is 0.011 behind LR; 88%-categorical `in_vehicle_coupon` is 0.063
  behind, and its loss is the nonlinear one (RF minus LR +0.075).
- **Missingness: no.** 8 datasets have any missing cell; their median gap equals the rest's.
- **Balance: weak, same direction as §102.** The rank of 96 rises with the minority share
  (ρ = +0.51, p = 0.006, unadjusted; ×9 covariates puts it at 0.05). The published checkpoint's
  prior draws its base rate log-uniformly up to `_RATE_CEILING = 0.30`
  (`src/fintfm/prior/financial.py:59`, unchanged since d32b125:59), so it never saw a task whose
  positive class held more than 30% of rows. 11 of 27 Lite datasets code a positive class above
  that, 5 of them above 50% (Amazon 0.94, credit-g 0.70, E-Commerce 0.60, coupon 0.57, airline
  0.55).
- **Nonlinearity: yes, and it is the main effect.** Where RF beats LR by more than 0.03
  (`churn`, `Bank_Customer_Churn`, `in_vehicle_coupon`, `polish_companies_bankruptcy`,
  `E-CommereShippingData`), FinTFM's median gap to CatBoost is -0.1085 against -0.0441 elsewhere
  (p = 0.003), while its gap to LR is the same in both groups. On `churn` FinTFM (0.7316) sits
  next to LR (0.7634) while RF reaches 0.9252. This matches §137 on two other financial-prior
  checkpoints: chance on any interaction of order 2 or more.
- **Head against representation.** The probe gains +0.0273 mean ROC-AUC over the head and
  reaches LR on average (-0.0024). Its gain is largest where the task is nonlinear (`churn`
  +0.104, `Marketing_Campaign` +0.066, `credit-g` +0.065, `E-Commerce` +0.054) and smallest on
  tables over 40 features (median +0.006 against +0.034, p = 0.026). The representation carries
  some nonlinear signal that the head does not use; still, the probe is 0.035 below CatBoost on
  average.

## 3. The adapter's path, checked against each pattern

Fork branch `fintfm-audit-fixes` (the branch both runs used), line numbers as of 2026-11-02; fintfm
at 82238f6.

**D1, defect: the probe falls back to the head whenever positives exceed 20,000.**
`tabarena/models/fintfm_linprobe/model.py:75-84` keeps every row coded 1 and pads with negatives up
to `TRAIN_SAMPLE_SIZE = 20_000` (line 18); `n_negative = max(20_000 - len(positive), 0)` is 0 once
positives pass 20,000, so the subsample holds one class and lines 87-100 fall back to the head.
Airline has about 47,400 coded positives in 86,586 rows (54.7%), so every fit falls back; Amazon's
refit-on-all fits (about 20,580 positives in 21,846 rows) do the same. The comment at lines 88-92
attributes this to AutoGluon's refit step; the arithmetic accounts for it without that. The 910
Elo therefore has 2 of 27 datasets scored by the head.

**D2, setting: the positive slot is AutoGluon's alphabetical choice, not the minority.**
`tabarena/models/fintfm/model.py:276` passes AutoGluon's 0/1 coding straight to
`FinancialTFMClassifier.fit`. AutoGluon codes the second sorted label as 1 ("Yes" over "No",
"good" over "bad", "satisfied" over "neutral or dissatisfied"), so on 5 datasets the class the
model treats as "default" is the majority. The financial prior always made class 1 the rarer
outcome, at most 30% of rows (section 2, balance). ROC-AUC is invariant to the swap; the model is not (§45's label-slot
bias; §165 measured this checkpoint's symmetry at pooled r = 0.891 on financial-prior tasks and
0.490 on Polish 3-year, not 1.0).

**D3, setting: balanced tasks are outside the base-rate range the checkpoint was trained on**,
and nothing in the adapter maps them back into it. The context draw is uniform
(`classifier.py:230-231`), so a 0.48-positive table yields a 0.48-positive context. The classifier
already corrects logits for a context whose rate differs from the pool's
(`classifier.py:808-811`, applied at 1708-1716), which is what a rate-capped context would need.

**D4, defect (small expected cost): the rank transform pins tied extremes to ±5.2 SD.**
`preprocess.py:200-205` fits sklearn's `QuantileTransformer(output_distribution="normal")` and
`preprocess.py:239` applies it. Run on a synthetic zero-inflated count (60% zeros): 0 maps to
-5.199 and 1 to 5 map to 0.29-1.20, so after the model's own z-scoring a count of 1 and a count of
5 differ by 0.30 SD while 0 and 1 differ by 1.80 SD. A three-level column maps to -5.199, 0.391,
+5.199. The prior never produced ties: every financial-prior column gets Gaussian jitter
(`financial.py:869`, `:882`). Seven Lite datasets have at least 30% of their columns tie-heavy
(modal value ≥ 30% of rows): coil2000 0.60, qsar 0.51, GiveMeSomeCredit 0.50, heloc 0.48,
seismic-bumps 0.47, online_shoppers 0.41, bank-marketing 0.31. The tie-heavy share does not
correlate with the gap (ρ = +0.03), so the cost is likely small.

**Checked and not a lever here:**

- *Context subsampling.* The adapter sets `max_context=1000` (`model.py:320`, `:334`), not the
  library's 2000 (`classifier.py:497`), with `n_ensemble=8` (`model.py:331`); each member draws
  its own 1,000 rows at seed `random_state + i` (`classifier.py:1657-1659`), so up to 8,000
  distinct rows inform a prediction. The 500 → 1000 arm moved +0.0008; pretraining contexts reach
  about 921 rows (TABARENA.md, AU5.14), so 2000 is outside the trained range as well.
- *Wide tables.* Nothing on Lite's eligible set exceeds the cap; `max_features_strategy` stays at
  `raise` (`model.py:261-271`), which is why the three wide tasks are skipped, not lost.
- *Categorical encoding.* Out-of-fold, 5 folds, smoothing 10 (`categorical.py:74`, `:193`), fitted
  on all training rows of the bag fold before the context is drawn, so context and query rows are
  encoded consistently. No defect found; categorical share does not predict the gap.
- *Target handling.* Binary 0/1 reaches `_supports_target_statistics` as integers and gets target
  statistics, as intended. The prior correction is near zero under a uniform draw.

## 4. Fixes to test (inference or adapter only)

Each is one Lite run under a new `FINTFM_RUN_NAME`. Lite reports and never selects (DR4.1), and a
change is scored on V4FinBench before it is believed (CLAUDE.md). The expected directions are
ESTIMATED from this analysis, not measured.

| # | change | where | datasets it should move | expected direction |
| --- | --- | --- | --- | --- |
| 1 | Fix D1 (draw the probe's 20,000 rows stratified, capping both classes) and rerun the probe arm | `fintfm_linprobe/model.py:75-84` | `customer_satisfaction_in_airline` (head 0.9043, LR 0.9649) and `Amazon_employee_access` (0.8220, LR 0.8411); the other 25 should be bit-identical | up on both; probe Elo up from 910. §132 measured +145 Elo for the probe with the defect, so the probe is also the adapter's best decision step on record |
| 2 | Code the minority class as 1 before fit, and swap the probability columns back at predict | `fintfm/model.py:276` plus a `_predict_proba` override | Amazon (0.94), credit-g (0.70), E-Commerce (0.60), coupon (0.57), airline (0.55); the other 22 bit-identical, a built-in control | up on the five if the label-slot asymmetry costs accuracy; null otherwise |
| 3 | Cap each member's context positive rate at 0.30 by subsampling the minority, relying on the existing prior correction | new `context_strategy` option beside `classifier.py:216-231`; correction at `:808` | hazelnut, heloc, airline, coupon, E-Commerce, NATICUSdroid, qsar, diabetes, credit-g, Fitness_Club (fix 2 first for those above 0.5) | up if §102's balance effect is the prior's 30% ceiling; down if losing minority rows costs more |
| 4 | Tree-partitioned contexts: wrap the classifier in `PartitionedTFMClassifier` (depth 2, 5% leaf floor), §169's lever, which gained +0.0375 mean AP on V4FinBench, 5 of 5 folds, and is untested on TabArena | `fintfm/model.py:276`; class at `src/fintfm/inference/large_n.py:81` | the nonlinear tables: churn, Bank_Customer_Churn, in_vehicle_coupon, polish_companies_bankruptcy, E-Commerce, jm1, airline | up on those (a tree split is the interaction the model cannot learn); flat or slightly down on small tables, where leaves fall back to the global context |
| 5 | `context_strategy="retrieval"` (§187's other proxy candidate: 8 wins, 1 loss, 7 ties) | adapter default params, `model.py:327-340` | the same nonlinear tables with more rows than one context | up on those; no change on tables that fit one context |
| 6 | Tie-aware rank: map tied values to their mid-quantile instead of pinning the extremes; first A/B `feature_transform="squash"` | `preprocess.py:200-205`, `:239` | coil2000, qsar, GiveMeSomeCredit, heloc, seismic-bumps, online_shoppers, bank-marketing | small up at most; low prior (ρ = +0.03; §187's `ens8-tf` was not a candidate) |

**Implementation status (2026-11-04), not yet run.** Fix 1 is in the fork's untracked
`fintfm_linprobe/model.py` and queued as `fintfm_linprobe_d1fix_1104`; fixes 5 and 6 (squash)
are queued as `fintfm_lite_retrieval_1104` and `fintfm_lite_squash_1104`. Fix 2 is opt-in
`FINTFM_MINORITY_POSITIVE=1` in the fork (branch `fix2-minority-positive`): a fit whose class 1
holds more than half of its rows is recoded and the probability columns are swapped back, so
datasets at or below one half stay bit-identical. Fix 3 is the classifier option
`context_positive_cap` (default `None`, byte-identical; `FINTFM_PARAMS='{"context_positive_cap":
0.3}'` on Lite): above the cap the context keeps the negatives that fit and draws positives down
to the cap, also when the pool fits in `max_context`, so credit-g's 875-row bag folds get
375-row contexts; the base-rate correction then reads the capped rate. Neither is merged into
the running queue's code until that queue finishes.

**Fix 5 could not run on this machine (2026-11-05).** `fintfm_lite_retrieval_1104` completed 0 of
the 4 datasets it attempted (Amazon_employee_access, Bank_Customer_Churn, Diabetes130US,
E-CommereShippingData). On MPS each fit ran out of Metal memory (86 GB of "other allocations"
against about 2 GB of live tensors), because retrieval runs one forward pass per query group and
that memory grows within a single dataset, so one process per dataset does not contain it. On CPU
(fork switch `FINTFM_DEVICE=cpu`, be201d7) the same datasets hit TabArena's per-fit time limit. The
arm was stopped by hand and needs a CUDA run; `runs/lite2611/fintfm_lite_retrieval_1104*.log`
hold both failure modes. Fixes 1, 2, 3 and 6 run in `runs/lite2611/queue_rest.sh`.

Not worth a run on this evidence: a larger context (+0.0008 for 500 → 1000), wide-table
strategies (no Lite table is wide), and categorical-encoding variants (ρ = -0.07).

**Ceiling, stated so a fix is not oversold.** The probe already sits at LR on average (-0.0024).
Fixes 1 to 3 can plausibly take FinTFM to LR and slightly past it on the balanced and
majority-coded sets. The gap to CatBoost (-0.0626 mean for the head, -0.035 for the probe) sits
on the nonlinear tasks and is a prior and model property. Fixes 4 and 5 are the only ones aimed at
it, and they work around the model by handing it local contexts rather than changing what it can
represent.

## Caveats

- One split per dataset (Lite). With 27 datasets and nine covariates, only the nonlinearity
  correlations survive any multiplicity allowance; the balance result is a direction.
- The head (2026-10-19, NaN passed through) and the probe (§132, zero fill) differ in missing-value
  handling on 6 datasets; DR3.5 measured that difference at +0.00008 mean ROC-AUC over 27.
- The coded positive rate is recomputed with AutoGluon's labelling rule on the full dataset; it was
  not read from the runs' stored predictions.
- "Best default TFM" excludes imputed rows (`TABPFNV2_GPU` and `MITRA_GPU` on Amazon,
  Diabetes130US and GiveMeSomeCredit, among others), so it is the best of the models that actually
  ran.
