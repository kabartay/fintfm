# DR4.3: Estonia and FDIC full-scale run plan (draft, 2026-11-07)

Status (2026-11-08): **full-scale runs made and finished** (results section at the end). Not in FINDINGS.
The SMOKE-TEST and ESTIMATE sections below are kept as written on 2026-11-07; the measured results
supersede them.

## Caveats that travel with the numbers

* Estonia: the label is survivor-conditioned (firms bankrupt then deleted are absent). A score
  there is never quoted as bankruptcy-prediction skill.
* FDIC: admitted FinTFM only. Read here as: the FinTFM family (synthetic-only checkpoint) and the
  suite's two baselines as its reference arms; never FinTFM-R (`fdic` is in `EVAL_ONLY_SOURCES`, 6.72%
  SEC-filer overlap floor). The task text lists "FinTFM, logistic-regression and LightGBM runs" for
  both datasets, so all three models run on both. If the owner meant no baselines on FDIC, drop
  `lightgbm_1000` and `logistic_regression` rows from the FDIC CSV; they add under 10 minutes.
* Both are evaluation-only; this is verification against held-out time windows, not validation
  against observed outcomes of another kind.

## How the other datasets were run (MEASURED, from `runs/credit_suite/`)

`python -m fintfm.evaluation.credit_suite --model runs/v4-cellattn-labels.pt --datasets <key> --root data/real --out <csv>`
with `nice -n 10`, CPU, one dataset per process, CSV rewritten after each row (dataset, split, model).
Splits are the suite's three fixed out-of-time splits (cuts at 1/2, 2/3, 5/6 of the date quantile,
embargo per dataset). Checkpoint sha256 `d231b062...c518d2`. FinTFM seconds per split: SBA 512 to 606
(about 80k test rows, 6.4 to 7.6 ms per test row); ULB 543 to 958 (95k test rows). LightGBM 9 to 30 s.

## Data present and loading (MEASURED 2026-11-07)

| dataset | location | root flag | rows | features | positives | loader time | loader peak RAM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FDIC | `data/real/fdic` (financials.csv 414 MB, failures.csv) | `--root data/real` | 733,528 | 79 | 2,303 | 6.5 s | 3.5 GB |
| Estonia | `data/estonia` (not under `data/real`) | `--root data` | 1,251,575 | 25 | 260 | 30.1 s | 5.9 GB |

Both feature counts are under the checkpoint's `max_features` 136, so FinTFM is not skipped. No
download needed. Estonia's root differs: the loader reads `<root>/estonia`.

## Full-scale commands (serial, run from .)

```
mkdir -p runs/dr43-fdic runs/dr43-estonia
PYTHONUNBUFFERED=1 nice -n 10 uv run --frozen --extra bench python -m fintfm.evaluation.credit_suite \
  --model runs/v4-cellattn-labels.pt --datasets fdic \
  --root data/real \
  --out runs/dr43-fdic/fdic.csv > runs/dr43-fdic/fdic.log 2>&1

PYTHONUNBUFFERED=1 nice -n 10 uv run --frozen --extra bench python -m fintfm.evaluation.credit_suite \
  --model runs/v4-cellattn-labels.pt --datasets estonia \
  --root data \
  --out runs/dr43-estonia/estonia.csv > runs/dr43-estonia/estonia.log 2>&1
```

Run one at a time, not during a solver campaign or another full-panel scoring (shared Mac).
`run()` checks `include=True` for both, which holds.

## SMOKE-TEST measurements

Harness: the suite's own `run()` on a row-capped panel (random rows in time order, 5% positives
oversampled, Estonia from the cached `frame.parquet`), CPU, `nice -n 19`, `/usr/bin/time -l`.
Outputs in `runs/dr43-smoke/`.

| run | rows | wall | peak RAM | FinTFM s per split | LightGBM s per split |
| --- | --- | --- | --- | --- | --- |
| Estonia | 3,000 | 33 s | 6.1 GB | 2.5, 4.1, 8.9 | 3.8, 4.9, 5.8 |
| Estonia | 20,000 | 133 s | 15.0 GB | 26.8, 28.6, 49.0 | 4.7, 8.7, 12.2 |
| FDIC | 20,000 | 218 s | 11.5 GB | 43.3, 49.2, 49.6 | 10.4, 33.8, 19.2 |

Logistic regression takes under 0.2 s everywhere. Mean AP at 20,000 rows (SMOKE-TEST, one seed,
not a result): Estonia FinTFM 0.2333; FDIC FinTFM 0.6320 (LightGBM above logistic regression on
most splits).

## ESTIMATES for the full runs

Basis: per-test-row FinTFM cost from the measured SBA and ULB runs (6.4 to 7.6 ms, up to 10 ms
for ULB), with the smoke's 8 to 15 ms per test row (which includes fixed costs) as the upper bound.
Test rows: FDIC 118,769 + 122,292 + 127,289 = 368,350 (MEASURED, DR4.3 FDIC draft); Estonia about
16.8% of rows per split, about 210,000 each, 630,000 in all (ESTIMATED from the 20,000-row smoke).

| run | FinTFM | LightGBM | logistic regression | total wall | peak RAM |
| --- | --- | --- | --- | --- | --- |
| FDIC | 40 to 90 min | 3 to 10 min (584k training rows at the largest split, 79 features) | under 1 min | about 0.7 to 1.7 h | about 12 to 15 GB |
| Estonia | 65 to 150 min | 2 to 5 min | under 1 min | about 1.2 to 2.6 h | about 15 GB |

RAM: the smoke's 11.5 to 15 GB is FinTFM's, not the data's (loaders alone 3.5 and 5.9 GB), and
earlier full runs on SBA and ULB sat at 12 to 14 GB, so 15 GB is the planning figure. Both runs
together are not safe on this Mac.

## Not done

Full-scale runs; any held-out score. Open after the runs: the per-split report must state the
survivor-conditioned label (Estonia) and the FinTFM-only scope and the crisis-tail test windows
(FDIC; splits 1 and 2 have 109 and 66 test positives).

## Full-scale results (MEASURED, 2026-11-08)

Source: `runs/dr43-fdic/fdic.csv`, `runs/dr43-estonia/estonia.csv`, logs `runs/local_queue/dr43-fdic.log`
and `dr43-estonia.log` (read from the main checkout; `runs/` is not tracked). Both runs exited 0 on the
published checkpoint `runs/v4-cellattn-labels.pt` (sha256 `d231b062...c518d2` in every FinTFM row), CPU,
`nice -n 10`, one process at a time. Command lines as above. Each split is out-of-time on the suite's three
fixed cuts. Mean AP is the plain mean over three splits, one seed; no paired test (FDIC splits 1 and 2 have
109 and 66 test positives, Estonia 56 to 74).

### Run totals

| run | wall | peak RSS | FinTFM s per split | LightGBM s per split | logistic regression s per split |
| --- | --- | --- | --- | --- | --- |
| FDIC | 2,817 s (46.9 min) | 18.13 GB (16.9 GiB) | 985.3, 897.1, 834.0 | 23.6, 27.1, 27.0 | 0.4, 1.2, 1.6 |
| Estonia | 5,490 s (91.5 min) | 20.02 GB (18.6 GiB) | 1,440.0, 1,725.8, 2,244.7 | 17.0, 8.9, 19.1 | 0.1, 0.1, 0.2 |

FinTFM per test row: FDIC 8.3, 7.3, 6.6 ms; Estonia 6.8, 8.3, 10.7 ms (MEASURED; SBA 6.4 to 7.6, ULB 7.1 to 8.5).

### FDIC (733,528 rows, 79 features, 2,303 positives = 0.314%; FinTFM family only, baselines as reference arms)

| split | train rows | test rows | test positives | test positive rate | model | AP | ROC-AUC | AP top decile | s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 (2008-2012 crisis tail) | 332,332 | 118,769 | 873 | 0.735% | FinTFM | 0.1831 | 0.9354 | 0.1770 | 985.3 |
| 0 | | | | | logistic regression | 0.2718 | 0.9554 | 0.2679 | 0.4 |
| 0 | | | | | LightGBM (1000) | 0.2434 | 0.8677 | 0.2460 | 23.6 |
| 1 | 455,969 | 122,292 | 109 | 0.089% | FinTFM | 0.2114 | 0.9331 | 0.2113 | 897.1 |
| 1 | | | | | logistic regression | 0.1501 | 0.9505 | 0.1500 | 1.2 |
| 1 | | | | | LightGBM (1000) | 0.2810 | 0.9937 | 0.2809 | 27.1 |
| 2 | 583,938 | 127,289 | 66 | 0.052% | FinTFM | 0.1118 | 0.8320 | 0.1112 | 834.0 |
| 2 | | | | | logistic regression | 0.1500 | 0.7092 | 0.1496 | 1.6 |
| 2 | | | | | LightGBM (1000) | 0.0107 | 0.6781 | 0.0101 | 27.0 |
| **mean** | | | | | FinTFM | **0.1688** | 0.9002 | 0.1665 | |
| **mean** | | | | | logistic regression | **0.1906** | 0.8717 | 0.1892 | |
| **mean** | | | | | LightGBM (1000) | **0.1783** | 0.8465 | 0.1790 | |

Split 0 is the crisis tail and the only split with a base rate near 1%. Rankings flip by split: FinTFM
leads logistic regression on AP on split 1 only; LightGBM leads on split 1 and trails on split 2 (AP 0.0107
against a 0.052% base rate). Treat the three means as indistinguishable until a paired test over more
than three units exists. FinTFM's mean predicted probability is below the base rate on split 0 (0.396% vs
0.735%) and 1.9x and 3.2x it on splits 1 and 2 (uncalibrated; AP and ROC-AUC do not use it).

### Estonia (1,251,575 rows, 25 features, 260 positives = 0.021%; survivor-conditioned label)

| split | train rows | test rows | test positives | test positive rate | model | AP | ROC-AUC | AP top decile | s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 116,652 | 210,245 | 56 | 0.027% | FinTFM | 0.0036 | 0.8813 | 0.0031 | 1,440.0 |
| 0 | | | | | logistic regression | 0.0052 | 0.9109 | 0.0048 | 0.1 |
| 0 | | | | | LightGBM (1000) | 0.0013 | 0.6511 | 0.0010 | 17.0 |
| 1 | 121,840 | 208,736 | 74 | 0.035% | FinTFM | 0.0023 | 0.8090 | 0.0017 | 1,725.8 |
| 1 | | | | | logistic regression | 0.0047 | 0.8592 | 0.0042 | 0.1 |
| 1 | | | | | LightGBM (1000) | 0.0046 | 0.7952 | 0.0042 | 8.9 |
| 2 | 165,461 | 209,004 | 62 | 0.030% | FinTFM | 0.0026 | 0.8668 | 0.0021 | 2,244.7 |
| 2 | | | | | logistic regression | 0.0069 | 0.9120 | 0.0066 | 0.2 |
| 2 | | | | | LightGBM (1000) | 0.0026 | 0.8113 | 0.0021 | 19.1 |
| **mean** | | | | | FinTFM | **0.0028** | 0.8524 | 0.0023 | |
| **mean** | | | | | logistic regression | **0.0056** | 0.8940 | 0.0052 | |
| **mean** | | | | | LightGBM (1000) | **0.0028** | 0.7525 | 0.0024 | |

These are not bankruptcy-prediction skill (survivor-conditioned label). The test base rate is 0.027 to
0.035%, over 100 times below the 5.20% of the 5,000-row smoke subsample and the 5% oversampled 20,000-row
smoke, so the SMOKE-TEST scores above (Estonia FinTFM mean AP 0.2333) do not transfer: the full-run mean AP
is 0.0028. FinTFM's AP is 6 to 14 times its split's base rate; ROC-AUC is the more stable column.
Logistic regression is first on AP and ROC-AUC on every split (split 1 AP: 0.0047 against LightGBM 0.0046).

### Errors, warnings, truncation, caps, guards (from the logs)

* No errors, no skipped models (`skipped` empty in both CSVs), no `--max-rows` cap (the commands carry
  none; every split uses the full panel and the train and test counts above are the loader's).
* One warning type, FDIC only, three times (one per split, from the baselines' imputer): sklearn
  `SimpleImputer` "Skipping features without any observed values: [25 26 53]". Three FDIC feature columns
  (indices 25, 26, 53) are entirely missing in the training window, so the logistic-regression and
  LightGBM arms drop them. The log does not say how FinTFM handles them; not checked. Estonia: no warnings.
* `uv` printed its `VIRTUAL_ENV` mismatch warning (harmless, `CLAUDE.md`).
* FinTFM conditions on a sampled context of at most `max_context` = 2,000 training rows per fit (suite
  design, §174), whatever the pool (116,652 to 583,938 rows here). A design cap, not a log event; the
  baselines train on the whole pool.
* Leakage guards: embargoed out-of-time splits only (none printed). The inner-fold `assert_no_leak`
  belongs to the `--hybrid` path, which these runs did not use.
* Against the estimates: FDIC 46.9 min (estimate 0.7 to 1.7 h, inside, low end); Estonia 91.5 min (1.2 to
  2.6 h, inside). Peak RSS 16.9 and 18.6 GiB exceeded the 15 GB planning figure (and FDIC's 12 to 15 GB
  estimate); the next such run needs about 19 GiB of headroom.

### Reading across the four admitted datasets (MEASURED means over three splits, one seed)

| dataset | FinTFM AP | logistic regression AP | LightGBM AP |
| --- | --- | --- | --- |
| SBA 7(a) (§177) | 0.1095 | 0.1295 | 0.3861 |
| ULB (§174) | 0.6830 | 0.7489 | 0.6752 |
| FDIC (this run) | 0.1688 | 0.1906 | 0.1783 |
| Estonia (this run) | 0.0028 | 0.0056 | 0.0028 |

ULB LightGBM is the mean of 0.8539, 0.4845, 0.6871 in `runs/credit_suite/results.csv`; the other SBA and
ULB cells are as in the DR4.3 task text. FinTFM's mean AP is below logistic regression on all four. Same
sign everywhere, but one seed and three non-independent splits per dataset make it a pattern, not a test;
FinTFM leads on individual splits (FDIC split 1).

### What this means for DR4.3

* **Verify** ("the suite runs end to end on the published checkpoint with runtimes recorded"): met for the
  four admitted datasets. FinTFM, logistic regression and LightGBM ran all three splits on SBA, ULB, FDIC
  and Estonia on the published checkpoint, with per-split seconds and peak RSS recorded. Not met for the
  suite as the task text names it: PPDai and Bondora cannot run (`include=False`, licence not passing
  DR4.2's rule; owner decision 2026-10-19) and have never run on real data. The box stays unticked.
* **Composition**: in and run: ULB (card fraud, not default), SBA 7(a), FDIC bank failures, Estonia
  (survivor-conditioned). Waiting: the SBA 7(a) row-hash check against TabArena-v0.1 and V4FinBench (FDIC
  and Estonia are done, 0.0000 share); PPDai and Bondora on licence, so the suite still has no
  licence-clean consumer-credit-default dataset; ULB keeps its ODbL notice. Also open: a paired comparison
  across datasets (the splits within a dataset are not independent units) and a second seed.
