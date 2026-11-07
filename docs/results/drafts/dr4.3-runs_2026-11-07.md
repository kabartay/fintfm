# DR4.3: Estonia and FDIC full-scale run plan (draft, 2026-11-07)

Status: prepared and smoke-tested; **no full-scale run made**. Not in FINDINGS. Every score below is
SMOKE-TEST scale and says nothing about model quality.

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

## Full-scale commands (serial, run from /Users/morganoko/proj/fintfm)

```
mkdir -p runs/dr43-fdic runs/dr43-estonia
PYTHONUNBUFFERED=1 nice -n 10 uv run --frozen --extra bench python -m fintfm.evaluation.credit_suite \
  --model /Users/morganoko/proj/fintfm/runs/v4-cellattn-labels.pt --datasets fdic \
  --root /Users/morganoko/proj/fintfm/data/real \
  --out /Users/morganoko/proj/fintfm/runs/dr43-fdic/fdic.csv > /Users/morganoko/proj/fintfm/runs/dr43-fdic/fdic.log 2>&1

PYTHONUNBUFFERED=1 nice -n 10 uv run --frozen --extra bench python -m fintfm.evaluation.credit_suite \
  --model /Users/morganoko/proj/fintfm/runs/v4-cellattn-labels.pt --datasets estonia \
  --root /Users/morganoko/proj/fintfm/data \
  --out /Users/morganoko/proj/fintfm/runs/dr43-estonia/estonia.csv > /Users/morganoko/proj/fintfm/runs/dr43-estonia/estonia.log 2>&1
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
