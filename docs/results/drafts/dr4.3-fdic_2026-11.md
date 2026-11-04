# DR4.3: FDIC bank failures as a credit-suite dataset (draft, 2026-11-04)

Status: loader, label, splits and tests built; data downloaded; **not admitted** (licence verdict
below needs an owner decision); no FinTFM number; SMOKE-TEST of two baselines only.
Code: `src/fintfm/evaluation/credit_fdic.py`, registered in `credit_suite.DATASETS["fdic"]`.
Not in `docs/results/FINDINGS.md`; the owner folds this in.

## Licence (read at source 2026-11-04)

| page | what it states |
| --- | --- |
| `https://api.fdic.gov/banks/docs/` (BankFind Suite API, "FDIC Bank Data API (Beta)") | No licence, copyright, public-domain or attribution term for the data. Keys "are currently not required"; the FDIC "is not responsible for the contents of any linked site". |
| `https://www.fdic.gov/about/website-policies` | No licence for FDIC data. Only reuse text: external sites "may contain information that is copyrighted with restrictions on reuse", and permission for that material "cannot be obtained from the FDIC". |

The FDIC is the producer and a US federal agency, so the data is a US government work
(17 U.S.C. section 105); that basis is statute, not a statement at the source (not re-read here).
DR4.2's rule decides on what the source states, so the verdict is `include=False` pending an
owner decision, recorded in `SourceLicence` (`credit_fdic.LICENCE`, checked 2026-11-04) and as an
`EVAL_ONLY` row in `docs/research/data_candidates.csv`. `run()` refuses it until the owner flips
the entry. FFIEC bulk call reports were not used. The 2026-10-19 note that the failures list is
"public domain" is therefore too strong as a source statement.

## Data (MEASURED 2026-11-04, API indexes of 2026-08-19 and 2026-08-25)

* Download 414 MB (`financials.csv`) plus 0.4 MB (`failures.csv`), under 5 GB, in
  `data/real/fdic/` (gitignored). 106 quarter-end pulls, `REPDTE` 2000-03-31 to 2026-06-30.
* 751,322 bank-quarter rows, 11,793 banks, 57 numeric Call Report fields plus bank class and
  state. After label rules: 733,528 rows, 79 features, **2,303 positive rows (0.314%)**.
* Failures list: 4,117 records (3,524 FAILURE, 593 ASSISTANCE). Since 2000: 581 FAILURE
  records, every one matched to a bank in the financials. Three pre-2000 CERTs failed twice,
  which is why only failures from 2000 on label rows.
* Failures per year since 2000: 2000: 7, 2001: 4, 2002: 11, 2003: 3, 2004: 4, 2007: 3,
  2008: 25, 2009: 140, 2010: 157, 2011: 92, 2012: 51, 2013: 24, 2014: 18, 2015: 8, 2016: 5,
  2017: 8, 2019: 4, 2020: 4, 2023: 5, 2024: 2, 2025: 2, 2026: 4. **465 of 581 (80%) fall in
  2008-2012.**

## Label, embargo, splits

* Row = bank-quarter. Positive when a FAILURE record's `FAILDATE` lies in the 366 days (four
  quarters) that start 35 days after `REPDTE`, the Call Report filing deadline. 267 rows with a
  failure inside the 35 days are dropped; 17,529 rows whose window ends after 2026-11-04 are
  dropped (censored). Embargo 401 days.
* Never features: `ACTIVE`, `ENDEFYMD`, `EFFDATE`, `RISDATE`, `INSAGNT1`, names.
* Shared suite splits (out-of-time, 1/2, 2/3, 5/6 quantiles of `REPDTE`), MEASURED:

| split | train rows (positive) | test rows (positive) | test window |
| --- | --- | --- | --- |
| 0 | 332,332 (580) | 118,769 (873) | 2010-03-31 to 2013-12-31, crisis tail |
| 1 | 455,969 (2,053) | 122,292 (109) | 2014-03-31 to 2018-12-31 |
| 2 | 583,938 (2,228) | 127,289 (66) | 2019-03-31 to 2025-06-30 |

  The crisis cluster is in test for split 0 and in training for 1 and 2, so the three splits
  ask different questions and must be reported per split, not averaged. Splits 1 and 2 have few
  test positives (109 and 66 positive rows; a failed bank contributes up to four).
* `crisis_split`: train to 2006-12-31, test 2008-01-01 to 2011-12-31, embargoed: 254,805 train
  rows (102 positive), 128,495 test rows (1,771 positive). Strict institution-grouped variant:
  43,304 train rows (102 positive). Of the 465 banks failing in the test window, 457 have
  training rows, so the ungrouped split lets the model see the failing bank's earlier healthy
  quarters, which is how a monitoring model is used; the grouped variant is the strict check.
* No institution grouping in the shared splits: a bank spans up to 105 quarters, and a
  borrower-style rule would leave only banks that exited before the window.

## SEC-filer overlap

Not checked. Many bank holding companies file with the SEC, and FinTFM-R's pretraining starts
on EDGAR. **FinTFM-only** (FinTFM, synthetic-only) until the check is done; no FinTFM-R number
was computed, and `fdic` is in `EVAL_ONLY_SOURCES` so the FinTFM-R sampler refuses it. Overlap
with TabArena and BeyondArena was not checked either (`name_clashes`, not run).

## SMOKE-TEST (pipeline check only; says nothing about model quality)

Logistic regression and LightGBM (1,000 trees, learning rate 0.05, the suite's settings), CPU,
2,000 training rows (400 positive random sample, so the train rate is 20%; crisis split: 102
positive, all of them) and 3,000 test rows at the natural rate, 5,000 rows per split. Median
imputation and standardisation fitted on the training sample. One seed. Test positives are 28,
4, 1 and 50, so every figure below is noise-level; the 1-positive split is meaningless.
Average precision first:

| split | test positives | LR AP | LightGBM AP | LR AUC | LightGBM AUC |
| --- | --- | --- | --- | --- | --- |
| shared 0 | 28 | 0.5771 | 0.3769 | 0.9673 | 0.9866 |
| shared 1 | 4 | 0.2715 | 0.5276 | 0.9979 | 0.9983 |
| shared 2 | 1 | 0.0006 | 0.0005 | 0.4135 | 0.3855 |
| crisis | 50 | 0.2228 | 0.4100 | 0.9291 | 0.9766 |

Mean predicted probability is 0.18 (LR) and 0.08 (LightGBM) on shared 0 against a 0.93% base
rate: the enriched training rate shows up as a level error, as expected. Source:
`runs/agents1104/DR4.3/smoke.csv`; the full-data counts are in `measure.txt` there.

## Open

* Owner decision on the licence (admit or not).
* SEC-EDGAR overlap check before any FinTFM-R number; TabArena overlap check.
* The full run (FinTFM, logistic regression, LightGBM at 1,000 trees) on the 733,528 rows is
  not done. FinTFM's cost is 6.4 to 8.5 ms per test row at `max_context` 2,000 (§174, §177,
  MEASURED on other datasets), so about 118,769 to 127,289 test rows per split is ESTIMATED at
  0.2 to 0.3 hours per split, not measured here.
* Whether the shared splits should be replaced by a crisis-aware pair for this dataset.
