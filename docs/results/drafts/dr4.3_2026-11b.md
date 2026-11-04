# DR4.3: open checks on the FDIC and Estonia datasets (draft, 2026-11-04)

Follows `dr4.3-fdic_2026-11.md` and `dr4.3-estonia_2026-11.md`. Neither dataset is admitted; no
FinTFM or FinTFM-R number was computed. Not in `docs/results/FINDINGS.md`; the owner folds it in.
Raw output: `runs/agents1104/DR4.3b/` (`sec_overlap.log`, `fdic_sec_matches_primary.csv`,
`rowhash.log`, `rowhash.csv`, `control.log`, `est5000.log`, `est30000.log`,
`estonia_selection_*.csv`).

**Result.** (1) FDIC cannot be declared clean of SEC filers: a conservative name match against
the only EDGAR filer list reachable (2023 filings) already flags 6.72% of rows, and it is a
floor, so FinTFM-R must not be scored on FDIC until a historical filer list is matched.
(2) Neither dataset shares a row or a column with any TabArena-v0.1 or V4FinBench dataset.
(3) The Estonia smoke result is not an emptiness artefact; it runs the other way (positives file
fuller reports), and it is mostly firm size and activity. The selection concern changes shape but
is not resolved.

## 1. FDIC and SEC EDGAR filer overlap

### Gaps first

* **The SEC filer list is incomplete.** `www.sec.gov` returned HTTP 403 ("Request Rate Threshold
  Exceeded") for `company_tickers.json`, `cik-lookup-data.txt` and `full-index` with the
  User-Agent `fintfm-research`, on repeated attempts over several minutes. Only `data.sec.gov` answered
  (per-CIK submissions, which need the CIK first). The filer list used is therefore the local
  EDGAR Financial Statement Data Sets that FinTFM-R's panel is built from
  (`data/edgar/raw/2023q1..q4/sub.txt`): 28,742 filings, **7,692 distinct filers (CIK)**, of
  which 1,332 carry a bank-like SIC (60xx, 6111-6399, 6500, 6711, 6712, 6770; 397 are SIC
  6021, 6022, 6035, 6036 or 6712). It lists filers of 2023 only.
* **Direction of error: the counts below are floors for the positives in particular.** 465 of 581
  failures since 2000 (80%) fall in 2008-2012, and their holding companies were mostly gone
  before 2023, so they cannot appear in a 2023 list. The share of positives affected is
  therefore understated by an amount not measured here.
* FDIC institution records come from the BankFind API (`/institutions`, index of 2026-10-30,
  27,835 institutions, fields `CERT, NAME, RSSDHCR, NAMEHCR`). 17,075 carry a holding-company
  name; 2,124 of the 11,613 institutions that have frame rows carry none. 171 frame CERTs are
  not in the institutions list (rows not matched, counted as not flagged).

### Matching rule

Names upper-cased, `&` to AND, state tags (`/DE/`), punctuation and the tokens INC, CORP,
CORPORATION, CO, COMPANY, LTD, LLC, LP, THE, DE, NEW, INCORPORATED removed.

* **Primary (conservative):** normalised `NAMEHCR` equals a normalised EDGAR filer name whose SIC
  is bank-like, **or** the bank's own `NAME` equals one. Exact match only. Errors toward
  undercounting (renamed or merged holding companies, spelling variants, filers outside 2023);
  a few false positives are possible for generic names shared by unrelated firms.
* **Upper (sensitivity):** primary, plus a match after also dropping generic words (BANCORP,
  BANCSHARES, FINANCIAL, BANK, HOLDING, GROUP, TRUST, SAVINGS, NATIONAL, FIRST, ...). Errors toward
  overcounting (it matched "PROGRESSIVE BANK INC" and "FIRST CITIZENS CORP" style pairs that are
  probably different firms). The truth lies between the two only for the 2023 list; see the
  floor note above.

### Counts (MEASURED 2026-11-04; frame 733,528 rows, 2,303 positive rows, 11,613 institutions)

| rule | institutions in frame | distinct holding-company names | rows (share) | positive rows (share) |
| --- | --- | --- | --- | --- |
| primary: holding company, bank-like SIC | 840 | 248 | 48,395 (6.60%) | 20 (0.87%) |
| primary: holding company, any SIC | 849 | 257 | 49,138 (6.70%) | 20 (0.87%) |
| bank's own name | 23 | 25 | 1,845 (0.25%) | 0 |
| **primary, combined** | **853** | 264 | **49,318 (6.72%)** | **20 (0.87%)** |
| upper (sensitivity) | 1,412 | 634 | 82,540 (11.25%) | 134 (5.82%) |

Failed banks (at least one positive row): 579; 424 of them have a holding-company name;
**5 match under the primary rule, 34 under the upper rule.** Largest primary matches by rows:
Synovus, Wells Fargo, Wintrust, Heartland, Simmons First, Fulton.

### Which subset would be clean, and what follows

* Banks with **no holding company recorded**: 2,124 institutions, **115,970 rows (15.8%), 616
  positive rows (26.7%)**. A bank without a parent has no holding company to file with the SEC,
  and a bank itself files its reports with its banking regulator, not EDGAR, so this is the
  subset least likely to be covered. It is not proven clean: a recorded absence is today's
  record, and the 2023-only list says nothing about a former parent.
* The complement of the primary rule (676,911 rows, 2,283 positives) is **not** a clean subset,
  for the floor reason above. The complement of the upper rule (643,689 rows, 2,169 positives)
  is the same.
* So: `fdic` stays FinTFM-only (synthetic-only). FinTFM-R scoring needs (a) a historical EDGAR
  filer list (2000 onwards, bank SIC or holding-company names), which needs access to
  `www.sec.gov` or `data.sec.gov` per-CIK lookups, and (b) the no-holding-company subset as the
  working candidate, with a per-split positive count (not measured here). Even a clean subset is
  a different population (standalone banks) from the whole panel.

## 2. Row-hash overlap against the benchmarks (MEASURED)

Tool: `fintfm.evaluation.overlap` (`content_fingerprint`, `parquet_fingerprint`, `content_match`;
flag at row share or column share of at least 0.5, `OVERLAP_SHARE`). References: all **51**
TabArena-v0.1 datasets in the local OpenML cache, and V4FinBench `company_years.parquet`
(1,106,879 rows, 1,106,835 hashed). Candidates: FDIC (733,528 rows, 79 features) and Estonia
(1,251,575 rows, 25 features).

Gaps: row hashes are from a **uniform random sample of 50,000 rows** of each candidate (seed 0;
FDIC 50,000 unique hashes, Estonia 49,888), compared with every reference row. A hit rate above
about 6 in 100,000 would be missed with 95% probability. Column fingerprints were taken on the
full columns (a sampled column's hash cannot match by construction); only columns with at least
10 distinct values are fingerprintable: FDIC 56 of 79, Estonia 14 of 25. BeyondArena was not
compared by contents (the existing name check covers it). The V4FinBench reference includes its
`Insolvency_flag` column (no target was dropped), which does not change a no-match result.

| candidate | references | row share, max over references | column share, max over references (shared columns) | flagged |
| --- | --- | --- | --- | --- |
| FDIC | 52 (51 TabArena + V4FinBench) | 0.0000 | 0.0000 (0) | no |
| Estonia | 52 | 0.0000 | 0.0000 (0) | no |

Positive controls (`control.log`): 2,000 V4FinBench rows against a 100,000-row V4FinBench
reference, row share 1.0; 500 rows of TabArena 46918 (credit-g) against its own parquet
fingerprint, row share 1.0, flagged. The first control in `rowhash.log` read 0.0 because it
dropped a column the reference kept; that is a control error, replaced by `control.log`, and it
does not touch the candidate rows. Per-reference values: `rowhash.csv`.

## 3. Estonia selection effect (MEASURED, SMOKE-TEST scale)

Logistic regression and LightGBM (1,000 trees, learning rate 0.05), no FinTFM, no torch imported
(asserted in the script), CPU, nice 19, 2 threads. Same protocol as the smoke run: median
imputation and standardisation fitted on the training side, three out-of-time splits (cuts at
1/2, 2/3, 5/6 of filing date, 365-day embargo, grouped by registry code), mean over the three
splits. Sample A reproduces the smoke run exactly (all 260 positives plus 4,740 random negatives,
seed 0, base rate 5.20%; LR 0.4726 AP / 0.8973 AUC and LightGBM 0.6314 / 0.9057, matching
`dr4.3-estonia_2026-11.md`). Sample B is all 260 positives plus 29,740 negatives (30,000 rows,
base rate 0.87%). One seed; test positives 57/71/61 (A) and 56/73/63 (B), so differences of a
few hundredths in AP are within noise.

**Emptiness score** = count of missing or zero values among 16 report-derived features (ten
ratios, log revenue, headcount, four equity/profit signs). Higher means emptier.

| | positives | negatives |
| --- | --- | --- |
| mean emptiness score (A) | 1.60 | 5.45 |
| rows with at least one missing field (A) | 25.4% | 63.6% |

**The hypothesis is reversed.** Positives file *fuller* reports than negatives (the
negatives include many near-empty micro-firm filings), not emptier ones. As a score, "emptiness"
has AUC 0.2513 (A) and 0.2467 (B), i.e. 0.749 and 0.753 when read as "fullness"; the
univariate AUC of low total assets is 0.168 (A), i.e. 0.832 as "large assets"; low revenue 0.228,
low headcount 0.207. A control feature unrelated to size (log firm age) gives 0.534 (A).

| arm (features used) | LR AP / AUC (A) | LightGBM AP / AUC (A) | LR AP / AUC (B) | LightGBM AP / AUC (B) |
| --- | --- | --- | --- | --- |
| all 25 | 0.4726 / 0.8973 | 0.6314 / 0.9057 | 0.1734 / 0.9001 | 0.2315 / 0.8669 |
| minus the totals (log assets, log revenue) | 0.3887 / 0.8623 | 0.5850 / 0.8823 | 0.1203 / 0.8510 | 0.1804 / 0.8457 |
| minus totals, signs, headcount | 0.2659 / 0.8210 | 0.4729 / 0.8505 | 0.0611 / 0.8248 | 0.1301 / 0.8024 |
| ten ratios only | 0.2669 / 0.8284 | 0.4897 / 0.8511 | 0.0811 / 0.8388 | 0.1084 / 0.7965 |
| age, audited, legal form only (no report field) | 0.1521 / 0.6303 | 0.1091 / 0.5783 | 0.0295 / 0.6312 | 0.0193 / 0.5659 |

Univariate rows (AP / AUC mean, A): emptiness 0.0590 / 0.2513; missing count 0.0636 / 0.2919;
low log assets 0.0439 / 0.1684; low log revenue 0.0481 / 0.2282; low headcount 0.0465 / 0.2070;
log firm age 0.0839 / 0.5339. Base rate A 5.2% (mean test base rate about 7.5%).

Reading, within the noise stated:

* Dropping everything that encodes emptiness or size leaves ROC-AUC 0.82 to 0.85 (A) with ten
  ratios, against 0.90 to 0.91 with all features. The AUC is not driven by the emptiness
  features alone, but the totals and field-presence signs carry a visible part of it:
  average precision falls by 0.08 to 0.21 (A) when they go (LR 0.4726 to 0.2659; LightGBM 0.6314
  to 0.4729).
* The ratios themselves can carry presence information (a ratio is missing when a field is
  absent), so the ten-ratio arm is not emptiness-free. This was not separated further.
* What the signal looks like is "larger, fuller-reporting firms are the ones with open
  bankruptcy proceedings in 2026", against a negative class dominated by small, sparse filers.
  That is a population-mix effect, consistent with the survivor conditioning already noted
  (positives are firms still in the register with proceedings open). The data cannot say how
  much is selection and how much real distress signal; a held-out set with deleted-firm history
  would be needed.
* The headline ROC-AUC of about 0.90 should not be quoted for Estonia as bankruptcy-prediction
  skill. Scale: 5,000 and 30,000 rows of 1,251,575, one seed, 260 positives in total.

## Open

* Historical SEC filer list (2000 onwards) for the FDIC check: `www.sec.gov` was blocked from this
  machine with the stated User-Agent; needs access or per-CIK `data.sec.gov` lookups from bank
  SICs.
* Row hashes use 50,000-row candidate samples; a full-row hash of both candidates would remove
  the sampling caveat (about 15 minutes at the observed speed for the V4FinBench reference, not
  repeated).
* Estonia: a full-size (all 1,251,575 rows) run and more seeds; separating presence from value
  in the ratios (add explicit missing indicators to an arm, or impute with a constant).
