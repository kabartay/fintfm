# DR4.3: Estonian register panel (draft, 2026-11-04)

**Result.** A firm-year bankruptcy panel can be built from the register's open data, but its
label is survivor-conditioned, not a default label, so the suite entry is registered with
`include=False`. The licence passes. The owner decides whether to admit it.

## Gaps first

1. **The open files hold only firms currently in the register** (statuses entered, in
   liquidation, in bankruptcy), as the download page states. A firm declared bankrupt and then
   deleted has no registry card, so its bankruptcy date is not in the data.
2. **Deleted firms stay in the annual-report files** (status `Kustutatud`, 129,890 of
   1,681,151 report rows, MEASURED) with no deletion date and no reason. Deleted after
   bankruptcy, after a voluntary liquidation or for not filing look the same, so their label is
   unobservable and they are dropped.
3. **Positives are therefore firms whose proceedings were still open on 2026-11-04**, and
   negatives are firms alive then. Both are selected on 2026 state. The registry cards hold
   765 bankruptcy declarations (755 firms, MEASURED) and the yearly count rises toward the
   present: 25 (2019), 40 (2020), 40 (2021), 39 (2022), 73 (2023), 114 (2024), 132 (2025), 140
   (2026 to 3 November). That shape is what deletion of older cases produces. No external count
   of Estonian bankruptcies was read, so how many are missing is not measured.
4. **Very few positives.** 260 positives in 1,251,575 firm-year rows (0.0208%, MEASURED). A
   5,000-row uniform sample would hold about one positive.
5. **Licence statement is a link, not a sentence.** The pages link the CC BY 4.0 legal code as
   "License information"; none says "this dataset is licensed under CC BY 4.0".
6. The label is an observed legal event, so a score on it would be a comparison with observed
   outcomes, but the conditioning above biases it. Do not quote it as a bankruptcy-prediction
   result.

## Licence (read 2026-11-04)

- URLs: `https://avaandmed.ariregister.rik.ee/en/downloading-open-data` and
  `https://avaandmed.ariregister.rik.ee/en/terms-service`.
- As stated: "Downloading open data requires acceptance of the license terms"; link text
  "License information" to `https://creativecommons.org/licenses/by/4.0/legalcode`; "Data, the
  reuse of which we have not limited (i.e. open data) and which may be used for other
  commercial or non-commercial purposes than the original purpose of collecting the information
  are published at ... rik.ee/en/open-data". The same page warns that personal data in the files
  is subject to GDPR. This module never reads the representative, shareholder or beneficial-owner
  files.
- Verdict: CC BY 4.0, from the register's operator (the data's producer), allows commercial
  evaluation use with attribution. Passes DR4.2's rule. Recorded in `credit_suite.DATASETS`
  with `include=False` because of the label, not the licence. Attribution text for any public
  use is still to write. `README.md`'s licensing section needs a line (not edited here).

## Data (MEASURED 2026-11-04)

Eleven files from the page, 0.58 GB zipped (about 10 GB unzipped; the 1.8 GB rulings file was
downloaded and deleted unused): the report list (1,681,151 rows; 7,313 consolidated), key
indicators for FY2019 to FY2025 reports (1,525,052 of the kept reports have at least one
indicator), registry cards, basic data. They sit under `data/estonia/` (gitignored); the zips
are in `runs/agents1104/DR4.3/`.

## Label and split (as implemented in `fintfm.evaluation.credit_estonia`)

- Row: one non-consolidated, non-deleted annual report per firm and year (earliest filing).
- Label: bankruptcy declared in (filing date, filing date + 365 days]. Reports filed on or after
  a declaration are dropped; reports whose window ends after 2026-09-30 are dropped as censored.
- Features (25): balance-sheet and income ratios, log assets, log revenue, log firm age, audited
  flag, headcount, legal form, signs of equity and profit. The register's current status is
  never a feature: it contains the label.
- Split: out-of-time on filing date, cuts at the 1/2, 2/3 and 5/6 quantiles, 365-day embargo,
  grouped by registry code (the suite's protocol).
- Kept rows 1,251,575; positives by filing year: 2019 1 of 225, 2020 7 of 147,650, 2021 18 of
  175,925, 2022 28 of 200,124, 2023 59 of 238,279, 2024 79 of 258,374, 2025 68 of 230,998.

## SMOKE-TEST (pipeline only, says nothing about model quality)

`credit_suite.run` on 5,000 rows: all 260 positives plus 4,740 negatives drawn at random
(seed 0), so the base rate is 5.20%, not the panel's 0.0208%. Logistic regression and LightGBM
at 1,000 trees, no FinTFM, 3 out-of-time splits, test sets of 853, 833 and 834 rows (57, 71 and
61 positives). CPU, nice 19, 2 threads. Output `runs/agents1104/DR4.3/smoke_estonia.csv`.

| model | mean AP | mean ROC-AUC | mean top-decile AP |
| --- | --- | --- | --- |
| logistic regression | 0.4726 | 0.8973 | 0.3479 |
| LightGBM (1,000 trees) | 0.6314 | 0.9057 | 0.5422 |

Test base rates were 6.68%, 8.52% and 7.31%; mean predicted probabilities were below them for
both models (LightGBM 2.84% to 4.45%). These APs sit above an enriched base rate and the ROC-AUC
near 0.9 is high for a one-year bankruptcy label. With positives selected on still-open
proceedings, part of that may be a signature of the selection (firms with open proceedings
filing near-empty reports) rather than predictive signal; not tested.

## Open

- Owner: admit as a survivor-conditioned stress set, or leave gated. A usable default label
  needs the register's deleted-firm history (deletion dates and reasons), which the open files
  do not carry; the paid/contract register services were not checked.
- FinTFM was not run. Its 25-feature input fits the checkpoint's limit; cost is the test rows.
- A full-size LightGBM/LR run on all 1,251,575 rows was not done (positives 260 make fold
  scores noisy: 28 to 79 positives per filing year).
