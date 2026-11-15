# TabBench v2: licence and published protocol

Checked 2026-11-15. Nothing was run, and no dataset or checkpoint was downloaded. The only
files fetched were the benchmark repository's `LICENSE`, `README.md`, `CHANGELOG.md`,
`pyproject.toml`, its file tree, OpenML metadata records (descriptions, not data), and one PDF.

Naming: `CLAUDE.md`'s no-commercial-peer-names rule covers this repository, so the benchmark's
publisher and the model it evaluates are written here as "the publisher" and "the vendor model".
The benchmark itself is named TabBench. The vendor's name appears in the sources below only
through quoted text, where it is replaced by `[vendor]`. A reader who needs the URLs can find the
repository link in `docs/results/FINDINGS.md` §134.

## What "TabBench v2" is

A public suite of **189 OpenML classification datasets** (binary and multi-class), evaluated with
5 stratified shuffle folds, 20% of rows held out per fold, ROC-AUC as the optimisation target and
accuracy as a secondary metric (report §3.1, p. 10). The publisher announced v2 on 3 June 2026
(blog post, not a technical report). The repository's own `CHANGELOG.md` has one entry (0.1.0,
2025-07-18) and the README never says "v2"; v2 exists only in the announcement and the report.

**The proposal's "18 of 22" and "15 of 22" are not TabBench v2 figures.** They come from a
private portfolio of 22 anonymised production problems in the vendor's own report (§4, pp. 14-16).
TabBench v2 itself has 189 datasets and no such count. See "Published figures" below. This matters
for task 35.5: the public suite cannot reproduce the 22-problem claim, and its data are not ours to
obtain.

## 35.2 Licence verdicts

Sources: the benchmark repository's `LICENSE` (raw file, read in full, 214 lines); GitHub's
licence detection for the repository (`NOASSERTION`, "Other"); the `README.md`; `pyproject.toml`;
OpenML dataset records for all 251 dataset configs in the repository's `src/tabbench/datasets/openml/`.

### Benchmark code (`tabbench`)

The README badge and its last line say "Apache 2.0", and `pyproject.toml` carries the classifier
"License :: OSI Approved :: Apache Software License". **The `LICENSE` file is not plain
Apache-2.0.** Its header reads:

> [vendor] License / Version 1.0, June 2025 / This license is a derivative of the Apache 2.0
> license (http://www.apache.org/licenses/) with a single modification: The added Paragraph 10
> introduces an enhanced attribution requirements.

Paragraph 10, verbatim apart from the vendor's name:

> 10. Attribution Requirements. As a condition of using, modifying, or sharing the materials, you
> agree to the following: a. Distribution of Materials: When you distribute or make publicly
> available the Work, any modifications of it, or any service or product that contains or is built
> upon its source code, you must: i. Include the full text of this License with the distributed
> materials. ii. Feature the notice "Built with [vendor]-Foundry" in a prominent location on all
> related websites, user interfaces, blog entries, and in all product documentation or "about"
> sections. b. Exception for Internal Use: The obligations outlined in the section 10 do not apply
> to private, internal use for the purposes of testing and benchmarking, so long as no part of the
> work or its results are communicated or made available to any external third party.

Verdict: **commercial use of the code is permitted** (Apache-2.0 terms, patent grant included), with
an attribution condition that the README and `pyproject.toml` do not mention.

- Running it privately for benchmarking is exempt from paragraph 10 **only while no part of the work
  or its results is communicated to an external third party**. Publishing a comparison table
  produced with the harness is a communication of results. Whether that triggers paragraph 10(a)
  is unclear from the text, because 10(a) is worded around distributing the Work or a product built
  on its source. This is a question for the owner (and counsel), not one to settle here.
- The harness depends on the vendor's workflow package (`*_foundry_ce`, `>=0.0.2`). Its PyPI
  classifier says Apache Software License; its repository `LICENSE` header matches the one above
  (confirmed through a fetch summariser, not the raw file, so treat the paragraph 10 text as
  unverified for this second package). Assume the same attribution condition until checked.
- Never import it into `src/fintfm`. The established pattern (FINDINGS §134) keeps the harness in a
  sibling checkout.

### Dataset families

The repository contains **one family**: OpenML datasets. It ships dataset loaders (`DataConfig`
classes with an OpenML id) and cached splits, not the data, and contains no dataset cards. The
publisher's Hugging Face Space, which the README names as the home of "documentation, leaderboards,
datasets", returned **HTTP 401** on 2026-11-15 (anonymous requests, both the page and its API), so
no dataset card could be read there. The report (§3.1) says the 189 datasets come from OpenML
CC-18, the AutoML benchmark, TabZilla and a curated extension; none of those four sources was
opened for this note.

Verdict per dataset comes from the `licence` field of each OpenML dataset record, which is
uploader-supplied and unaudited. Census over all 251 ids in the repository (the 189-dataset
subset used by the report could not be identified: no `base_datasets.json` is in the repository
tree, so this is a superset):

| OpenML licence string | count | verdict |
| --- | ---: | --- |
| CC BY 4.0, CC BY, CC0 (three spellings), Public Domain, BSD (from scikit-learn) | 63 | commercial use permitted, attribution where CC BY |
| CC BY-SA (6), ODbL + DbCL (1), DbCL v1.0 (1), GPL (>= 2) (1) | 9 | conditions apply (share-alike, database-licence terms, copyleft); do not use without per-dataset review |
| "Non-commercial scholarly and teaching purposes." (id 44987, `socmob`) | 1 | **not permitted** for commercial use |
| "Public" (157), "public" (13), "Publicly available" (1), "See source" (3), "Unknown" (1), "CC-4Y" (1), "Unknown (Kaggle)/Custom (FICO website)" (1), a UCI-and-university credit line (1) | 178 | **unknown, excluded.** OpenML's "Public" states visibility, not terms; I read it as no licence statement, which is an inference |

The 178 unknown datasets include FICO-HELOC (id 45553) and credit (id 44089), which sit closest to
our domain. Nothing here clears them.

Practical reading: evaluating a model on a public benchmark's datasets is the case `CLAUDE.md`
already permits in principle, and no dataset enters the repository. The licence column matters if
we publish derived numbers per dataset or redistribute any split. Only the 63 clear rows could be
named without a further check.

### Verdict on the task

Task 35.2's Verify ("the licence quoted in the finding, not summarised") is met for the code
licence above, with the vendor's name replaced. It is **not closed**, for three reasons: the
README licensing line the task asks for was outside this session's remit (README is off limits
here); the publisher's dataset cards are unreachable (401); and the licence quote is redacted.

## 35.3 Published protocol and figures

Source: the vendor's technical report, 23 PDF pages, LaTeX with hyperref, stamped "TECHNICAL REPORT |
June 12, 2026" and "arXiv:submit/7708190 [stat.ML] 12 Jun 2026", first author Atienza, ten authors.
Read from the PDF with `pdftotext -layout` and the tables copied from that text. The vendor's web
page for the same report carries a date of 25 August 2026, which does not match the PDF's stamp; I
did not establish which is correct, and no arXiv identifier was resolved (the stamp is a
submission number). Page numbers below are the **printed** numbers; PDF page = printed + 1.

### Protocol (§3.1, pp. 10-11)

- 189 OpenML classification datasets, "up to ~2,000 features after preprocessing", "a few hundred to
  several hundred thousand rows" (the announcement post says 100 to 150k rows and does not agree).
- 5 stratified shuffle folds, 20% held out per fold. Per-dataset numbers are averaged over folds,
  then aggregated across datasets.
- Tuned baselines (XGBoost, CatBoost, LightGBM, RealMLP, TabM, ModernNCA): 100 Optuna trials per
  dataset on a nested 5-fold split, top trials bagged.
- Tabular foundation models run zero-shot with each one's default ensemble strategy. The vendor
  model is called through the vendor's public API, not as local weights.
- **Imputation rule (p. 11):** where a model produces no result, the XGBoost score is imputed
  (the text says "GBDT average performance" in one place and "XGBoost's score" in the Table 2
  caption; I report the caption's wording). The imputed share is reported per model.
- ROC-AUC is the optimisation target, accuracy secondary.

### Table 2, p. 11 (copied from the text layer, which was clean)

Held-out: yes, 20% test split per fold. Metric: ROC-AUC and accuracy, mean ± std across 189 datasets.

| Method | ROC-AUC | Accuracy | Rank | Imputed (%) | Won (%) |
| --- | --- | --- | --- | ---: | ---: |
| XGBoost | 0.884 ± 0.115 | 0.828 ± 0.163 | 9.1 ± 2.6 | 0.0 | 1.1 |
| TabM | 0.885 ± 0.119 | 0.844 ± 0.151 | 9.0 ± 3.0 | 0.0 | 2.1 |
| TabICL v1 | 0.901 ± 0.108 | 0.854 ± 0.143 | 5.6 ± 2.6 | 0.0 | 3.2 |
| Mitra | 0.895 ± 0.111 | 0.843 ± 0.161 | 6.4 ± 2.6 | 28.6 | 4.8 |
| TabPFN-2.5 | 0.894 ± 0.113 | 0.842 ± 0.161 | 5.9 ± 2.8 | 16.4 | 6.3 |
| LimiX | 0.898 ± 0.111 | 0.844 ± 0.167 | 5.5 ± 2.8 | 19.0 | 7.9 |
| TabICL v2 | 0.902 ± 0.113 | 0.859 ± 0.148 | 3.3 ± 2.6 | 2.1 | 22.8 |
| TabPFN v3 | 0.906 ± 0.106 | 0.862 ± 0.142 | 3.3 ± 2.2 | 0.0 | 27.5 |
| Vendor model | 0.904 ± 0.109 | 0.861 ± 0.143 | 3.6 ± 2.5 | 0.0 | 14.8 |

Rows for LightGBM, CatBoost, RealMLP, ModernNCA and TabDPT are in the report and omitted here
because the comparison needs only the leaders and the strongest tree ensemble. "Won" means strictly
the best AUC on a dataset, ties credit nobody.

Statements the text makes on pp. 11-12: TabPFN v3's mean AUC is 0.2 pp above the vendor model and
0.4 pp above TabICL v2, "all inside the noise floor of a 189-dataset benchmark"; the three leaders'
mean ranks "cannot be separated at the standard alpha = 0.05 level" (Fig. 1, p. 12; test not
described in the passage read); the vendor model runs natively on 100% of the suite; TFMs together
win 88.4% of the 189 datasets and the six non-TFM models 6.4% (p. 12). The announcement post gives
82.5% for a different head-to-head and no method, so it is not used.

### The 22-problem figures (§4, pp. 14-16): private data, not TabBench

- A "private portfolio of 22 problems spanning five sectors" (retail, behavioural, equity,
  transportation, energy), mixing "private datasets shared by client engagements with public
  datasets" (p. 15). Client identities, names and schemas are withheld. **Not reproducible by us.**
- Compared: XGBoost, LightGBM and the three leading TFMs only.
- Vendor model beats TabPFN v3 on 18 of 22 problems and TabICL v2 on 15 of 22; beats XGBoost on
  12 of 22 and LightGBM on 14 of 22 (p. 16).
- Mean rank across five models: vendor model 2.21, XGBoost 2.27, LightGBM 3.27, TabICL v2 3.55,
  TabPFN v3 3.71 (p. 16). Outright wins within the three TFMs: 50%, 18% (TabPFN v3), 9% (TabICL v2),
  23% ties (Fig. 3, p. 15).
- The report's own text says tuned trees lead on retail, behavioural and equity sectors (p. 16-17).
- The vendor's authors wrote the report, own the model and built the benchmark. These are
  vendor-reported figures, not independent ones.

## Gaps stated before capabilities

- No dataset card for any TabBench dataset could be read; the Hugging Face Space answered 401.
- The 189-dataset list was not obtained, so licence counts are over a 251-config superset.
- OpenML licence strings are uploader-supplied; 178 of 251 carry none that can be read as terms.
- The report's date is inconsistent between PDF (12 June 2026) and web page (25 August 2026).
- Our checkpoint still scores below its own gate (FINDINGS §42), so tasks 35.1 and 35.4 are open and
  35.5 cannot be written.

## Sources

1. TabBench repository, files `LICENSE` (214 lines), `README.md`, `CHANGELOG.md`, `pyproject.toml`,
   and its git tree listing, fetched 2026-11-15. GitHub licence detection: `NOASSERTION`. Link:
   see `docs/results/FINDINGS.md` §134.
2. OpenML dataset records (`/api/v1/json/data/<id>`, field `licence`), 251 ids, fetched 2026-11-15.
   Metadata only.
3. Vendor technical report, PDF, 23 pages (see REFERENCES.md entry "Atienza et al. (2026)").
4. Publisher's TabBench v2 announcement, 3 June 2026 (web page; used only for the date, the dataset
   count and the absence of licence or fold information).
5. Hugging Face Space for TabBench: HTTP 401 on 2026-11-15, not readable.
