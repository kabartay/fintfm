# Licences of the `inria-soda/tabular-benchmark` classification datasets

Checked **2026-11-15** for task 37.1 of `openspec/changes/diverse-eval-suite`. Scope: the 23
classification files in the Hugging Face collection `inria-soda/tabular-benchmark`
(Grinsztajn, Oyallon and Varoquaux, NeurIPS 2022): 16 under `clf_num/` and 7 under `clf_cat/`.
The regression folders are out of scope.

**Result: 10 commercial use allowed, 0 non-commercial only, 13 unknown and excluded.**

## Why this needed doing

The collection card declares `license: []` and its "Licensing Information" section reads
"[More Information Needed]". The collection's code repository (`LeoGrin/tabular-benchmark`) has
no `LICENSE` file. So nothing at the collection level grants anything, and each dataset's terms
come from wherever it was first published.

OpenML's `licence` field is not a substitute. For most of these datasets it reads `Public`,
which OpenML uses as its default label for "openly downloadable". It is not a grant of
commercial rights, so a dataset whose only evidence is `Public` is treated as unknown. Where
OpenML says `CC0` it disagrees with the originating repository (below), and the repository is
taken as the authority.

## Method

1. File list: the Hugging Face tree API
   (`https://huggingface.co/api/datasets/inria-soda/tabular-benchmark/tree/main?recursive=true`)
   and the raw card
   (`https://huggingface.co/datasets/inria-soda/tabular-benchmark/raw/main/README.md`), whose
   "Source Data" tables give each file's original and "new" OpenML ids. No summariser was used.
2. OpenML: `https://www.openml.org/api/v1/json/data/<id>` for both the original id and the
   benchmark's re-upload id, reading `licence`, `description` and the source link.
3. Primary source for each dataset's rights:
   - UCI Machine Learning Repository dataset pages (fetched as raw HTML; each carries a
     "licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license"
     statement and the `licenses/by/4.0/legalcode` link).
   - UK Department for Transport on data.gov.uk for road-safety (licence field, and the site
     footer).
   - For the rest: the originating page where it could be read (PASCAL eye-movements challenge
     page, ChaLearn AutoML data page, ProPublica's `compas-analysis` README via the GitHub API,
     the Kaggle competition pages).
4. A dataset is **allowed** only when a primary source states a licence that permits commercial
   use. Anything else is **excluded**. "Could not read the terms" counts as excluded.

## Table

Verdicts: `commercial use allowed`, `non-commercial only`, `unknown: excluded`. The OpenML id is
the benchmark's re-upload. The licence column is the licence at the primary source.

| File | OpenML id (original) | Licence | Verdict | Source |
|---|---|---|---|---|
| `clf_num/covertype` | 44121 (293) | CC BY 4.0 | commercial use allowed | https://archive.ics.uci.edu/dataset/31/covertype |
| `clf_num/MagicTelescope` | 44125 (1120) | CC BY 4.0 | commercial use allowed | https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope |
| `clf_num/bank-marketing` | 44126 (1461) | CC BY 4.0 | commercial use allowed | https://archive.ics.uci.edu/dataset/222/bank+marketing |
| `clf_num/MiniBooNE` | 44128 (41150) | CC BY 4.0 (OpenML says CC0) | commercial use allowed | https://archive.ics.uci.edu/dataset/199/miniboone+particle+identification |
| `clf_num/default-of-credit-card-clients` | 45020 (42477) | CC BY 4.0 (OpenML says CC0) | commercial use allowed | https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients |
| `clf_num/Higgs` | 44129 (42769) | CC BY 4.0 | commercial use allowed | https://archive.ics.uci.edu/dataset/280/higgs |
| `clf_num/Diabetes130US` | 45022 (4541) | CC BY 4.0 | commercial use allowed | https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008 |
| `clf_cat/covertype` | 44159 (1596) | CC BY 4.0 | commercial use allowed | https://archive.ics.uci.edu/dataset/31/covertype |
| `clf_cat/default-of-credit-card-clients` | 45036 (42477) | CC BY 4.0 (OpenML says CC0) | commercial use allowed | https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients |
| `clf_cat/road-safety` | 45038 (42803) | UK Open Government Licence v3.0 (OpenML says Public) | commercial use allowed | https://data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data |
| `clf_num/pol` | 44122 (722) | none found; OpenML source "Unknown" | unknown: excluded | https://www.openml.org/d/722 |
| `clf_num/house_16H` | 44123 (821) | none found; OpenML source "Unknown" | unknown: excluded | https://www.openml.org/d/821 |
| `clf_num/Bioresponse` | 45019 (4134) | Kaggle competition data; rules page not readable | unknown: excluded | https://www.kaggle.com/c/bioresponse |
| `clf_num/credit` | 44089 (Kaggle) | OpenML says "See source"; Kaggle rules page not readable | unknown: excluded | https://www.kaggle.com/c/GiveMeSomeCredit |
| `clf_num/heloc` | 45026 (Kaggle re-share) | OpenML says "See source"; a third-party Kaggle upload of FICO's challenge data, no licence readable | unknown: excluded | https://www.kaggle.com/datasets/averkiyoliabev/home-equity-line-of-creditheloc |
| `clf_num/california` | 45028 (StatLib) | OpenML says "See source"; the source page did not respond | unknown: excluded | https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html |
| `clf_num/jannis` | 45021 (41168) | ChaLearn AutoML data page states no licence | unknown: excluded | https://automl.chalearn.org/data |
| `clf_cat/albert` | 45035 (41147) | ChaLearn AutoML data page states no licence | unknown: excluded | https://automl.chalearn.org/data |
| `clf_num/electricity` | 44120 (151) | none found; OpenML says Public; the author page did not respond | unknown: excluded | https://www.openml.org/d/151 |
| `clf_cat/electricity` | 44156 (151) | as above | unknown: excluded | https://www.openml.org/d/151 |
| `clf_num/eye_movements` | 44130 (1044) | PASCAL challenge page states no licence | unknown: excluded | http://www.cis.hut.fi/eyechallenge2005/ |
| `clf_cat/eye_movements` | 44157 (1044) | as above | unknown: excluded | http://www.cis.hut.fi/eyechallenge2005/ |
| `clf_cat/compas-two-years` | 45039 (42192) | ProPublica's repository README and repository metadata state no licence for the data | unknown: excluded | https://github.com/propublica/compas-analysis |

## Where sources disagreed

- **MiniBooNE** and **default-of-credit-card-clients** (both folders): OpenML records `CC0`;
  the UCI pages say CC BY 4.0. Both permit commercial use. The stricter UCI statement is
  recorded, so attribution is owed.
- **covertype, MagicTelescope, bank-marketing, Higgs, Diabetes130US, road-safety**: OpenML
  records `Public`, which is no licence. The UCI or DfT statement is used instead.
- **Diabetes130US**: OpenML lists the author as Attila Reiss, who belongs to a different UCI
  dataset; the description and citation are the UCI 296 hospital data (Strack et al. 2014). The
  licence is read from UCI 296.
- **road-safety**: the OpenML copy is Cerda and Varoquaux's 2020 join of the DfT road-safety
  data with vehicle make and model, up to 2015. The OGL statement is read from DfT's current
  data.gov.uk record for the series, not from a statement on the OpenML copy, so the verdict
  rests on the upstream publisher's licence. The DfT record now says the data has moved to the
  Road Safety Data page.

## Caveats

- CC BY 4.0 and OGL v3.0 both require attribution. Anything that ships or reports on these
  datasets should carry the citations given on the UCI and DfT pages.
- "Excluded" means "terms not established", not "known to prohibit". If a dataset's terms are
  later read from a primary source, move its row and cite the page. Do not infer from another
  copy of the data.
- The Kaggle pages and the ChaLearn page are rendered by script, and the fetched HTML did not
  contain the competition rules. A browser read of those pages could change an exclusion, only
  on a stated licence.
- This is evidence for an internal evaluation-suite decision, not legal advice.
