# Licences of the proposed extra baselines: code and weights, checked separately

Checked **2026-11-15** for task 18.6 ("Added 2026-09-14, licensing-gated") of
`openspec/changes/strong-baselines`. Candidates named there: TabICLv2, TabDPT, TabPFN,
FT-Transformer, TabM, RealMLP. Nothing was installed, run or downloaded: the only files read were
licence texts, READMEs, `pyproject.toml` files, file trees and registry metadata, fetched raw with
`curl -sL` (no summariser).

This note answers a licensing question only. The standing rule in `CLAUDE.md` is independent of
licence: **no code, weights or training data from any tabular-foundation-model product enters this
repository.** A permissive licence changes only whether a model may be *evaluated outside* this
repository (verdict (a)) and whether it could sit in a commercial setting (verdict (b)).

## Result

| Baseline | Code licence | Weights licence | (a) run as an evaluation baseline outside this repo | (b) commercial use |
|---|---|---|---|---|
| TabICLv2 | BSD-3-Clause (Apache-2.0 for `src/tabicl/forecast` only) | BSD-3-Clause, **by the Hub card field only** (no LICENSE file in the weights repo) | Yes | Code yes. Weights yes as stated, but the grant is a card field, not a licence text: confirm with Soda-Inria before any commercial reliance |
| TabDPT | Apache-2.0 | Apache-2.0 (Hub LICENSE file and card agree) | Yes | Licences permit it. Provenance of the real-data training corpus not checked, so not cleared |
| TabPFN | Apache-2.0 (package 10.0.0) | **Non-commercial**: 3.5 (default), 3, 2.6, 2.5. TabPFN-2: Prior Labs License v1.1 (Apache-2.0 plus a "Built with PriorLabs-TabPFN" attribution paragraph) | 3.5, 3, 2.6, 2.5: **excluded** here, see below. TabPFN-2: permitted subject to attribution, not run | 3.5, 3, 2.6, 2.5: **no**, outputs included. TabPFN-2: permitted with attribution. A commercial licence is sold separately |
| FT-Transformer | Apache-2.0 (`rtdl-revisiting-models`, `rtdl`) | none distributed | Yes | Yes for the code |
| TabM | Apache-2.0 | none distributed | Yes | Yes for the code. Dependency `rtdl_num_embeddings` is MIT on GitHub, Apache on PyPI (see disagreements) |
| RealMLP (via `pytabkit`) | Apache-2.0 | none distributed (the model is trained per dataset) | Yes | Yes for the code. The `models`, `autogluon`, `hpo` and `extra` extras were not checked |

"None distributed" means no `.pt`, `.pth`, `.ckpt`, `.safetensors` or `.bin` file in the file tree
at the commit named below (pytabkit and rtdl-revisiting-models trees complete; the TabM tree was
truncated by the GitHub API at 51,995 entries, so for TabM this rests on its README and
`pyproject.toml`, which describe a library that trains from scratch and mention no checkpoint).

## Excluded on licensing grounds

**TabPFN 3.5, 3, 2.6 and 2.5 weights are excluded as baselines for this project.** Their licences
permit only "Non-Commercial Purposes" and bar use of the model, derivatives and **outputs** in
commercial decision-making, client deliverables, paid products or revenue-generating activity
(TABPFN-3.5 License v1.0, revised 2026-09-09, clauses on Non-Commercial Purposes and Outputs).
Access also requires accepting the licence through a Prior Labs account (README, 2026-11-13
commit). FinTFM is developed towards a commercial product, so the grant is not one to assume;
whether any single research run counts as non-commercial is the owner's decision, not ours. Until
then TabPFN is a **published-numbers peer only**, as `docs/paper/RELATED_WORK.md` already treats it.
TabPFN-2 is the one TabPFN weight set with a commercial-compatible licence, and it is older and
weaker than the current default. It is a candidate only if the owner wants a TabPFN-family run.

No other candidate is excluded on licence grounds. TabICLv2's exclusion risk is the thin weights
grant noted above; TabDPT's is the unverified training-data provenance. Neither blocks evaluation.

## Evidence

All fetched 2026-11-15. "Commit" is the default-branch head at fetch time.

### TabICLv2
- Code: `github.com/soda-inria/tabicl`, commit `a626a1b67049b0b1748ebe8c92f7df40aaebd334`
  (2026-11-12). `LICENSE` is "BSD 3-Clause License, Copyright (c) 2025, Soda team @ Inria", followed
  by a second text: "Code in the src/tabicl/forecast directory is currently derived work from
  TabPFN-TS", under Apache-2.0. The GitHub licence API returns `NOASSERTION` because of the two
  texts. PyPI `tabicl` 2.2.0 (2026-09-02): licence text BSD 3-Clause.
  [LICENSE](https://raw.githubusercontent.com/soda-inria/tabicl/a626a1b67049b0b1748ebe8c92f7df40aaebd334/LICENSE)
- Weights: the code downloads from `huggingface.co/jingang/TabICL`
  (`src/tabicl/_sklearn/classifier.py` line 349, `repo_id = "jingang/TabICL"`). The repo holds
  `tabicl-classifier-v2-20260212.ckpt` and `tabicl-regressor-v2-20260212.ckpt` among others; model
  sha `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`, last modified 2026-02-18. Card front matter:
  `license: bsd-3-clause`. The repo has **no LICENSE file**, only the README. The
  `soda-inria/tabicl` Hub path returned an error, so `jingang/TabICL` is the only weights source.
  [card](https://huggingface.co/jingang/TabICL/raw/main/README.md),
  [API](https://huggingface.co/api/models/jingang/TabICL)

### TabDPT
- Code: `github.com/layer6ai-labs/TabDPT`, commit `3cef828a0bbfe8818bcdd6d855940e1949f2b4cd`
  (2026-11-12). `LICENSE` is Apache License 2.0; README: "The code in this repository is licensed
  under the Apache License 2.0". PyPI `tabdpt` 1.3.2 (2026-11-02): Apache-2.0.
  [LICENSE](https://raw.githubusercontent.com/layer6ai-labs/TabDPT/3cef828a0bbfe8818bcdd6d855940e1949f2b4cd/LICENSE)
- Weights: `huggingface.co/Layer6/TabDPT`, model sha `a5ca6e01c0fa09ec68c73e958e5199d1932abb3a`
  (2026-09-08), not gated. Card `license: apache-2.0`; the repo carries a `LICENSE` file whose text
  is Apache License 2.0. Files: `tabdpt1_1/1_2/1_3.safetensors`.
  [LICENSE](https://huggingface.co/Layer6/TabDPT/raw/main/LICENSE)
- Not checked: the licences of the real tables the model was trained on. The weights grant does
  not settle that.

### TabPFN
- Code: `github.com/PriorLabs/TabPFN`, commit `61912fe03115b684559ded7d821d9611ed37c77a`
  (2026-11-13). `LICENSE` is Apache License 2.0 (GitHub licence API: Apache-2.0). PyPI `tabpfn`
  10.0.0 (2026-11-13): Apache-2.0.
  [LICENSE](https://raw.githubusercontent.com/PriorLabs/TabPFN/61912fe03115b684559ded7d821d9611ed37c77a/LICENSE)
- Weights, each from its own Hub repo (all not gated, all with a LICENSE file):
  - `Prior-Labs/tabpfn_3_5`, sha `06bf2ba3...`: TABPFN-3.5 License v1.0, revised 2026-09-09,
    "non-commercial and non-production use". Outputs "may be used only for Non-Commercial
    Purposes", and use of Outputs in production systems, business processes, client deliverables or
    revenue-generating activities needs a separate commercial licence. README (2026-11-13): "TabPFN-3.5
    is used by default".
    [LICENSE](https://huggingface.co/Prior-Labs/tabpfn_3_5/raw/main/LICENSE)
  - `Prior-Labs/tabpfn_3`, sha `24a16a89...`: TABPFN-3 License v1.0 (revised 2026-03-24),
    non-commercial.
  - `Prior-Labs/tabpfn_2_6`, sha `24148873...`: TABPFN-2.6 License v1.0, non-commercial.
  - `Prior-Labs/tabpfn_2_5`, card `license_name: tabpfn-2.5-license-v1.1`, "other". The LICENSE text
    was not read; the README groups 2.5 with the non-commercial releases. Treated as non-commercial.
  - `Prior-Labs/TabPFN-v2-clf`, sha `275f19d2...`: "Prior Labs License, Version 1.1, May 2025":
    Apache 2.0 with an added paragraph 10, "Additional. attribution", requiring "Built with
    PriorLabs-TabPFN" on related websites, interfaces and blog posts. The regressor repo
    (`TabPFN-v2-reg`) was listed but its LICENSE.txt was not opened; the README links the same licence.
- README: commercial licence available from the vendor ("request a commercial license").

### FT-Transformer
- Code: `github.com/yandex-research/rtdl-revisiting-models`, commit
  `e3ed46cac38568785289d8fa16b8cfa585bde27e` (2024-11-12): Apache License 2.0 (GitHub licence API
  Apache-2.0). `github.com/yandex-research/rtdl`, commit `4ca77be57aae445648df63e9ef3a5566b1f0a835`
  (2026-04-17): Apache License 2.0. PyPI `rtdl` 0.0.13 (2022-03-16): Apache Software License
  classifier only.
- Weights: none distributed.
  [LICENSE](https://raw.githubusercontent.com/yandex-research/rtdl-revisiting-models/e3ed46cac38568785289d8fa16b8cfa585bde27e/LICENSE)

### TabM
- Code: `github.com/yandex-research/tabm`, commit `b53fd7cf2d9be2d66b918ca6cf66e0365e4ce11c`
  (2026-11-07): Apache License 2.0. PyPI `tabm` 0.0.3 (2025-08-14): Apache Software License
  classifier; requires `torch` and `rtdl_num_embeddings`.
- `github.com/yandex-research/rtdl-num-embeddings`, commit
  `a8fc25025c83f2321c63ff127a3bcef83bb1bfb5` (2025-04-16): `LICENSE` is MIT (copyright "Authors of
  On Embeddings for Numerical Features in Tabular Deep Learning"). PyPI `rtdl-num-embeddings`
  0.0.12 carries the Apache Software License classifier.
- Weights: none distributed.
  [LICENSE](https://raw.githubusercontent.com/yandex-research/tabm/b53fd7cf2d9be2d66b918ca6cf66e0365e4ce11c/LICENSE)

### RealMLP
- Code: `github.com/dholzmueller/pytabkit`, commit `d232eecd47b127b29da0378423cb475b88fb0e80`
  (2026-11-01). The licence file is `LICENSE.txt` (a top-level `LICENSE` returns 404): Apache License
  2.0; `pyproject.toml` `license = "Apache-2.0"`. PyPI `pytabkit` 1.8.0 (2026-11-01): Apache-2.0.
  README lists RealMLP (TD, HPO, Ensemble), and TabM, as models in the package.
  [LICENSE.txt](https://raw.githubusercontent.com/dholzmueller/pytabkit/d232eecd47b127b29da0378423cb475b88fb0e80/LICENSE.txt)
- Weights: none distributed.
- Not checked: optional extras (`models`, `autogluon`, `hpo`, `extra`, `bench`), which pull in
  further packages, among them `xrfm`, `structured-data-models`, `probmetrics` and `kditransform`.
  Anything installed through an extra needs its own check.

## Reconciliation with existing notes

Existing notes grepped: `docs/paper/RELATED_WORK.md`, `docs/research/REFERENCES.md`,
`docs/research/RESEARCH_NOTES.md`, `README.md`.

- **TabICL/TabICLv2, BSD-3-Clause for both** (`docs/paper/RELATED_WORK.md` line 1141; `REFERENCES.md` line
  545): agrees. Added precision: the `LICENSE` file has a second, Apache-2.0 text for `forecast/`,
  and the weights grant is the card field of `jingang/TabICL` with no licence file.
- **TabDPT Apache-2.0 for both** (`docs/paper/RELATED_WORK.md` line 1144): agrees; now backed by the Hub
  LICENSE file as well as the card.
- **TabPFN code Apache-2.0, 2.5 onwards non-commercial, TabPFN-2 Prior Labs licence**
  (`docs/paper/RELATED_WORK.md` lines 1141 to 1143, 1208 to 1224): agrees on weights. **One update, not a
  conflict:** `docs/paper/RELATED_WORK.md` line 1219 records PyPI `tabpfn` 9.2.0 on 2026-11-13; PyPI now lists
  10.0.0, uploaded 2026-11-13, still Apache-2.0. The older wording "TabPFN code Apache-2.0 with
  attribution requirement" (line 1211) is already flagged there as out of date for current code, and
  this check confirms the repository `LICENSE` is plain Apache-2.0. The attribution paragraph
  survives only in the TabPFN-2 weights licence.
- **TabM, RealMLP, FT-Transformer:** no licence note existed (`RESEARCH_NOTES.md` line 71 lists
  them with "none"; "none" there meant no foundation-model constraint, and is now backed by the
  checks above).
- **Disagreement inside the sources, resolved by taking the licence text:**
  `rtdl-num-embeddings` is MIT in its GitHub `LICENSE` and Apache Software License in the PyPI
  classifier. The `LICENSE` file at a named commit is taken as the authority. Both are permissive,
  so neither changes a verdict; record MIT, and re-read the PyPI sdist before shipping it.

## Gaps

- The TabPFN-2.5 and TabPFN-2 regressor licence texts were not opened (card fields and README only).
- TabICLv2 and TabDPT training-data licences were not checked.
- Transitive dependency licences (torch and the rest) were not audited. `CLAUDE.md` requires a
  licence statement for any dependency proposed for the platform; none is proposed here, and every
  baseline above runs outside this repository.
- No baseline has been added, so the "licence note recorded alongside each new baseline before it is
  used" part of 18.6 is satisfied as a record only. The finding that states the exclusions belongs in
  `docs/results/FINDINGS.md`, which this task did not edit.
