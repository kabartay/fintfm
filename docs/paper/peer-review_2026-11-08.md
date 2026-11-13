# Peer review, 2026-11-08

Previous pass: `docs/paper/RELATED_WORK.md`, "Peer read of 2026-11-06" (FTFM, TabSTAR-v1.1) and `docs/research/REFERENCES.md` addendum 2026-11-06. This pass covers 2026-11-06 to
2026-11-08 plus items the earlier passes had not entered. Sources were read as abstract pages and,
for two papers, as PDFs (`pdftotext`). Nothing here enters the repository: no peer code or weights.

## Gaps first

- **The TabArena board itself was not readable** (the Space renders client-side). The only board
  evidence is the `autogluon/tabarena` pull-request list to 2026-11-06: Xiaomi-TabLDM pin updates (#708 closed, #710 merged), no new TabPFN, TabICL, LimiX, Mitra,
  TabDPT, Kumo or CARTE entry. A board snapshot is still owed.
- **The LimiX-2 replacement was not re-checked** at source this pass (issue #648 was last read
  2026-11-05); the standing recheck of task 48.26(b) carries over.
- **No new credit- or finance-specific tabular foundation model** turned up. The credit papers
  found (arXiv:2605.18635, 2605.18147) were already in `docs/research/REFERENCES.md`; 2605.18147 is now at v3
  (revised 2026-10-23; the page does not say what changed).
- Several items below were read from the abstract only. Where a number is missing it is missing
  in the source page and is not estimated.

## Already recorded, nothing new

TabAbax (arXiv:2611.00562; `docs/research/REFERENCES.md` has the repository, the weights licence CC-BY-NC-4.0 and
the classification/regression split), Mitra-v2 (arXiv:2609.04540, Apache-2.0 per its abstract),
TabDPT-Turbo (arXiv:2608.01400), Xiaomi-TabLDM (arXiv:2609.03880), LimiX-2M, TabPFN-3, TabFlex,
FTFM. One caution on quoting TabAbax: its abstract gives TabArena overall Elo 1,676 on 51
datasets and 11th of 99 (+115 over TabICLv2), a different view from the classification-only 1,647
in our reference entry; name the view every time.

## New items

### 1. Empirical Evaluation of Out-Of-Distribution Performance of Tabular Foundation Models (arXiv:2607.26000)

- **What.** Loza, Chushig-Muzo et al., v1 2026-07-28. Nine TFMs (TabPFNv2, v2.5, v2.6, v3, TabICL,
  TabICLv2, Mitra, LimiX, TabFM) on three TableShift datasets: HELOC (label shift), Childhood Lead
  (poverty-level shift), Voting (region shift).
- **Licence.** Paper CC BY-SA 4.0. No code link on the page.
- **Numbers.** Metric ROC-AUC; shift gap is ID minus OOD ROC-AUC. Shift gaps range 0.003 to 0.060
  depending on shift type (abstract). HELOC split (Table 1): 2,220 train, 278 validation, 278 ID
  test, 6,914 OOD test, 38 features. On HELOC the models sit in a narrow ID ROC-AUC band of
  0.70 to 0.74, and TabICL and TabICLv2 reach the highest OOD scores. By pre-training corpus the
  OOD ROC-AUC median is about 0.78 (real), 0.775 (mixed), 0.77 (synthetic), as stated in the text,
  not from a table. Mitra shows the largest HELOC and Childhood Lead degradation. The protocol
  (seeds, context size, preprocessing) is not given in the parts read.
- **Implication.** It supports Claim 17's direction (real-data pre-training buys level, not
  robustness) and says nothing against our results. It is the only third-party credit-shift
  comparison of our peers, and it uses AUC only. We have never scored the published checkpoint
  under label shift on a public credit set, and the TableShift HELOC split is small enough to run
  on CPU. It cannot move the TabArena-Lite Elo; it could give robustness evidence and an AP column
  the paper lacks. Filed as task 48.28.

### 2. GEAR: Generative Expansion and Real Anchoring for Two-Stage Distillation of TFMs (arXiv:2608.18849)

- **What.** Qin et al., v2 2026-08-25. Distils a frozen in-context teacher (TabICL, TabPFN,
  TabDPT) into an MLP or GBDT student: stage 1 trains on teacher soft targets at generated
  covariates, stage 2 re-anchors on real rows with out-of-fold teacher predictions
  (soft-label weight 0.7). Generators: TabPFGen, TabDiff, Copula, CTGAN.
- **Licence.** Paper CC BY 4.0. No code repository found.
- **Numbers.** TALENT and TabArena, ROC-AUC (macro one-vs-rest for multiclass), mean of five
  random runs. Two-stage MLP over supervised MLP: +1.81 to +2.00 AUC points binary, +1.19 to
  +1.35 multiclass. At 40 generated rows per real row with TabPFGen: +1.91 binary, +1.27
  multiclass. Inference time down 57x to 2,866x, peak memory down 1.9x to 3.3x. The students are
  not in-context models and are scored on a different scale from Elo.
- **Implication.** Nothing for the Elo gap: a student cannot exceed its teacher, and the teacher is
  the weak side. It matters only if a deployment story needs a CPU-only scorer later. Not filed.

### 3. Memory Efficient Tabular Foundation Models (arXiv:2607.27546)

Accepted at FMSD at ICML 2026, v1 2026-07-30, paper CC BY-NC-SA 4.0, no code link. Abstract only:
compression "up to 7.6" (no unit on the page) and "nearly 87%" lower deployment requirement at
"similar levels of performance"; methods, models, datasets and protocol are not stated. Nothing to
act on: our control model is 885,650 parameters and memory is not our constraint.

### 4. Tabular foundation models for non-tabular tasks (arXiv:2608.22594)

v1 2026-08-23, CC BY 4.0, no code. TabPFN v3 only, on MNIST, French versus German word language
identification and Tiny ImageNet, accuracy against context size; "comparable" in some cases, no
numbers on the abstract page. Nothing for credit risk. Not filed.

## Ranking by expected value

For closing the TabArena-Lite binary Elo gap (756 against a bar of about 1,010) and for V4FinBench
AP. None of the items is expected to close the gap; the ranking is of small values.

| rank | item | Elo gap | V4FinBench AP | action |
| --- | --- | --- | --- | --- |
| 1 | TabAbax prior-quality finding (already recorded) | low: prior-only gain is +12 classification against +228 regression | possible, through the prior | existing prior tasks (48.13, 48.18) cover it |
| 2 | HELOC shift evaluation (arXiv:2607.26000) | none | robustness evidence, new AP column | task 48.28 |
| 3 | Context construction for credit (arXiv:2605.18635, already recorded; balanced context adds 3 to 4 AUC points) | none | direct | existing task 48.24 |
| 4 | GEAR distillation | none | none | not filed |
| 5 | Memory-efficient TFMs; non-tabular tasks | none | none | not filed |

Reading for the gap: the one peer lever with a reported size, TabAbax's prior, is +12 Elo on
classification in its own prior-only ablation, far short of the roughly 254 Elo we lack. Nothing
read this pass explains the remaining gap.

## Status notes, 2026-11-13

Added after the record above, which is otherwise unchanged. References to commercial
companies were removed from this file on 2026-11-13 by owner decision; the peer list above is
shortened accordingly.

- **Item 1, HELOC shift evaluation (task 48.28):** deferred 2026-11-09 by the owner. TableShift's
  HELOC data sits behind FICO's data access agreement, the agreement page no longer resolves to
  it, and only unofficial copies remain, so it cannot be used under known terms
  (`docs/paper/NEGATIVE_RESULTS.md`, section (e)). The ID-to-OOD shift gap stays unmeasured.
- **The Elo gap quoted in "Ranking by expected value" has changed.** The 756 Elo on 27 binary
  datasets was the zero-shot default (§204). A trained linear readout with squash and partitioned
  contexts reached 1,000.2 on the same 27 (§215), level with RF (default); on all 30 binary datasets
  it is 974 (§220), and on all 51 datasets 903, 107th of 110 (§224). The "bar of about 1,010" did not
  carry over to the later leaderboard (§215).
- **Board snapshot (gap 1):** the 110-method leaderboard used by §221 to §224 is the fork's own
  merged copy, not a re-read of the public board; a public-board snapshot is not recorded in
  FINDINGS.
- **Ranking row 1, TabAbax prior-quality finding:** the prior experiments since then are null or
  harmful at the current recipe (§180, §191, §211, §213); none reproduced a prior gain on
  V4FinBench.
- **Item 3, context construction for credit (task 48.24):** the task is in progress
  (`openspec/changes/learn-from-peers/tasks.md`, 2026-11-11 entry). Related V4FinBench results:
  the prototype context is +0.0002 mean AP (§211) and the positive floor +0.0055 (§200), both
  null.
