DRAFT, not posted. Do not open a TabArena pull request from this text until every box in "Before posting" is ticked.

# Draft text: TabArena resubmission of FinTFM (follows autogluon/tabarena#616)

Prepared 2026-11-15. Every figure below is taken from `docs/results/FINDINGS.md` (section
number in brackets) or from the task notes named beside it, and is TabArena-Lite, repeat 0,
fold 0, run from the tabarena fork on CPU. None of it is the public leaderboard or a full
TabArena run. The owner's bar for resubmitting is 1,000 overall Elo (DR7.6, owner,
2026-11-13, `openspec/changes/deferred-work-2026-11/tasks.md` task 49.6). **The entry is
below that bar at 931.9 Elo, so under the current rule this PR is not to be posted.**

## What #616 said, and where this draft answers it

The pull request autogluon/tabarena#616 (closed 2026-09-25, withdrawn by us before the
maintainers' full run) received three concerns from the TabArena collaborator who reviewed it,
and one positive finding.

| # | Reviewer concern in #616 | Answer in this draft |
| --- | --- | --- |
| 1 | The entry declared binary only. Excluding datasets or problem types because the model does worse there is cherry-picking and is not eligible. | The new entry declares binary, multiclass and regression and scores every dataset. No exclusion. The per-type Elo values are reported, including the weak ones. |
| 2 | Datasets were missing, and "there are failures that should be fixed". We had answered that three binary tables were excluded by the old 136-feature cap and that lifting it meant retraining. The reviewer replied: "Yes, this would be needed. We would like new submissions to cover all datasets." | Coverage is 51 of 51 Lite datasets (§235). The wide tables run through column selection, not a new checkpoint (§221). No fit is imputed (§221). |
| 3 | The model does not beat a tuned KNN or other simple baselines, so it does not meet the competitiveness bar for the leaderboard. Recommendation: withdraw until the performance meets a reasonable minimum. | Not met in the sense the reviewer meant. The entry is 106th of 110 at 931.9 Elo, 3.7 Elo below tuned linear regression with ensembling, and 447 Elo below tuned LightGBM with ensembling. A tuned KNN figure on this board is not sourced in the repository, so this draft does not quote one. The draft states the shortfall instead of arguing it away. |
| 4 | Positive: "Based on the submission's code and numbers you shared, I think everything is correct." | The wrapper has changed since that review (partitioning, readout, checkpoints); the changes are listed below so the reviewer can see which parts need a fresh read. |

In #616 we also wrote that we would "come back with full coverage from the start" and "try
to beat the next foundation model". The first is now true. The second is not: the entry sits
below the zero-shot foundation models on the same board (see the reference rows below).

## Draft pull request body (follows the TabArena template)

### Summary

FinTFM is a tabular foundation model pretrained on a synthetic prior only. This resubmission
covers all 51 TabArena-Lite datasets (30 binary, 8 multiclass, 13 regression), labelled
"FinTFM with a trained linear readout": it is not zero-shot, because it fits a linear
readout on training rows inside each partition leaf. On TabArena-Lite (repeat 0, fold 0) it
scores 931.9 Elo, 106th of 110, which is below tuned linear regression and well below tuned
tree ensembles. We are submitting it with that position stated, not as a competitive entry.

<details>
<summary>Details</summary>

**What the reviewers need to know first**

- **The entry is not zero-shot.** It fits a linear readout per partition leaf on the training
  rows: a logistic regression for binary and multiclass, a ridge regression for regression.
  The readout also takes the row's own squashed features beside the network's representation.
  The label is "FinTFM with a trained linear readout".
- **Checkpoints, one per problem type** (§221, §227):
  - binary: the published FinTFM checkpoint;
  - multiclass: `dr74-mc-s0`, not public;
  - regression: the 48,000-task pure-regression checkpoint (`v4-binned48_3-control.pt`), not
    public.
  The multiclass and regression weights are not yet published; that is pending an owner
  decision, and so is their licence. Until they are public a maintainer cannot reproduce two
  of the three problem types.
- **Numbers are TabArena-Lite, repeat 0, fold 0**, run from the fork on CPU, 110-method board.
  They are not a full TabArena run and are not on the public leaderboard.
- **Positions** are one plus the number of methods with a strictly higher Elo, computed from
  the stored leaderboard `fintfm_linprobe_all51_1115`, re-read on 2026-11-15: 931.9 Elo
  (+82.7 / -97.9), 106th of 110.

**Results**

| quantity | value | source |
| --- | --- | --- |
| Overall, 51 of 51 datasets | 931.9 Elo (+83 / -98), 106th of 110 | §235 |
| Binary, 30 datasets | 997 Elo (+118 / -127) | §232 |
| Multiclass, 8 datasets | 874 Elo (+110 / -273) | §227 |
| Regression, 13 datasets | 689 Elo (+91 / -206), 99th of 103 on the regression board | §235 |
| Coverage | 51 of 51, none imputed | §235, §221 |
| Tuned linear regression with ensembling, same board | 935.6 Elo, so the entry is 3.7 Elo below | §235 |
| `GBM (tuned + ensemble)`, same board | 1379.3 Elo (+34.6 / -38.0), 37th of 110 | `openspec/changes/tabarena-integration/tasks.md` 45.4, re-read from the stored leaderboard |

The gap to a tuned tree ensemble is about 447 Elo (1379.3 minus 931.9). The owner's bar for
this submission is 1,000 Elo (DR7.6) and the entry is below it by 68.1 Elo.

**What changed since #616** (the withdrawn entry was binary only, zero-shot head, 94th of 95,
mean ROC-AUC 0.7823 over 27 datasets, `docs/competition/SUBMISSION.md`, §122):

- **Partitioned contexts.** A tree routes rows to leaves and each leaf gets its own context,
  with bags of such partitions. Binary §215; multiclass §224 and §227 (eight bags); regression
  §231 (one bag) and §235 (four bags, 639 to 689 Elo, better on 12 of 13).
- **A linear readout per leaf, fitted on training rows, with the row's own features.** Binary
  §215, with the scale, cv and raw options in §232 (974 to 997 Elo); multiclass §222 (751 to
  825 Elo, better on 8 of 8); regression §223 (436 to 578 Elo, better on 12 of 13).
- **Per-type checkpoints.** One checkpoint per problem type, §221; the pure-regression
  checkpoint replaced the mixed one in §227.
- **Column selection for wide tables.** `max_features_strategy: select` admits the five tables
  over 136 columns without a new checkpoint, §221. Selection changes what the model sees on
  those tables and is declared in the entry. It does not make those tables good: the five wide
  tables rank 90th to 103rd (§221), and §190 found that selection does not rescue wide tasks.

**Controls the reviewers should read beside the Elo values**

Each problem type has a matched control in which the readout sees only the squashed raw
features, with no FinTFM representation (`no_rep`). In effect it is a decision tree with a
linear model per leaf.

| problem type | entry | raw-only control | representation better on | source |
| --- | --- | --- | --- | --- |
| binary | 997 Elo | 1009 Elo | 24 of 30 datasets by ROC-AUC, but the Elo intervals overlap completely | §239 |
| multiclass | 874 Elo | 829 Elo | 7 of 8 datasets (sign test p = 0.07) | §243 |
| regression | 689 Elo | 537 Elo | 12 of 13 datasets (sign test p = 0.003) | §240 |

**The binary Elo is not evidence for the representation.** A tree with a logistic regression
per leaf, which uses no FinTFM representation, scores 1009 against 997 (§239). Binary Elo
should not be read as a FinTFM result. The representation is supported on multiclass and
regression (§243, §240), not on binary Elo. This is TabArena breadth only; it is not a credit
result.

**Cost.** The regression run took 269 min for 13 datasets on CPU under a high load average from other sessions (§235). Per-type times for the other runs are in their findings.

**Dependency and licences.** `fintfm` is Apache-2.0. The fork's `fintfm` extra pinned
`fintfm==0.10.1` on 2026-11-13 (fork commit af7da60, `docs/competition/SUBMISSION.md`); the
§235 runs used an editable checkout, so the pin must be re-confirmed against the run. Weights
licence: not decided.

**Hardware and entry-point script.** CPU. Entry-point script and hardware description to be
filled in from the run log before posting (`runs/merge_all51_1115.log`).

</details>

### Tests

To be filled with the commands actually run before posting. Required:
`ruff check` and `ruff format --check`; `pytest -m models -k FinTFM`; the TabArena-Lite run
with the official pipeline and no environment variables set. None of these has been run for
the resubmission wrapper yet. Not claimed here.

### Model or system submission

Kind: model. It is a new entry, not a replacement for a leaderboard entrant (the #616 entry
was withdrawn before any run).

Results sign-off for the maintainer re-run: the owner's GitHub handle, to be added before
posting.

By submitting this pull request, I confirm that you can use, modify, copy, and redistribute
this contribution, under the terms of your choice.

## Before posting (all must be true)

- [ ] **Bar decision.** The owner decides whether to post below the 1,000 Elo bar. The entry is
      at 931.9 Elo. Under DR7.6 as written, the answer is no.
- [ ] **Weights published with a licence.** The multiclass (`dr74-mc-s0`) and regression
      (`v4-binned48_3-control.pt`) checkpoints are public with a stated licence, and the
      binary checkpoint's licence is confirmed. Decision pending with the owner.
- [ ] **Reproduction with no environment variables.** The fork's entry reproduces the merged
      51-dataset Elo (931.9) with `FINTFM_PARTITION`, `FINTFM_PROBE`, `FINTFM_MAX_CONTEXT` and
      `FINTFM_PARAMS` unset, because they are now the entry's own hyperparameters (task 49.6).
- [ ] **Fit-test pass.** `pytest -m models -k FinTFM` passes in the fork, and `ruff check` and
      `ruff format --check` are clean. Confirm that the wrapper uses the `X_val` / `y_val` that
      TabArena passes, or states why the readout fits on training rows only.
- [ ] **Search space decided.** The template asks for about 200 configurations where a search
      space exists; the readout entry's search space is undecided
      (`docs/results/TABARENA.md`, `docs/competition/SUBMISSION.md`).
- [ ] **Label.** The entry label is "FinTFM with a trained linear readout" everywhere (task
      49.6), and the PR says plainly that it is not zero-shot.
- [ ] **Pin and numbers re-read.** The `fintfm` pin matches the code that produced the numbers;
      every figure above is re-read from the leaderboard and the position re-derived with
      `scripts/lite_positions.py` on the board current at posting (the field was 110 methods on
      2026-11-15).
- [ ] **Support status.** `docs/competition/SUBMISSION.md` records that the project is no
      longer actively supported from mid-November 2026. Decide who answers the reviewers.
- [ ] **Hardware, script and sign-off handle** filled into the body.
- [ ] **No confidential or peer-product names** in the posted text.

## Numbers this draft could not source

- KNN on the same 110-method board (read 2026-11-15 from `../tabarena/eval/fintfm_linprobe_all51_1115/tabarena_leaderboard.csv`, position = 1 + methods with strictly higher Elo): KNN (tuned + ensemble) 965.3, 103rd of 110; KNN (tuned) 813.4, 108th; KNN (default) 584.6, 110th. The entry (931.9, 106th) is above tuned KNN and below tuned-and-ensembled KNN, so #616's "beat a tuned KNN" condition is met only for the non-ensembled variant.