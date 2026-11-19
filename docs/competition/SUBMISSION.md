# Submitting FinTFM to TabArena

> **Status, 2026-10-15.** This procedure was followed for `fintfm==0.5.5`. The pull request
> ([autogluon/tabarena#616](https://github.com/autogluon/tabarena/pull/616)) was reviewed on
> 2026-09-25: a TabArena collaborator confirmed the integration correct and declined the entry
> on competitiveness, at rank 94 of 95 and 0.7823 mean ROC-AUC over 27 binary datasets. The
> reviewer also ruled the binary-only coverage below ineligible, so a future submission needs
> all 51 datasets. The submission was withdrawn before the maintainers' full run, so stages 2-5
> did not take place (`docs/results/FINDINGS.md` §122). The rest of this file is kept as the
> procedure for a future submission.

> **Support status, 2026-11-13.** The project states it is no longer actively supported from
> mid-November 2026, so a resubmission is unlikely to be made by this project.
> As of 2026-11-19 the owner (GitHub handle kabartay) will answer the reviewers of the
> 2026-11 TabArena pull request himself.

> **Update, 2026-11-13.** Nothing has been resubmitted. The owner holds the PR (DR7.6,
> 2026-11-11) until multiclass and regression improve. The entry that would be submitted is no
> longer the one tabulated below: it is "FinTFM with a trained linear readout" over all 51
> datasets, 903 Elo, 107th of 110 on TabArena-Lite (`docs/results/FINDINGS.md` §224), and the
> tabarena fork's `fintfm` extra pins `fintfm==0.10.1` (fork commit af7da60, pushed 2026-11-13).
> The checkpoint, configuration and coverage rows, the sign-off checks and the "roughly 0.78 mean
> ROC-AUC" sanity range below describe the `fintfm==0.5.5` binary entry and would all change.
> Details: the last section of [`../results/TABARENA.md`](../results/TABARENA.md).

The procedure, and the two points where it needs a person rather than a script. Written down
because stage 2 can take weeks, by which time the details of stage 1 are no longer in anyone's
head.

Upstream reference: [Contributing a Model or
System](https://github.com/autogluon/tabarena#contributing-a-model-or-system).

## The five stages

| # | What happens | Who |
| --- | --- | --- |
| 1 | Integrate the model, run TabArena-Lite, open the pull request | us |
| 2 | Maintainers re-run on the full task set on their hardware | them |
| 3 | Maintainers verify against our Lite results; **we sign off** | both |
| 4 | Results hosted, method registered, pull request merged | them |
| 5 | Leaderboard regenerated, in batches, usually within days of the merge | them |

Stage 2 is the long pole: every split of every dataset on a benchmark cluster. Nothing we do
shortens it.

## What we submit

| | |
| --- | --- |
| package | `fintfm==0.5.5` from PyPI |
| checkpoint | `kabartay/fintfm-binary`, commit `f116bfd43a2b15c65ed3551ea8c38e3364629ddc` |
| registry key | `ag_key="TA-FINTFM"`, displayed as `FinTFM` |
| configurations | the default only; `hpo.py` declares no search space and `can_hpo=False` |
| coverage | 27 of 51 datasets, 53% (21 excluded as not binary, 3 over the 136-feature cap) |

The run is produced from the **published** checkpoint rather than a local file, so a maintainer
fetches the same artifact we measured. A run made from a path only one machine has cannot be
verified at stage 3.

**For a 51-dataset resubmission** (2026-11-13; sources §221 to §224): the entry declares three
problem types, each with its own checkpoint (binary: `kabartay/fintfm-binary`; multiclass:
`dr74-mc-s0`; regression: `v4-regression.pt`), and the last two are private today, so the owner
would have to publish both first. In the fork the measured runs set their configuration through environment
variables (`FINTFM_PARTITION`, `FINTFM_PROBE`, `FINTFM_PARAMS` with `max_features_strategy: select`);
a submission would need that configuration fixed in the entry itself, which has not been done.
The readout takes the row's own features for multiclass and regression, which the entry must
state. Whether the search space stays empty is undecided. The sanity range for a
maintainer would be the §224 figures by type (binary 974, multiclass 865, regression 578
Elo), not 0.78 ROC-AUC. A dataset near the time limit (Amazon_employee_access, §215) is a risk
on the maintainers' hardware.

## Handling the sign-off

Stage 3 is the one step that cannot be automated, and it is small.

**The trigger.** A maintainer posts a comment in the pull request with the results of their
full run: the leaderboard position, and usually notes on failed splits or time-limit hits.
GitHub notifies the pull request author, so there is nothing to watch.

**What to check**, against what they posted:

1. **Version** is `fintfm==0.5.5` (or whatever `pip_extra` pinned at the time).
2. **Checkpoint** is the commit above, not `main` and not a local file.
3. **Configuration** is the default alone. If tuned or ensembled variants appear, the empty
   search space was not honoured and the numbers describe something else.
4. **The numbers are in the right region.** Our Lite result is roughly 0.78 mean ROC-AUC, near
   the bottom of the board. A large discrepancy in either direction means a misconfiguration,
   not a discovery, and goes back for a fix.

**What to write.** A comment, roughly:

> Confirmed on behalf of the authors. The run used `fintfm==0.5.5` and checkpoint
> `kabartay/fintfm-binary@f116bfd4`, default configuration only, which is the method as
> intended. The position matches what we measured on Lite.
>
> [Note anything that differs, or say "no discrepancies".]

**What it does and does not mean.** It confirms the run reflects the method as intended. It is
not a claim that the results are good. They are not, and the repository says so in the README,
the model card and this file.

**If we skip it.** The entry still appears, flagged **unverified**. For a project whose whole
argument is that its numbers are checkable, that would undercut the point for no reason.

## The other thing that needs a person

The search space is empty, and the pull request checklist asks for about 200 configurations.
`can_hpo=False` has precedent upstream (`causilo`), and `info.py` records the measured reasons:
context size is nearly flat ([FINDINGS](../results/FINDINGS.md) §83/§84), retrieval harms at
low prevalence (§70), and ensembling over column-identity draws is a correctness setting rather
than a hyperparameter ([DECISIONS](../design/DECISIONS.md) D12). Expect a reviewer to ask, and
answer with the measurements rather than the precedent.

## Pickle audit (2026-11-16, MEASURED)

`python -P -m tabarena.tools.audit_pickle --model <name>` on the PR branch (`fintfm-pr-2026-11`,
ecd4ac5) passes for FinTFM and FinTFM-LinProbe (binary) in a fresh environment with the branch's own
AutoGluon floor (`autogluon.tabular>=1.6.4b20261109,<1.7`; checked with 1.6.4b20261115, fintfm 0.10.1):

| model | `model.pkl` at 200 rows | at 400 rows | growth per row | result |
| --- | --- | --- | --- | --- |
| FinTFM | 50,838 B | 78,058 B | 136 B | OK |
| FinTFM-LinProbe | 70,033 B | 87,643 B | 88 B | OK |

The flag limit is 1,024 B per row; `model_tensors.pt` (the weights, 3,580,115 B) does not grow with
rows. The earlier crash (`AttributeError: ... no attribute 'model_tensors_file_name'`) came from the
older nightly in the local development venv (1.6.4b20260919), which predates the tensors-file split
the audit reads; the wrappers are not involved. Regression cannot be audited by a maintainer until
the regression checkpoint is published (`CheckpointNotPublishedError` otherwise).
