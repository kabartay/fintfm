# Submitting FinTFM to TabArena

The procedure, and the two points where it needs a person rather than a script. Written down
because stage 2 can take weeks, by which time the details of stage 1 are no longer in anyone's
head.

| record | state |
| --- | --- |
| [`PR_2026-09.md`](PR_2026-09.md) | autogluon/tabarena#616, `fintfm==0.5.5`, 27 binary datasets: declined on competitiveness and withdrawn, 2026-09-25 |
| [`PR_2026-11.md`](PR_2026-11.md) | the resubmission, [autogluon/tabarena#736](https://github.com/autogluon/tabarena/pull/736), "FinTFM with a trained linear readout", all 51 datasets, fork 6f353cf, then 9d99144 (thin adapter on fintfm 0.13.0): posted 2026-11-19 and closed by the maintainers the same day, on competitiveness and on selection by internal cross-validation instead of `X_val` (`docs/results/FINDINGS.md` §306) |

Upstream reference: [Contributing a Model or
System](https://github.com/autogluon/tabarena#contributing-a-model-or-system).

> **Support status.** 2026-11-13: the project states it is no longer actively supported from
> mid-November 2026. 2026-11-19: the owner (GitHub handle kabartay) answers the reviewers of
> the 2026-11 pull request directly. 2026-11-19, later: development is limited to that submission. 2026-11-19, later still: TabArena closed it the same day (FINDINGS §306), the owner posted
> no reply, and development ended with v1.0.0. The stages below were never reached for it.

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

Every run we submit is produced from **published** checkpoints rather than local files, so a
maintainer fetches the same artifacts we measured. A run made from a path only one machine has
cannot be verified at stage 3.

## Handling the sign-off

Stage 3 is the one step that cannot be automated, and it is small.

**The trigger.** A maintainer posts a comment in the pull request with the results of their
full run: the leaderboard position, and usually notes on failed splits or time-limit hits.
GitHub notifies the pull request author, so there is nothing to watch.

**What to check**, against what they posted (values for the 2026-11 entry):

1. **Version** is what `pip_extra` pins: `fintfm==0.13.0`.
2. **Checkpoints** are the three revisions pinned by SHA-256 in
   `fintfm.readout.checkpoints`, not `main` and not local files.
3. **Configuration** is the default alone. If tuned or ensembled variants appear, the empty
   search space was not honoured and the numbers describe something else.
4. **The numbers are in the right region.** Our Lite reproduction is about 1,020 Elo overall
   (binary 1,083, multiclass 947, regression 853 on per-type boards; FINDINGS §297, §303). The
   full run uses every fold, so expect movement inside the Elo intervals (about ±75 overall). A
   large discrepancy in either direction means a misconfiguration, not a discovery, and goes back
   for a fix. Watch the time limit: on a 4-CPU machine QSAR-TID-11 sits at it (§301).

**What to write.** A comment, roughly:

> Confirmed on behalf of the authors. The run used `fintfm==0.13.0` and the three checkpoints
> pinned in `fintfm.readout.checkpoints`, default configuration only, which is the method as intended. The
> position matches what we measured on Lite.
>
> [Note anything that differs, or say "no discrepancies".]

**What it does and does not mean.** It confirms the run reflects the method as intended. It is
not a claim that the results are good: the entry is general-tabular breadth, and on real credit
panels FinTFM loses to gradient boosting, as the README says.

**If we skip it.** The entry still appears, flagged **unverified**. For a project whose whole
argument is that its numbers are checkable, that would undercut the point for no reason.

## The other thing that needs a person: the search space

The pull request checklist asks for about 200 configurations, and for the default plus about 25
sampled ones on Lite. The 2026-11 entry submits the default only (`can_hpo=False`, owner decision
2026-11-19): `hpo.py` keeps a proposed space of at most 108 combinations, none of them measured.
Expect a reviewer to ask, and answer with the measurements rather than precedent: each lever was
chosen on Lite fold 0, several of the size lines after the results were seen (FINDINGS §283 to
§301), so a sampled space would be a new measurement, not a check of this one.

## Pickle audits (MEASURED)

`python -P -m tabarena.tools.audit_pickle --model <name> [--problem-type <type>]` on the PR branch.

**2026-11-19, fork 633dcc2, FinTFM-LinProbe on binary, multiclass and regression:** all three
"OK: nothing that scales with the training rows sits in model.pkl" (outputs in
`runs/pr_checks_9c7db11/`). The unpatched tool crashes on this model: scipy's `BSpline` keeps the
array namespace module in `_xp`, and the tool's tensor walk follows it into `sys.modules` until a
`urllib3` mapping refuses iteration. The audit was run with module objects skipped in that walk;
the pull request text describes the one-line fix.

**2026-11-16, fork ecd4ac5, binary only,** in a fresh environment with the branch's own
AutoGluon floor (`autogluon.tabular>=1.6.4b20261109,<1.7`; checked with 1.6.4b20261115, fintfm
0.10.1):

| model | `model.pkl` at 200 rows | at 400 rows | growth per row | result |
| --- | --- | --- | --- | --- |
| FinTFM | 50,838 B | 78,058 B | 136 B | OK |
| FinTFM-LinProbe | 70,033 B | 87,643 B | 88 B | OK |

The flag limit is 1,024 B per row; `model_tensors.pt` (the weights, 3,580,115 B) does not grow with
rows. The earlier crash (`AttributeError: ... no attribute 'model_tensors_file_name'`) came from the
older nightly in the local development venv (1.6.4b20260919), which predates the tensors-file split
the audit reads; the wrappers are not involved.
