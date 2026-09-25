# Submitting FinTFM to TabArena

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
context size is nearly flat ([FINDINGS](../results/FINDINGS.md) S83/S84), retrieval harms at
low prevalence (S70), and ensembling over column-identity draws is a correctness setting rather
than a hyperparameter ([DECISIONS](../design/DECISIONS.md) D12). Expect a reviewer to ask, and
answer with the measurements rather than the precedent.
