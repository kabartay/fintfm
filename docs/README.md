# docs/

The project's reasoning, grouped by what each document is *for*. Start with
[`paper/CLAIMS.md`](paper/CLAIMS.md) — it decides what could honestly be written, and it
records supersessions and retractions as prominently as wins.

[`index.html`](index.html) is not a document in this list: it is the landing page served at
<https://kabartay.github.io/fintfm/>. Its figures must not go beyond the status section of the
repository's root `README.md`. [`CHANGELOG.md`](CHANGELOG.md) records what changed in each
release.

## results/ — what was measured

| file | what it is for |
| --- | --- |
| [FINDINGS.md](results/FINDINGS.md) | The measurement log, numbered and dated, each entry labelled by how it was produced. **§N** anywhere in this repository refers to an entry here. |
| [TABARENA.md](results/TABARENA.md) | External evaluation under someone else's protocol: the reproduction recipe, the coverage fraction any score must carry, and four silent failure modes — including a results cache that returns stale numbers after a preprocessing change. |
| [POSTMORTEM.md](results/POSTMORTEM.md) | Wrong diagnoses, each caught by measurement rather than review, kept on the record deliberately. Three chains: the first week's seven diagnoses, a cheap proxy standing in for an expensive measurement, and harness defects read as model results (§219, §220). |
| [drafts/](results/drafts/) | Working drafts and pre-registrations (`dr*`, `au*`, `48.*`, `preregs_2026-11.md`, `path-to-bar_2026-11-08.md` and others), named by task id and month. Not findings: a result becomes one only when it has a section in FINDINGS.md. Grouped here rather than listed, because the folder grows with every audit round. |

## design/ — how it works, and why it is built this way

| file | what it is for |
| --- | --- |
| [ARCHITECTURE.md](design/ARCHITECTURE.md) | Package layout, the model stages, the inference corrections, and the invariants that must not break. |
| [DECISIONS.md](design/DECISIONS.md) | Decisions with the alternatives priced against them and **what would reverse each one**. |
| [DESIGN_REVIEW_2026-10-16.md](design/DESIGN_REVIEW_2026-10-16.md) | Six reviews of the design (architecture, prior, training recipe and compute, inference, evaluation, code architecture), run on 2026-10-16 after the 2026-10-15 audit; the source of the DR-numbered tasks. |
| [attention_cost_model.md](design/attention_cost_model.md) | The cost model for the cell-attention encoder (openspec task 44.1): measured figures copied from FINDINGS, predictions ESTIMATED, reproduced by `scripts/attention_cost_model.py`. |
| [GLOSSARY.md](design/GLOSSARY.md) | The ubiquitous language. Credit-risk terms are legally loaded; `[verify]` marks what has not been checked against a primary source. |

## roadmap/ — where it is going

| file | what it is for |
| --- | --- |
| [STRATEGY.md](roadmap/STRATEGY.md) | The plan of record: thesis, buyer, phases with falsifiable exit conditions, and what is deliberately not being built. |
| [NEXT.md](roadmap/NEXT.md) | The near-term queue, in tiers. Larger proposals live in [`openspec/changes/`](../openspec/changes/) as structured tasks. |
| [ROADMAP.md](roadmap/ROADMAP.md) | The ordered execution list toward one goal: listable on TabArena. Overrides NEXT.md's tier ordering where the two disagree, and says why. |
| [TABARENA_BAR.md](roadmap/TABARENA_BAR.md) | What the bar actually is, in Elo, after the submission was reviewed and declined. |

## competition/ — who else is in this space

| file | what it is for |
| --- | --- |
| [LANDSCAPE.md](competition/LANDSCAPE.md) | Competitors and what each claims about itself — positioning intelligence, not verified fact — plus how the open projects present themselves and what is worth copying. |
| [SUBMISSION.md](competition/SUBMISSION.md) | The TabArena submission procedure and its stages. The review's outcome is not recorded there: the entry was confirmed correct, declined on competitiveness and withdrawn ([`FINDINGS`](results/FINDINGS.md) §122). |

## leaderboard/ — public standing, own comparison

| file | what it is for |
| --- | --- |
| [index.html](leaderboard/index.html) | fintfm's own TabArena comparison — not the official board. Labelled as such on the page itself, since Elo is only comparable within the population it was computed over. |

## research/ — the literature, and the map through it

| file | what it is for |
| --- | --- |
| [REFERENCES.md](research/REFERENCES.md) | Every entry **verified against the source**. Unverified leads are quarantined at the bottom, because an abstract is a lead and not a citation. |
| [RESEARCH_NOTES.md](research/RESEARCH_NOTES.md) | Concepts, a syllabus audited against what this repository actually implements, baseline families and open threads. **Nothing here is a measurement.** |
| [IMPROVEMENT_PLAN_2026-11.md](research/IMPROVEMENT_PLAN_2026-11.md) | Ranked experiments written 2026-11-02 against FINDINGS up to §202. Planning only: every cost in it is ESTIMATED unless it names a finding. Later entries (§203 onwards) supersede parts of it. |

## infra/ — how to run it

| file | what it is for |
| --- | --- |
| [COMPUTE.md](infra/COMPUTE.md) | Measured throughput on Metal, T4, L4 and Kaggle T4x2, device guidance, what a real run costs, and the rules for a shared machine. |
| [HF_JOBS.md](infra/HF_JOBS.md) | The GPU pretraining recipe on Hugging Face Jobs, with the failures kept because each one cost a run. Kaggle T4x2 step costs are in COMPUTE.md. |

## paper/ — the workspace for a potential write-up

A **claims ledger** mapping every candidate claim to its evidence and status, an outline,
related work, limitations, and the figure list with the command behind each.
**Nothing enters it without a `results/FINDINGS.md` section number** — which is why
`research/RESEARCH_NOTES.md` stays out of it.

| file | what it is for |
| --- | --- |
| [README.md](paper/README.md) | What the workspace is, and why it is deliberately not a draft yet. |
| [CLAIMS.md](paper/CLAIMS.md) | The claims ledger: each claim tagged SURVIVES / SINGLE DRAW / SUPERSEDED / RETRACTED / OPEN against its evidence. |
| [NEGATIVE_RESULTS.md](paper/NEGATIVE_RESULTS.md) | A sourced ledger of what did not work: harmful levers, nulls, refuted predictions, closed and deferred work. Supersedes the short list in OUTLINE.md section 5.5. |
| [OUTLINE.md](paper/OUTLINE.md) | The structure a paper would take, with the claim carrying each section, ordered by what a reviewer checks first. |
| [FIGURES.md](paper/FIGURES.md) | The figures and tables a paper needs, with the command behind each and whether the run exists. |
| [LIMITATIONS.md](paper/LIMITATIONS.md) | The limitations page. |
| [RELATED_WORK.md](paper/RELATED_WORK.md) | Positioning against the field, including dated peer reads. |
| [peer-review_2026-11-08.md](paper/peer-review_2026-11-08.md) | A peer-research pass covering 2026-11-06 to 2026-11-08, read from abstract pages and sources; follows the 2026-11-06 read in RELATED_WORK. |

[`paper/RELATED_WORK.md`](paper/RELATED_WORK.md) and [`research/REFERENCES.md`](research/REFERENCES.md)
overlap and do different jobs: REFERENCES records **what has been read and verified**, for the
whole project; RELATED_WORK argues **what it means for this project's positioning**, for a
paper.

## Conventions, and the reason each exists

- **Every number says how it was produced** — SMOKE-TEST, MEASURED, SIMULATED or ESTIMATED
  (`../CLAUDE.md`). In an ML repository a wrong number does not crash; it looks like a result.
- **Citations are opened before they are cited.** `research/REFERENCES.md` separates verified
  entries from leads for this reason.
- **Findings live here, not in commit messages.** A measurement that cost a training run is an
  asset; a commit message is where it goes to die.
- **Negative and superseded results are kept**, because they are what stops the same wrong
  conclusion being reached twice.
