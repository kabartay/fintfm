# docs/

The project's reasoning, grouped by what each document is *for*. Start with
[`paper/CLAIMS.md`](paper/CLAIMS.md) — it decides what could honestly be written, and it
records supersessions and retractions as prominently as wins.

## results/ — what was measured

| file | what it is for |
| --- | --- |
| [FINDINGS.md](results/FINDINGS.md) | The measurement log, numbered and dated, each entry labelled by how it was produced. **§N** anywhere in this repository refers to an entry here. |
| [TABARENA.md](results/TABARENA.md) | External evaluation under someone else's protocol: the reproduction recipe, the coverage fraction any score must carry, and four silent failure modes — including a results cache that returns stale numbers after a preprocessing change. |
| [POSTMORTEM.md](results/POSTMORTEM.md) | Wrong diagnoses, each caught by measurement rather than review, kept on the record deliberately. |

## design/ — how it works, and why it is built this way

| file | what it is for |
| --- | --- |
| [ARCHITECTURE.md](design/ARCHITECTURE.md) | Package layout, the model stages, the inference corrections, and the invariants that must not break. |
| [DECISIONS.md](design/DECISIONS.md) | Decisions with the alternatives priced against them and **what would reverse each one**. |
| [GLOSSARY.md](design/GLOSSARY.md) | The ubiquitous language. Credit-risk terms are legally loaded; `[verify]` marks what has not been checked against a primary source. |

## roadmap/ — where it is going

| file | what it is for |
| --- | --- |
| [STRATEGY.md](roadmap/STRATEGY.md) | The plan of record: thesis, buyer, phases with falsifiable exit conditions, and what is deliberately not being built. |
| [NEXT.md](roadmap/NEXT.md) | The near-term queue, in tiers. Larger proposals live in [`openspec/changes/`](../openspec/changes/) as structured tasks. |

## competition/ — who else is in this space

| file | what it is for |
| --- | --- |
| [LANDSCAPE.md](competition/LANDSCAPE.md) | Competitors and what each claims about itself — positioning intelligence, not verified fact — plus how the open projects present themselves and what is worth copying. |

## research/ — the literature, and the map through it

| file | what it is for |
| --- | --- |
| [REFERENCES.md](research/REFERENCES.md) | Every entry **verified against the source**. Unverified leads are quarantined at the bottom, because an abstract is a lead and not a citation. |
| [RESEARCH_NOTES.md](research/RESEARCH_NOTES.md) | Concepts, a syllabus audited against what this repository actually implements, baseline families and open threads. **Nothing here is a measurement.** |

## infra/ — how to run it

| file | what it is for |
| --- | --- |
| [COMPUTE.md](infra/COMPUTE.md) | Measured throughput, device guidance, what a real run costs, and the rules for a shared machine. |
| [HF_JOBS.md](infra/HF_JOBS.md) | The GPU pretraining recipe, with the failures kept because each one cost a run. |

## paper/ — the workspace for a potential write-up

A **claims ledger** mapping every candidate claim to its evidence and status, an outline,
related work, limitations, and the figure list with the command behind each.
**Nothing enters it without a `FINDINGS.md` section number** — which is why
`research/RESEARCH_NOTES.md` stays out of it.

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
