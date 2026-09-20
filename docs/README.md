# docs/

| file | what it is for |
| --- | --- |
| [STRATEGY.md](STRATEGY.md) | The plan of record: thesis, buyer, phases with falsifiable exit conditions, and what is deliberately not being built. Read this first. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the system works: package layout, the three model stages, the two inference corrections, and the invariants that must not break. |
| [DECISIONS.md](DECISIONS.md) | Decisions with the alternatives priced against them and **what would reverse each one**. |
| [GLOSSARY.md](GLOSSARY.md) | The ubiquitous language. Credit-risk terms are legally loaded; `[verify]` marks what has not been checked against a primary source. A spec introducing a concept adds it here. |
| [FINDINGS.md](FINDINGS.md) | Measured results and the reasoning they forced, numbered and dated, each labelled by how it was produced. |
| [TABARENA.md](TABARENA.md) | External evaluation on TabArena: the reproduction recipe, the coverage fraction any score must carry, and four silent failure modes — including a results cache that returns stale numbers after a preprocessing change. |
| [HF_JOBS.md](HF_JOBS.md) | The GPU pretraining recipe on Hugging Face Jobs, with the failures kept because each one cost a run. |
| [COMPUTE.md](COMPUTE.md) | Measured throughput, device guidance, what a real run costs, and the rules for running on this shared machine. |
| [POSTMORTEM.md](POSTMORTEM.md) | Wrong diagnoses, each caught by measurement rather than review, kept on the record deliberately. |
| [LANDSCAPE.md](LANDSCAPE.md) | Competitors, what each claims about itself, and what it implies for positioning. Claims are theirs, not verified. |
| [REFERENCES.md](REFERENCES.md) | Literature, every entry verified against the source. Unverified leads are quarantined at the bottom. |

Conventions, and the reason they exist:

- **Every number says how it was produced** — SMOKE-TEST, MEASURED, SIMULATED or ESTIMATED
  (`../CLAUDE.md`). In an ML repository a wrong number does not crash; it looks like a result.
- **Citations are opened before they are cited.** `REFERENCES.md` separates verified entries
  from leads for this reason.
- **Findings live here, not in commit messages.** A measurement that cost a training run is
  an asset; a commit message is where it goes to die.
- **Negative and superseded results are kept**, because they are what stops the same wrong
  conclusion being reached twice.

## paper/

[`paper/`](paper/) is the workspace for a potential paper: a **claims ledger** mapping every
candidate claim to its evidence and status, an outline, related work, limitations, and the
figure list with the command behind each. Nothing enters it without a `FINDINGS.md` section
number.

Read [`paper/CLAIMS.md`](paper/CLAIMS.md) first — it is the file that decides what could
honestly be written, and it records supersessions and retractions as prominently as wins.
