# docs/

| file | what it is for |
| --- | --- |
| [STRATEGY.md](STRATEGY.md) | The plan of record: thesis, buyer, phases with falsifiable exit conditions, and what is deliberately not being built. Read this first. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the system works: package layout, the three model stages, the two inference corrections, and the invariants that must not break. |
| [DECISIONS.md](DECISIONS.md) | Decisions with the alternatives priced against them and **what would reverse each one**. |
| [FINDINGS.md](FINDINGS.md) | Measured results and the reasoning they forced, numbered and dated, each labelled by how it was produced. |
| [COMPUTE.md](COMPUTE.md) | Measured throughput, device guidance, what a real run costs, and the rules for running on this shared machine. |
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
