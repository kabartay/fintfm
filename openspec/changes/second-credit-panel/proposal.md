# A second, independent credit panel

## Why

Every real-data number in this repository comes from one dataset: UCI Polish companies
bankruptcy. A result on a single panel is a result about that panel. Polish firms in
2000-2013 are one economy, one accounting regime and one crisis, and the model could be
learning that rather than credit risk.

`docs/STRATEGY.md` lists this as a Phase 1 step for exactly this reason: results must not
rest on one dataset before anything is claimed externally.

## What

Acquire and wire in at least one further corporate-default panel with commercially usable
licensing. Candidates, none yet verified:

- **Taiwanese bankruptcy prediction** (UCI) — 6,819 firms, 95 features. Different economy,
  similar shape, likely CC BY.
- **V4FinBench** — described in conversation as ~1M company-year observations, 131 features,
  multiple horizons, with TabPFN baselines. **Unverified — not yet opened**, and quarantined
  as a lead in `docs/REFERENCES.md`.
- SEC EDGAR XBRL filings joined to Chapter 11 filings — free and public domain, but the join
  is real work and the label definition is a design decision, not a given.

## Non-goals

- Not a licensed commercial database (Compustat, Capital IQ, Moody's DRD) at this stage. Those
  are the right answer eventually and the wrong answer before Phase 1 returns a result.
- Not building a data pipeline. One extra panel, loaded the same way as the first.

## Falsified by

Not applicable — this reduces the risk that existing results are dataset artefacts. The
informative outcome is a *disagreement* between panels, which would be a genuine finding
about generalisation rather than a setback.

## Blocked by / blocks

- **Blocked by** nothing.
- **Blocks** any external claim about model quality, and `time-based-evaluation` if the UCI
  files turn out to carry no period information.
