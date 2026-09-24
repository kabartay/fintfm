# Start the forward prediction register now

## Why

The only evaluation immune to both leakage modes is predicting outcomes that do not yet exist
(`docs/results/FINDINGS.md` §3). Every vendor arrives holding excellent retrospective numbers and a
model-risk committee has no way to tell skill from a well-tuned backtest.

**A track record cannot be bought, back-dated, or acquired with a funding round.** It has to
be lived. That makes it the one axis where a solo founder starting today beats a funded
competitor entering next year, and `docs/competition/LANDSCAPE.md` says every other axis favours them.

Its value is purely a function of elapsed time, so the cost of delay is the whole asset.
**Start it before the model is good** — a mediocre model with a two-year honest record is
more persuasive to a supervisor than an excellent model with none.

## What

- Fix a cohort of companies from public filings, with a documented, mechanical inclusion rule
  so the cohort cannot be curated after the fact.
- Publish predictions with a hash and a timestamp nobody can backdate: a signed git tag
  pushed to a public repository, or an OSF/arXiv registration.
- Record the power calculation up front — at a few percent base rate, a few thousand firms
  yield only tens of defaults a year. **A null result from an underpowered cohort is not
  evidence of parity**, and saying so in advance is what stops it being spun later.
- Score against public insolvency records as outcomes arrive.

## Non-goals

- Not a product feature. This is evidence generation.
- Not requiring a good model to start. The first register can use whatever exists.

## Falsified by

The register cannot be falsified, only underpowered. The cheapest check on its worth is the
power calculation in task 6.2: if the achievable cohort cannot detect a difference worth
caring about, this becomes a credibility exercise rather than a statistical one, and should
be described as such.

## Blocked by / blocks

- **Blocked by** a licensed source of company financials with observable outcomes. Expect
  this to be the binding constraint (`docs/results/FINDINGS.md` §4). UK Companies House is the most
  promising lead — free bulk data plus published insolvency notices — but its licence terms
  and whether accounts data is included were **not** established when checked on 2026-09-08.
- **Blocks** nothing, and blocked by nothing that should delay starting a minimal version.
