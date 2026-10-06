# Open FinTFM-R, a second model family trained on real data

## Why

D16 scored this project's entire architecture/prior-mechanism effort honestly: seven measured
levers (§93, §108, §144-§148), seven nulls or losses, zero wins. That leaves the lever D2/D14
named and deliberately gave up — real training data — as the only one with a credible ceiling
that has never been tested here. Every peer near or above 1200 TabArena Elo either trains on
real tables or is a far larger synthetic effort than this project's own scale tests (§93, §108)
support as a path forward.

D17 (`docs/design/DECISIONS.md`) opened this as a **second family, not a change to FinTFM**:
the existing model keeps its name and its auditability claim intact; **FinTFM-R** ("R" for
real data) is new, and its provenance argument does not exist by construction. That cost is
accepted and documented (`CLAUDE.md`, "Two families"), not hidden.

**Design constraint, from D17: no parallel codebase.** `PriorConfig` already mixes task
generators by weight. A real source is one more weighted generator that *samples* rows
instead of generating them; `collate()`, the training loop, checkpointing and the evaluation
harness are reused unchanged. The synthetic-versus-real question becomes measurable on the
exact paired-bootstrap V4FinBench protocol every lever in §144-§150 was already judged by.

## What this is not

Not a claim that real data helps. D16's own point is that hope is not evidence; this proposal
exists to measure the one lever that has not been measured, honestly, the same way every other
lever this session was.

## Non-goals

- **Not a change to FinTFM.** The existing model, its name, package, DOI and TabArena entry
  are untouched. Every claim about FinTFM in `docs/paper/CLAIMS.md` stays exactly as measured.
- **Not a retraction of D2/D14's auditability bet.** FinTFM-R is an addition, not a reversal —
  the synthetic-only family continues to exist and continues to be the one this project's
  auditability pitch rests on.
- **Not adopting any third-party tabular-foundation-model's code, weights or training data.**
  `CLAUDE.md`'s licensing boundary stands, independent of this proposal — EDGAR and Freddie
  Mac are public/licensed real-world data sources, not another TFM product's artefacts.
- **Not touching Freddie Mac data, code, or derived artefacts before B1 closes.** Phase B is
  listed for completeness and sequencing; nothing in it is permitted to start early.
- **Not a claim that two real sources beat one**, or that real beats synthetic at all — both
  are the open questions A6/A7 and B5/B6 exist to answer, not assumptions this proposal makes.

## Blocked by

- **A1 is blocked by nothing** and is the first thing to run — a feasibility probe needs no
  licence, no GPU, and no repository changes.
- **A2-A7 are blocked by A1 passing.** A bad panel shape (label censoring too severe, too few
  or too clean rare events) stops the phase at A1 rather than being discovered mid-build.
- **All of Phase B is blocked by B1**, a human action (a commercial licensing agreement plus
  Clarity Data Intelligence registration) that only the project owner can close.

## Sources, and why they are sequenced

**Phase A — SEC EDGAR (starts immediately).** Financial Statement Data Sets (XBRL fundamentals,
2009-2026, all filers) for features; 8-K Item 1.03 bankruptcy/receivership disclosures from the
same archive for labels. US federal government work: public domain, no registration, no
agreement, commercially usable. Verified via `scripts/check_doc_links.py`'s standing and via
direct access (`curl` returns 200 with a descriptive `User-Agent`; SEC's Akamai front end
returns 403 without one — not a licensing gate, a bot filter).

**Phase B — Freddie Mac Single-Family Loan-Level Dataset (blocked on a human action).** A
genuinely different real credit domain (consumer mortgage), chosen deliberately for diversity
rather than as an afterthought — but its free tier is non-commercial, and this project is
commercial. **No data, code path, or derived artefact for this source may exist before a
commercial licensing agreement is signed and registration on Clarity Data Intelligence is
complete.** That is tracked as a single blocking task (B1) that only the project owner can
close; nothing downstream of it starts until it does.

## Verify

- A feasibility probe (A1) measures whether the EDGAR+8-K panel is actually usable — a rare
  -event panel comparable to V4FinBench's regime, not too censored by firms that stop filing
  rather than filing a bankruptcy disclosure — **before** any loader is built. A bad shape here
  is D17's own named reversal condition and should stop the phase, not be worked around.
- Any FinTFM-R checkpoint is scored against a matched synthetic-only control on V4FinBench
  under the exact protocol this session's other eight entries used (§144-§150), paired
  bootstrap, Holm-corrected, with the §74 Bayes-ceiling probe run first to confirm the
  checkpoint is not broken before any accuracy claim is trusted.
- Every benchmark a FinTFM-R checkpoint is scored on is checked for SEC-filer (Phase A) or
  Freddie-Mac-loan (Phase B) overlap before scoring, per `CLAUDE.md`'s standing rule.
