# Tasks

## Phase A — SEC EDGAR (unblocked, start now)

- [ ] A1 **Feasibility probe: is the EDGAR + 8-K Item 1.03 panel actually usable?** Download a
      handful of quarters of the Financial Statement Data Sets and the 8-K full-text/structured
      index for the same period, join bankruptcy/receivership disclosures to filers by CIK.
      This is D17's own named reversal condition; a bad shape here stops the phase rather than
      being worked around. No repository code changes in this task — a scratch script and a
      written-up result only. Verify: report the resulting panel's positive rate (is it a
      rare-event panel in V4FinBench's regime, 0.19-4%, or something else), the censoring rate
      (firms that stop filing without ever filing an Item 1.03 — silently missing, not
      labelled), and row/feature coverage.
- [ ] A2 **Build the EDGAR loader**, gated on A1 passing. A `Task`-compatible generator
      (`prior/real_edgar.py` or similar) that samples real firm-quarter rows rather than
      generating synthetic ones, matching the existing `sample_financial_task` signature
      closely enough that `mixture.py` can call it the same way. Verify: unit tests mirroring
      `tests/test_prior.py`'s conventions — shapes, label correctness, no leakage of query rows
      into any fitted statistic.
- [ ] A3 **Wire `p_real_edgar` into `PriorConfig`/`mixture.py`**, D17's "no parallel codebase"
      constraint made concrete. Verify: `collate()` and the training loop accept a batch drawn
      partly or wholly from the real source with no code path divergent from the synthetic
      case; a test trains a few steps at `p_real_edgar=1.0` and confirms loss is finite and
      decreasing, mirroring the smoke tests already in `tests/test_train.py`.
- [ ] A4 **Document the new source** in `README.md`'s licensing section and
      `docs/research/REFERENCES.md`. Verify: the standing instruction already in
      `README.md`'s licensing section ("add a line there when a new source is added") is
      followed, naming the source, its licence and the date verified.
- [ ] A5 **Local smoke run**: a short training run (hundreds of steps, not thousands) at
      `p_real_edgar > 0`, checked the same way every GPU launch this session was. Verify:
      `uptime` checked first if local, `--checkpoint-every` set if remote, and a throughput
      check taken within the first 15 minutes against a known-good baseline, before any real
      compute is spent on the decisive run (A6).
- [ ] A6 **The decisive run**: a full matched-compute checkpoint (same architecture, step
      count and seed as the current `v4-cellattn-labels.pt` control) with `p_real_edgar` at a
      scoped weight. Per `CLAUDE.md`'s benchmark-ordering rule: V4FinBench before TabArena,
      always. Verify: scored against the matched control on V4FinBench under the exact
      protocol §144-§150 used — paired bootstrap, Holm-corrected — with the §74 Bayes-ceiling
      probe run first to confirm the checkpoint is not broken before any accuracy claim is
      trusted.
- [ ] A7 **Write up the result, honestly, either direction.** Verify: a new `FINDINGS.md`
      entry states whether a weighted real-EDGAR mixture moved V4FinBench AP against the
      matched synthetic-only control either way; `docs/paper/CLAIMS.md` gains a FinTFM-R-scoped
      claim without retroactively broadening Claim 2; `docs/design/DECISIONS.md` D17 is updated
      with the result if it bears on the reversal condition stated there.

## Phase B — Freddie Mac Single-Family Loan-Level Dataset (blocked)

- [ ] B1 **[BLOCKED — human action, not engineering]** Execute a commercial licensing
      agreement with Freddie Mac and complete registration on Clarity Data Intelligence. The
      free tier is academic/non-commercial; this project is commercial. **Nothing below this
      line may start before this task is closed** — no data download, no loader code, no
      derived artefact. This is the same rule `CLAUDE.md`'s licensing boundary applies to every
      other third-party source, made concrete for this one. Verify: a signed agreement and a
      completed Clarity Data Intelligence registration both exist before B2 begins.
- [ ] B2 Read the signed agreement's actual terms (not the public terms page) for any
      restriction beyond "commercial use requires a licence" — redistribution limits, retention
      limits, required attribution. Verify: those terms are recorded in `README.md`'s licensing
      section before any code in B3 onward touches the data.
- [ ] B3 Build the mortgage-loan loader (`prior/real_mortgage.py` or similar), mirroring A2's
      shape, adapted for loan-level rather than firm-level rows. Verify: the same test
      conventions A2 used, adapted for loan-level fields — shapes, label correctness, no query
      -row leakage into any fitted statistic.
- [ ] B4 Wire `p_real_mortgage` into `PriorConfig`/`mixture.py`, alongside `p_real_edgar` from
      Phase A — both present from Phase A's first commit (per D17) so this is additive, not a
      rework. Verify: the same smoke-run convention A3/A5 used, at `p_real_mortgage=1.0`.
- [ ] B5 A decisive run comparing EDGAR-only against EDGAR-plus-mortgage at matched compute.
      Tests D17's diversity argument directly: does a second real domain buy anything over
      one. Verify: scored the same way as A6, against both the synthetic-only control and the
      EDGAR-only checkpoint from A6, on V4FinBench, paired bootstrap, Holm-corrected.
- [ ] B6 Write up the result honestly, same discipline as A7. Verify: a new `FINDINGS.md`
      entry and the corresponding `CLAIMS.md`/`DECISIONS.md` updates, stated either direction.
