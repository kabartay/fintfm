# Tasks

## Phase A — SEC EDGAR (unblocked, start now)

- [x] A1 **Done: passes (§151).** 5,506 unique filers with XBRL fundamentals in one quarter
      alone; a core set of common GAAP tags (`StockholdersEquity`, `NetIncomeLoss`,
      `Revenues`, `Assets`) recurs tens of thousands of times, comparable density to
      V4FinBench's own 131-feature panel. "Item 1.03 Bankruptcy" disclosures run 49-80 a
      quarter, stable across five quarters spanning 2010-2024; one was read in full and
      confirmed a genuine Chapter 11 event, not a boilerplate mention. Crude base rate lands in
      V4FinBench's rare-event regime. Censoring (firms that stop filing without ever disclosing
      distress) is real — Form 15 deregistrations outnumber bankruptcy disclosures the same
      quarter — but maps directly onto `fintfm.modeling.hazard`'s existing `CENSORED`
      machinery rather than requiring new design. D17's reversal condition does not trigger.
      Original text follows.
      **Feasibility probe: is the EDGAR + 8-K Item 1.03 panel actually usable?** Download a
      handful of quarters of the Financial Statement Data Sets and the 8-K full-text/structured
      index for the same period, join bankruptcy/receivership disclosures to filers by CIK.
      This is D17's own named reversal condition; a bad shape here stops the phase rather than
      being worked around. No repository code changes in this task — a scratch script and a
      written-up result only. Verify: report the resulting panel's positive rate (is it a
      rare-event panel in V4FinBench's regime, 0.19-4%, or something else), the censoring rate
      (firms that stop filing without ever filing an Item 1.03 — silently missing, not
      labelled), and row/feature coverage.
- [x] A2 **Done.** `prior/real_edgar.py`: `sample_real_edgar_task` samples real firm-quarter
      rows from a panel (fixture-injectable for tests, path-loaded and cached otherwise),
      returns a standard `Task` (`source="real_edgar"`), raises rather than truncates if the
      panel's feature count falls outside `[min_features, max_features]` (mirroring
      `collate()`'s own invariant), and repairs a degenerate single-class draw the same way
      the synthetic priors do. `tests/test_real_edgar.py`: 11 tests, shapes, without
      -replacement sampling, degenerate-label repair, refusal paths, and a mixed real+synthetic
      batch collating with no `collate()` changes. One real bug caught by running the tests
      rather than reasoning about them: `DataFrame.to_numpy()` can return a read-only array,
      which `y[flip] = ...` then fails on -- fixed with `copy=True`.
      Original text follows.
      **Build the EDGAR loader**, gated on A1 passing. A `Task`-compatible generator
      (`prior/real_edgar.py` or similar) that samples real firm-quarter rows rather than
      generating synthetic ones, matching the existing `sample_financial_task` signature
      closely enough that `mixture.py` can call it the same way. Verify: unit tests mirroring
      `tests/test_prior.py`'s conventions — shapes, label correctness, no leakage of query rows
      into any fitted statistic.
- [x] A3 **Done.** `PriorConfig.p_real_edgar` (default `0.0`) and `.real_edgar_panel_path`;
      `sample_task` checks it before `p_financial`, and `_sample_tasks_reusing_graphs`'s
      replicated decision chain was updated to match so `scm_reuse_graph > 1` cannot misroute
      a real-edgar draw. `--p-real-edgar`/`--real-edgar-panel-path` added to `fintfm-train`.
      `collate()` needed no changes at all -- the whole point of D17's "no parallel codebase"
      constraint. A real test trains 60 steps at `p_real_edgar=1.0` on a fixture panel with a
      genuine (not noise) feature-label relationship and confirms the loss actually decreases,
      not merely that nothing crashes. **Also found and fixed a real provenance-check gap**:
      `openspec/tools/validate.py --provenance` (spec P1) scans `prior/` for real-data-reading
      patterns and correctly flagged `real_edgar.py`'s `read_parquet` call. Added a narrow,
      named exemption (`PROVENANCE_EXEMPT`) rather than weakening the check, with its own
      self-verifying test that fails if `PriorConfig.p_real_edgar`'s default ever stops being
      `0.0` -- `openspec/specs/pretraining-provenance/spec.md`'s P1 updated to document the
      exception and point to D17. Original text follows.
      **Wire `p_real_edgar` into `PriorConfig`/`mixture.py`**, D17's "no parallel codebase"
      constraint made concrete. Verify: `collate()` and the training loop accept a batch drawn
      partly or wholly from the real source with no code path divergent from the synthetic
      case; a test trains a few steps at `p_real_edgar=1.0` and confirms loss is finite and
      decreasing, mirroring the smoke tests already in `tests/test_train.py`.
- [x] A4 **Done.** `README.md`'s licensing section names both real sources (EDGAR verified
      and in use; Freddie Mac adopted but gated on B1) with the verification date;
      `docs/research/REFERENCES.md` gained an entry pointing to §151. Original text follows.
      **Document the new source** in `README.md`'s licensing section and
      `docs/research/REFERENCES.md`. Verify: the standing instruction already in
      `README.md`'s licensing section ("add a line there when a new source is added") is
      followed, naming the source, its licence and the date verified.
- [x] A5 **Done (§152).** `scripts/edgar/build_panel.py` built a real panel (2023, four
      quarters, 24,677 rows, 1.39% positive -- in V4FinBench's regime) and a 300-step,
      109K-parameter checkpoint trained against it end to end: sampling, collation, training,
      mid-run checkpointing, reload, a finite forward pass. Two real bugs in the panel
      builder caught by running it against known values rather than trusting the arithmetic
      (quarter-end dates a month early; the bankruptcy-label search window not extended past
      the panel's own last quarter, undercounting events 314 -> 613 once fixed), plus an
      undocumented SEC API pagination ceiling (`from >= 100` 500s) found by deliberately
      probing the boundary. `uptime` checked first (load ~10, moderate); this is a correctness
      check, not a measurement -- task A6 remains the decisive one. Original text follows.
      **Local smoke run**: a short training run (hundreds of steps, not thousands) at
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
- [ ] A7 **Write up the result, honestly, either direction.** Verify: a new `docs/results/FINDINGS.md`
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
- [x] B3 **Done (§154).** `scripts/freddie_mac/build_panel.py` parses the 2018-vintage sample
      (31 origination + 35 performance pipe-delimited fields, layout verified against the
      current user guide and a real row's exact field count) into a 33-feature panel and a
      binary distress label (Zero Balance Code 03/09). `prior/real_mortgage.py` and
      `tests/test_real_mortgage.py` mirror `real_edgar.py`/`test_real_edgar.py` field for
      field, 15 tests. **One open problem, not yet resolved**: the measured positive rate is
      0.048% (24 of 50,000), rarer than this project's own working regime — see §154 for why
      and what B4/B5 need before a decisive run is trustworthy. Original text follows.
      Build the mortgage-loan loader (`prior/real_mortgage.py` or similar), mirroring A2's
      shape, adapted for loan-level rather than firm-level rows. Verify: the same test
      conventions A2 used, adapted for loan-level fields — shapes, label correctness, no query
      -row leakage into any fitted statistic.
- [x] B4 **Done (§154).** `PriorConfig.p_real_mortgage`/`.real_mortgage_panel_path` wired into
      `mixture.py` alongside `p_real_edgar`, and into `fintfm-train` as
      `--p-real-mortgage`/`--real-mortgage-panel-path`; `openspec/tools/validate.py`'s
      `PROVENANCE_EXEMPT` gained a matching entry. `test_training_at_p_real_mortgage_one_decreases_loss`
      trains 60 steps on a fixture panel with a genuine feature-label relationship and confirms
      the loss decreases, matching A3's bar. Original text follows.
      Wire `p_real_mortgage` into `PriorConfig`/`mixture.py`, alongside `p_real_edgar` from
      Phase A — both present from Phase A's first commit (per D17) so this is additive, not a
      rework. Verify: the same smoke-run convention A3/A5 used, at `p_real_mortgage=1.0`.
- [ ] B5 A decisive run comparing EDGAR-only against EDGAR-plus-mortgage at matched compute.
      Tests D17's diversity argument directly: does a second real domain buy anything over
      one. Verify: scored the same way as A6, against both the synthetic-only control and the
      EDGAR-only checkpoint from A6, on V4FinBench, paired bootstrap, Holm-corrected.
- [ ] B6 Write up the result honestly, same discipline as A7. Verify: a new `docs/results/FINDINGS.md`
      entry and the corresponding `docs/paper/CLAIMS.md`/`docs/design/DECISIONS.md` updates, stated either direction.

## Phase C — Further real sources, audited one at a time (§155-§157)

- [x] C1 **Done (§155).** LendingClub (Zenodo 11295916, CC-BY-4.0) built, loaded, smoke-trained;
      its BeyondArena overlap (`lending_club_1m`) found late and recorded as a bounded forfeit.
      Verify: panel builds, loss decreases on a fixture, a CPU smoke run on the real panel saves a
      checkpoint that reloads and runs a finite forward pass.
- [x] C2 **Done (§156).** Candidate registry (`docs/research/data_candidates.csv`) against a
      TabArena + BeyondArena snapshot, with a mechanical overlap check run in CI. Verify:
      `tests/test_data_candidates.py` passes, including the regression test for the
      one-word/two-word name mismatch that missed LendingClub.
- [x] C3 **Done (§156).** Generic panel sampler (`prior/real_panel.py`, `--real-panels`) with a
      real-positive floor; PPDai, Bondora, ULB card fraud and SBA 7(a) panels built; feasibility
      benchmark run on all seven real panels. Verify: `tests/test_real_panel.py` passes, including
      that floor-enforced positives are genuine positive rows; feasibility table recorded.
- [x] C4 **Done (§157): six arms scored on identical fold-0 rows; none separates from the control
      after Holm, SBA's lead shrinks to +0.0073 mean over five folds (3 of 5 positive), PPDai
      second; the pre-registered base-rate hypothesis is rejected.**
      **Continued-pretraining pilot**, one arm per source, all from the A6 control checkpoint
      (`scripts/candidates/pilot.sh`), scored on V4FinBench fold 0. Verify: every arm scored on
      identical rows, paired-bootstrap AP difference against the control with Holm correction
      (`scripts/candidates/pilot_compare.py`), written up as §157 either direction.
- [ ] C5 **Prepared 2026-10-07, not launched:** SBA at seeds 0 and 1, `max_classes 10`, kernels
      `scripts/kaggle/fintfm_r_sba_s{0,1}` (pinned d03af3d, smoke-tested 5 steps locally, 888,090
      parameters). §158 measured seed noise at −0.0073 mean AP, the size of the pilot's SBA
      effect, so one seed cannot decide this; at head size 10 the two controls are
      `task_158_matched_control` (seed 0, queued anyway) and the scored s1 replicate (seed 1).
      Order: matched control, SBA s0, SBA s1; PPDai (`fintfm_r_ppdai_s{0,1}`) only if SBA's
      difference has the same sign at both seeds. Panels staged in `runs/kaggle_staging/`; the
      private dataset is not yet created.
      **The decisive run for whatever C4 ranks first**, on GPU once quota allows: from scratch
      at the A6 recipe and matched compute, V4FinBench five folds before TabArena. Verify: the
      same protocol as A6, and a pilot-scale result is not quoted as if it were this one.
- [ ] C6 **Freddie Mac at a usable scale** -- the full 2018 vintage file or several combined
      vintages, since the 50k sample's 24 positives cannot be learned from (§154, §156). Verify:
      the feasibility benchmark shows a learnable label (GBM AP clearly above base rate) before it
      enters any pilot arm.
