# Tasks

- [x] 35.1 **Done (2026-11-15, evidence): the probe table is recorded in FINDINGS §44 (trained 0.685 / 0.692 against untrained control 0.367 / 0.298 on `linear` / `conjunction`, with the old prior and logistic regression and boosting beside them), §46 (0.712 / 0.693 against 0.357 / 0.351) and §141 for the checkpoint now published (`linear` 0.973, `conjunction` 0.971, six structured probes above the control floor). Caveat recorded in §44: the control scores below chance on these two probes, so the margin shows the model is not at the floor, not that it matches a tuned baseline (logistic regression 1.000 and 0.955).** **Gate:** `fintfm-capability` shows the trained model clearing the untrained control
      by a wide margin on `linear` and `conjunction`. Verify: the probe table recorded in
      `docs/results/FINDINGS.md`. Until this passes, everything below measures §42.
- [ ] 35.2 **Progress (2026-11-15): licence of the benchmark code read from the raw `LICENSE` and recorded in `docs/research/tabbench_v2.md`: Apache-2.0 plus an added attribution paragraph 10 (not plain Apache-2.0), commercial use permitted; per-dataset licences censused from OpenML (63 permissive, 9 conditional, 1 non-commercial, 178 unknown and excluded); the Hugging Face dataset Space returned HTTP 401. Still open: README licensing line (not added in this pass) and the owner's call on paragraph 10 when results are published.** Fetch TabBench v2 and check its licence against commercial use **before** running
      anything, adding a line to `README.md`'s licensing section. Verify: the licence quoted in
      the finding, not summarised.
- [x] 35.3 **Done (2026-11-15): report read from the PDF text layer (23 pages); protocol, Table 2 (p. 11) and the 22-problem figures (pp. 14-16) recorded with page references in `docs/research/tabbench_v2.md` and `docs/research/REFERENCES.md`. The 18-of-22 and 15-of-22 figures belong to the vendor's private industrial pool, not to TabBench v2, which the proposal's Why section conflates.** Read their technical report from the PDF, not a summary, and record the published
      leading-model / TabPFN v3 / TabICL v2 figures with their protocol. Verify: a finding citing
      page-level numbers — §36 is the standing reminder that a fetched summary invented an
      entire results table.
- [ ] 35.4 **Progress (2026-11-15): not met. FINDINGS §134 ran the published checkpoint on TabBench's public academic set (189 OpenML ids; the same suite `docs/research/tabbench_v2.md` describes), fold 0 only, 90 of 189 eligible under the 136-feature binary constraint and 84 completed, so native coverage is recorded (84 of 189 run; 6 failed on encoding). Missing: the untrained control arm, the suite's 5-fold protocol, and accuracy as the secondary metric.** Run our checkpoint plus the untrained control under the suite's protocol. Verify:
      accuracy **and native-coverage fraction**, since a model that skips datasets is flattered
      by the average.
- [ ] 35.5 Write the comparison, stating distance from the frontier plainly and stating that
      general tabular accuracy is not the product claim (D3). Verify: a reader of the first
      paragraph learns both.
