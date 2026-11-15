# Tasks

- [ ] 35.1 **Gate:** `fintfm-capability` shows the trained model clearing the untrained control
      by a wide margin on `linear` and `conjunction`. Verify: the probe table recorded in
      `docs/results/FINDINGS.md`. Until this passes, everything below measures §42.
- [ ] 35.2 **Progress (2026-11-15): licence of the benchmark code read from the raw `LICENSE` and recorded in `docs/research/tabbench_v2.md`: Apache-2.0 plus an added attribution paragraph 10 (not plain Apache-2.0), commercial use permitted; per-dataset licences censused from OpenML (63 permissive, 9 conditional, 1 non-commercial, 178 unknown and excluded); the Hugging Face dataset Space returned HTTP 401. Still open: README licensing line (not added in this pass) and the owner's call on paragraph 10 when results are published.** Fetch TabBench v2 and check its licence against commercial use **before** running
      anything, adding a line to `README.md`'s licensing section. Verify: the licence quoted in
      the finding, not summarised.
- [x] 35.3 **Done (2026-11-15): report read from the PDF text layer (23 pages); protocol, Table 2 (p. 11) and the 22-problem figures (pp. 14-16) recorded with page references in `docs/research/tabbench_v2.md` and `docs/research/REFERENCES.md`. The 18-of-22 and 15-of-22 figures belong to the vendor's private industrial pool, not to TabBench v2, which the proposal's Why section conflates.** Read their technical report from the PDF, not a summary, and record the published
      leading-model / TabPFN v3 / TabICL v2 figures with their protocol. Verify: a finding citing
      page-level numbers — §36 is the standing reminder that a fetched summary invented an
      entire results table.
- [ ] 35.4 Run our checkpoint plus the untrained control under the suite's protocol. Verify:
      accuracy **and native-coverage fraction**, since a model that skips datasets is flattered
      by the average.
- [ ] 35.5 Write the comparison, stating distance from the frontier plainly and stating that
      general tabular accuracy is not the product claim (D3). Verify: a reader of the first
      paragraph learns both.
