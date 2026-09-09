# Tasks

- [ ] 35.1 **Gate:** `fintfm-capability` shows the trained model clearing the untrained control
      by a wide margin on `linear` and `conjunction`. Verify: the probe table recorded in
      `docs/FINDINGS.md`. Until this passes, everything below measures §42.
- [ ] 35.2 Fetch TabBench v2 and check its licence against commercial use **before** running
      anything, adding a line to `README.md`'s licensing section. Verify: the licence quoted in
      the finding, not summarised.
- [ ] 35.3 Read their technical report from the PDF, not a summary, and record the published
      Seldon / TabPFN v3 / TabICL v2 figures with their protocol. Verify: a finding citing
      page-level numbers — §36 is the standing reminder that a fetched summary invented an
      entire results table.
- [ ] 35.4 Run our checkpoint plus the untrained control under the suite's protocol. Verify:
      accuracy **and native-coverage fraction**, since a model that skips datasets is flattered
      by the average.
- [ ] 35.5 Write the comparison, stating distance from the frontier plainly and stating that
      general tabular accuracy is not the product claim (D3). Verify: a reader of the first
      paragraph learns both.
