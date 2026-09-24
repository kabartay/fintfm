# Tasks

- [ ] 12.1 Read ExplainerPFN in full and record how attribution targets are generated and
      what fidelity they claim. Verify: entry expanded in `docs/research/REFERENCES.md`; currently
      only the abstract has been read.
- [ ] 12.2 Establish the baseline to beat: SHAP over gradient boosting on the UCI panels,
      including runtime. Verify: numbers into `docs/results/FINDINGS.md`, re-derived from the run.
- [ ] 12.3 Emit ground-truth attributions from `prior/financial.py`, which knows the true
      driver weights. Verify: a test that a feature with zero weight receives near-zero
      attribution.
- [ ] 12.4 Add an attribution head and train it jointly. Verify: agreement with SHAP-on-GBM
      reported as a rank correlation per dataset, not as an average.
- [ ] 12.5 Measure attribution stability across context resamples and seeds. Verify: a test
      asserting a stability threshold, chosen and recorded before it is measured.
