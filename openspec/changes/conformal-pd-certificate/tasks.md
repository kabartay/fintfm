# Tasks

- [ ] 5.1 Confirm the licence boundary in writing before any code moves: `finkele-axiom` is
      AGPLv3, this repo Apache-2.0, and neither may import the other. Verify: `grep -rn
      "^\s*\(import\|from\)\s\+axiom" src/` prints nothing, asserted in a test.
- [ ] 5.2 Split-conformal PD intervals on a pre-registered calibration slice. Verify: a test
      that empirical coverage on synthetic data matches the nominal level within its
      binomial interval.
- [ ] 5.3 Report coverage **with interval width**, always. Verify: a test that the metrics
      object cannot be constructed with coverage and no width — a coverage number alone is
      inadmissible.
- [ ] 5.4 OOD detection over features with escalation, reported as correct behaviour rather
      than failure. Verify: a synthetic disjoint family is escalated at a high rate.
- [ ] 5.5 Coverage under regime shift: calibrate on one period, evaluate on another. Verify:
      numbers into `docs/FINDINGS.md`, re-derived from `results.json`.
- [ ] 5.6 Generate `certificate.md` where every clause cites a `results.json` field. Verify:
      a test that each numeric claim in the generated document resolves to a field.
