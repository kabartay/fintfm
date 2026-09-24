# Tasks

- [ ] 6.1 Establish the data source. Check UK Companies House bulk products and API for
      accounts data, insolvency status and licence terms permitting commercial use. Verify:
      record the licence verbatim in `docs/research/REFERENCES.md`; do not proceed on a recalled
      licence. **The earlier check reached only the overview page and settled nothing.**
- [ ] 6.2 Power calculation before any cohort is fixed. Verify: state the cohort size, the
      expected default count per year, and the smallest AUC difference detectable, in
      `docs/results/FINDINGS.md` §3. If it cannot detect anything useful, say so and reconsider scope.
- [ ] 6.3 Write the inclusion rule as executable code, not prose, so the cohort is mechanical
      and reproducible. Verify: rerunning it on the same snapshot yields an identical cohort
      hash.
- [ ] 6.4 Publish v1 of the register: cohort hash, predictions, model commit, timestamp.
      Verify: a signed git tag exists and the hash is independently recomputable from the
      published inputs.
- [ ] 6.5 Set a recurring reminder to score outcomes. Verify: the first scoring pass runs 12
      months after 6.4, against public insolvency records.
