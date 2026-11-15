# Tasks

- [x] 6.1 **Done (2026-11-15): every licence and terms quote in `docs/research/companies_house_source.md` and the Companies House and Gazette section of `docs/research/REFERENCES.md` was re-checked verbatim against the raw page text (curl, tags stripped, exact substring search); two differed in wording (the OGL commercial-use sentence lacked a comma, the Crown insignia re-use list had a bracketed paraphrase) and were corrected, the rest are verbatim. No Companies House API terms page exists (guessed URLs return 404, the developer site states none); the Gazette fair use policy is now read. Results per URL are in the "Verification (2026-11-15, raw page text)" subsection.** **Progress (2026-11-15, reopened at merge): not closed, because the licence quotes below came through a fetch summariser, not the pages themselves (CLAUDE.md: a summary is a lead, not a citation); re-read each quote verbatim on its page, then close.** **Draft (2026-11-15, `docs/research/companies_house_source.md`, `docs/research/REFERENCES.md`).** Sources read and licence text recorded verbatim; the bulk products carry no formal licence (Companies House states it "imposes no rules or requirements" on register use, the Gazette is OGL v3.0), and cohort size, third-party copyright in filed accounts and dissolved-company handling are listed there as not established. Establish the data source. Check UK Companies House bulk products and API for
      accounts data, insolvency status and licence terms permitting commercial use. Verify:
      record the licence verbatim in `docs/research/REFERENCES.md`; do not proceed on a recalled
      licence. **The earlier check reached only the overview page and settled nothing.**
- [x] 6.2 **Done (2026-11-11, FINDINGS §3, scripts/register_power.py).** Power calculation before any cohort is fixed. Verify: state the cohort size, the
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
