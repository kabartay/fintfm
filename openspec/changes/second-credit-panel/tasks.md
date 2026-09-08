# Tasks

- [x] 7.1 Open the Taiwanese bankruptcy dataset page and record instances, features, target
      definition, period and **licence verbatim**. Verify: entry added to
      `docs/REFERENCES.md` under verified sources, not leads.
      **Done 2026-09-08: 6,819 firms, 95 features, 3.23% positives, 1999-2009, zero missing
      values, CC BY 4.0 (commercial OK). Downloaded and inspected: NO dates, NO identifiers.**
- [ ] 7.2 Open V4FinBench and either promote it from lead to verified reference, or record
      why it is unusable. Verify: `docs/REFERENCES.md` no longer lists it as unverified.
- [ ] 7.3 Add a loader alongside `load_polish_bankruptcy`, carrying its own licence and
      attribution fields. Verify: `uv run pytest tests/ -q` plus attribution printed by
      `fintfm-bench --credit`.
- [ ] 7.4 Rerun the credit benchmark across both panels. Verify: numbers into
      `docs/FINDINGS.md`, and state explicitly whether the panels agree.
