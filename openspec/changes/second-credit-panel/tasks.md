# Tasks

- [x] 7.1 Open the Taiwanese bankruptcy dataset page and record instances, features, target
      definition, period and **licence verbatim**. Verify: entry added to
      `docs/research/REFERENCES.md` under verified sources, not leads.
      **Done 2026-09-08: 6,819 firms, 95 features, 3.23% positives, 1999-2009, zero missing
      values, CC BY 4.0 (commercial OK). Downloaded and inspected: NO dates, NO identifiers.**
- [x] 7.2 **Done (2026-11-15, evidence): FINDINGS §8 (2026-09-08) verified V4FinBench by opening it; `docs/research/REFERENCES.md` lists it among the verified sources (data CC BY 4.0, code MIT) and its only 'unverified' section ("Still unverified") does not contain it; ingested and measured in §22 and §168.** Open V4FinBench and either promote it from lead to verified reference, or record
      why it is unusable. Verify: `docs/research/REFERENCES.md` no longer lists it as unverified.
- [x] 7.3 **Done (2026-11-15, evidence): `load_taiwan_bankruptcy` (CC-BY-4.0, own attribution field) sits beside `load_polish_bankruptcy` in `src/fintfm/evaluation/datasets.py`, and `load_v4finbench` carries licence and attribution too; `run_credit` prints `source: <attribution>` per panel (`fintfm-bench --credit`), covered by `tests/test_bench_credit.py` (3 passed, 2026-11-15).** Add a loader alongside `load_polish_bankruptcy`, carrying its own licence and
      attribution fields. Verify: `uv run pytest tests/ -q` plus attribution printed by
      `fintfm-bench --credit`.
- [x] 7.4 **Done (2026-11-15, evidence): FINDINGS §35 (Polish and Taiwan, three seeds, retrieval best on both panels, rank transform better in 7 of 8 configurations), and §75, which states that the panels disagree with V4FinBench (the financial prior's benefit does not generalise past V4FinBench).** Rerun the credit benchmark across both panels. Verify: numbers into
      `docs/results/FINDINGS.md`, and state explicitly whether the panels agree.
