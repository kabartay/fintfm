# Tasks

> **SUPERSEDED 2026-09-25 — do not run this grid.** The question it was designed to answer has
> been answered three independent ways, all negative, and the grid would cost GPU hours to
> re-derive a conclusion already in hand:
>
> - **Parameters, ours, matched:** 5.7× the parameters on the *same* 48,000 tasks scores
>   **−0.0049** on TabArena binary, 12/27, p = 0.701 (§114). §108's earlier −0.0077 was
>   confounded by half the task count; removing that recovered +0.0028 and no more.
> - **Volume, ours:** null at 5× (§93), **+0.0028** at 2× (§114).
> - **Parameters, a peer's, clean:** Nori publishes 6M → 100M for **+0.0049 R²** — a 16.7×
>   increase returning single-digit thousandths against this project's 0.035 deficit (§110).
> - **The other corner:** the deep-narrow arm **could not be scored at all**, hitting the
>   harness time limit after 8 of 27 datasets (§114).
>
> `docs/roadmap/STRATEGY.md`'s Phase 1 target of 10–50M parameters is **withdrawn** on this
> evidence rather than merely questioned. The tasks below are kept unticked because they were
> never run — marking them done would claim work that did not happen — and this banner records
> why they should not be.

- [ ] 9.1 Decide and record the grid before running: task counts, model sizes, seeds.
      Verify: written into this file as a table, so the grid cannot be trimmed after seeing
      results.
- [ ] 9.2 Price the grid in GPU hours from measured step times. Verify: numbers from
      `docs/infra/COMPUTE.md`, not estimated afresh.
- [ ] 9.3 Extend the ablation harness to sweep scale as well as prior mixture, writing one
      `results.json` per cell. Verify: `uv run pytest tests/test_experiments.py -q`.
- [ ] 9.4 Run the grid on rented GPU. Verify: every cell has a `results.json` with its commit.
- [ ] 9.5 Plot real-data performance against pretraining volume and record the verdict in
      `docs/results/FINDINGS.md`, explicitly stating whether it is monotone, saturating or flat.
      Verify: the figure is regenerated from the per-cell `results.json` files by a script, so
      no number in it is retyped.
