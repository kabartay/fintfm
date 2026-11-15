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

- [ ] 9.1 **Progress (2026-11-16): owner decision: the grid is deferred; §233 found no lift from 480,000 tasks, so quota goes to Elo levers first and the grid is revisited after the TabArena PR. The proposal in docs/results/drafts/scaling_grid_2026-11-15.md stands as written.** **Progress (2026-11-15): a PROPOSED grid awaits the owner's decision; this task stays open.** `docs/results/drafts/scaling_grid_2026-11-15.md` proposes tasks 48,000 / 480,000 / 4,800,000, model sizes 885K and 5.0M parameters, and 3 / 2 / 1 seeds (12 runs); nothing is written into this file as a table until the owner accepts it. Decide and record the grid before running: task counts, model sizes, seeds.
      Verify: written into this file as a table, so the grid cannot be trimmed after seeing
      results.
- [ ] 9.2 **Progress (2026-11-15): priced, not closed.** The proposed grid is priced per cell in `docs/results/drafts/scaling_grid_2026-11-15.md`: 280.2 Kaggle T4x2 quota hours (560.4 GPU-h, 9.3 weeks at 30 h) for all 12 runs, ESTIMATED from the MEASURED 0.54 s/step in COMPUTE.md (§197, §198). The 885K step time is COMPUTE.md's; the 5.0M step time is a ratio (0.98 / 0.64, pre-cell-attention T4) applied to it, which is estimated afresh, so the Verify is not met literally. Missing: a measured 5.0M published-recipe step time. Price the grid in GPU hours from measured step times. Verify: numbers from
      `docs/infra/COMPUTE.md`, not estimated afresh.
- [x] 9.3 **Done (2026-11-15): `fintfm-ablate --grid configs/scaling/grid-2026-11-15.yaml` expands a (task count, model size, seed) grid from one YAML spec (`src/fintfm/experiments/scaling_grid.py`), training each cell to its own `results.json` with commit, recipe path and hash, steps x batch = tasks seen and parameter count; `--plan` prints every cell's training command (with `--checkpoint-every`) and its ESTIMATED cost from the draft's step times without training. The new tests are in `tests/test_scaling_grid.py`; the Verify command still passes. The grid itself (9.1) remains the owner's decision.** Extend the ablation harness to sweep scale as well as prior mixture, writing one
      `results.json` per cell. Verify: `uv run pytest tests/test_experiments.py -q`.
- [ ] 9.4 Run the grid on rented GPU. Verify: every cell has a `results.json` with its commit.
- [ ] 9.5 Plot real-data performance against pretraining volume and record the verdict in
      `docs/results/FINDINGS.md`, explicitly stating whether it is monotone, saturating or flat.
      Verify: the figure is regenerated from the per-cell `results.json` files by a script, so
      no number in it is retyped.
