# Tasks

- [ ] 9.1 Decide and record the grid before running: task counts, model sizes, seeds.
      Verify: written into this file as a table, so the grid cannot be trimmed after seeing
      results.
- [ ] 9.2 Price the grid in GPU hours from measured step times. Verify: numbers from
      `docs/COMPUTE.md`, not estimated afresh.
- [ ] 9.3 Extend the ablation harness to sweep scale as well as prior mixture, writing one
      `results.json` per cell. Verify: `uv run pytest tests/test_experiments.py -q`.
- [ ] 9.4 Run the grid on rented GPU. Verify: every cell has a `results.json` with its commit.
- [ ] 9.5 Plot real-data performance against pretraining volume and record the verdict in
      `docs/FINDINGS.md`, explicitly stating whether it is monotone, saturating or flat.
