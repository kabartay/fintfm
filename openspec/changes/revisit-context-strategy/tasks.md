# Tasks

- [ ] 32.1 Replicate §29 across three seeds on the survival path before anything else. Verify:
      `uv run fintfm-ctxsweep --seeds 0,1,2`; if the ordering does not hold, §29 is; if the ordering does not hold, §29 is
      downgraded and this proposal closes.
- [ ] 32.2 Add the strategy sweep to the binary benchmark across all four panels, correction on.
      Verify: `uv run fintfm-bench` reporting a strategy column per dataset.
- [ ] 32.3 Paired bootstrap with Holm-Bonferroni over the strategy comparisons. Verify: adjusted
      p-values recorded beside every AUC difference quoted.
- [ ] 32.4 Test the majority-class mechanism: gap against portfolio default rate. Verify: the
      relationship plotted and stated numerically, with the prediction marked hit or miss.
- [ ] 32.5 Resolve D9 either way, in `docs/DECISIONS.md`, and change the class default if the
      evidence supports it. Verify: `uv run pytest -q` green, and the default's justification in
      `classifier.py`'s docstring updated to cite the measurement rather than the paper.
