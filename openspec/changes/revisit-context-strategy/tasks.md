# Tasks

- [x] 32.1 Replicate §29 across three seeds on the survival path before anything else. Verify:
      `uv run fintfm-ctxsweep --seeds 0,1,2`; if the ordering does not hold, §29 is; if the ordering does not hold, §29 is
      downgraded and this proposal closes.
- [x] 32.2 Add the strategy sweep to the binary benchmark across all four panels, correction on.
      Verify: `uv run fintfm-bench` reporting a strategy column per dataset.
- [ ] 32.3 Paired bootstrap with Holm-Bonferroni over the strategy comparisons. Verify: adjusted
      p-values recorded beside every AUC difference quoted.
- [ ] 32.4 Test the majority-class mechanism: gap against portfolio default rate. Verify: the
      relationship plotted and stated numerically, with the prediction marked hit or miss.
- [x] 32.5 Resolve D9 either way, in `docs/DECISIONS.md`, and change the class default if the
      evidence supports it. Verify: `uv run pytest -q` green, and the default's justification in
      `classifier.py`'s docstring updated to cite the measurement rather than the paper.

**Resolved 2026-09-09.** 32.1 replicated §29 on three seeds (§33): retrieval's worst seed
beats uniform's best at every context size, and the blind ordering holds. 32.2 measured the
binary path on Polish and Taiwan (§35) and found the strategies **nearly tied** there —
uniform − balanced is −0.0004 (not significant) on Polish and +0.0077 on Taiwan — so §29's
10-12 point margin does **not** generalise. 32.5 changed the class default to `uniform` and
recorded D10.

**The mechanism was narrower than §29 proposed:** "balanced" is not one operation. The smaller
panels lack the positives for it to reach 50/50, so it lands near 25% and behaves like hybrid.
The effect scales with **how extreme the rebalancing actually is**, not with the strategy's
name.

Task 32.3's Holm-corrected bootstrap was run for the retrieval-versus-balanced comparison
(significant on both panels) but **not** for every pairwise strategy comparison, so it stays
open. 32.4's explicit gap-against-base-rate regression also stays open — §35 establishes the
direction from two panels rather than fitting the relationship.
