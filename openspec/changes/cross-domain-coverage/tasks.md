# Tasks

- [x] 38.1 **Settle whether base rate or domain content drives the regime effect.** Closed by
      `docs/FINDINGS.md` §62 and §63: domain content is worth +0.0982 AP (Holm p 0.002),
      density is worth +0.0047 (p 0.895). Verify: both findings record the paired-bootstrap
      intervals and the base-rate-curriculum framing this proposal originally carried is
      withdrawn in the proposal text.
- [ ] 38.2 **Find which generator property predicts transfer.** Using the five existing
      checkpoints, correlate their off-domain performance against the measurable differences
      between their priors — missingness, per-column separability, correlation structure.
      Verify: recorded as a finding that names which property tracks transfer and which does
      not. **Gates 38.4-38.6** — building variants before this is guessing.
- [ ] 38.3 **Report both axes for every checkpoint, always.** Verify: the arm evaluator emits
      symmetry-probe scores *and* protocol AP side by side, because §63 showed a prior can
      transfer well while teaching badly and a single column hides that.
- [ ] 38.4 **Strip missingness from the financial generator** and pretrain at §58's
      configuration. Verify: antisymmetric probe and protocol AP both recorded against
      `colid-finbal`; missingness is the cheapest of §62's three remaining candidates.
- [ ] 38.5 **Test the separability candidate.** Sharpen the financial prior's per-column signal
      toward the SCM prior's 0.911 without changing its identities. Verify: as 38.4.
- [ ] 38.6 **Test the correlation candidate.** Verify: as 38.4, with the caveat recorded that
      weakening the accounting identities makes it progressively less a financial prior — if
      this is the property that matters, the finding must say what that costs.
- [ ] 38.7 **Evaluate every candidate on two domains, not one.** Verify: each pretraining run
      above is scored on V4FinBench *and* a non-credit panel, with paired bootstrap intervals.
      A single-domain score cannot see the failure this proposal exists to fix.
- [x] 38.8 **Re-run the V4FinBench protocol properly before quoting anything**: five folds,
      `--tune`, AP first per D13. Done in §69: fintfm ties logistic regression (Holm p=0.309)
      and loses to all three tuned boosters (Holm p<0.001 each, CatBoost's gap roughly
      doubling from untuned). Confirms §60's caveat that the untuned gap understated the
      boosters' lead. Raises a new task below rather than closing the question: none of
      fintfm's own context-quality levers were engaged for this comparison.
- [ ] 38.12 **Re-score V4FinBench with fintfm's own levers turned on** before concluding the
      §69 gap is architectural. Retrieval context strategy (§32: +0.066 to +0.095 AUC measured
      on this exact protocol), `n_ensemble > 1` with distinct `column_id_seed` per member
      (D12's stated mitigation, never applied to a real-data comparison), and a `max_context`
      sweep given the training pool is 63,588 rows against a 2,000-row context. Verify: same
      five folds, same paired-bootstrap methodology as §69, so the *before* and *after* are
      directly comparable. If this closes most of the gap to the boosters, §69's reading
      changes from "architectural deficit" to "unfair comparison, now fixed."
- [ ] 38.9 **Record any regression explicitly.** Verify: if a transfer-improving variant loses
      on credit, the loss is written into `docs/FINDINGS.md` with the trade stated. Second-best
      everywhere may still be the right product, but that must be argued rather than silent.
- [x] 38.10 **Run the crossed design §66 specifies.** Done in §67: SCM features under a
      financial-style label collapse from 0.9932 to 0.5647 on the antisymmetric probe,
      landing with the financial arms rather than the SCM baseline. **Tracks the features,
      not the label function.** The financial-features/SCM-label cell was left at a cheap
      gradient-boosting sanity check (raw AUC ~0.54-0.56, near-unlearnable at this scale, so
      a full pretraining run there would not have been readable) rather than pretrained.
- [ ] 38.11 **Isolate the feature-side cause.** §67 implicates the financial generator's
      features without naming the mechanism; §65 already eliminated near-duplicate columns,
      overall correlation and label-dependence concentration as continuous statistics on the
      *whole* task. The remaining candidate is the accounting-identity structure itself —
      ratios as deterministic functions of a small set of latent account balances. §68: a
      cheap screen adding post-hoc noise to the assembled feature matrix was tried and found
      inconclusive (hump-shaped, an artefact of raw/sorted AUC both collapsing toward chance
      as noise grows, not a clean identity-structure signal). **The valid design perturbs the
      shared latent accounts before deriving ratios** — independent copies of an account for
      each ratio that currently shares the literal array, preserving each account's own
      marginal and its contribution to the label, breaking only the cross-ratio identity —
      which requires modifying `_accounts`/`_ratio_family` directly rather than the assembled
      `Task.X`. Verify: measure the antisymmetric probe on a pretrained checkpoint, not a
      raw/sorted GB proxy alone, since §65 already showed such proxies can mislead. State the
      concern explicitly if this test requires weakening the identities the ratios are drawn
      from, since doing so trades against the generator being recognisably financial.
