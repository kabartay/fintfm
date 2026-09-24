# Tasks

- [x] 38.1 **Settle whether base rate or domain content drives the regime effect.** Closed by
      `docs/results/FINDINGS.md` §62 and §63: domain content is worth +0.0982 AP (Holm p 0.002),
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
- [x] 38.12 **Re-score V4FinBench with fintfm's own levers turned on.** Done in §70-§71.
      Retrieval actively hurt (§70: -0.133 AP alone on one fold, not rescued by the other two
      levers). Dropping retrieval and keeping `n_ensemble=8` + `max_context=4000` gave a real,
      five-fold, paired-bootstrap-significant gain of +0.0311 AP (§71: Holm p<0.001) — the
      first configuration-only change all session to clear significance. It changes the
      standing vs logistic regression from a tie to a numerical (not yet significant) lead,
      and leaves -0.14 to -0.17 AP against all three tuned boosters, still significant. So the
      §69 gap was *partly* configurational (now fixed, +0.031 AP recovered) and remains mostly
      real: the boosters are not caught by inference-side fixes alone.
- [ ] 38.13 **Chase why retrieval hurts here.** §70 found retrieval alone drops AP by 0.133,
      opposite the +0.066 to +0.095 AUC gain §32 measured on this same protocol. Not yet
      isolated: candidate mechanism is retrieval's per-group logit correction or
      `retrieval_min_positive=8` behaving differently at 79 positives across 63,588 rows than
      whatever denser regime §32 measured. Verify: if the mechanism is found and fixable,
      re-measure whether retrieval (fixed) beats `n_ensemble` + `max_context` alone; if not,
      record retrieval as regime-dependent and move on rather than disabling it silently.
- [ ] 38.9 **Record any regression explicitly.** Verify: if a transfer-improving variant loses
      on credit, the loss is written into `docs/results/FINDINGS.md` with the trade stated. Second-best
      everywhere may still be the right product, but that must be argued rather than silent.
- [x] 38.10 **Run the crossed design §66 specifies.** Done in §67: SCM features under a
      financial-style label collapse from 0.9932 to 0.5647 on the antisymmetric probe,
      landing with the financial arms rather than the SCM baseline. **Tracks the features,
      not the label function.** The financial-features/SCM-label cell was left at a cheap
      gradient-boosting sanity check (raw AUC ~0.54-0.56, near-unlearnable at this scale, so
      a full pretraining run there would not have been readable) rather than pretrained.
- [x] 38.11 **Isolate the feature-side cause.** Done: `identity_shuffle` (permute each
      account independently, breaking cross-account identities exactly while preserving every
      account's marginal and leaving the label untouched — verified before spending the GPU
      run) is implemented and pretrained. §72: **not a clean result.** The antisymmetric probe
      moved further than any other manipulation tried (0.7488, against 0.535-0.586 for every
      other financial-only arm) — real support for the hypothesis — but `symmetric_sum` and
      `symmetric_count` collapsed to 0.30 and 0.26, *below* the 0.5 chance floor and stable
      across 16 seeds, where every other checkpoint in this project scores >=0.92. The
      checkpoint is also the worst of four financial-only arms on its own in-distribution
      per-task AUC (0.5772). Some support for the hypothesis, a new unexplained failure mode,
      and not yet a candidate fix.
- [ ] 38.14 **Explain the symmetric-probe inversion.** §72's candidate mechanism, untested:
      identity-shuffle may produce far more extreme derived ratios than the true accounts
      (independently-permuted numerator/denominator pairs), and the model may have learned
      "extreme values across many columns" as a spurious cue specific to this prior that
      misfires in the opposite direction on a probe whose positive class is a large sum/count.
      Verify: measure whether identity-shuffled tasks have heavier-tailed exposed columns than
      the unshuffled prior, and whether the inversion's sign tracks feature extremity in a
      constructed probe. **Gates any further identity-shuffle work** — building a production
      variant on a mechanism this poorly understood would repeat the §47/§54 pattern of
      shipping a fix whose side effects were not characterised.
- [ ] 38.15 **Explain §74's capacity cap.** Training predominantly on the financial prior caps
      basic signal extraction at ~0.73 AUC regardless of true task difficulty, on a probe with
      no column-identity structure at all; the identical architecture under the generic SCM
      prior tracks the true Bayes curve almost exactly. Candidates, none yet tested: effective
      SNR lower in practice than the sharpness parameter implies once accounting-identity
      correlations and missingness are accounted for; the cross-entropy objective interacting
      badly with the prior's base-rate distribution to teach systematic under-confidence; the
      MNAR missingness mechanism training the model to hedge. Verify: isolate one candidate at
      a time (matching the discipline of §64-§67), rerun the exact §74 Bayes-ceiling probe on
      the resulting checkpoint, and report against the fin00/fin07/fin10 baseline curve.
      **Gates 38.4-38.7 and any recommendation to keep the financial prior in a production
      mixture** -- shipping a prior with an uncharacterised severe defect because it wins on
      one benchmark repeats the §47 pattern this project was founded on catching.
- [ ] 38.16 **Re-run §73/§75's cross-dataset comparison at five folds with paired bootstrap**,
      the same upgrade §69 gave V4FinBench. Verify: `run_credit` gains a `--tune`/multi-split
      mode matching `v4_protocol.py`'s standard, or the comparison is ported into that harness
      directly. Both §73 and §75 are single-split and explicitly caveated as directional; the
      SCM-beats-financial reversal is large enough (Taiwan: +0.065 AP) that it likely survives,
      but "likely" is not the standard this project has held every other real-data claim to.
