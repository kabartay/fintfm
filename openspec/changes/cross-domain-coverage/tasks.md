# Tasks

- [x] 38.1 **Settle whether base rate or domain content drives the regime effect.** Closed by
      `docs/results/FINDINGS.md` §62 and §63: domain content is worth +0.0982 AP (Holm p 0.002),
      density is worth +0.0047 (p 0.895). Verify: both findings record the paired-bootstrap
      intervals and the base-rate-curriculum framing this proposal originally carried is
      withdrawn in the proposal text.
- [ ] 38.2 **Progress (2026-11-15): analysis drafted in `docs/results/drafts/38.2_2026-11-15.md` from recorded results only (FINDINGS §58, §64-§67, §74-§77), no new runs. Across the five checkpoints with a recorded off-domain score (fin10, finbal, finsharp, crossed, fin00; Bayes-ceiling probe, §76), missingness and correlation structure are two-valued and collinear with generator family, so they cannot be told apart (Spearman rho -0.866 for both, n = 5, approx. p 0.058); per-column separability does not track (finsharp, raw AUC 0.8885, transfers 0.716 against fin00's 0.984 at Bayes 0.990); `crossed` (SCM features, financial label) transfers like the financial arms, which points at the label mechanism (§76, §77), not a listed property. Five points cannot establish a predictor. Not met: the Verify needs a FINDINGS entry, which this draft is not, and no recorded real-domain off-domain score exists for four of the five checkpoints (the proposal's asset-failure panel has no figure in FINDINGS). Left open.** **Find which generator property predicts transfer.** Using the five existing
      checkpoints, correlate their off-domain performance against the measurable differences
      between their priors — missingness, per-column separability, correlation structure.
      Verify: recorded as a finding that names which property tracks transfer and which does
      not. **Gates 38.4-38.6** — building variants before this is guessing.
- [x] 38.3 **Done (2026-11-11, `scripts/dr64_arm_verdict.py` `teaching_axis` and `cmd_verdict`, `tests/test_dr64_arm_verdict.py::test_verdict_emits_probe_and_ap_side_by_side`).** **Report both axes for every checkpoint, always.** Verify: the arm evaluator emits
      symmetry-probe scores *and* protocol AP side by side, because §63 showed a prior can
      transfer well while teaching badly and a single column hides that.
- [ ] 38.4 **Strip missingness from the financial generator** and pretrain at §58's
      configuration. Verify: antisymmetric probe and protocol AP both recorded against
      `colid-finbal`; missingness is the cheapest of §62's three remaining candidates.
- [ ] 38.5 **Progress (2026-11-15): the pretraining half exists. The sharpened financial prior (`configs/sharp-financial.yaml`, raw AUC 0.8885 against the SCM prior's 0.8306 as measured in §65, not the 0.911 named below) was pretrained and its antisymmetric probe recorded at 0.5861, against 0.5352 for the default envelope and 0.5669 for the balanced arm (§65); §76 adds its Bayes-ceiling row (0.667 at target 0.900, 0.733 at 0.999, no cap relief). The identities were left unchanged. Not met: no protocol AP on V4FinBench against `colid-finbal` (0.1900, §62) was located for the sharpened arm. Left open.** **Test the separability candidate.** Sharpen the financial prior's per-column signal
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
- [ ] 38.13 **Progress (2026-11-15): FINDINGS §253: the recurring-positives cause in §194 was a diagnostic artefact (withdrawn); a zero-IQR scale floor is a real defect; none of three cheap fixes beats the default on one fold; the ranking loss itself is not isolated, folds 1 to 4 not run.** **Progress (2026-11-15, second note, draft `docs/results/drafts/38.13_2026-11-15.md`): the §194 recurrence is a diagnostic artefact: `scripts/retrieval_plan_diag.py` planned from unconditioned queries against a conditioned pool (context rate 4.04% on the subsample, 0.64% in the scoring path; median 52 of 64 groups per positive against 5 of 63), now fixed in the script. Found instead: 10 of 130 features have IQR 0 and take the 1e-6 scale floor, so 5 columns hold 89.8% of the squared distance. On a 60,000-row subsample, fold 0, 44 test positives (one fold, not a result): default AP 0.1856, retrieval 0.0943, retrieval with standard-deviation fallback 0.0528, `retrieval_per_class=8` 0.0127, 0.5 default + 0.5 retrieval 0.1469, `n_ensemble` 8 + `max_context` 4000 without retrieval 0.1989. No tested fix beats the default; why retrieval loses ranking with a sound scale is not isolated, and folds 1 to 4 and the full panel are not run. Left open.** **Progress (2026-11-15): partly isolated in §194 (fold 0, `scripts/retrieval_plan_diag.py`, commit 981d13c), on the full panel and the published checkpoint rather than the 63,588-row subsample: `retrieval_min_positive=8` is not the mechanism (no group sits at the floor); contexts hold 9.74% defaults against the pool's 0.3546%, the same 540 of 2,128 positives recur across 136 groups, and within-group ROC-AUC is 0.8688 against 0.9886 for the default. Why the same positives sit near every centroid is stated there as not measured. Not met: no fix was attempted, and the Verify's re-measure against `n_ensemble` + `max_context` alone, or an explicit regime-dependent record, is not in a finding. Left open.** **Chase why retrieval hurts here.** §70 found retrieval alone drops AP by 0.133,
      opposite the +0.066 to +0.095 AUC gain §32 measured on this same protocol. Not yet
      isolated: candidate mechanism is retrieval's per-group logit correction or
      `retrieval_min_positive=8` behaving differently at 79 positives across 63,588 rows than
      whatever denser regime §32 measured. Verify: if the mechanism is found and fixable,
      re-measure whether retrieval (fixed) beats `n_ensemble` + `max_context` alone; if not,
      record retrieval as regime-dependent and move on rather than disabling it silently.
- [x] 38.9 **Done (2026-11-16, FINDINGS §260): every recorded transfer gain was checked against its credit result: tree prior (§115, §116, §166), pure SCM (§63, §75), general-trunk priors (§191), volume (§188), leaf probe (§215, §218, §230), 512 columns (§234). Each loss was already written with its trade; §260 collects them and states the position. Corrected §75's V4FinBench cell by a dated note (0.1853, not 0.1900, for `fin07`; the original text is kept). Not covered: 38.4 to 38.6 are unbuilt, so any loss they produce must be recorded under 38.7's two-domain rule; §254 has no V4FinBench row. Verify met for the variants that exist.** **Record any regression explicitly.** Verify: if a transfer-improving variant loses
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
- [ ] 38.14 **Progress (2026-11-15): the constructed-probe half is built, not run.** `src/fintfm/experiments/tail_inversion.py` runs the symmetric probes with features from Gaussian to Student-t (df 10, 5, 3, 2) at a matched bulk (equal interquartile range) and an identical noise-free label rule, scoring each checkpoint, a seeded untrained control, logistic regression and gradient boosting. It reports per-family excess kurtosis, AUC and AP, the Spearman correlation of AUC with tail weight, and whether any family falls below 0.5. `tests/test_tail_inversion.py` checks the kurtosis ordering, matched signal, determinism and a tiny end-to-end run (SMOKE-TEST only). Run command (checkpoints are those §72 and §74 used; paths verified present): `cd /Users/morganoko/proj/fintfm && env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 15 env -u VIRTUAL_ENV uv run --frozen python -m fintfm.experiments.tail_inversion --models idshuffle=runs/dl/v4-colid-idshuffle.pt,fin10=runs/dl/v4-colid-fin10.pt,fin00=runs/dl/v4-colid-fin00.pt --seeds 0,1,2,3,4,5,6,7 --out runs/tail_inversion` (check `uptime` first). Remains: run it; measure whether identity-shuffled prior tasks have heavier-tailed exposed columns than unshuffled ones (the first Verify clause, a property of the prior that this harness does not cover); if the inversion does not track tail weight, §72's mechanism is falsified and another is needed. Left open.** **Explain the symmetric-probe inversion.** §72's candidate mechanism, untested:
      identity-shuffle may produce far more extreme derived ratios than the true accounts
      (independently-permuted numerator/denominator pairs), and the model may have learned
      "extreme values across many columns" as a spurious cue specific to this prior that
      misfires in the opposite direction on a probe whose positive class is a large sum/count.
      Verify: measure whether identity-shuffled tasks have heavier-tailed exposed columns than
      the unshuffled prior, and whether the inversion's sign tracks feature extremity in a
      constructed probe. **Gates any further identity-shuffle work** — building a production
      variant on a mechanism this poorly understood would repeat the §47/§54 pattern of
      shipping a fix whose side effects were not characterised.
- [ ] 38.15 **Progress (2026-11-15): §76 bisected four content axes on the exact §74 Bayes-ceiling probe with existing checkpoints (base rate, raw SNR, features against label mechanism, accounting identities), and none closes the cap (0.728 to 0.755 at target 0.999, against fin00's 0.997). Not met: the label functional form (named in §76 as the untested candidate), the cross-entropy and MNAR-missingness candidates are untested, no new checkpoint was trained with one candidate varied, and §76's own status line says the task is not closed. Left open.** **Explain §74's capacity cap.** Training predominantly on the financial prior caps
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
- [ ] 38.16 **Progress (2026-11-15): the multi-split mode now exists, not yet run. `fintfm-bench --credit --compare A,B --folds 5` (`run_credit_compare` in `src/fintfm/evaluation/bench.py`) scores the Polish 1y/3y/5y and Taiwan panels at k disjoint stratified folds with both checkpoints fitted per fold (`n_ensemble=8`, as §73), reports per-fold and mean AP first then ROC-AUC, bootstraps A minus B paired on the pooled out-of-fold rows per panel (`paired_auc_difference`, AP via its `metric` argument), Holm-adjusts across panels (`holm_adjusted_p`), and writes a `Coverage` record of skipped panels; tests use stub classifiers on synthetic panels only. The pair is §75's: `colid-fin00` (pure SCM) and `colid-fin07` (§73's checkpoint). Command, from the fintfm repo root, after an `uptime` check: `uv run fintfm-bench --credit --folds 5 --compare /Users/morganoko/proj/fintfm/runs/dl/v4-colid-fin00.pt,/Users/morganoko/proj/fintfm/runs/dl/v4-colid-fin07.pt --out runs/credit_compare_fin00_fin07.json > run.log 2>&1; echo EXIT=$?`. Caveat: the pooled bootstrap mixes folds fitted on different training parts; the §73 tables also used the boosters, which this mode does not re-run. Not met: no five-fold numbers exist. Left open.** **Re-run §73/§75's cross-dataset comparison at five folds with paired bootstrap**,
      the same upgrade §69 gave V4FinBench. Verify: `run_credit` gains a `--tune`/multi-split
      mode matching `v4_protocol.py`'s standard, or the comparison is ported into that harness
      directly. Both §73 and §75 are single-split and explicitly caveated as directional; the
      SCM-beats-financial reversal is large enough (Taiwan: +0.065 AP) that it likely survives,
      but "likely" is not the standard this project has held every other real-data claim to.
