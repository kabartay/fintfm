# Mine the published TFM literature for methods, deliberately rather than by accident

## Why

The single most valuable thing found in two days of measurement came from the *literature*,
not from our data: `docs/FINDINGS.md` §42 traced the model's failure to learn back to a prior
clamped to a narrow difficulty band, and the fix — sample difficulty across orders of
magnitude, from trivial to impossible — is how TabPFN-style priors are built. That was found
by accident, while chasing a different hypothesis.

Decision D3 chose not to compete on general tabular accuracy against far better-funded
competitors. **That governs what we claim, not what we read.** Staying ignorant of frontier
methods is a separate mistake and a cheaper one to avoid.

Two published data points make the case that the frontier is reachable rather than magical:

- On Neuralk's own TabBench v2, Seldon beats TabPFN v3 on 18 of 22 problems and TabICL v2 on
  15 of 22, while TabPFN v3 edges it by 0.2 AUC inside the noise floor. The honest read is a
  three-way tie at the top, not a runaway leader.
- Kostrzewa et al.'s fine-tuned TabPFN matches or beats gradient-boosted trees on V4FinBench's
  ROC-AUC at every horizon — and their *prototype undersampling*, which we independently
  arrived at as §29's mechanism and they published first (§36), is a context-construction
  trick, not a scale advantage.

## What

Five methods, each with a status here, tracked so none is quietly forgotten:

| method | source | status |
| --- | --- | --- |
| priors spanning difficulty | TabPFN | **applied** (§42); needs validating, task 36.1 |
| prototype undersampling | Kostrzewa et al. | implemented and measured (§38); binary path open under `revisit-context-strategy` 32.6 |
| ensembling over context draws and feature permutations | TabPFN standard practice | **untried** — task 36.2 |
| in-context scaling to larger tables | TabICL | **untried** — task 36.3. Directly relevant: we degrade past 2,000 context rows (§33) |
| native coverage as a reported metric | Seldon runs 100% of TabBench where two of eight models failed on 19% and 29% | **not measured** — task 36.4 |

## Non-goals

- **Not vendoring anyone's code or weights.** `CLAUDE.md`'s boundary stands: public papers and
  blog posts are legitimate to read and cite, never to copy from, and a weights licence is
  checked separately from a code licence every time.
- Not a general-tabular pivot. These are methods for the credit model, not a change of target.
- **Fine-tuning on customer data is deliberately excluded here** and belongs in a decision, not
  a task list. Kostrzewa et al. show it works, and decision D2 forbids it because the
  synthetic-only provenance argument *is* the product. Whether an auditable model and a
  fine-tuned variant can both exist is a positioning question for the founder, not an
  engineering one — see task 36.5, which only asks that the decision be recorded either way.

## Falsified by

Nothing collectively — each item stands or falls on its own measurement, and a method that
does not help is recorded as not helping rather than quietly dropped.

## Blocked by

Task 36.1 gates the rest: until the capability probes show the model clearing its untrained
control by a wide margin, every other measurement here is reading §42 rather than the method
under test.
