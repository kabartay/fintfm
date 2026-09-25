# Working conventions

Instructions for anyone, human or agent, working in this repository. Short by design: the
project is one day old. Grows as mistakes actually happen here, following the pattern in the
sibling `finkele`/`finkele-axiom`/`jile` repos — a rule earns its place by being violated once,
not by being imaginable in advance.

## Identity

| purpose | identity |
| --- | --- |
| Commit author, human contact | `mukharbek.organokov@gmail.com` |

**Never use `temelion.ai`, anything named Temelion, or `finkele.com`.** The shell environment on
this machine carries a `temelion.ai` address by default for anything that reads "the user's
email" — it has reached a monitoring channel on an unrelated project before. Check before
creating anything (a commit, a cloud resource, a registration) that carries an address.

Commits carry no `Co-Authored-By` or Claude attribution trailers.

## Which Python

`uv`-managed (`pyproject.toml` + `uv sync`). Run everything through `uv run`:

```bash
uv sync --extra bench
uv run pytest
uv run fintfm-train --steps 5000 --out runs/name.pt
uv run fintfm-bench --model runs/name.pt
```

This machine has other projects' venvs (e.g. `finkele-alert/.venv`) that leak in via
`$VIRTUAL_ENV` and print a harmless `uv` warning; ignore the warning, don't `source` another
project's activate script here, and don't invoke a bare `python`/`pytest` and assume it resolves
to this project.

## The machine is shared

`bwa` runs 24/7 on roughly three cores as part of an unrelated pipeline. Never launch a real
pretraining run (anything past a smoke test — thousands of steps, GPU-scale batch/model size)
without checking `uptime` first and getting an explicit go-ahead; that combination has frozen
this Mac before. Ports `8080` (mlflow), `8001` (an unrelated API) and `3000` (Docker) are taken.

## Claims: label every number by how it was produced

This is the rule that matters most in an ML repo, because a wrong number here doesn't crash —
it just looks like a result. Today's session shipped a benchmark harness with three bugs that
would each have silently produced a plausible-looking but meaningless headline: a feature-width
mismatch (`0/N` scorable, but no error), an unguarded `OSError` that quietly dropped LightGBM
from the comparison, and a degenerate single-class test split whose `NaN` poisoned every
aggregate mean via `np.mean`. All three were caught only by actually running the pipeline
end-to-end rather than reading the code and reasoning that it looked right.

So: **run it, don't reason about it**, and say what kind of number you're reporting:

- **SMOKE-TEST** — proves the pipeline executes (training loop runs, checkpoint saves/loads,
  benchmark harness produces numbers). Says nothing about model quality. The 172K-parameter,
  300-step checkpoint from the first session is this and only this; do not quote its AUC as if
  it were evidence about the architecture or the prior.
- **MEASURED** — produced by a real pretraining run (meaningful step count and model size) and
  a benchmark run against held-out data, with the command and checkpoint path recorded.
- **SIMULATED / ESTIMATED** — a back-of-envelope number, not from running the actual code. Never
  act on one of these as if it were MEASURED.

When a benchmark run reports `N/M scorable tasks`, `M > N` means tasks were skipped (currently:
degenerate single-class splits) — that's informative, not noise to suppress.

## Two ways a long run lies about itself

Both cost a three-hour Phase 1 run on 2026-09-08, and neither surfaced as a failure.

**A pipe hides the exit status.** `uv run ... | tee run.log` reports *tee's* exit code, so a
Python traceback arrives as **exit code 0** and the harness says "completed". Never pipe a
run whose success you intend to believe. Redirect instead, and check the status:

```bash
PYTHONUNBUFFERED=1 uv run fintfm-ablate ... > run.log 2>&1; echo "EXIT=$?"
```

`PYTHONUNBUFFERED=1` matters for the same reason in reverse: without it, Python's stdout
buffering plus a pipe hides every progress line until the process exits, so a run that died
at step 500 looks identical to one that is working. The first Phase 1 attempt was
unmonitorable for four hours for exactly this reason, and it had already been dead for
almost all of it.

**A code path that has only ever run on CPU is untested, not working.** `_eval_accuracy`
sampled batches without moving them to the model's device. The training loop moved its own,
so everything passed on CPU and everything passed in CI, and the crash landed at the first
`eval_every` checkpoint — five minutes into the run, on the one path no test exercised off
CPU. Two rules follow:

- **Derive the device from the model** (`next(model.parameters()).device`) rather than passing
  it alongside; two sources of the same fact will desynchronise.
- **Parametrise device-sensitive tests over every device actually available** and skip rather
  than omit. `tests/test_model.py::test_training_completes_on_each_available_device` is that
  guard; it runs a four-step training with `eval_every` below `steps` so the evaluation path
  is genuinely entered.

Verifying a run is alive by watching accumulated CPU time grow is still right, but it only
proves the process was alive *at that moment* — it cannot distinguish training from a process
about to hit an untested branch. Read the log.

## A correction that lives in one method will be bypassed by the next caller

Cost a 6,000-step retrain and a wrong root-cause diagnosis on 2026-09-09
(`docs/results/FINDINGS.md` §28, decision D8).

The base-rate correction of decision D5 was implemented in `predict_proba` and only there. The
term-structure path was added later, built its own forward pass through
`FinancialTFM.term_structure`, and walked straight past it — so the out-of-time evaluation
reported a **12.8% default rate against a 0.47% truth** and stated it as a finding. The cause
was diagnosed as the synthetic prior being unable to reach low default rates, and a retrain was
spent widening the prior before anyone checked the context's base rate against the
population's.

- **Never call a model's forward method directly when a wrapper exists.** If you find yourself
  reaching into `_ctx_X`, `_ctx_y` or any other private on a fitted estimator, you are building
  a second inference path that will diverge from the corrected one. Add a public method
  instead.
- **A level error is a context problem before it is a prior problem.** Print the context's base
  rate beside the population's. One line, and it is now printed on every out-of-time run.
- **Print the level, not only the score.** Three guards each failed to see a 27× error: AUC
  cannot see it because every prediction inflates alike, the coherence check cannot see it
  because a wrong curve can be perfectly monotone, and `mean_predicted` was computed, stored in
  the JSON, and never rendered. Any harness reporting a probability reports its **mean beside
  the observed rate**.
- **Keep the broken configuration as an arm.** The uncorrected context is now a permanent arm
  of the out-of-time harness, so the distortion is measured next to the fix instead of being
  assumed absent.

## Sync with the extras, and distrust a jump in the skip count

`uv sync` without `--extra bench` silently *uninstalls* the benchmark dependencies, and the
suite then reports skips rather than failures — which reads as success. On 2026-09-09 a bare
`uv sync -q` in the middle of a session took the skip count from 1 to 4 and broke the
V4FinBench loader that every out-of-time finding depends on.

```bash
uv sync --extra bench --extra hf   # not a bare `uv sync`, and not one extra either
uv run pytest -q                   # 1 skip is the expected count; more means look
```

**Naming one extra drops the others.** On 2026-09-22 a `uv sync --extra bench` mid-session
removed the `hf` extra and with it the `hf` CLI, so every `hf jobs` call in a session with five
GPU runs in flight failed with `Failed to spawn: hf`. Nothing said a dependency had been
uninstalled; the command simply stopped existing. Sync every extra the session needs, together.

**And `uv pip install` obeys the leaked `$VIRTUAL_ENV`.** The obvious recovery — `uv pip
install huggingface_hub` — reported `Using Python 3.13.7 environment at
/Users/morganoko/proj/finkele-alert/.venv` and installed into **another project's venv**, which
is the leak this file already warns about one section above, now with a way to act on it:

```bash
env -u VIRTUAL_ENV uv pip install <pkg>    # or just re-run the full `uv sync` line above
```

That incident also surfaced a real defect: `pyarrow` was declared only in the `kaggle` extra,
so reading already-fetched parquet worked solely by accident of a previous sync. **CI stayed
green throughout**, because CI has no data and the test skips — a dependency that only the
data-carrying machine needs is invisible to CI by construction. Declare a read dependency in
every extra whose code path reads.

## Releases

A release is a git tag **and** a GitHub Release. A bare tag ships nothing and, once a release
workflow exists, fires nothing.

**Title carries the version:** `vX.Y.Z — Title`, with an em dash.

**Create the release as a draft, then publish it.** Not a style preference — it is the only
way the Zenodo deposit fires reliably. GitHub sends `created`, `published` and `released` for
one release within about 200 ms; Zenodo accepts whichever arrives first, rejects the rest with
`409`, and archives only on `published`. v0.5.3 was lost because `created` won by 0.17 s. A
draft sends no webhook at all, so publishing it fires only `published` and `released`, and
`released` has been rejected in every delivery observed.

```bash
gh release create vX.Y.Z --draft --target main --title "..." --notes-file notes.md
gh release edit vX.Y.Z --draft=false     # this is what fires Zenodo
```

**A DOI is not minted until the records API says so.** Zenodo's interface lists a failed
release as *received*, which is also the state a successful deposit passes through, and
GitHub's delivery log shows `202` either way. Neither one can tell you it worked:

```bash
curl -s "https://zenodo.org/api/records?q=fintfm&all_versions=true" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['hits']['total'])"
```

**Two documents, opposite conventions, on purpose:**

| document | shape | why |
| --- | --- | --- |
| `docs/CHANGELOG.md` | hard-wrapped at ~90 columns | read in an editor and a diff |
| the GitHub Release body | **unwrapped**, one paragraph per line | read in a browser at full width |

**Never paste the changelog entry straight into the release body.** That is the specific way
the rule gets broken while appearing to be followed — the wrapping comes along and the
release renders as a narrow column of unstyled prose beside its neighbours. Write the body
separately, or unwrap the changelog text. Check rather than eyeball it:

```bash
gh release view vX.Y.Z --json body -q .body | awk '{ if (length($0) > m) m = length($0) } END { print m }'
```

A correct body measures 300+. One at 80-95 is prose broken at a column.

**Commit messages stay hard-wrapped** at ~78 columns, because they are read in a terminal by
`git log`. The same text does not serve both, so never reuse a tag message as a release body.

**Confirm the run is green afterwards.** Creating the release is not the end of the job.

**There IS CI, and on 2026-09-23 a release went out while it was failing.** This paragraph used
to say there was none, that sentence went stale when CI landed, and the v0.4.0 release body
repeated it — claiming "no CI in this repository yet, so this release is verified by a local run
and nothing else" while a `ruff` error sat red on the release commit. Local `pytest` was green;
lint was not, and `uv run pytest` does not run `ruff`.

```bash
uv run ruff check src/ tests/     # CI runs this and pytest does not
uv run pytest -q
gh run list --limit 3             # after pushing a tag, before announcing anything
```

A stale sentence in this file is worse than no sentence, because it gets quoted into a release
body as if it were checked.

## Licensing: what this repo carries

**Apache-2.0**, chosen 2026-09-08. `LICENSE` holds the canonical text fetched from
apache.org; `pyproject.toml` carries the SPDX expression and ships the file in the wheel.
Every dependency is permissive (numpy/pandas/scikit-learn/scipy BSD-3, torch Apache-2.0,
PyYAML MIT, pyarrow/xgboost/catboost Apache-2.0, lightgbm
MIT), so nothing constrained the choice — verified from installed package metadata.

The reasoning, so it is not relitigated: **the moat is the trained weights and the mature
prior, not the training code.** This architecture is reproducible from the public
TabPFN/TabICL literature by any competent engineer in days, so protecting it buys little,
while Apache's permissiveness and its patent grant buy adoption and clear the procurement
review at exactly the banks and insurers this targets. AGPL-plus-dual-licence was considered
and rejected: customers consume predictions through an API or licensed weights and never
deploy the training code, so there is no copyleft obligation for them to pay to escape, and
AGPL is blanket-banned at many of the target buyers.

**The repository went public on 2026-09-25, so the Apache-2.0 grant has now reached everyone
who takes a copy and the choice is one-way from here.** It cannot be narrowed retroactively for
anyone already holding the code. Relicensing future work is still possible; taking back what
has shipped is not.

**Two consequences that were theoretical while it was private and are not now.**

A bad release can no longer be replaced. v0.4.0 was re-tagged on 2026-09-24 because its
original tag pointed at a commit with a failing lint check, and that was only safe because
nobody held the tag. From now on a flawed release is **superseded by the next version**, never
rewritten — someone may already have fetched it.

The prior generator in `src/fintfm/prior/` is now published. `docs/competition/LANDSCAPE.md`
recorded the tension this resolves: a provenance claim nobody can check is a slogan, and the
three peer projects that publish their generators (TabICL, LimiX, Nori) are the ones whose
synthetic-only claims can be verified. **Trained weights are still not published and remain
gitignored** — publishing one is a separate decision with a separate licence, which is the
distinction this file already requires for everyone else's checkpoints and now applies to
ours.

**Trained weights never enter git**, and neither does the mature prior if it diverges from
the reference version here — those are the private asset. `.gitignore` excludes `*.pt`; keep
it that way. If outside contributions are ever accepted and relicensing might matter, a CLA
has to be in place *before* the first pull request is merged, not after.

## Licensing boundary: what must never come in

No code, model weights, or **training** data from Neuralk (Seldon), Fundamental (NEXUS),
Google TabFM, TabPFN/TabICL, or any other tabular-foundation-model product may enter this
repository.

**Training and evaluation are different questions, and the rule used to conflate them.**
Training on such data would destroy decision D2's auditability claim — that the model cannot
have memorised a benchmark because it never saw real data — and nothing would warn us.
*Evaluating* on it does not: the model never learns from it and the provenance argument is
untouched.

So evaluation is permitted in principle, and gated on something else entirely: **whose data is
it, and what did they agree to?** Data that reached us through a client engagement belongs to
that client, and reusing it to benchmark a different commercial project is a permission
question, not a contamination one. Before evaluating on any third-party data: confirm the
engagement terms allow it, keep the data outside this repository, keep derived numbers
internal, and do not publish a comparison without asking. Ask rather than assume — this is not
a judgement call to make on the owner's behalf.

**Check the weights licence separately from the code licence, every time.** Google's TabFM
and TimesFM 3.0 both ship Apache-2.0 code with **non-commercial** weights, and TimesFM's
weights were Apache-2.0 through 2.5 and are not at 3.0 — so a licence checked once is not
checked. Evaluating against published *numbers* is always fine; running someone's checkpoint
inside anything commercial is not. See `docs/results/FINDINGS.md` §24. Their public papers/blog posts are legitimate research context to read and cite in
discussion, never a source to copy from. Before adding any third-party dataset or dependency,
check its license against commercial use — see `README.md`'s licensing section for the current
policy and add a line there when a new source is added.

## Secrets: `.env`, and never anywhere else

`.env` holds credentials and is gitignored; `.env.example` documents every key it may contain
and is committed. Same convention as `finkele-axiom`.

**Nothing loads `.env` automatically, on purpose.** A file sourced implicitly is a file nobody
re-reads. Source it for the call that needs it:

```bash
set -a; . ./.env; set +a
```

Two rules that cost someone a debugging session already:

- **The negation must follow the pattern it exempts.** `.env.*` swallows `!.env.example` if
  the negation comes first, silently. That happened in `finkele-axiom`.
- **`git check-ignore -v` does not answer this question.** It exits 0 when *any* pattern
  matches, negations included, so a correctly-exempted file looks ignored. Use
  `git status --porcelain --ignored`: `!!` means ignored, `??` means visible.

Never paste a token into chat, a commit, a config under `configs/`, or a run log. Note that
credential *scope* cannot enforce this repository's licensing boundary — Hugging Face grants
every token read access to all public repo contents, so no scope prevents pulling
TabPFN-family weights. That boundary stays a policy rule enforced by review.

## The first diagnostic for any in-context model: shuffle the labels

Before any accuracy discussion, ask whether the context matters at all. Randomise the context
labels, destroying any feature-label relationship, and predict:

- predictions barely move → **the model ignores its context**, and every accuracy number is
  measuring a fixed function of the features
- predictions collapse toward chance → the model reads its context, and accuracy means what
  you think it means

This took three days to run and it cost most of them (`docs/results/FINDINGS.md` §47). The
financial-only checkpoint scored **higher with random labels than with true ones** — 0.6895
against 0.6841, rank correlation 0.977 — so its apparent 0.68 was unsupervised feature
structure, not learning. Four hypotheses were tested and discarded before anyone asked the one
question that separates in-context learning from a lookup table.

**And check the prior admits no global rule.** §47's cause was a single line: driver
orientations were hardcoded and weights were `np.abs(...)`, so in every task higher leverage
meant riskier. A universal feature-to-label mapping can be memorised once, so the model never
had a reason to read a label. If one fixed rule solves every task your prior generates, the
prior cannot teach in-context inference — whatever else is right about it.

## Measuring a model that might not work

Five rules, each of which cost real time before it was written down
(`docs/results/FINDINGS.md` §42, §43).

**A prior must span difficulty.** Ours was clamped so synthetic difficulty *matched* real
difficulty, which sounds correct and left only 8% of tasks with a learnable boundary. The model
learned to do as well as anything can on noise and never learned to extract a clean boundary,
scoring 0.68 on a linear task logistic regression solves at 0.9997. A prior's job is to teach,
not to resemble the test set. Now 42%, and `tests/test_prior.py` pins the span in both
directions — a prior of only *easy* tasks would never teach the model to abstain.

**Every evaluation needs a floor.** Without an untrained model of the same architecture, a
number like "0.66 on prior tasks" cannot be read: competence and "the architecture plus the
context did it" look identical. `fintfm-capability` builds the control automatically for this
reason. **A trained model scoring *below* its untrained control is a specific signature** — it
means training actively steered the model wrong, which points at the training signal rather
than at the architecture.

**A benchmark that one column solves cannot diagnose anything.** V4FinBench horizon 0 gives
0.9811, and `Working_capital/total_assets` alone gives 0.9799. Keep probes with *known*
ceilings alongside any real benchmark, or a broken model will look competent for days.

**The training metric must be a proper scoring rule against a baseline.** The loop reported
accuracy, which at a 4.7% base rate a constant predictor scores 0.953 on; three runs logged
"0.935-0.945" as progress. This file already said accuracy is not a proper scoring rule and
the training loop did not follow it.

**ROC-AUC hides failure at the top of the ranking**, which is usually the part that matters —
a top-K credit decision, a top-K dispatch. Report average precision and precision@K beside it.
Twice in one day a model with ~0.88 AUC was near-useless where it counted.

## Never truncate a validator's output

`uv run python openspec/tools/validate.py | tail -2` prints the task count and hides every
`FAIL` line above it. That is how a failing `test_openspec` reached a push on 2026-09-10: the
validator had been reporting three malformed tasks for an hour and the pipe cut them off.

Read the whole output, or grep for what you want to see rather than for where it ends. Same
family as the `| tee` trap above — a pipe that quietly discards the part that says "no".

## Count the tasks, not the steps

Training volume is `steps × batch_size`, and that number belongs in every discussion of whether
the model works. Every checkpoint here has come from **48,000 tasks** against a field norm
around 10⁷ (§43) — visible in every command for two days, and never once multiplied out.
Before concluding anything about architecture or scale, check that the model has been trained
at a defensible volume. On a T4 it is roughly $13 for a million tasks.

## Configuration, not constants — and not everything

Experiment numbers live in `src/fintfm/configs/default.yaml`, layered as packaged default →
`--config`/`$FINTFM_CONFIG` → explicit CLI flag, deep-merged. New entry points take
`--config` and default their other flags to `None` so the configuration supplies the value
and an explicit flag still wins.

**Not every literal is configuration.** The rule: a value belongs in the config if a
reasonable experiment would want it different; a value whose change would make results
incomparable or the code incorrect stays in code. So the `CENSORED = -1` sentinel, the
float32 numerical guards, and the prior's accounting identities are not configurable, and
`configs/README.md` records why. The prior's rate constants are the middle case — the values
are overridable by argument, but they stay declared in `prior/financial.py` beside the
measured reasoning for them.

**Two rules that earned their place:**

- **A misspelled key must be an error.** `load_config` refuses unknown keys and names the
  path. A tolerant loader completes the run, reports numbers, and used the default — which is
  the same failure shape as `docs/results/FINDINGS.md` §28.
- **Mirrored defaults need a drift guard.** The `inference` section duplicates
  `FinancialTFMClassifier`'s literal defaults on purpose, because a library must behave the
  same without reading a file. That duplication is only safe because
  `tests/test_config.py::test_inference_section_matches_the_classifier_defaults` fails when
  they diverge. Any future mirror needs the same treatment.

## Code style

New code gets full treatment in the same pass it's written, not after being asked separately:
a module docstring saying what the file is for, a docstring on every public function/class
(Args/Returns, not restating the name), and explicit type hints throughout. This mirrors
`finkele-axiom`'s convention.

**This used to say "don't retrofit — nothing here predates the convention", and by 2026-09-25
that was false:** an AST sweep before going public found **47 functions and classes without a
docstring** and six signatures with incomplete hints. The rule was applied to each new file and
never checked across the tree, so exceptions accumulated silently.

The sweep is three lines and worth running before any release:

```python
import ast, pathlib
for p in pathlib.Path("src").rglob("*.py"):
    for n in ast.walk(ast.parse(p.read_text())):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not ast.get_docstring(n):
            print(f"{p}:{n.lineno} {n.name}")
```

Two conventions came out of doing it. A constructor whose class docstring already documents
every argument gets a one-line pointer rather than a second copy — two argument lists drift.
And a private helper still gets a docstring when its *reason* is not obvious from its body:
`_auc_safe` returning NaN rather than 0.0 for a single-class split is a decision, not an
implementation detail.

## Commits

`git add` by path, not `git add -A`/`git add .` — cheap now while the repo is small and single-
session, worth keeping as the habit before it stops being cheap.

## Documentation

Two destinations, not one: this file (`CLAUDE.md`) is standing working rules that would
otherwise get relearned the hard way; `README.md` is what the repository is and how to run it
right now. Keep `README.md`'s status section honest — it's the one document nothing tests, so
it goes stale silently if a claim in it stops being true and nobody rereads it.

## Score on the benchmark this project claims, before the one that is convenient

Cost four checkpoints and two days on 2026-09-22/23, and produced a result that reversed
(`docs/results/FINDINGS.md` §115, §116).

A tree-structured prior was selected on a measured criterion, trained at two seeds against
matched controls, and scored **+0.0094 on TabArena** — replicating in sign against a measured
noise floor, one seed reaching p = 0.019. It was written into the README as an improvement. It
was then scored on **V4FinBench**, the low-default credit panel this project's claims actually
rest on, and came back at **−0.0221 average precision: negative on 5 of 5 folds, three of them
surviving Holm correction at p < 0.001**, dropping the model below untuned logistic regression
where its own control cleared it.

Both results are real. The general-tabular one is not the one that decides anything here.

**TabArena is not a proxy for credit, and it is seductive because it is fast and returns a
rank.** A 27-dataset single-fold run takes about an hour and prints a leaderboard position;
V4FinBench is five folds over a million rows and prints no rank at all. That asymmetry is why
this happened, and it will recur unless the order is fixed rather than remembered:

- **A prior, architecture or training change is scored on V4FinBench before it is believed.**
  TabArena afterwards, for breadth. Not the reverse.
- **A general-tabular gain is not a project result** until it has a credit number beside it.
  Presenting one as a result is `docs/results/FINDINGS.md` §107's error — reporting a favourable
  measurement of the wrong thing — in a new place.
- **Read average precision first**, always, and never report a low-base-rate result on ROC-AUC
  alone. In §116 ROC-AUC moved 0.9856 → 0.9840, a 0.16% relative change any reader would call a
  null, while AP fell 13% relative. A ROC-AUC-only report would have shipped the regression.

The general form, worth keeping when the specific benchmarks change: **when two benchmarks
disagree, the one you optimised against is the one you should distrust**, and the cheap one is
almost always the one you optimised against.

## Never test against `main`, and never on a public repository

On 2026-09-25 a branch ruleset was enabled and then verified by pushing an **empty commit
named `probe`** to `main` to see whether direct pushes still worked. They did. The commit is
now permanent in public history, because the same ruleset blocks the force-push that would
remove it — the protection worked exactly as designed, against its author, within a minute of
being created.

Test a ruleset on a throwaway branch (`git push origin HEAD:refs/heads/probe-delete-me`), read
the rule's own documentation, or accept not knowing. A repository that other people can clone
is not a place to find out what a setting does.

## Install the pre-commit hook, once per clone

```bash
uv run pre-commit install
```

`ruff` errors reached `main` **twice on 2026-09-24** — once on the v0.4.0 release commit, which
shipped with CI red, and once in the commit that fixed it. Both times `uv run pytest` was green
locally and `ruff` was simply not run. The release checklist in this file said to run it; a
checklist is a thing a human has to remember, and the hook is not.

Only the fast checks are hooked. `pytest`, the licence and link checkers and the openspec
validator stay in CI, because a hook that takes a minute gets bypassed with `--no-verify`,
which is worse than not having one.

## Read the field periodically, and write what you find into `docs/paper/`

Standing instruction, 2026-09-21. **The competitive landscape is a research input, not
background.** One afternoon spent opening seven TabArena entrants produced three things no
experiment here would have produced:

- **A published scaling curve that caps a hypothesis.** Nori reports 6M → 100M parameters
  buying +0.0049 R². Our residual is 0.035 (§101), and §93 had already measured a 5× *data*
  increase at −0.0012 AP. Scale was the leading remaining explanation and a *perfectly
  executed* parameter-scaling programme would not have closed the gap — knowable for free, and
  not knowable from our own runs at any price. **Read such a curve for which axis it varies:**
  TabDPT reports power laws in both model and data, but its own contribution is that *real*
  data beats synthetic, so its laws are not evidence about synthetic-prior scaling.
- **A differentiator quietly lost.** Nori is synthetic-only, in-context, Apache-2.0 in both
  code and weights. "Synthetic-only prior" was half of Claim 2 and is no longer ours to claim.
  Finding that in a reviewer's report instead would have been much more expensive.
- **A design axis we had never varied.** Nori-6M is 16 layers at width 128; our scale-up went
  wide, not deep, and lost (§108). The peer's aspect ratio is a cheap experiment nobody here
  had thought to run.

So, periodically — when a benchmark run lands, or a peer appears on a leaderboard:

- **Open the primary source.** An abstract or a relayed summary is a lead, not a citation
  (`docs/research/REFERENCES.md` quarantines these for a reason, and `docs/results/POSTMORTEM.md` records a
  WebFetch summariser fabricating a results table).
- **Check the weights licence separately from the code licence, every time** — the rule above
  applies here and nowhere is it more tempting to skip.
- **Write it into `docs/paper/RELATED_WORK.md`** (positioning and what is left for us) and
  `docs/research/REFERENCES.md` (the verified literature list). A finding about the field belongs in
  the paper workspace, not in a chat log — same argument as "findings live here, not in commit
  messages".
- **Update `docs/paper/CLAIMS.md` when a claim narrows.** It records supersessions as
  prominently as wins; a claim that quietly stopped being ours is the failure mode the ledger
  exists to prevent.
- **File the adoptable ideas as openspec tasks with named falsifications**, not as a wish list.
  `openspec/changes/learn-from-peers` is the pattern.

**Reading is not ingesting.** The licensing boundary above is independent of licence: no code,
weights or training data from any tabular-foundation-model product enters this repository, and
a permissive licence changes only whether we may *evaluate* it.

## Re-read a number before quoting it, and check what it was measured on

Cost a wrong headline caveat in `docs/results/FINDINGS.md` §80, propagated into the claims ledger and
the limitations page before it was caught a day later (§82).

§80 compared a new architecture's five-fold mean on the **full** 1,000,087-row V4FinBench
panel against "§71's 0.2116" and concluded the change might be a net loss. Two errors, neither
visible in the number:

- **0.2116 was §71's fold 0**, not its result. §71's five-fold mean is 0.1676, and §71's own
  text says the single-fold figure "was optimistic". The number quoted was the one the source
  was warning about.
- **§69-§71 ran on a ~10x subsample** — `n_train` 63,588 and 105,900 test rows total, against
  the full panel's 1,000,087. The two were never measured on the same data.

Both are one line away in the source finding, and both were printed in each run's own output.

- **Quote from the finding, not from memory of the finding.** Re-open it. A number that has
  been carried through two or three summaries has usually lost its qualifiers.
- **Check the row count of both runs before comparing their metrics.** Average precision
  depends on the row set; two AP values from different subsamples are not comparable even at
  identical configuration, and nothing in either number says so.
- **A single fold is not a result.** If a finding reports per-fold values and a mean, the mean
  is the result; quoting the best fold is a silent cherry-pick even when it is accidental.
- **`max_context` is a fraction of the pool it samples from.** A context conclusion drawn at
  63,588 training rows does not transfer to 800,000. State the pool whenever stating a context
  finding.
