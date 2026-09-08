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

## Releases

A release is a git tag **and** a GitHub Release. A bare tag ships nothing and, once a release
workflow exists, fires nothing.

**Title carries the version:** `vX.Y.Z — Title`, with an em dash.

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

**Confirm the run is green afterwards.** Creating the release is not the end of the job. There
is no CI in this repository yet (`openspec/changes/ci-and-release-gates`), so until there is,
say so explicitly rather than implying a release was verified.

## Licensing: what this repo carries

**Apache-2.0**, chosen 2026-09-08. `LICENSE` holds the canonical text fetched from
apache.org; `pyproject.toml` carries the SPDX expression and ships the file in the wheel.
Every dependency is permissive (numpy/pandas/scikit-learn BSD-3, torch Apache-2.0, lightgbm
MIT), so nothing constrained the choice — verified from installed package metadata.

The reasoning, so it is not relitigated: **the moat is the trained weights and the mature
prior, not the training code.** This architecture is reproducible from the public
TabPFN/TabICL literature by any competent engineer in days, so protecting it buys little,
while Apache's permissiveness and its patent grant buy adoption and clear the procurement
review at exactly the banks and insurers this targets. AGPL-plus-dual-licence was considered
and rejected: customers consume predictions through an API or licensed weights and never
deploy the training code, so there is no copyleft obligation for them to pay to escape, and
AGPL is blanket-banned at many of the target buyers.

**The repository is still private, so the grant has reached nobody yet and the choice remains
changeable until it goes public.** After that it is one-way for anyone holding a copy.

**Trained weights never enter git**, and neither does the mature prior if it diverges from
the reference version here — those are the private asset. `.gitignore` excludes `*.pt`; keep
it that way. If outside contributions are ever accepted and relicensing might matter, a CLA
has to be in place *before* the first pull request is merged, not after.

## Licensing boundary: what must never come in

No code, model weights, or training/eval data from Neuralk (Seldon), Fundamental (NEXUS),
Google TabFM, TabPFN/TabICL, or any other tabular-foundation-model product may enter this
repository. Their public papers/blog posts are legitimate research context to read and cite in
discussion, never a source to copy from. Before adding any third-party dataset or dependency,
check its license against commercial use — see `README.md`'s licensing section for the current
policy and add a line there when a new source is added.

## Code style

New code gets full treatment in the same pass it's written, not after being asked separately:
a module docstring saying what the file is for, a docstring on every public function/class
(Args/Returns, not restating the name), and explicit type hints throughout. This mirrors
`finkele-axiom`'s convention. Don't retrofit this onto anything — there's nothing here yet that
predates the convention.

## Commits

`git add` by path, not `git add -A`/`git add .` — cheap now while the repo is small and single-
session, worth keeping as the habit before it stops being cheap.

## Documentation

Two destinations, not one: this file (`CLAUDE.md`) is standing working rules that would
otherwise get relearned the hard way; `README.md` is what the repository is and how to run it
right now. Keep `README.md`'s status section honest — it's the one document nothing tests, so
it goes stale silently if a claim in it stops being true and nobody rereads it.
