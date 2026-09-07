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
