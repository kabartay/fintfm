# CI and release gates

## Why

There is no CI. Tests pass because they were run locally, on one machine, in a working tree
that has files the repository does not — the cached UCI download among them. That is exactly
the configuration in which a sibling project shipped red releases from a green local suite,
repeatedly, because a test opened a file that only existed locally.

This repository is about to cut its first release. A release built on an unverified tree is
how a run of broken releases starts silently.

## What

- A GitHub Actions workflow running `ruff check`, `ruff format --check` and `pytest` on push,
  on a clean checkout.
- A test-suite audit for hidden dependence on local state, chiefly the `data/cache/`
  download. Tests that need network must skip cleanly, not fail, and must say which they are.
- A release workflow, and the discipline recorded in `CLAUDE.md`: a tag alone ships nothing;
  the GitHub Release is what fires the workflow; watch the run to green.

## Non-goals

- Not publishing to PyPI. The package is private and the weights are the asset.
- Not GPU CI. The Metal path is exercised locally; CI runs the CPU parametrisation of the
  device test and skips the rest.

## Falsified by

Not applicable.

## Blocked by / blocks

- **Blocked by** nothing.
- **Blocks** trustworthy releases. Cut `v0.1.0` first if desired, but treat its green status
  as unverified until this lands.
