# Tasks

- [ ] 8.1 Audit the suite for local-state dependence, especially `data/cache/`. Verify: run
      `git clone` into a temp dir, `uv sync`, `uv run pytest -q`, and record the pass/skip
      counts. Expect more skips than locally; a *failure* there is the real finding.
- [ ] 8.2 Make network-dependent tests skip with a stated reason rather than fail. Verify:
      the clean-clone run above is green.
- [ ] 8.3 Add `.github/workflows/ci.yml` on push and pull_request. Verify: `gh run list -L 3`
      shows it green on the commit that adds it.
- [ ] 8.4 Add a release workflow firing on `release: published`, and record in `CLAUDE.md`
      that a bare tag ships nothing. Verify: the next release shows a green run.
- [ ] 8.5 Retrospectively verify `v0.1.0` against the clean-clone procedure and note the
      result on the release, since it was cut before any of this existed.
