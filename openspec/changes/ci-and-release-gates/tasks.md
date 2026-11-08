# Tasks

- [x] 8.1 Audit the suite for local-state dependence, especially `data/cache/`. Verify: run
      `git clone` into a temp dir, `uv sync`, `uv run pytest -q`, and record the pass/skip
      counts. Expect more skips than locally; a *failure* there is the real finding.
      **Done 2026-09-09: clean clone gives 72 passed, 2 skipped in 119s. The skips are the
      Taiwan and V4FinBench loaders correctly detecting absent data. No hidden local-state
      dependence.**
- [x] 8.2 Make network-dependent tests skip with a stated reason rather than fail. Verify:
      the clean-clone run above is green. **Done — already satisfied; the clean-clone run is
      green with 2 explained skips.**
- [x] 8.3 Add `.github/workflows/ci.yml` on push and pull_request. Verify: `gh run list -L 3`
      shows it green on the commit that adds it. **Added 2026-09-09: lint, the openspec
      validator, tests, a wheel build, and an assertion that the Apache-2.0 licence and
      py.typed marker ship inside the wheel. Installs the `bench` extra so the boosting
      baselines cannot silently vanish as in §25.**
- [ ] 8.6 Add `ruff format --check` as a gate. **Deliberately excluded 2026-09-09**: it would
      reformat 18 of 29 files, and that diff must not be mixed into substantive work. Verify:
      run the reformat as its own isolated commit, then add the gate.
- [ ] 8.4 **Progress (2026-11-08): `.github/workflows/release.yml` already existed (tag-push trigger, gates, wheel, PyPI Trusted Publishing); its trigger is now `release: published` and `CLAUDE.md` "Releases" records that a bare tag fires nothing. `actionlint` reports only an info-level SC2012 (`ls` in the wheel-install step, unchanged). Open: the Verify (a green run on the next release) cannot be met until the next release.** Add a release workflow firing on `release: published`, and record in `CLAUDE.md`
      that a bare tag ships nothing. Verify: the next release shows a green run.
- [x] 8.5 **Done (2026-11-08, v0.1.0 release body addendum):** a clean clone at the `v0.1.0` tag, `uv sync --extra bench` then `uv run pytest -q`, gives 30 passed, 0 skipped in 28.6 s, and `ruff check src/ tests/` passes (MEASURED); README training and benchmark steps not re-run. Recorded as a dated addendum on the GitHub release. **Progress (2026-11-08): clean clone of `v0.1.0` (`uv sync --extra bench`, `uv run pytest -q`) gives 30 passed, 0 skipped in 28.6 s, and `ruff check src/ tests/` passes. The README quickstart train and bench steps were not run (5000-step training). Open: the note is not yet on the release (`gh release edit` writes to GitHub); the owner posts it.** Retrospectively verify `v0.1.0` against the clean-clone procedure and note the
      result on the release, since it was cut before any of this existed. Verify:
      `gh release view v0.1.0` carries a note stating the clean-clone pass/skip counts.
