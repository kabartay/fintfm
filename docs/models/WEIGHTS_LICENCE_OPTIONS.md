# Weights licence options for the three entry checkpoints

**Decided 2026-11-16 (owner): Apache-2.0 for all three checkpoints.** LICENCE: Apache-2.0. This note restates what `CLAUDE.md` records; it recommends nothing.

## Facts

- Code is Apache-2.0 (chosen 2026-09-08, public since 2026-09-25, one-way: it cannot be
  narrowed for anyone holding it).
- `CLAUDE.md`: "Trained weights are still not published and remain gitignored. Publishing one is
  a separate decision with a separate licence." Trained weights never enter git.
- `CLAUDE.md`: the moat is the trained weights and the mature prior, not the training code.
- `CLAUDE.md`: check the weights licence separately from the code licence, every time (Google
  TabFM and TimesFM 3.0 ship Apache-2.0 code with non-commercial weights).
- Reality check (public API, 2026-11-15): `kabartay/fintfm-binary` is public and declares
  `apache-2.0` in its metadata, with the binary file's SHA-256 identical to the local one. The
  multiclass and regression repositories were not publicly visible on that date (both are public
  since 2026-11-17, per their model cards). The binary grant is therefore
  already out, and `CLAUDE.md`'s "not published" sentence is stale on that point.
- A TabArena resubmission needs all three checkpoints public with a licence.

## Option A: Apache-2.0, matching the code

- Adoption and procurement: the permissive licence and patent grant are the reasons the code
  chose it, and the same banks and insurers read the weights terms.
- One licence to explain; consistent with the binary repository as it stands.
- Irrevocable for each copy fetched; later versions can change, shipped ones cannot.
- Gives away the asset `CLAUDE.md` names as the moat, for three small (about 3.6 MB each)
  checkpoints.

## Option B: a separate weights licence (for example non-commercial, or custom terms)

- Keeps commercial use of the weights under the owner's control.
- Departs from the code licence; the owner must state the terms and check whether they pass the
  procurement review the code licence was chosen to clear.
- The binary repository is already Apache-2.0 for the file at the revision fetched; a different
  licence on later revisions does not withdraw that grant.
- TabArena's own terms for entries were not checked here.

## Open for the owner

- Which licence for multiclass and regression, and whether binary's stays as it is (decided on
  2026-11-16: Apache-2.0 for all three).
- Attribution wording and any CLA are not covered here.
