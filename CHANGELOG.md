# Release Notes

All notable changes to CollisionOperators.jl.

`Project.toml` declares `version = "1.0.0-DEV"`, so the first release will be 1.0.0 and
[SemVer](https://semver.org) applies from then on in its usual sense: a minor bump is
additive, a major bump is breaking. The sections below name what actually changed, so that a
compat-only bump can be told apart from a rename or a change in results.

This file was started on 2026-08-31 and deliberately holds no entries. Nothing has been
released yet — there are no tags, and the repository holds two commits, both from 2026-06-15.

## [Unreleased] — targeting 1.0.0

### New Features

### Bug Fixes

### Breaking Changes

### Dependencies

- `[compat]` now bounds every registered dependency instead of only `julia`: CairoMakie 0.15.12,
  DelimitedFiles 1.9.1, FileIO 1.19.0, ForwardDiff 1, GLMakie 0.13, Mantis 0.6 and PNGFiles 0.4.5.
  Until now the package carried no bound on any of them, so a resolve could pick a version whose
  API no longer matches this source. The lower bounds are the versions this code is known to work
  with.
- `docs/Project.toml` bounds Documenter at 1.17.0, so the documentation build resolves the same
  Documenter everywhere.

## Open Issues
