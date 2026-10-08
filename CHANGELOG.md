# Release Notes

All notable changes to CollisionOperators.jl.

`Project.toml` declares `version = "1.0.0-DEV"`, so the first release will be 1.0.0 and
[SemVer](https://semver.org) applies from then on in its usual sense: a minor bump is
additive, a major bump is breaking. The sections below name what actually changed, so that a
compat-only bump can be told apart from a rename or a change in results.

This file was started on 2026-08-31. Nothing has been released yet — there are no tags.

## [Unreleased] — targeting 1.0.0

### New Features

- **Jacobian-free Newton–Krylov solve for the implicit step** (`6d48033`). `--solver=newton`
  selects `step_newton!`, which solves the same Picard fixed point as `step_anderson!` by inexact
  Newton with a matrix-free GMRES modeled on Krylov.jl, Eisenstat–Walker forcing and Armijo
  backtracking (`newton_krylov.jl`). New parameters: `solver` (default `:anderson`),
  `nk_krylov_max = 30`, `nk_eta_max = 0.9` and `nk_fd_rel = 0`, which picks the √eps of the
  collision kernel's precision as the finite-difference step. Both solvers share `tol`,
  `abs_floor` and `max_iter`, and the `iter` column counts Picard-map evaluations for both, so
  runs compare one-to-one; Anderson runs are unchanged. Measured, it does not beat Anderson in
  FP32 at the production tolerance and comes out roughly even in FP64, so Anderson with
  `abs_floor = 1e-8` stays the production setting — see `docs/src/newton_krylov.md`.

### Performance

- **The Anderson window is updated through a ring cursor** (`49197af`). Each new residual and
  map difference overwrites the oldest column of `ΔF`/`ΔG` instead of shifting the whole window
  left, which recopied 2N(m − 1) entries on every iteration. The least-squares solve is invariant
  under a common column permutation, so the iterates are unchanged up to round-off. At
  N = 40,000, m = 8 the update drops from 2.12 ms to 0.08 ms per iteration: about 6 % of an FP32
  iteration, 1.5 % of an FP64 one — see `docs/src/anderson_window.md`.

### Bug Fixes

- **Physical time survives a resume at a different `DT`** (`1769872`). `time` was written as
  `step * DT` with the `DT` in force at the time, so resuming with a new `DT` rescaled the whole
  history. It is now accumulated from the resume point, and checkpoints carry `t` so the resume
  recovers it exactly.
- **Resuming from an older checkpoint no longer duplicates rows** (`e4c09d8`). The conservation
  and particle CSVs are trimmed to the rows at or before the resume step before new rows are
  appended.

### Breaking Changes

- **Conservation CSV layout** (`1769872`). A 12th column `dt` holds each step's `DT`, with
  `cumsum(dt) == time`, and `time` is accumulated physical time. Code that rebuilds time as
  `step * DT` is wrong for any run whose `DT` changed. Legacy files are migrated in place on
  `--resume`; `particle_snapshots_*.csv` is not, so its rows written before a resume keep
  `step * DT`.

### Dependencies

- `[compat]` now bounds every registered dependency instead of only `julia`: CairoMakie 0.15.12,
  DelimitedFiles 1.9.1, FileIO 1.19.0, ForwardDiff 1, Mantis 0.6 and PNGFiles 0.4.5.
  Until now the package carried no bound on any of them, so a resolve could pick a version whose
  API no longer matches this source. The lower bounds are the versions this code is known to work
  with.
- **`GLMakie` is no longer a dependency. CairoMakie is the plotting backend.** GLMakie initialises
  GLFW when the module loads, which fails on a headless runner with
  `GLFWError(65550, "X11: The DISPLAY environment variable is missing")`. Every CI job that
  precompiles it therefore dies before a test runs. Julia 1.13 downgrades that to a warning for a
  dependency nothing loads, so the breakage is invisible on the `1` row and fatal on `min` and
  `nightly`. CairoMakie was already a dependency, renders headless, and covers everything this
  package needs.
- `docs/Project.toml` bounds Documenter at 1.17.0, so the documentation build resolves the same
  Documenter everywhere.
- Stdlib dependencies now have `[compat]` bounds: LinearAlgebra, Random, Serialization, and
  SparseArrays = "1" so Aqua's deps_compat check passes.
- Test suite restructured with `SafeTestsets` with GROUPS; test dependencies moved to
  `test/Project.toml`, `[extras]`/`[targets]` removed, and empty template testset deleted.
  `test/quality/aqua.jl` runs Aqua with stale_deps marked @test_broken for issue #22.
