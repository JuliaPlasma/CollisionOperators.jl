# CollisionOperators

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaPlasma.github.io/CollisionOperators.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaPlasma.github.io/CollisionOperators.jl/dev/)
[![Build Status](https://github.com/JuliaPlasma/CollisionOperators.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaPlasma/CollisionOperators.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaPlasma/CollisionOperators.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaPlasma/CollisionOperators.jl)
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/C/CollisionOperators.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/C/CollisionOperators.html)

Implementations of various collision operators such as Landau or Lenard–Bernstein.

## 2D Lenard–Bernstein operator

Structure-preserving particle discretisation of the conservative 2D
Lenard–Bernstein operator: the density is reconstructed in a tensor-product
B-spline finite-element space (via [`Mantis`](https://github.com/JuliaPlasma/Mantis.jl)),
particles are advected by implicit midpoint, and a 3×3 Lagrange-multiplier
system enforces exact discrete conservation of momentum and energy. Follows
Jeyakumar et al. (2024), specialised to the LB kernel. See the
[documentation](https://JuliaPlasma.github.io/CollisionOperators.jl/dev/lenard_bernstein/)
for the scheme and the conservation algebra.

### Source layout

| File | Role |
|------|------|
| `main_LB.jl` | Driver: time loop, implicit solve (Picard / Anderson), checkpointing, CSV + PNG output |
| `functions.jl` | LB physics: L² projection, log-gradient, drift multipliers, velocity update, diagnostics |
| `MantisWrappers.jl` | FEM scaffolding around `Mantis` (mesh, mass matrix, particle location/evaluation, `Workspace`) |
| `Parameters.jl` | `SimParameters` struct + CLI override parsing + Gaussian IC sampling |
| `parameters_LB*.jl` | Presets, each building a `PARAMS::SimParameters` |
| `plot_dashboard_LB*.jl`, `plot_scatter_LB*.jl` | Post-processing plots of the CSV output |

### Run

```sh
julia --project=. main_LB.jl parameters_LB2D_v3.jl
# scalar overrides:
julia --project=. main_LB.jl parameters_LB2D_v3.jl --N_STEPS=200 --suffix=foo
# resume from last checkpoint:
julia --project=. main_LB.jl parameters_LB2D_v3.jl --resume=auto
```

`ARGS[1]` is the preset file; `ARGS[2:]` are `--key=value` scalar overrides.
Vector fields (`bp1`, `bp2`) are not CLI-overridable — edit the preset.

### Output (per `suffix`)

- `conservation_history_<suffix>.csv` — one row per step:
  `step,time,entropy,energy,p1,p2,iter,residual,fp_minus_fs,neg_part`
- `fs_snapshot_<suffix>_step#####.csv` — B-spline coefficients of `f_s`
  (mesh breakpoints in the header) every `snap_every` steps + final
- diagnostic PNG + particle dump at each snapshot step
