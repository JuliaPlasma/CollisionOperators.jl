# CollisionOperators

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaPlasma.github.io/CollisionOperators.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaPlasma.github.io/CollisionOperators.jl/dev/)
[![Build Status](https://github.com/JuliaPlasma/CollisionOperators.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaPlasma/CollisionOperators.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaPlasma/CollisionOperators.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaPlasma/CollisionOperators.jl)
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/C/CollisionOperators.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/C/CollisionOperators.html)

Implementations of various collision operators such as Landau or Lenard–Bernstein.

## 2D Landau operator (Gonzalez discrete gradient)

Structure-preserving particle discretisation of the 2D Landau collision
operator in the metriplectic frame: the density is reconstructed in a
tensor-product B-spline finite-element space (via
[`Mantis`](https://github.com/JuliaPlasma/Mantis.jl)), the entropy gradient is
evaluated through the FE field, and the markers are advanced by the **Gonzalez
discrete-gradient** integrator so that energy and momentum are conserved while
the discrete entropy increases exactly. Follows Jeyakumar et al. (2024). See the
[documentation](https://JuliaPlasma.github.io/CollisionOperators.jl/dev/landau_gonzalez/)
for the scheme and the discrete-gradient construction.

### Source layout

| File | Role |
|------|------|
| `main_Gonzalez.jl` | Driver: time loop, discrete-gradient Picard map, implicit solve (Picard / Anderson), checkpointing, CSV + PNG output |
| `functions.jl` | Landau physics: L² projection, entropy & entropy-gradient seed, particle gradient, collision velocity update, diagnostics |
| `MantisWrappers.jl` | FEM scaffolding around `Mantis` (mesh, mass matrix, particle location/evaluation, `Workspace`) |
| `Parameters.jl` | `SimParameters` struct + CLI override parsing + Gaussian IC sampling |
| `parameters_*.jl` | Presets, each building a `PARAMS::SimParameters` |

### Run

```sh
julia --project=. main_Gonzalez.jl parameters_default.jl
# Picard instead of Anderson, scalar overrides:
julia --project=. main_Gonzalez.jl parameters_default.jl --N_STEPS=200 --use_anderson=false --suffix=picard_short
# resume from last checkpoint:
julia --project=. main_Gonzalez.jl parameters_default.jl --resume=auto
```

`ARGS[1]` is the preset file; `ARGS[2:]` are `--key=value` scalar overrides.
Vector fields (`bp1`, `bp2`) are not CLI-overridable — edit the preset.

### Output (per `suffix`)

- `conservation_history_<suffix>.csv` — one row per step:
  `step,time,entropy,energy,momentum_1,momentum_2,iter,residual,fp_minus_fs,neg_part`
- `fs_snapshot_<suffix>_step####.csv` — B-spline coefficients of `f_s`
  (mesh breakpoints in the header) every 25 steps + final
- `dashboard_<suffix>.png` — per-run quick-look dashboard
- diagnostic PNG + particle dump + checkpoint at each snapshot step
