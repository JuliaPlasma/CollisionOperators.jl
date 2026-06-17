# CollisionOperators

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaPlasma.github.io/CollisionOperators.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaPlasma.github.io/CollisionOperators.jl/dev/)
[![Build Status](https://github.com/JuliaPlasma/CollisionOperators.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaPlasma/CollisionOperators.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaPlasma/CollisionOperators.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaPlasma/CollisionOperators.jl)
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/C/CollisionOperators.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/C/CollisionOperators.html)

Implementations of various collision operators such as Landau or Lenard–Bernstein.

## Examples

- [`examples/lenard_bernstein_2d`](examples/lenard_bernstein_2d) — structure-preserving
  particle discretisation of the conservative 2D Lenard–Bernstein operator
  (implicit midpoint, exact momentum/energy conservation). See the
  [documentation](https://JuliaPlasma.github.io/CollisionOperators.jl/dev/lenard_bernstein/).
