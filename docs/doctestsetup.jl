# Shared setup for the documentation: included by docs/make.jl and, separately,
# by the Doctests CI job, which calls doctest() without running make.jl. Both
# need `Driver` bound in Main — doctest() evaluates the `CurrentModule = X` of
# every manual page there.
using CollisionOperators
using Documenter: DocMeta

# The driver is a set of top-level scripts, not part of the package module, so
# running main.jl would put its docstrings in `Main`, out of Documenter's reach.
# Load the documented files into a module of their own instead; `makedocs` lists
# it in `modules` and docs/src/solver.md pulls the docstrings from it.
# plots.jl is left out: it only wraps CairoMakie.
module Driver
include(joinpath(@__DIR__, "..", "MantisWrappers.jl"))
using .MantisWrappers
include(joinpath(@__DIR__, "..", "io.jl"))
include(joinpath(@__DIR__, "..", "solver.jl"))
end

DocMeta.setdocmeta!(CollisionOperators, :DocTestSetup, :(using CollisionOperators);
    recursive = true)
