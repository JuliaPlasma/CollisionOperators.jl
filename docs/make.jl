using CollisionOperators
using Documenter

# The driver is a set of top-level scripts, not part of the package module, so
# `main.jl` would put its docstrings in `Main` where Documenter cannot reach
# them. Load the documented files into a module of their own and point
# `makedocs` at that as well. `plots.jl` is left out: it only wraps CairoMakie.
module Driver
include(joinpath(@__DIR__, "..", "MantisWrappers.jl"))
using .MantisWrappers
include(joinpath(@__DIR__, "..", "io.jl"))
include(joinpath(@__DIR__, "..", "solver.jl"))
end

DocMeta.setdocmeta!(CollisionOperators, :DocTestSetup, :(using CollisionOperators); recursive = true)

makedocs(;
    modules = [CollisionOperators, Driver],
    authors = "Michael Kraus, Junyi Xu <junyixu0@gmail.com>",
    sitename = "CollisionOperators.jl",
    format = Documenter.HTML(;
        canonical = "https://JuliaPlasma.github.io/CollisionOperators.jl",
        edit_link = "main",
        assets = String[]
    ),
    pages = [
        "Home" => "index.md",
        "Operators" => [
            "Lenard–Bernstein (2D)" => "lenard_bernstein.md"
        ],
        "Anderson window update" => "anderson_window.md",
        "Driver API" => "solver.md"
    ]
)

deploydocs(;
    repo = "github.com/JuliaPlasma/CollisionOperators.jl",
    devbranch = "main"
)
