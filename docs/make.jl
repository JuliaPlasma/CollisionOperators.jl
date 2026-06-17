using CollisionOperators
using Documenter

DocMeta.setdocmeta!(CollisionOperators, :DocTestSetup, :(using CollisionOperators); recursive=true)

makedocs(;
    modules=[CollisionOperators],
    authors="Michael Kraus",
    sitename="CollisionOperators.jl",
    format=Documenter.HTML(;
        canonical="https://JuliaPlasma.github.io/CollisionOperators.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Operators" => [
            "Lenard–Bernstein (2D)" => "lenard_bernstein.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/JuliaPlasma/CollisionOperators.jl",
    devbranch="main",
)
