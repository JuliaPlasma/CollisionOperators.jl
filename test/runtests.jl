using SafeTestsets

const GROUPS = isempty(ARGS) ? ["core", "slow"] : ARGS

if "core" in GROUPS
    @safetestset "Aqua" include("quality/aqua.jl")
    # io.jl needs only Parameters.jl, so its helpers run on every core pass.
    @safetestset "io.jl helpers" include("unit/io_helpers.jl")
end

# main.jl is a script that pulls CairoMakie and Mantis, so the kernels reachable
# only through it are exercised in the slow group rather than on every core run.
if "slow" in GROUPS
    @safetestset "main.jl helpers" include("unit/main_helpers.jl")
end
