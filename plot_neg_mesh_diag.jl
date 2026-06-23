# Negative-part mesh-alignment diagnostic for a single fs_snapshot.
# Reconstructs f_s on a FINE grid and plots ONLY where f_s<0, with mesh lines,
# to test whether the negative ring is locked to the (dense bp2) cell pitch.
#   julia --project=. plot_neg_mesh_diag.jl <tag> <step>
include("MantisWrappers.jl")
using .MantisWrappers
using GLMakie
GLMakie.activate!()

tag  = ARGS[1]
step = parse(Int, ARGS[2])
path = "fs_snapshot_$(tag)_step$(lpad(step,4,'0')).csv"
isfile(path) || error("missing $path")

function parse_header(p)
    bp1=Float64[]; bp2=Float64[]; ndof=0; ds=0
    open(p) do io
        for (i,ln) in enumerate(eachline(io))
            startswith(ln,"# bp1=") && (bp1=parse.(Float64, split(ln[7:end],',')))
            startswith(ln,"# bp2=") && (bp2=parse.(Float64, split(ln[7:end],',')))
            startswith(ln,"# n_dofs=") && (ndof=parse(Int, ln[10:end]))
            ln=="coeff" && (ds=i+1; break)
        end
    end
    bp1,bp2,ndof,ds
end
function load_coeffs(p,ndof,ds)
    c=Vector{Float64}(undef,ndof)
    open(p) do io
        for _ in 1:ds-1; readline(io); end
        for k in 1:ndof; c[k]=parse(Float64, readline(io)); end
    end
    c
end

bp1,bp2,ndof,ds = parse_header(path)
p  = SimParameters(; bp1=bp1, bp2=bp2)
ws = build_workspace(p)
c  = load_coeffs(path, ndof, ds)
fld = build_field(ws, c)

ε=1e-6
v1g = range(bp1[1]+ε, bp1[end]-ε; length=600)
v2g = range(bp2[1]+ε, bp2[end]-ε; length=600)
F = evaluate_on_grid(ws, fld, v1g, v2g)
println("step=$step fmin=$(minimum(F)) fmax=$(maximum(F))  Δv2_core=$(round(bp2[15]-bp2[14];digits=3))")

negmag = map(x -> x<0 ? -x : NaN, F)         # magnitude of negative part
nmax = maximum(filter(!isnan, negmag))

fig = Figure(; size=(1700, 560))

# A: full f_s for context
axA = Axis(fig[1,1]; title="f_s  step=$step", xlabel="v₁", ylabel="v₂", aspect=DataAspect())
heatmap!(axA, v1g, v2g, F; colormap=:viridis)
vlines!(axA, bp1; color=:red, linewidth=0.3, alpha=0.4); hlines!(axA, bp2; color=:red, linewidth=0.3, alpha=0.4)

# B: negative part full domain + mesh
axB = Axis(fig[1,2]; title="|negative part|  (mesh red)", xlabel="v₁", ylabel="v₂", aspect=DataAspect())
heatmap!(axB, v1g, v2g, negmag; colormap=:hot, colorrange=(0, nmax))
vlines!(axB, bp1; color=:cyan, linewidth=0.4, alpha=0.6); hlines!(axB, bp2; color=:cyan, linewidth=0.4, alpha=0.6)

# C: zoom upper tail band where the ring sits, to read pitch vs Δv2
axC = Axis(fig[1,3]; title="zoom v₂∈[0.8,2.6]  (Δv₂ cells)", xlabel="v₁", ylabel="v₂", aspect=DataAspect())
heatmap!(axC, v1g, v2g, negmag; colormap=:hot, colorrange=(0, nmax))
vlines!(axC, bp1; color=:cyan, linewidth=0.5, alpha=0.7); hlines!(axC, bp2; color=:cyan, linewidth=0.5, alpha=0.7)
xlims!(axC, -2.5, 2.5); ylims!(axC, 0.8, 2.6)

Colorbar(fig[1,4], colorrange=(0,nmax), colormap=:hot, label="|f_s<0|")
out="neg_mesh_diag_$(tag)_step$(lpad(step,4,'0')).png"
save(out, fig); println("Saved $out")
