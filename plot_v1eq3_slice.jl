# v1 ≈ 3 narrow slice, full v2 — direct strip pitch read.
using GLMakie, DelimitedFiles
GLMakie.activate!()

include("Parameters.jl")
include("parameters_bpmesh40k_v4.jl"); p = PARAMS

raw, _ = readdlm("particle_snapshots_bpmesh40k_v4.csv", ',', Any, '\n'; header=true)
mask = Int.(raw[:, 1]) .== 100
v1s = Float64.(raw[mask, 4])
v2s = Float64.(raw[mask, 5])

# Slice v1 ∈ [2.9, 3.1] — narrow band at v1=3
slice = (v1s .>= 2.9) .& (v1s .<= 3.1)
v1sl = v1s[slice]
v2sl = v2s[slice]
println("slice v1∈[2.9, 3.1]  N = $(length(v2sl))")

Δv2 = p.bp2[4] - p.bp2[3]

fig = Figure(; size=(1800, 900))

# Panel 1: scatter slice — v1 narrow strip, full v2
ax1 = Axis(fig[1, 1]; xlabel="v₁", ylabel="v₂",
    title="v4 step=100  slice v₁∈[2.9, 3.1]  N=$(length(v2sl))")
scatter!(ax1, v1sl, v2sl; markersize=5.0, color=(:steelblue, 0.8))
hlines!(ax1, p.bp2; color=:red, linewidth=0.6, alpha=0.7)
xlims!(ax1, 2.85, 3.15); ylims!(ax1, -2.5, 2.5)

# Panel 2: v2 hist of slice, bin Δv₂/4 = 0.05
edges = collect(-2.5:Δv2/4:2.5)
ax2 = Axis(fig[1, 2]; xlabel="v₂", ylabel="count",
    title="v₂ hist (slice)  bin=Δv₂/4=$(Δv2/4)  bp2(red)")
hist!(ax2, v2sl; bins=edges, color=:steelblue)
vlines!(ax2, p.bp2; color=:red, linewidth=0.6, alpha=0.7)
xlims!(ax2, -2.5, 2.5)

# Panel 3: same hist finer bin Δv₂/10
edges_finer = collect(-2.5:Δv2/10:2.5)
ax3 = Axis(fig[1, 3]; xlabel="v₂", ylabel="count",
    title="v₂ hist finer bin=$(Δv2/10)")
hist!(ax3, v2sl; bins=edges_finer, color=:steelblue)
vlines!(ax3, p.bp2; color=:red, linewidth=0.4, alpha=0.6)
xlims!(ax3, -2.5, 2.5)

Label(fig[0, 1:3], "v₁≈3 slice — strip pitch direct read (sparse band, signal/noise OK)";
    fontsize=18, tellwidth=false)

save("v4_slice_v1eq3.png", fig)
println("Saved v4_slice_v1eq3.png")
