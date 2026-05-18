# Side-by-side v3 vs v4 step=100 zoom — measure honeycomb pitch vs mesh.
# Hypothesis: if pitch tracks Δv₁ (v3=0.5, v4=0.333), it's cell-locking artifact.
using GLMakie, DelimitedFiles
GLMakie.activate!()

function load_step(csv, step)
    raw, _ = readdlm(csv, ',', Any, '\n'; header=true)
    mask = Int.(raw[:, 1]) .== step
    v1s = Float64.(raw[mask, 4])
    v2s = Float64.(raw[mask, 5])
    return v1s, v2s
end

include("Parameters.jl")

# v3 mesh
include("parameters_bpmesh40k.jl"); p3 = PARAMS
v1_v3, v2_v3 = load_step("particle_snapshots_bpmesh40k_v3.csv", 100)
# v4 mesh
include("parameters_bpmesh40k_v4.jl"); p4 = PARAMS
v1_v4, v2_v4 = load_step("particle_snapshots_bpmesh40k_v4.csv", 100)

println("v3 N=$(length(v1_v3))  Δv₁ inner=$(round(p3.bp1[4]-p3.bp1[3]; digits=4))")
println("v4 N=$(length(v1_v4))  Δv₁ inner=$(round(p4.bp1[4]-p4.bp1[3]; digits=4))")

fig = Figure(; size=(1600, 1300))

# Row 1: full zoom [-2,2]×[-1,1] particles + mesh
for (k, (lbl, v1, v2, p)) in enumerate([("v3 Δv₁=0.5",  v1_v3, v2_v3, p3),
                                         ("v4 Δv₁=0.333", v1_v4, v2_v4, p4)])
    ax = Axis(fig[1, k]; xlabel="v₁", ylabel = k==1 ? "v₂" : "",
        title="$lbl  step=100 zoom [-2,2]×[-1,1]",
        aspect=DataAspect())
    scatter!(ax, v1, v2; markersize=2.0, color=(:steelblue, 0.45))
    vlines!(ax, p.bp1; color=:red, linewidth=0.5, alpha=0.6)
    hlines!(ax, p.bp2; color=:red, linewidth=0.5, alpha=0.6)
    xlims!(ax, -2, 2); ylims!(ax, -1, 1)
end

# Row 2: tighter zoom [-1,1]×[-0.5,0.5] for pitch measure
for (k, (lbl, v1, v2, p)) in enumerate([("v3", v1_v3, v2_v3, p3),
                                         ("v4", v1_v4, v2_v4, p4)])
    ax = Axis(fig[2, k]; xlabel="v₁", ylabel = k==1 ? "v₂" : "",
        title="$lbl  tight [-1,1]×[-0.5,0.5]  pitch?",
        aspect=DataAspect())
    scatter!(ax, v1, v2; markersize=3.5, color=(:steelblue, 0.5))
    vlines!(ax, p.bp1; color=:red, linewidth=0.6, alpha=0.7)
    hlines!(ax, p.bp2; color=:red, linewidth=0.6, alpha=0.7)
    xlims!(ax, -1, 1); ylims!(ax, -0.5, 0.5)
end

# Row 3: 1D projection histogram on v₁ for both
for (k, (lbl, v1, p)) in enumerate([("v3 v₁ hist", v1_v3, p3),
                                     ("v4 v₁ hist", v1_v4, p4)])
    ax = Axis(fig[3, k]; xlabel="v₁", ylabel="count",
        title="$lbl  bin=Δ/3 to resolve sub-cell pitch")
    Δ = p.bp1[4] - p.bp1[3]
    edges = -2:Δ/3:2
    hist!(ax, v1; bins=collect(edges), color=:steelblue, strokewidth=0.3)
    vlines!(ax, p.bp1; color=:red, linewidth=0.5, alpha=0.7)
    xlims!(ax, -2, 2)
end

Label(fig[0, 1:2],
    "v3 (Δv₁=0.5, 17 inner) vs v4 (Δv₁=0.333, 25 inner) — step=100 honeycomb pitch test";
    fontsize=18, tellwidth=false)

save("v3v4_honeycomb_compare.png", fig)
println("Saved v3v4_honeycomb_compare.png")
