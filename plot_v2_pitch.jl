# v₂ 1D hist pitch test: if banding tracks bp2 Δv₂=0.2, sub-bin should show modulation.
using GLMakie, DelimitedFiles
GLMakie.activate!()

function load_step(csv, step)
    raw, _ = readdlm(csv, ',', Any, '\n'; header=true)
    mask = Int.(raw[:, 1]) .== step
    return Float64.(raw[mask, 4]), Float64.(raw[mask, 5])
end

include("Parameters.jl")
include("parameters_bpmesh40k.jl"); p3 = PARAMS
v1_v3_0, v2_v3_0 = load_step("particle_snapshots_bpmesh40k_v3.csv", 0)
v1_v3,   v2_v3   = load_step("particle_snapshots_bpmesh40k_v3.csv", 100)
include("parameters_bpmesh40k_v4.jl"); p4 = PARAMS
v1_v4_0, v2_v4_0 = load_step("particle_snapshots_bpmesh40k_v4.csv", 0)
v1_v4,   v2_v4   = load_step("particle_snapshots_bpmesh40k_v4.csv", 100)

Δv2_3 = p3.bp2[4] - p3.bp2[3]   # = 0.2
Δv2_4 = p4.bp2[4] - p4.bp2[3]   # = 0.2 (same — bp2 unchanged)
println("Δv₂ v3=$Δv2_3  v4=$Δv2_4 (both should be 0.2)")

fig = Figure(; size=(1600, 900))

# Row 1: v₂ hist, bin=Δv₂/4 = 0.05, range [-1, 1]
for (k, (lbl, v2_0, v2_100, p)) in enumerate([("v3 v₂ hist", v2_v3_0, v2_v3, p3),
                                                ("v4 v₂ hist", v2_v4_0, v2_v4, p4)])
    Δ = p.bp2[4] - p.bp2[3]
    edges = collect(-1.5:Δ/4:1.5)
    ax = Axis(fig[1, k]; xlabel="v₂", ylabel="count",
        title="$lbl  bin=$(round(Δ/4; digits=3))  Δv₂=$(round(Δ; digits=3))")
    hist!(ax, v2_0;   bins=edges, color=(:gray, 0.4), label="step=0")
    hist!(ax, v2_100; bins=edges, color=(:crimson, 0.6), label="step=100")
    vlines!(ax, p.bp2; color=:blue, linewidth=0.4, alpha=0.5)
    axislegend(ax; position=:rt)
    xlims!(ax, -1.5, 1.5)
end

# Row 2: v₁ hist sanity, bin=Δv₁/4
for (k, (lbl, v1_0, v1_100, p)) in enumerate([("v3 v₁ hist", v1_v3_0, v1_v3, p3),
                                                ("v4 v₁ hist", v1_v4_0, v1_v4, p4)])
    Δ = p.bp1[4] - p.bp1[3]
    edges = collect(-3.0:Δ/4:3.0)
    ax = Axis(fig[2, k]; xlabel="v₁", ylabel="count",
        title="$lbl  bin=$(round(Δ/4; digits=3))  Δv₁=$(round(Δ; digits=3))")
    hist!(ax, v1_0;   bins=edges, color=(:gray, 0.4), label="step=0")
    hist!(ax, v1_100; bins=edges, color=(:crimson, 0.6), label="step=100")
    vlines!(ax, p.bp1; color=:blue, linewidth=0.4, alpha=0.5)
    axislegend(ax; position=:rt)
    xlims!(ax, -3.0, 3.0)
end

Label(fig[0, 1:2],
    "Pitch test — 1D hist v₂ (top) & v₁ (bottom), step=0 (gray) vs step=100 (red)";
    fontsize=18, tellwidth=false)

save("v3v4_pitch_hist.png", fig)
println("Saved v3v4_pitch_hist.png")
