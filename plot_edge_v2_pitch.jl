# Edge ring only v₂ hist — exclude bulk to expose strip pitch.
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
v1_v3, v2_v3 = load_step("particle_snapshots_bpmesh40k_v3.csv", 100)
include("parameters_bpmesh40k_v4.jl"); p4 = PARAMS
v1_v4_0, v2_v4_0 = load_step("particle_snapshots_bpmesh40k_v4.csv", 0)
v1_v4, v2_v4 = load_step("particle_snapshots_bpmesh40k_v4.csv", 100)

# Edge ring: 2 ≤ |v₁| ≤ 5
function edge_mask(v1, v2)
    (abs.(v1) .>= 2.0) .& (abs.(v1) .<= 5.0)
end

fig = Figure(; size=(1800, 1200))

# Row 1: zoom scatter edge ring v3 vs v4 step=100, larger markers
for (k, (lbl, v1, v2, p)) in enumerate([("v3 edge step=100", v1_v3, v2_v3, p3),
                                         ("v4 edge step=100", v1_v4, v2_v4, p4)])
    m = edge_mask(v1, v2)
    ax = Axis(fig[1, k]; xlabel="v₁", ylabel = k==1 ? "v₂" : "",
        title="$lbl  N_edge=$(count(m))",
        aspect=DataAspect())
    scatter!(ax, v1[m], v2[m]; markersize=5.0, color=(:steelblue, 0.7))
    vlines!(ax, p.bp1; color=:red, linewidth=0.4, alpha=0.5)
    hlines!(ax, p.bp2; color=:red, linewidth=0.4, alpha=0.5)
    xlims!(ax, -5, 5); ylims!(ax, -2.5, 2.5)
end

# Row 2: v₂ hist edge ring only, very fine bin (Δv₂/8 = 0.025)
for (k, (lbl, v1_0, v2_0, v1, v2, p)) in enumerate([
        ("v3 edge v₂", v1_v3_0, v2_v3_0, v1_v3, v2_v3, p3),
        ("v4 edge v₂", v1_v4_0, v2_v4_0, v1_v4, v2_v4, p4)])
    m0 = edge_mask(v1_0, v2_0)
    m  = edge_mask(v1, v2)
    Δ = p.bp2[4] - p.bp2[3]
    edges = collect(-1.5:Δ/8:1.5)
    ax = Axis(fig[2, k]; xlabel="v₂", ylabel="count",
        title="$lbl  bin=Δ/8=$(round(Δ/8; digits=4))")
    hist!(ax, v2_0[m0]; bins=edges, color=(:gray, 0.4), label="step=0")
    hist!(ax, v2[m];    bins=edges, color=(:crimson, 0.6), label="step=100")
    vlines!(ax, p.bp2; color=:blue, linewidth=0.5, alpha=0.6)
    axislegend(ax; position=:rt)
    xlims!(ax, -1.5, 1.5)
end

# Row 3: v₁ hist edge ring only, very fine bin
for (k, (lbl, v1_0, v2_0, v1, v2, p)) in enumerate([
        ("v3 edge v₁", v1_v3_0, v2_v3_0, v1_v3, v2_v3, p3),
        ("v4 edge v₁", v1_v4_0, v2_v4_0, v1_v4, v2_v4, p4)])
    m0 = edge_mask(v1_0, v2_0)
    m  = edge_mask(v1, v2)
    Δ = p.bp1[4] - p.bp1[3]
    edges = collect(-5.0:Δ/8:5.0)
    ax = Axis(fig[3, k]; xlabel="v₁", ylabel="count",
        title="$lbl  bin=Δ/8=$(round(Δ/8; digits=4))")
    hist!(ax, v1_0[m0]; bins=edges, color=(:gray, 0.4), label="step=0")
    hist!(ax, v1[m];    bins=edges, color=(:crimson, 0.6), label="step=100")
    vlines!(ax, p.bp1; color=:blue, linewidth=0.5, alpha=0.6)
    axislegend(ax; position=:rt)
    xlims!(ax, -5, 5)
end

Label(fig[0, 1:2], "Edge ring 2≤|v₁|≤5 only — strip pitch test, exclude bulk";
    fontsize=18, tellwidth=false)

save("v3v4_edge_pitch.png", fig)
println("Saved v3v4_edge_pitch.png")
