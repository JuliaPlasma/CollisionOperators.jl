# Edge honeycomb check: full domain v3 vs v4 step=100, focus outer ring.
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
v1_v3, v2_v3     = load_step("particle_snapshots_bpmesh40k_v3.csv", 100)
include("parameters_bpmesh40k_v4.jl"); p4 = PARAMS
v1_v4_0, v2_v4_0 = load_step("particle_snapshots_bpmesh40k_v4.csv", 0)
v1_v4, v2_v4     = load_step("particle_snapshots_bpmesh40k_v4.csv", 100)

fig = Figure(; size=(1800, 1300))

# Row 1: step=0 full domain
for (k, (lbl, v1, v2, p)) in enumerate([("v3 step=0 full", v1_v3_0, v2_v3_0, p3),
                                         ("v4 step=0 full", v1_v4_0, v2_v4_0, p4)])
    ax = Axis(fig[1, k]; xlabel="v₁", ylabel = k==1 ? "v₂" : "", title=lbl, aspect=DataAspect())
    scatter!(ax, v1, v2; markersize=1.5, color=(:steelblue, 0.4))
    vlines!(ax, p.bp1; color=:red, linewidth=0.4, alpha=0.6)
    hlines!(ax, p.bp2; color=:red, linewidth=0.4, alpha=0.6)
    xlims!(ax, p.bp1[1], p.bp1[end]); ylims!(ax, p.bp2[1], p.bp2[end])
end

# Row 2: step=100 full domain
for (k, (lbl, v1, v2, p)) in enumerate([("v3 step=100 full", v1_v3, v2_v3, p3),
                                         ("v4 step=100 full", v1_v4, v2_v4, p4)])
    ax = Axis(fig[2, k]; xlabel="v₁", ylabel = k==1 ? "v₂" : "", title=lbl, aspect=DataAspect())
    scatter!(ax, v1, v2; markersize=1.5, color=(:steelblue, 0.4))
    vlines!(ax, p.bp1; color=:red, linewidth=0.4, alpha=0.6)
    hlines!(ax, p.bp2; color=:red, linewidth=0.4, alpha=0.6)
    xlims!(ax, p.bp1[1], p.bp1[end]); ylims!(ax, p.bp2[1], p.bp2[end])
end

# Row 3: step=100 edge ring focus — annulus 2.5 < |v₁| < 5
for (k, (lbl, v1, v2, p)) in enumerate([("v3 edge ring 2≤|v₁|≤5", v1_v3, v2_v3, p3),
                                         ("v4 edge ring 2≤|v₁|≤5", v1_v4, v2_v4, p4)])
    sel = (abs.(v1) .>= 2.0) .& (abs.(v1) .<= 5.0)
    ax = Axis(fig[3, k]; xlabel="v₁", ylabel = k==1 ? "v₂" : "",
        title="$lbl  N_sel=$(count(sel))",
        aspect=DataAspect())
    scatter!(ax, v1[sel], v2[sel]; markersize=4.0, color=(:steelblue, 0.6))
    vlines!(ax, p.bp1; color=:red, linewidth=0.5, alpha=0.7)
    hlines!(ax, p.bp2; color=:red, linewidth=0.5, alpha=0.7)
    xlims!(ax, -5, 5); ylims!(ax, -2.5, 2.5)
end

Label(fig[0, 1:2], "v3 vs v4 step=0→100 edge honeycomb test (full domain + edge ring)";
    fontsize=18, tellwidth=false)

save("v3v4_edge_compare.png", fig)
println("Saved v3v4_edge_compare.png")
