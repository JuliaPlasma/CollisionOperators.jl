# Zoom v4 step 100 scatter — check honeycomb pattern claim.
using GLMakie, DelimitedFiles
GLMakie.activate!()
include("Parameters.jl")
include("parameters_bpmesh40k_v4.jl")
p = PARAMS

raw, _ = readdlm("particle_snapshots_bpmesh40k_v4.csv", ',', Any, '\n'; header=true)
mask = Int.(raw[:, 1]) .== 100
v1s = Float64.(raw[mask, 4])
v2s = Float64.(raw[mask, 5])

cnt = zeros(Int, length(p.bp1)-1, length(p.bp2)-1)
for (a, b) in zip(v1s, v2s)
    i = searchsortedlast(p.bp1, a); j = searchsortedlast(p.bp2, b)
    (1 <= i <= size(cnt,1) && 1 <= j <= size(cnt,2)) || continue
    cnt[i, j] += 1
end

f0(v1, v2) = exp(-v1^2/(2p.σ1^2) - v2^2/(2p.σ2^2)) / (2π*p.σ1*p.σ2)
v1_axis = range(p.bp1[1], p.bp1[end]; length=400)
v2_axis = range(p.bp2[1], p.bp2[end]; length=400)
F = [f0(v1, v2) for v1 in v1_axis, v2 in v2_axis]
peak = maximum(F)

fig = Figure(; size=(1600, 700))

# Left: full
ax1 = Axis(fig[1,1]; xlabel="v₁", ylabel="v₂",
    title="v4 step=100  full  N=$(length(v1s))  empty=$(count(==(0),cnt))/$(length(cnt))",
    aspect=DataAspect())
contour!(ax1, v1_axis, v2_axis, F;
    levels=peak .* [exp(-0.5), exp(-2.0), exp(-4.5)], color=:gray)
scatter!(ax1, v1s, v2s; markersize=1.2, color=(:steelblue, 0.35))
vlines!(ax1, p.bp1; color=:red, linewidth=0.4, alpha=0.6)
hlines!(ax1, p.bp2; color=:red, linewidth=0.4, alpha=0.6)
xlims!(ax1, p.bp1[1], p.bp1[end]); ylims!(ax1, p.bp2[1], p.bp2[end])

# Right: zoom inner bulk
ax2 = Axis(fig[1,2]; xlabel="v₁", ylabel="v₂",
    title="zoom [-3,3]×[-1.5,1.5]  inner bulk",
    aspect=DataAspect())
contour!(ax2, v1_axis, v2_axis, F;
    levels=peak .* [exp(-0.5), exp(-2.0), exp(-4.5)], color=:gray)
scatter!(ax2, v1s, v2s; markersize=3.0, color=(:steelblue, 0.5))
vlines!(ax2, p.bp1; color=:red, linewidth=0.6, alpha=0.7)
hlines!(ax2, p.bp2; color=:red, linewidth=0.6, alpha=0.7)
xlims!(ax2, -3, 3); ylims!(ax2, -1.5, 1.5)

# Cell-count heatmap third panel
ax3 = Axis(fig[1,3]; xlabel="v₁ cell idx", ylabel="v₂ cell idx",
    title="cell count (log10(1+n))", aspect=DataAspect())
hm = heatmap!(ax3, log10.(1 .+ cnt); colormap=:viridis)
Colorbar(fig[1,4], hm)

save("scatter_v4_step100_zoom.png", fig)
println("Saved scatter_v4_step100_zoom.png")
println("cell stats: empty=$(count(==(0),cnt))  max=$(maximum(cnt))  median=$(round(Int, sum(cnt)/length(cnt)))")
