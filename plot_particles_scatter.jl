# Scatter of particles at chosen steps from a particle_snapshots_*.csv, overlaid on
# the bp1/bp2 mesh of a chosen preset (default = parameters_default.jl).
# Run as:
#   julia --project=. plot_particles_scatter.jl <particle_csv> [preset.jl] [steps]
# steps: comma-separated list, e.g. "0,200,400". Default "0,200,400".
using GLMakie, DelimitedFiles
GLMakie.activate!()
include("Parameters.jl")

length(ARGS) >= 1 || error("usage: plot_particles_scatter.jl <particle_csv> [preset.jl] [steps]")
csv_path = ARGS[1]
preset   = length(ARGS) >= 2 ? ARGS[2] : "parameters_default.jl"
steps_arg = length(ARGS) >= 3 ? ARGS[3] : "0,200,400"
steps = parse.(Int, split(steps_arg, ','))
include(preset)
p = PARAMS

# CSV cols: step,time,particle_idx,v1,v2 — read raw matrix
raw, _ = readdlm(csv_path, ',', Any, '\n'; header=true)
available = unique(Int.(raw[:, 1]))
println("Available steps: $available")
# clamp requested steps to nearest available (handy if run incomplete)
steps = [available[argmin(abs.(available .- s))] for s in steps]
println("Plotting steps: $steps")

f0(v1, v2) = exp(-v1^2/(2p.σ1^2) - v2^2/(2p.σ2^2)) / (2π*p.σ1*p.σ2)
v1_axis = range(p.bp1[1], p.bp1[end]; length=400)
v2_axis = range(p.bp2[1], p.bp2[end]; length=400)
F = [f0(v1, v2) for v1 in v1_axis, v2 in v2_axis]
peak = maximum(F)

# Get N from first chosen step
mask0 = Int.(raw[:, 1]) .== steps[1]
N = count(mask0)
msize = max(1.0, 5.0 / sqrt(N / 1000))
tag = N >= 30_000 ? "40k" : "10k"

ncols = length(steps)
fig = Figure(; size=(420 * ncols, 800))

for (k, s) in enumerate(steps)
    mask = Int.(raw[:, 1]) .== s
    v1s = Float64.(raw[mask, 4])
    v2s = Float64.(raw[mask, 5])
    t   = Float64(raw[findfirst(mask), 2])

    # cell stats vs new bp mesh
    cnt = zeros(Int, length(p.bp1)-1, length(p.bp2)-1)
    for (a, b) in zip(v1s, v2s)
        i = searchsortedlast(p.bp1, a); j = searchsortedlast(p.bp2, b)
        (1 <= i <= size(cnt,1) && 1 <= j <= size(cnt,2)) || continue
        cnt[i, j] += 1
    end
    n_empty = count(==(0), cnt)
    println("step=$s  N=$(length(v1s))  empty_cells=$n_empty/$(length(cnt))  max_cell=$(maximum(cnt))")

    ax = Axis(fig[1, k];
        xlabel="v₁", ylabel=k==1 ? "v₂" : "",
        title="step=$s   t=$(round(t; digits=3))   empty=$n_empty/$(length(cnt))",
        aspect=DataAspect())
    contour!(ax, v1_axis, v2_axis, F;
        levels=peak .* [exp(-0.5), exp(-2.0), exp(-4.5)],
        color=:gray, linewidth=1.2)
    scatter!(ax, v1s, v2s; markersize=msize, color=(:steelblue, 0.35), strokewidth=0)
    vlines!(ax, p.bp1; color=:red, linewidth=0.5, alpha=0.7)
    hlines!(ax, p.bp2; color=:red, linewidth=0.5, alpha=0.7)

    # inner zoom
    ax2 = Axis(fig[2, k];
        xlabel="v₁", ylabel=k==1 ? "v₂" : "",
        title="zoom [-3,3] × [-1.5,1.5]",
        aspect=DataAspect())
    xlims!(ax2, -3, 3); ylims!(ax2, -1.5, 1.5)
    contour!(ax2, v1_axis, v2_axis, F;
        levels=peak .* [exp(-0.5), exp(-2.0), exp(-4.5)],
        color=:gray, linewidth=1.2)
    scatter!(ax2, v1s, v2s; markersize=msize*1.5, color=(:steelblue, 0.4), strokewidth=0)
    vlines!(ax2, p.bp1; color=:red, linewidth=0.5, alpha=0.7)
    hlines!(ax2, p.bp2; color=:red, linewidth=0.5, alpha=0.7)
    poly!(ax2,
        Point2f[(-p.σ1, -p.σ2), (p.σ1, -p.σ2), (p.σ1, p.σ2), (-p.σ1, p.σ2)];
        color=(:white, 0.0), strokecolor=:yellow, strokewidth=1.5)
end

Label(fig[0, 1:ncols],
    "Particles N=$N over bp mesh ($(p.suffix))  source=$(basename(csv_path))";
    fontsize=16, tellwidth=false)

out = "scatter_particles_$(tag)_evolution.png"
save(out, fig)
println("Saved $out")
