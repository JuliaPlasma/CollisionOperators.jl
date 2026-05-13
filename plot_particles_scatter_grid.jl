# Grid scatter — full-domain only, square panels in N-col grid for readability.
# Run as:
#   julia --project=. plot_particles_scatter_grid.jl <particle_csv> [preset.jl] [steps] [ncol]
using GLMakie, DelimitedFiles
GLMakie.activate!()
include("Parameters.jl")

length(ARGS) >= 1 || error("usage: plot_particles_scatter_grid.jl <particle_csv> [preset.jl] [steps] [ncol]")
csv_path  = ARGS[1]
preset    = length(ARGS) >= 2 ? ARGS[2] : "parameters_default.jl"
steps_arg = length(ARGS) >= 3 ? ARGS[3] : "0,200,400"
ncol      = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 5
steps = parse.(Int, split(steps_arg, ','))
include(preset)
p = PARAMS

raw, _ = readdlm(csv_path, ',', Any, '\n'; header=true)
available = unique(Int.(raw[:, 1]))
steps = [available[argmin(abs.(available .- s))] for s in steps]
println("Plotting steps: $steps")

f0(v1, v2) = exp(-v1^2/(2p.σ1^2) - v2^2/(2p.σ2^2)) / (2π*p.σ1*p.σ2)
v1_axis = range(p.bp1[1], p.bp1[end]; length=400)
v2_axis = range(p.bp2[1], p.bp2[end]; length=400)
F = [f0(v1, v2) for v1 in v1_axis, v2 in v2_axis]
peak = maximum(F)

mask0 = Int.(raw[:, 1]) .== steps[1]
N = count(mask0)
msize = max(0.6, 4.0 / sqrt(N / 1000))
tag = N >= 30_000 ? "40k" : "10k"

nrow = ceil(Int, length(steps) / ncol)
fig = Figure(; size=(360 * ncol, 360 * nrow + 60))

for (k, s) in enumerate(steps)
    r = div(k - 1, ncol) + 1
    c = mod(k - 1, ncol) + 1
    mask = Int.(raw[:, 1]) .== s
    v1s = Float64.(raw[mask, 4])
    v2s = Float64.(raw[mask, 5])
    t   = Float64(raw[findfirst(mask), 2])

    cnt = zeros(Int, length(p.bp1)-1, length(p.bp2)-1)
    for (a, b) in zip(v1s, v2s)
        i = searchsortedlast(p.bp1, a); j = searchsortedlast(p.bp2, b)
        (1 <= i <= size(cnt,1) && 1 <= j <= size(cnt,2)) || continue
        cnt[i, j] += 1
    end
    n_empty = count(==(0), cnt)

    ax = Axis(fig[r, c];
        xlabel="v₁", ylabel = c==1 ? "v₂" : "",
        title="step=$s  t=$(round(t; digits=3))  empty=$n_empty/$(length(cnt))",
        aspect=DataAspect())
    contour!(ax, v1_axis, v2_axis, F;
        levels=peak .* [exp(-0.5), exp(-2.0), exp(-4.5)],
        color=:gray, linewidth=1.0)
    scatter!(ax, v1s, v2s; markersize=msize, color=(:steelblue, 0.30), strokewidth=0)
    vlines!(ax, p.bp1; color=:red, linewidth=0.3, alpha=0.5)
    hlines!(ax, p.bp2; color=:red, linewidth=0.3, alpha=0.5)
    poly!(ax,
        Point2f[(-p.σ1, -p.σ2), (p.σ1, -p.σ2), (p.σ1, p.σ2), (-p.σ1, p.σ2)];
        color=(:white, 0.0), strokecolor=:yellow, strokewidth=1.0)
    xlims!(ax, p.bp1[1], p.bp1[end]); ylims!(ax, p.bp2[1], p.bp2[end])
end

Label(fig[0, 1:ncol],
    "Particles N=$N — $(basename(csv_path))  (yellow box = ±σ; red = bp mesh)";
    fontsize=16, tellwidth=false)

out = "scatter_particles_$(tag)_grid.png"
save(out, fig)
println("Saved $out")
