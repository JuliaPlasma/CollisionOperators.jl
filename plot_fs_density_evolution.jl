# Plot f_s(v₁,v₂) heatmap evolution from fs_snapshot_<tag>_step*.csv files.
# Useful when particle CSV missing (e.g. crashed run). bp1/bp2 read from header.
# Run as:
#   julia --project=. plot_fs_density_evolution.jl <tag> [dt] [steps]
# steps = comma-list (default "0,100,200,300").
include("MantisWrappers.jl")
using .MantisWrappers
using GLMakie
GLMakie.activate!()

length(ARGS) >= 1 || error("usage: plot_fs_density_evolution.jl <tag> [dt] [steps]")
tag = ARGS[1]
dt  = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.001
steps = length(ARGS) >= 3 ? parse.(Int, split(ARGS[3], ',')) : [0, 100, 200, 300]

function parse_header(path)
    bp1 = Float64[]; bp2 = Float64[]; ndof = 0; data_start = 0
    open(path) do io
        for (i, ln) in enumerate(eachline(io))
            if startswith(ln, "# bp1=")
                bp1 = parse.(Float64, split(ln[length("# bp1=")+1:end], ','))
            elseif startswith(ln, "# bp2=")
                bp2 = parse.(Float64, split(ln[length("# bp2=")+1:end], ','))
            elseif startswith(ln, "# n_dofs=")
                ndof = parse(Int, ln[length("# n_dofs=")+1:end])
            elseif ln == "coeff"
                data_start = i + 1
                break
            end
        end
    end
    return bp1, bp2, ndof, data_start
end

function load_coeffs(path, ndof, data_start)
    coeffs = Vector{Float64}(undef, ndof)
    open(path) do io
        for _ in 1:data_start-1; readline(io); end
        for k in 1:ndof
            coeffs[k] = parse(Float64, readline(io))
        end
    end
    coeffs
end

# Build workspace from first snapshot
first_path = "fs_snapshot_$(tag)_step$(lpad(steps[1], 4, '0')).csv"
isfile(first_path) || error("missing $first_path")
bp1, bp2, ndof, _ = parse_header(first_path)
p = SimParameters(; bp1=bp1, bp2=bp2)
ws = build_workspace(p)
println("Workspace: n_dofs=$(ws.n_dofs)  n_elements=$(ws.n_elements)")

# Dense eval grid (slightly inside domain to avoid edge NaN)
ε = 1e-6
v1g = range(bp1[1]+ε, bp1[end]-ε; length=300)
v2g = range(bp2[1]+ε, bp2[end]-ε; length=300)

# Precompute all fields + global colorscale
fields = []
for s in steps
    path = "fs_snapshot_$(tag)_step$(lpad(s, 4, '0')).csv"
    isfile(path) || error("missing $path")
    _, _, _, ds = parse_header(path)
    c = load_coeffs(path, ndof, ds)
    fld = build_field(ws, c)
    F = evaluate_on_grid(ws, fld, v1g, v2g)
    push!(fields, (s, F))
    println("step=$s  fmin=$(minimum(F))  fmax=$(maximum(F))  ∫(neg)≈$(sum(max.(-F, 0)) * step(v1g) * step(v2g))")
end

fmax = maximum(maximum(F) for (_, F) in fields)
fmin = minimum(minimum(F) for (_, F) in fields)
println("Global colorscale: [$fmin, $fmax]")

ncol = length(steps)
fig = Figure(; size=(400 * ncol, 800))

# Row 1: full domain
for (k, (s, F)) in enumerate(fields)
    ax = Axis(fig[1, k];
        xlabel="v₁", ylabel = k==1 ? "v₂" : "",
        title="step=$s  t=$(round(s*dt; digits=3))",
        aspect=DataAspect())
    hm = heatmap!(ax, v1g, v2g, F;
        colormap=:viridis, colorrange=(fmin, fmax))
    contour!(ax, v1g, v2g, F;
        levels = fmax .* [0.01, 0.1, 0.5, 0.9],
        color=:white, linewidth=0.8)
    vlines!(ax, bp1; color=:red, linewidth=0.3, alpha=0.5)
    hlines!(ax, bp2; color=:red, linewidth=0.3, alpha=0.5)
    if k == ncol
        Colorbar(fig[1, ncol+1], hm; label="f_s")
    end
end

# Row 2: zoom inner [-3,3] × [-1.5,1.5] + negative-part mask
for (k, (s, F)) in enumerate(fields)
    ax = Axis(fig[2, k];
        xlabel="v₁", ylabel = k==1 ? "v₂" : "",
        title="negative-part (red) over zoom",
        aspect=DataAspect())
    xlims!(ax, -3, 3); ylims!(ax, -1.5, 1.5)
    heatmap!(ax, v1g, v2g, F;
        colormap=:viridis, colorrange=(fmin, fmax))
    # negative regions overlay
    neg = map(x -> x < 0 ? 1.0 : NaN, F)
    heatmap!(ax, v1g, v2g, neg;
        colormap=[:red, :red], colorrange=(0.5, 1.5))
    vlines!(ax, bp1; color=:white, linewidth=0.3, alpha=0.4)
    hlines!(ax, bp2; color=:white, linewidth=0.3, alpha=0.4)
end

Label(fig[0, 1:ncol],
    "f_s density evolution — $(tag)   (red = f_s < 0  Gibbs)";
    fontsize=16, tellwidth=false)

out = "fs_density_evolution_$(tag).png"
save(out, fig)
println("Saved $out")
