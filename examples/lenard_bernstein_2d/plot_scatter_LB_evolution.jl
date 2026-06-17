#! /usr/bin/env -S julia --color=yes --startup-file=no
# Montage of LB particle scatter across ALL snapshot steps in one figure, to see
# the time evolution (initial → intermediate → final) of any banding/clustering.
# Reads particle_snapshots_<suffix>.csv once; one bulk-zoom panel per snap step.
#
#   julia --project=. plot_scatter_LB_evolution.jl [suffix]
#
# Default suffix=LB2D_v3. Also dumps a standalone step-0 (initial-condition) PNG.

using GLMakie
using DelimitedFiles

suffix = length(ARGS) >= 1 ? ARGS[1] : "LB2D_v3"

# mesh breakpoints come straight from the run's preset (single source of truth)
include("Parameters.jl")
include("parameters_$(suffix).jl")

# ---- run descriptor: all output/input strings derived from one suffix --------
struct LBScatterRun
    suffix::String
    snap_csv::String
    montage_png::String
end

LBScatterRun(suffix::AbstractString) =
    LBScatterRun(String(suffix), "particle_snapshots_$(suffix).csv", "scatter_evolution_$(suffix).png")

# cache the parsed CSV so per-panel convert_arguments does not re-read disk
const _CSV_CACHE = Dict{String,Matrix{Float64}}()

function load_snapshots(path::String)
    get!(_CSV_CACHE, path) do
        isfile(path) || error("Snapshot CSV not found: $path")
        data, _ = readdlm(path, ',', Float64; header=true)
        data
    end
end

function step_points(path::String, step::Integer)
    data = load_snapshots(path)
    mask = Int.(@view data[:, 1]) .== step
    return Point2f.(@view(data[mask, 4]), @view(data[mask, 5]))
end

# ---- Makie recipe: one bulk-zoom scatter panel for a (run, step) -------------
# convert_arguments dispatches on the LBScatterRun struct (multiple dispatch),
# turning string-keyed run metadata into the Point2f cloud the recipe draws.
@recipe LBPanel (points,) begin
    markersize = 1.5
    color = (:navy, 0.25)
    meshcolor = (:gray, 0.3)
    meshwidth = 0.4
    bp1 = PARAMS.bp1
    bp2 = PARAMS.bp2
    Makie.mixin_generic_plot_attributes()...
end

Makie.convert_arguments(::Type{<:LBPanel}, run::LBScatterRun, step::Integer) =
    (step_points(run.snap_csv, step),)

function Makie.plot!(p::LBPanel)
    vlines!(p, p.bp1; color=p.meshcolor, linewidth=p.meshwidth)
    hlines!(p, p.bp2; color=p.meshcolor, linewidth=p.meshwidth)
    scatter!(p, p.points; markersize=p.markersize, color=p.color)
    return p
end

# ---- standalone step-0 (initial condition) two-panel figure -----------------
function plot_single(lbrun::LBScatterRun, step::Integer)
    csv_path = lbrun.snap_csv
    data = load_snapshots(csv_path)
    i = findfirst(==(step), Int.(@view data[:, 1]))
    t = data[i, 2]
    N = count(==(step), Int.(@view data[:, 1]))
    fig = Figure(; size=(1500, 700))
    ax = Axis(
        fig[1, 1];
        xlabel="v₁",
        ylabel="v₂",
        aspect=DataAspect(),
        title="LB scatter  (suffix=$(lbrun.suffix), step=$step, t=$t, N=$N)",
    )
    p = lbpanel!(ax, lbrun, step; markersize=2, color=(:navy, 0.25), meshwidth=0.5)
    ax2 = Axis(
        fig[1, 2]; xlabel="v₁", ylabel="v₂", aspect=DataAspect(), title="bulk zoom  v₁∈[-4,4], v₂∈[-2.5,2.5]"
    )
    lbpanel!(ax2, lbrun, step; markersize=3, color=(:navy, 0.3), meshwidth=0.6)
    xlims!(ax2, -4, 4)
    ylims!(ax2, -2.5, 2.5)
    name = "scatter_$(lbrun.suffix)_step$(lpad(step, 5, '0')).png"
    save(name, fig)
    println("Saved $name  (N=$N)")
    return fig
end

# ---- evolution montage: bulk-zoom panel per snapshot step -------------------
function plot_evolution(lbrun::LBScatterRun)
    data = load_snapshots(lbrun.snap_csv)
    steps_all = Int.(@view data[:, 1])
    uniq_steps = sort(unique(steps_all))
    println("Snapshot steps: ", uniq_steps)
    nstep = length(uniq_steps)
    ncol = 5
    nrow = cld(nstep, ncol)
    fig = Figure(; size=(300 * ncol, 300 * nrow))
    Label(
        fig[0, 1:ncol],
        "LB particle-scatter evolution  (suffix=$(lbrun.suffix), bulk zoom v₁∈[-4,4] v₂∈[-2.5,2.5])";
        fontsize=18,
        tellwidth=false,
    )
    for (i, step) in enumerate(uniq_steps)
        r = cld(i, ncol)
        c = mod1(i, ncol)
        t = data[findfirst(==(step), steps_all), 2]
        ax = Axis(fig[r, c]; aspect=DataAspect(), title="step=$step  t=$(round(t; digits=3))", titlesize=12)
        lbpanel!(ax, lbrun, step)
        xlims!(ax, -4, 4)
        ylims!(ax, -2.5, 2.5)
        hidedecorations!(ax; label=false)
    end
    rowsize!(fig.layout, 0, Fixed(40))
    save(lbrun.montage_png, fig)
    println("Saved $(lbrun.montage_png)  ($nstep panels)")
    return fig
end

lbrun = LBScatterRun(suffix)

plot_single(lbrun, 0)    # initial-condition standalone
plot_evolution(lbrun)
