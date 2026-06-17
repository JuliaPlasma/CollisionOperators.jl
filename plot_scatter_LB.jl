#! /usr/bin/env -S julia --color=yes --startup-file=no
# Particle scatter of an LB run, to inspect for spikes / honeycomb clustering in
# the raw particle positions (not the projected f_s field). Mesh breakpoints are
# overlaid so any grid-aligned clumping is obvious.
#
#   julia --project=. plot_scatter_LB.jl [suffix] [step]
#
# Defaults: suffix=LB2D_v3, step=last in the snapshot CSV.

using CairoMakie
using DelimitedFiles

suffix = length(ARGS) >= 1 ? ARGS[1] : "LB2D_v3"
want_step = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : -1

# v3 baseline mesh breakpoints — fallback only.
bp1 = [-6.0; -5.0; collect(LinRange(-4.0, 4.0, 17)); 5.0; 6.0]
bp2 = [-6.0; collect(LinRange(-2.5, 2.5, 26)); 6.0]

# Prefer the actual mesh used by this run: fs_snapshot CSVs carry `# bp1=`/`# bp2=`
# header lines, so the overlay matches the run instead of the hardcoded baseline.
let snaps = filter(f -> startswith(f, "fs_snapshot_$(suffix)_step") && endswith(f, ".csv"), readdir())
    if !isempty(snaps)
        hdr = readlines(sort(snaps)[1])
        parse_bp(line) = parse.(Float64, split(strip(line[(findfirst('=', line) + 1):end]), ','))
        i1 = findfirst(l -> startswith(strip(l), "# bp1="), hdr)
        i2 = findfirst(l -> startswith(strip(l), "# bp2="), hdr)
        i1 === nothing || (global bp1 = parse_bp(hdr[i1]))
        i2 === nothing || (global bp2 = parse_bp(hdr[i2]))
        println("Mesh overlay from $(sort(snaps)[1])")
    else
        println("No fs_snapshot for $suffix — using baseline mesh overlay")
    end
end

snap_csv = "particle_snapshots_$(suffix).csv"
isfile(snap_csv) || error("Snapshot CSV not found: $snap_csv")

# CSV header: step,time,particle_idx,v1,v2
data, _ = readdlm(snap_csv, ','; header=true)
steps = Int.(@view data[:, 1])
maxstep = maximum(steps)
step = want_step < 0 ? maxstep : want_step
mask = steps .== step
any(mask) || error("No rows for step=$step (available max=$maxstep)")

v1 = Float64.(data[mask, 4])
v2 = Float64.(data[mask, 5])
N = length(v1)
t = Float64(data[findfirst(mask), 2])

fig = Figure(; size=(1500, 700))

# Full-domain scatter + mesh overlay
ax = Axis(fig[1, 1];
    xlabel="v₁", ylabel="v₂", aspect=DataAspect(),
    title="LB particle scatter  (suffix=$suffix, step=$step, t=$t, N=$N)")
vlines!(ax, bp1; color=(:gray, 0.35), linewidth=0.5)
hlines!(ax, bp2; color=(:gray, 0.35), linewidth=0.5)
scatter!(ax, v1, v2; markersize=2, color=(:navy, 0.25))
xlims!(ax, bp1[1], bp1[end]); ylims!(ax, bp2[1], bp2[end])

# Zoom into the bulk core where spikes/honeycomb would bite physics
ax2 = Axis(fig[1, 2];
    xlabel="v₁", ylabel="v₂", aspect=DataAspect(),
    title="bulk zoom  v₁∈[-4,4], v₂∈[-2.5,2.5]")
vlines!(ax2, bp1; color=(:gray, 0.4), linewidth=0.6)
hlines!(ax2, bp2; color=(:gray, 0.4), linewidth=0.6)
scatter!(ax2, v1, v2; markersize=3, color=(:navy, 0.3))
xlims!(ax2, -4, 4); ylims!(ax2, -2.5, 2.5)

png_name = "scatter_$(suffix)_step$(lpad(step, 5, '0')).png"
save(png_name, fig)
println("Saved $png_name  (N=$N, step=$step, t=$t)")
