# Diagnostic plots for understanding why Anderson stalls in late-step regime.
# Reads conservation_history_<tag>.csv (per-step) + reconstructs f_s from
# fs_snapshot_<tag>_step*.csv to look at coefficient amplitude growth.
#
# Run as:
#   julia --project=. analyze_convergence.jl <tag>
include("MantisWrappers.jl")
using .MantisWrappers
using GLMakie, DelimitedFiles, Statistics
GLMakie.activate!()

length(ARGS) >= 1 || error("usage: analyze_convergence.jl <tag>")
tag = ARGS[1]

cons_path = "conservation_history_$(tag).csv"
raw, _ = readdlm(cons_path, ',', Any, '\n'; header=true)
step = Int.(raw[:, 1])
t    = Float64.(raw[:, 2])
S    = Float64.(raw[:, 3])
E    = Float64.(raw[:, 4])
iter = Int.(raw[:, 7])
nrm_r = Float64.(raw[:, 8])
fpL2  = Float64.(raw[:, 9])
neg   = Float64.(raw[:, 10])
N     = length(step)
println("Loaded $N rows. Last step=$(step[end])")

# tol*‖v‖ threshold for the run. ‖v‖ ≈ sqrt(N_part) * sqrt(σ1² + σ2²).
# Look up N_PARTICLES from a preset file matching the tag, or hard-code 40k.
# Read first fs_snapshot to grab n_dofs; coefficient amplitude tracked later.
fs_files = sort(filter(f -> occursin(Regex("^fs_snapshot_$(tag)_step\\d+\\.csv\$"), f), readdir(".")))
println("Found $(length(fs_files)) fs_snapshots")

# Plain header parser (same as in compute_entropy_from_fs.jl)
function parse_header(path)
    bp1 = Float64[]; bp2 = Float64[]; ndof = 0; data_start = 0
    open(path) do io
        for (i, ln) in enumerate(eachline(io))
            startswith(ln, "# bp1=") && (bp1 = parse.(Float64, split(ln[length("# bp1=")+1:end], ',')))
            startswith(ln, "# bp2=") && (bp2 = parse.(Float64, split(ln[length("# bp2=")+1:end], ',')))
            startswith(ln, "# n_dofs=") && (ndof = parse(Int, ln[length("# n_dofs=")+1:end]))
            if ln == "coeff"; data_start = i + 1; break; end
        end
    end
    bp1, bp2, ndof, data_start
end

function load_coeffs(path)
    bp1, bp2, ndof, ds = parse_header(path)
    coeffs = Vector{Float64}(undef, ndof)
    open(path) do io
        for _ in 1:ds-1; readline(io); end
        for k in 1:ndof; coeffs[k] = parse(Float64, readline(io)); end
    end
    bp1, bp2, coeffs
end

# Per-snapshot coefficient stats
snap_steps = Int[]
coef_max  = Float64[]
coef_min  = Float64[]
coef_neg_frac = Float64[]
coef_l2   = Float64[]
for f in fs_files
    m = match(r"step(\d+)\.csv$", f)
    s = parse(Int, m.captures[1])
    _, _, c = load_coeffs(f)
    push!(snap_steps, s)
    push!(coef_max, maximum(c))
    push!(coef_min, minimum(c))
    push!(coef_neg_frac, count(<(0), c) / length(c))
    push!(coef_l2, sqrt(sum(c .^ 2)))
end

# Detect non-converged steps: iter == max_iter
max_iter = maximum(iter)
non_conv_mask = iter .== max_iter
n_non_conv = count(non_conv_mask)
println("max_iter observed: $max_iter")
println("Non-converged steps: $n_non_conv / $N")
if n_non_conv > 0
    nc_steps = step[non_conv_mask]
    println("First non-conv step: $(nc_steps[1])  Last: $(nc_steps[end])")
end

# Estimated tol threshold (from main_Gonzalez: tol * (‖v‖ + 1e-30))
# Use neg + fpL2 levels as proxy for solution quality
println("Mean ‖r‖ on converged steps: $(mean(nrm_r[.!non_conv_mask]))")
println("Mean ‖r‖ on max-iter steps: $(mean(nrm_r[non_conv_mask]))")

fig = Figure(; size=(1400, 1500))

# Row 1: iter vs step + residual vs step
ax_iter = Axis(fig[1, 1]; xlabel="step", ylabel="iter",
    title="Anderson iter count per step (red = max_iter exhausted)")
scatter!(ax_iter, step, iter; color=ifelse.(non_conv_mask, :red, :steelblue), markersize=4)
lines!(ax_iter, step, iter; color=:gray, linewidth=0.5, alpha=0.4)
hlines!(ax_iter, [max_iter]; color=:red, linestyle=:dash, linewidth=1)

ax_nrm = Axis(fig[1, 2]; xlabel="step", ylabel="‖r‖",
    title="Final residual ‖r‖ per step (log)", yscale=log10)
scatter!(ax_nrm, step, max.(nrm_r, 1e-15);
    color=ifelse.(non_conv_mask, :red, :steelblue), markersize=4)
lines!(ax_nrm, step, max.(nrm_r, 1e-15); color=:gray, linewidth=0.5, alpha=0.4)

# Row 2: neg part vs step + iter
ax_neg = Axis(fig[2, 1]; xlabel="step", ylabel="∫max(−f_s,0)",
    title="Negative-part L¹ — Gibbs growth")
lines!(ax_neg, step, neg; color=:darkred, linewidth=1.5)
non_conv_steps_for_neg = step[non_conv_mask]
if !isempty(non_conv_steps_for_neg)
    vlines!(ax_neg, non_conv_steps_for_neg; color=:red, alpha=0.3, linewidth=0.5)
end

ax_fpL2 = Axis(fig[2, 2]; xlabel="step", ylabel="‖f_p − f_s‖₂",
    title="Projection error ‖f_p − f_s‖₂")
lines!(ax_fpL2, step, fpL2; color=:orange, linewidth=1.5)
if !isempty(non_conv_steps_for_neg)
    vlines!(ax_fpL2, non_conv_steps_for_neg; color=:red, alpha=0.3, linewidth=0.5)
end

# Row 3: coefficient stats
ax_cmax = Axis(fig[3, 1]; xlabel="step", ylabel="amplitude",
    title="Spline coeff range (max & |min|) — diverging = Gibbs")
scatterlines!(ax_cmax, snap_steps, coef_max; color=:blue, marker=:circle, label="max c")
scatterlines!(ax_cmax, snap_steps, abs.(coef_min); color=:red, marker=:diamond, label="|min c|")
axislegend(ax_cmax; position=:lt)

ax_cneg = Axis(fig[3, 2]; xlabel="step", ylabel="fraction",
    title="Fraction of negative coefficients (Gibbs in coef space)")
scatterlines!(ax_cneg, snap_steps, coef_neg_frac; color=:purple, markersize=10)

# Row 4: iter vs neg scatter (does iter spike track Gibbs?)
ax_corr1 = Axis(fig[4, 1]; xlabel="∫max(−f_s,0)", ylabel="iter",
    title="iter vs negative-part L¹  (correlation?)")
scatter!(ax_corr1, neg, iter;
    color=ifelse.(non_conv_mask, :red, :steelblue), markersize=4)

ax_corr2 = Axis(fig[4, 2]; xlabel="step", ylabel="iter",
    title="Cumulative non-conv count")
cum_nc = cumsum(non_conv_mask)
lines!(ax_corr2, step, cum_nc; color=:black, linewidth=2)

# Row 5: zoom on iter near transition
trans_start = max(1, findfirst(non_conv_mask) - 5)
if !isnothing(trans_start) && trans_start > 0
    ax_zoom = Axis(fig[5, 1:2]; xlabel="step", ylabel="iter",
        title="iter zoom near first non-conv step (step $(step[findfirst(non_conv_mask)]))")
    rng = trans_start:N
    scatter!(ax_zoom, step[rng], iter[rng];
        color=ifelse.(non_conv_mask[rng], :red, :steelblue), markersize=8)
    lines!(ax_zoom, step[rng], iter[rng]; color=:gray, linewidth=1, alpha=0.4)
    hlines!(ax_zoom, [max_iter]; color=:red, linestyle=:dash)
end

Label(fig[0, 1:2],
    "Convergence analysis — $(tag)\n" *
    "$(N) steps recorded, $(n_non_conv) hit max_iter=$(max_iter)";
    fontsize=18, tellwidth=false)

out = "convergence_analysis_$(tag).png"
save(out, fig)
println("Saved $out")
