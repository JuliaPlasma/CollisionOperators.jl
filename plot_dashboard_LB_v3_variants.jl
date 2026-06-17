#! /usr/bin/env -S julia --color=yes --startup-file=no
# Dashboard comparing the LB2D v3 baseline + its parameter-variant probes,
# mirroring Gonzalez-anderson-fix/plot_dashboard_v3_variants.jl (Landau).
# Rows:
#   1) ‖f_s − f_p‖₂ (projection error) vs step
#   2) ‖G(v) − v‖₂ (Anderson/Picard fixed-point residual) vs step, log scale
#   3) log10|f_s| heatmap + negative-region mask at the final step, rebuilt from
#      the fs_snapshot coefficient CSV (no PNG embed, no sim rerun).
# Columns: one per variant, header labels (P_DEG, bp2_inner shift, seed).
#
#   julia --project=. plot_dashboard_LB_v3_variants.jl
using GLMakie, DelimitedFiles

# Row 5 reconstructs f_s from stored FE coefficients, so it needs the same
# FEM scaffolding the simulation uses.
include("MantisWrappers.jl")
using .MantisWrappers

# (id == suffix). final_step = last fs_diag PNG step written (N_STEPS).
variants = [
    (id="LB2D_v3",       P=2, bp2=26, seed=42, note="baseline"),
    (id="LB2D_v3_shift", P=2, bp2=26, seed=42, note="bp2+0.05"),
    (id="LB2D_v3_pdeg3", P=3, bp2=26, seed=42, note="cubic spline"),
]
final_step = 800
max_step   = 800

# LB conservation CSV cols: 1 step,2 time,3 entropy,4 energy,5 mom1,6 mom2,
#                           7 iter,8 residual,9 fp_minus_fs,10 neg_part
function load_hist(id; max_step=max_step)
    csv = "conservation_history_$(id).csv"
    raw, _ = readdlm(csv, ',', Any, '\n'; header=true)
    step = Int.(raw[:, 1])
    keep = step .<= max_step
    step  = step[keep]
    fp_l2 = Float64.(raw[keep, 9])
    res   = Float64.(raw[keep, 8])
    neg   = Float64.(raw[keep, 10])
    return step, fp_l2, res, neg
end

# fs_snapshot CSV: three `# bp1=`/`# bp2=`/`# n_dofs=` comment lines, a `coeff`
# header, then one FE coefficient per line. (id == suffix; preset file is
# parameters_<id>.jl.) n_grid matches plot_fs_diagnostics in main_LB.jl.
const N_GRID = 200

snapshot_path(id) = "fs_snapshot_$(id)_step$(lpad(final_step, 5, '0')).csv"

function load_coeffs(id)
    csv = snapshot_path(id)
    isfile(csv) || return nothing
    lines = strip.(readlines(csv))
    i = findfirst(==("coeff"), lines)
    i === nothing && error("no 'coeff' header in $csv")
    return parse.(Float64, filter(!isempty, lines[(i + 1):end]))
end

# Rebuild f_s on a 2D grid from the snapshot coefficients (no sim rerun).
function load_fs_grid(id)
    coeffs = load_coeffs(id)
    coeffs === nothing && return nothing
    p = include(joinpath(@__DIR__, "parameters_$(id).jl"))::SimParameters
    ws = build_workspace(p)
    length(coeffs) == ws.n_dofs ||
        error("coeff count $(length(coeffs)) ≠ workspace n_dofs $(ws.n_dofs) for $id")
    field = build_field(ws, coeffs)
    v1g = collect(range(p.bp1[1], p.bp1[end]; length=N_GRID))
    v2g = collect(range(p.bp2[1], p.bp2[end]; length=N_GRID))
    F = evaluate_on_grid(ws, field, v1g, v2g)
    return v1g, v2g, F
end

ncol   = length(variants)
fig    = Figure(; size=(420 * ncol, 1280))
colors = [:steelblue, :darkorange, :crimson]

# Top caption: LB drift form. Plain unicode (MathTeXEngine chokes on \text+\quad).
Label(fig[0, 1:ncol],
    "LB drift:  v̇_α = -ν (∇f_s/f_s + A + B v_α)   (conservative Lenard–Bernstein)";
    fontsize=18, tellwidth=false)

for (k, v) in pairs(variants)
    Label(fig[1, k],
          "$(v.id)\nP_DEG=$(v.P)  bp2_inner=$(v.bp2)  seed=$(v.seed)\n($(v.note))";
          fontsize=12, tellwidth=false)
end

# Row 2: projection error ‖f_s − f_p‖₂(t)
for (k, v) in pairs(variants)
    step, fp_l2, _, _ = load_hist(v.id)
    ax = Axis(fig[2, k];
              xlabel="step",
              ylabel = k==1 ? L"\Vert f_s - f_p \Vert_2" : "",
              title  = L"\Vert f_s - f_p \Vert_2")
    lines!(ax, step, fp_l2; color=colors[k], linewidth=2)
    xlims!(ax, 0, max_step)
end

# Row 3: fixed-point residual ‖G(v) − v‖₂(t), log scale
for (k, v) in pairs(variants)
    step, _, res, _ = load_hist(v.id)
    res_safe = max.(res, 1e-30)
    ax = Axis(fig[3, k];
              xlabel="step",
              ylabel = k==1 ? L"\Vert G(v) - v \Vert_2" : "",
              title  = L"\Vert G(v) - v \Vert_2",
              yscale = log10)
    lines!(ax, step, res_safe; color=colors[k], linewidth=2)
    xlims!(ax, 0, max_step)
end

# Row 4: negative-part L¹ ∫max(−f_s,0)(t) — Gibbs/projection-noise probe
for (k, v) in pairs(variants)
    step, _, _, neg = load_hist(v.id)
    ax = Axis(fig[4, k];
              xlabel="step",
              ylabel = k==1 ? L"\int\max(-f_s,0)\,dv" : "",
              title  = L"\int\max(-f_s,0)\,dv")
    lines!(ax, step, neg; color=colors[k], linewidth=2)
    xlims!(ax, 0, max_step)
end

# Row 5: log10|f_s| heatmap + negative-region mask at final step, redrawn from CSV
for (k, v) in pairs(variants)
    g = load_fs_grid(v.id)
    ax = Axis(fig[5, k]; xlabel="v₁", ylabel = k==1 ? "v₂" : "",
              title="log10|f_s| step=$(final_step)", aspect=DataAspect())
    if g === nothing
        text!(ax, "missing\n$(snapshot_path(v.id))"; position=(0.0, 0.0), align=(:center, :center))
    else
        v1g, v2g, F = g
        F_log = map(x -> abs(x) > 1e-30 ? log10(abs(x)) : -30.0, F)
        neg_mask = map(x -> x < 0.0 ? 1.0 : NaN, F)
        heatmap!(ax, v1g, v2g, F_log; colormap=:viridis)
        heatmap!(ax, v1g, v2g, neg_mask; colormap=[:transparent, :red], colorrange=(0.0, 1.0))
    end
end

rowsize!(fig.layout, 0, Fixed(40))
rowsize!(fig.layout, 1, Fixed(70))

save("dashboard_LB_v3_variants.png", fig)
println("Saved dashboard_LB_v3_variants.png")
