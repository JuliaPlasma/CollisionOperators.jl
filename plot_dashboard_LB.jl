#! /usr/bin/env -S julia --color=yes --startup-file=no
# Standalone run dashboard for a single LB2D run. Reads everything from the
# streamed conservation CSV — no in-memory state, no simulation rerun.
# (Extracted from main_LB.jl's old `plot_run_dashboard`.)
#
#   julia --project=. plot_dashboard_LB.jl <suffix>
#   julia --project=. plot_dashboard_LB.jl LB2D_v3
#
# Reads  conservation_history_<suffix>.csv
# Writes dashboard_<suffix>.png
#
# CSV cols: 1 step,2 time,3 entropy,4 energy,5 mom1,6 mom2,
#           7 iter,8 residual,9 fp_minus_fs,10 neg_part
using GLMakie, DelimitedFiles

function plot_dashboard(suffix::AbstractString)
    csv = "conservation_history_$(suffix).csv"
    isfile(csv) || error("CSV not found: $csv")
    raw, _ = readdlm(csv, ',', Any, '\n'; header=true)

    step    = Int.(raw[:, 1])
    entropy = Float64.(raw[:, 3])
    energy  = Float64.(raw[:, 4])
    mom1    = Float64.(raw[:, 5])
    mom2    = Float64.(raw[:, 6])
    iter    = Float64.(raw[:, 7])
    res     = Float64.(raw[:, 8])
    fp_l2   = Float64.(raw[:, 9])
    neg     = Float64.(raw[:, 10])

    E0 = energy[1]
    P0 = (mom1[1], mom2[1])
    E_err = abs.(energy .- E0) ./ max(abs(E0), 1e-30)
    P_err = hypot.(mom1 .- P0[1], mom2 .- P0[2]) ./ max(hypot(P0[1], P0[2]), 1e-30)

    # iter / residual / projection / neg-part are per-evolution-step (step ≥ 1);
    # the step-0 row holds placeholder zeros for iter/res, so exclude it.
    ev = step .>= 1

    fig = Figure(; size=(1200, 1500))

    ax_S = Axis(fig[1, 1]; xlabel="step", ylabel="H_h",
        title="Entropy H_h = -∫f log f dv  (monotone increase expected)")
    lines!(ax_S, step, entropy; color=:red, linewidth=2)

    ax_E = Axis(fig[2, 1]; xlabel="step", ylabel="rel. error",
        title="Energy conservation error", yscale=log10)
    lines!(ax_E, step, max.(E_err, 1e-18); color=:blue, linewidth=2)

    ax_P = Axis(fig[3, 1]; xlabel="step", ylabel="rel. error",
        title="Momentum conservation error", yscale=log10)
    lines!(ax_P, step, max.(P_err, 1e-18); color=:green, linewidth=2)

    ax_I = Axis(fig[4, 1]; xlabel="step", ylabel="iter", title="Inner-iteration count")
    lines!(ax_I, step[ev], iter[ev]; color=:black, linewidth=2)

    ax_R = Axis(fig[5, 1]; xlabel="step", ylabel="‖r‖",
        title="Picard fixed-point residual ‖G(v) − v‖₂", yscale=log10)
    lines!(ax_R, step[ev], max.(res[ev], 1e-30); color=:purple, linewidth=2)

    ax_F = Axis(fig[6, 1]; xlabel="step", ylabel="‖f_s − f_p‖₂",
        title="Histogram-based projection error  ‖f_s − f_p‖₂")
    lines!(ax_F, step[ev], fp_l2[ev]; color=:orange, linewidth=2)

    ax_N = Axis(fig[7, 1]; xlabel="step", ylabel="∫max(−f_s,0)",
        title="Negative-part L¹ of f_s")
    lines!(ax_N, step[ev], neg[ev]; color=:darkred, linewidth=2)

    png_name = "dashboard_$(suffix).png"
    save(png_name, fig)
    println("Saved $png_name")
    return png_name
end

function main(args=ARGS)
    isempty(args) && error("usage: julia --project=. plot_dashboard_LB.jl <suffix>")
    plot_dashboard(args[1])
end

main()
