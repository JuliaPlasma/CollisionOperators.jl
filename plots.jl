# plots.jl — per-run output artifacts that need the FEM workspace: the f_s
# coefficient snapshot (CSV, for post-processing scripts) and the CairoMakie
# diagnostics PNGs. The only part of the driver that depends on a plotting
# stack; `rclone_upload` comes from io.jl.

using CairoMakie

# Save f_s coefficient vector + breakpoints so post-run scripts can rebuild
# the spline. (CSV chosen for diff-friendliness with the conservation CSV.)
function save_fs_snapshot(ws::Workspace, suffix::String, step::Int,
        f_coeffs::AbstractVector)
    fname = "fs_snapshot_$(suffix)_step$(lpad(step, 4, '0')).csv"
    open(fname, "w") do io
        println(io, "# bp1=", join(ws.bp1, ","))
        println(io, "# bp2=", join(ws.bp2, ","))
        println(io, "# n_dofs=", ws.n_dofs)
        println(io, "coeff")
        for c in f_coeffs
            println(io, c)
        end
    end
    rclone_upload(suffix, fname)
    return fname
end

# Plot 1D slices f_s(v1_fixed, v2) along v₂ at three fixed v₁ values, plus the
# log10|f_s| heatmap with negative-value red overlay. Saves PNG.
function plot_fs_diagnostics(ws::Workspace, f_coeffs::AbstractVector,
        suffix::String, step::Int;
        v1_slices::Vector{Float64} = [0.0, 0.5, 1.0],
        n_v2::Int = 400, n_grid::Int = 200)
    p = ws.p
    f_s = build_field(ws, f_coeffs)

    # 1D slices along v₂
    v2_grid = collect(range(p.bp2[1], p.bp2[end]; length = n_v2))
    fig = Figure(; size = (1300, 900))
    ax_slice = Axis(fig[1, 1:2];
        xlabel = "v₂", ylabel = "f_s",
        title = "f_s slices along v₂  (suffix=$suffix, step=$step)")
    palette = [:blue, :red, :green, :purple]
    for (i, v1f) in enumerate(v1_slices)
        vals = [begin
                    loc = locate_particle(ws, v1f, v2)
                    isnothing(loc) ? 0.0 : (evaluate(ws, f_s, loc)[1][1][1])
                end
                for v2 in v2_grid]
        lines!(ax_slice, v2_grid, vals;
            color = palette[mod1(i, length(palette))],
            linewidth = 2, label = "v₁ = $v1f")
    end
    hlines!(ax_slice, [0.0]; color = :black, linestyle = :dash, linewidth = 1)
    axislegend(ax_slice; position = :rt)

    # log10|f_s| heatmap + negative mask
    v1_grid = collect(range(p.bp1[1], p.bp1[end]; length = n_grid))
    v2_grid_h = collect(range(p.bp2[1], p.bp2[end]; length = n_grid))
    F = evaluate_on_grid(ws, f_s, v1_grid, v2_grid_h)

    F_log = similar(F)
    @inbounds for I in eachindex(F)
        a = abs(F[I])
        F_log[I] = a > 1e-30 ? log10(a) : -30.0
    end
    ax_h = Axis(fig[2, 1];
        xlabel = "v₁", ylabel = "v₂",
        title = "log10|f_s|", aspect = DataAspect())
    hm = heatmap!(ax_h, v1_grid, v2_grid_h, F_log; colormap = :viridis)
    Colorbar(fig[2, 1, Right()], hm)

    neg_mask = map(x -> x < 0.0 ? 1.0 : NaN, F)
    ax_n = Axis(fig[2, 2];
        xlabel = "v₁", ylabel = "v₂",
        title = "negative-region mask (red = f_s < 0)", aspect = DataAspect())
    hm2 = heatmap!(ax_n, v1_grid, v2_grid_h, F_log; colormap = :viridis)
    Colorbar(fig[2, 2, Right()], hm2)
    heatmap!(ax_n, v1_grid, v2_grid_h, neg_mask;
        colormap = [:transparent, :red], colorrange = (0.0, 1.0))

    png_name = "fs_diag_$(suffix)_step$(lpad(step, 4, '0')).png"
    save(png_name, fig)
    println("Saved $png_name")
    return png_name
end

# Per-run quick-look dashboard: conservation, residuals, projection-error,
# negative-part. Main analytical payload is the conservation CSV.
function plot_run_dashboard(ws::Workspace,
        entropy_history, energy_history, momentum_history,
        iter_history, res_history, fp_l2_history,
        neg_history, suffix::String)
    p = ws.p
    steps = 0:p.N_STEPS

    E0 = energy_history[1]
    P0 = momentum_history[1]
    E_err = [abs(energy_history[n + 1] - E0) / abs(E0) for n in steps]
    P_err = [hypot(momentum_history[n + 1][1] - P0[1],
                 momentum_history[n + 1][2] - P0[2]) /
             max(hypot(P0[1], P0[2]), 1e-30) for n in steps]

    fig = Figure(; size = (1200, 1500))

    ax_S = Axis(fig[1, 1]; xlabel = "step", ylabel = "H_h",
        title = "Entropy H_h (monotone increase expected)")
    lines!(ax_S, collect(steps), entropy_history; color = :red, linewidth = 2)

    ax_E = Axis(fig[2, 1]; xlabel = "step", ylabel = "rel. error",
        title = "Energy conservation error", yscale = log10)
    lines!(ax_E, collect(steps), max.(E_err, 1e-18); color = :blue, linewidth = 2)

    ax_P = Axis(fig[3, 1]; xlabel = "step", ylabel = "rel. error",
        title = "Momentum conservation error", yscale = log10)
    lines!(ax_P, collect(steps), max.(P_err, 1e-18); color = :green, linewidth = 2)

    ax_I = Axis(fig[4, 1]; xlabel = "step", ylabel = "iter",
        title = "Inner-iteration count")
    lines!(ax_I, 1:p.N_STEPS, iter_history; color = :black, linewidth = 2)

    ax_R = Axis(fig[5, 1]; xlabel = "step", ylabel = "‖r‖",
        title = "Picard fixed-point residual ‖G(v) − v‖₂", yscale = log10)
    lines!(ax_R, 1:p.N_STEPS, max.(res_history, 1e-30); color = :purple, linewidth = 2)

    ax_F = Axis(fig[6, 1]; xlabel = "step", ylabel = "‖f_s − f_p‖₂",
        title = "Histogram-based projection error  ‖f_s − f_p‖₂")
    lines!(ax_F, 1:p.N_STEPS, fp_l2_history; color = :orange, linewidth = 2)

    ax_N = Axis(fig[7, 1]; xlabel = "step", ylabel = "∫max(−f_s,0)",
        title = "Negative-part L¹ of f_s  (Gibbs oscillation indicator)")
    lines!(ax_N, 1:p.N_STEPS, neg_history; color = :darkred, linewidth = 2)

    png_name = "dashboard_$(suffix).png"
    save(png_name, fig)
    println("Saved $png_name")
    return png_name
end
