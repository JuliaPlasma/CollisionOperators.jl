#! /usr/bin/env -S julia --color=yes --startup-file=no
# -*- coding: utf-8 -*-
#
# Gonzalez discrete-gradient time integration of the Landau collision
# operator. Refactored entry point with CLI configuration:
#
#   julia --project=. main_Gonzalez.jl parameters_default.jl
#   julia --project=. main_Gonzalez.jl parameters_picard.jl
#   julia --project=. main_Gonzalez.jl parameters_default.jl \
#         --N_STEPS=200 --use_anderson=false --suffix=picard_short
#
# ARGS[1]   : path to a preset file that defines `PARAMS::SimParameters`
# ARGS[2:]  : zero or more `--key=value` overrides applied on top of PARAMS
#
# Diagnostics pushed to CSV every step:
#   iter        : Picard iterations used
#   residual    : ‖G(v) − v‖₂ at the converged iterate
#   fp_minus_fs : ‖f_s − f_p‖₂  (histogram-based projection-error norm)
#   neg_part    : ∫ max(−f_s, 0) dv  (Gibbs negative-part L¹)

include("MantisWrappers.jl")
using .MantisWrappers

using GLMakie
using Random
using LinearAlgebra
using LinearAlgebra: ldiv!, mul!


# Compute ∂S_h/∂v_α = -w_α G_α for every particle. Workspace-aware.
function compute_entropy_gradient!(ws::Workspace, dS, v_parts, w_parts,
                                    f_coeffs_buf, r_vec, L_vec, G_buf)
    l2_project!(ws, f_coeffs_buf, v_parts, w_parts)
    f_s = build_field(ws, f_coeffs_buf)
    compute_r!(ws, r_vec, f_s)
    ldiv!(L_vec, ws.M_lu, r_vec)
    compute_G!(ws, G_buf, v_parts, L_vec)
    @inbounds for α in axes(v_parts, 1)
        dS[α, 1] = -w_parts[α] * G_buf[α, 1]
        dS[α, 2] = -w_parts[α] * G_buf[α, 2]
    end
    return nothing
end

# One Picard map: v_out = v0 + dt · G̃(v_mid) · ∇̄S
function picard_map!(ws::Workspace, v_out, v_in, v0, w_parts, S0, dt,
                     v_mid, dv, dS_mid, G_eff, dot_v_buf, f_buf,
                     r_vec, L_vec, G_buf)
    N = size(v0, 1)
    @. v_mid = 0.5 * (v0 + v_in)
    @. dv    = v_in - v0

    compute_entropy_gradient!(ws, dS_mid, v_mid, w_parts,
                               f_buf, r_vec, L_vec, G_buf)
    l2_project!(ws, f_buf, v_in, w_parts)
    S1 = compute_entropy(ws, build_field(ws, f_buf))

    dot_dv_dS = 0.0
    nrm2_dv   = 0.0
    @inbounds for α in 1:N
        dot_dv_dS += dv[α, 1] * dS_mid[α, 1] + dv[α, 2] * dS_mid[α, 2]
        nrm2_dv   += dv[α, 1]^2               + dv[α, 2]^2
    end
    correction = nrm2_dv > 1e-30 ? (S1 - S0 - dot_dv_dS) / nrm2_dv : 0.0

    @inbounds for α in 1:N
        inv_w = 1.0 / w_parts[α]
        G_eff[α, 1] = -(dS_mid[α, 1] + correction * dv[α, 1]) * inv_w
        G_eff[α, 2] = -(dS_mid[α, 2] + correction * dv[α, 2]) * inv_w
    end

    compute_collision!(ws, dot_v_buf, v_mid, w_parts, G_eff)
    @. v_out = v0 + dt * dot_v_buf
    return nothing
end


# Anderson-accelerated fixed-point iteration. `use_anderson=false` falls back
# to plain damped Picard.
function step_anderson!(ws::Workspace,
                        v1, v0, w_parts, S0, dt,
                        v_mid, dv, dS_mid, G_eff, dot_v_buf, f_buf,
                        r_vec, L_vec, G_buf,
                        Gv, r_curr, r_prev, Gv_prev, v_old, ΔF, ΔG;
                        m=5, max_iter=1000, tol=1e-12,
                        restart_factor=Inf, damping=0.5,
                        reg_factor=1e-10, verbose=false,
                        use_anderson::Bool=true)
    v1_v   = vec(v1)
    Gv_v   = vec(Gv)
    r_v    = vec(r_curr)
    rp_v   = vec(r_prev)
    Gp_v   = vec(Gv_prev)
    vold_v = vec(v_old)

    history = 0
    nrm_r0  = 0.0
    nrm_r   = 0.0
    nrm_best = Inf
    n_restart = 0

    for k in 1:max_iter
        vold_v .= v1_v
        picard_map!(ws, Gv, v1, v0, w_parts, S0, dt,
                    v_mid, dv, dS_mid, G_eff, dot_v_buf, f_buf,
                    r_vec, L_vec, G_buf)
        @. r_v = Gv_v - v1_v
        nrm_r = norm(r_v)
        k == 1 && (nrm_r0 = nrm_r)

        if nrm_r < tol * (norm(v1_v) + 1e-30)
            v1 .= Gv
            verbose && println("    k=$k  ‖r‖=$nrm_r  history=$history  [converged]")
            return k, nrm_r, n_restart
        end

        just_restarted = false
        if k > 1 && nrm_r > restart_factor * nrm_best
            history = 0
            n_restart += 1
            just_restarted = true
        end
        nrm_r < nrm_best && (nrm_best = nrm_r)

        verbose && println("    k=$k  ‖r‖=$nrm_r  history=$history" *
                           (just_restarted ? "  [restart]" : ""))

        if k == 1 || just_restarted || !use_anderson
            @. v1_v = damping * Gv_v + (1 - damping) * vold_v
        else
            if history < m
                history += 1
                new_col = history
            else
                @views ΔF[:, 1:m-1] .= ΔF[:, 2:m]
                @views ΔG[:, 1:m-1] .= ΔG[:, 2:m]
                new_col = m
            end
            @views ΔF[:, new_col] .= r_v  .- rp_v
            @views ΔG[:, new_col] .= Gv_v .- Gp_v

            ΔFv = @view ΔF[:, 1:history]
            ΔGv = @view ΔG[:, 1:history]
            ATA = ΔFv' * ΔFv
            ATr = ΔFv' * r_v
            diag_mean = 0.0
            @inbounds for j in 1:history
                diag_mean += ATA[j, j]
            end
            diag_mean /= history
            λ2 = reg_factor * diag_mean + 1e-30
            @inbounds for j in 1:history
                ATA[j, j] += λ2
            end
            γ = ATA \ ATr
            v1 .= Gv
            mul!(v1_v, ΔGv, γ, -1.0, 1.0)
            @. v1_v = damping * v1_v + (1 - damping) * vold_v
        end

        rp_v .= r_v
        Gp_v .= Gv_v
    end

    @warn "Solver did not converge" max_iter tol nrm_r0 nrm_r nrm_best n_restart
    return max_iter, nrm_r, n_restart
end


function compute_momentum(v_parts, w_parts)
    p1 = sum(w_parts[α] * v_parts[α, 1] for α in axes(v_parts, 1))
    p2 = sum(w_parts[α] * v_parts[α, 2] for α in axes(v_parts, 1))
    return (p1, p2)
end

function compute_energy(v_parts, w_parts)
    return 0.5 * sum(w_parts[α] * (v_parts[α, 1]^2 + v_parts[α, 2]^2)
                     for α in axes(v_parts, 1))
end


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
    return fname
end


# Plot 1D slices f_s(v1_fixed, v2) along v₂ at three fixed v₁ values, plus the
# log10|f_s| heatmap with negative-value red overlay. Saves PNG.
function plot_fs_diagnostics(ws::Workspace, f_coeffs::AbstractVector,
                              suffix::String, step::Int;
                              v1_slices::Vector{Float64}=[0.0, 0.5, 1.0],
                              n_v2::Int=400, n_grid::Int=200)
    p = ws.p
    f_s = build_field(ws, f_coeffs)

    # 1D slices along v₂
    v2_grid = collect(range(p.bp2[1], p.bp2[end]; length=n_v2))
    fig = Figure(; size=(1300, 900))
    ax_slice = Axis(fig[1, 1:2];
        xlabel="v₂", ylabel="f_s",
        title="f_s slices along v₂  (suffix=$suffix, step=$step)")
    palette = [:blue, :red, :green, :purple]
    for (i, v1f) in enumerate(v1_slices)
        vals = [begin
                    loc = locate_particle(ws, v1f, v2)
                    isnothing(loc) ? 0.0 : (evaluate(ws, f_s, loc)[1][1][1])
                end for v2 in v2_grid]
        lines!(ax_slice, v2_grid, vals;
            color=palette[mod1(i, length(palette))],
            linewidth=2, label="v₁ = $v1f")
    end
    hlines!(ax_slice, [0.0]; color=:black, linestyle=:dash, linewidth=1)
    axislegend(ax_slice; position=:rt)

    # log10|f_s| heatmap + negative mask
    v1_grid   = collect(range(p.bp1[1], p.bp1[end]; length=n_grid))
    v2_grid_h = collect(range(p.bp2[1], p.bp2[end]; length=n_grid))
    F = evaluate_on_grid(ws, f_s, v1_grid, v2_grid_h)

    F_log = similar(F)
    @inbounds for I in eachindex(F)
        a = abs(F[I])
        F_log[I] = a > 1e-30 ? log10(a) : -30.0
    end
    ax_h = Axis(fig[2, 1];
        xlabel="v₁", ylabel="v₂",
        title="log10|f_s|", aspect=DataAspect())
    hm = heatmap!(ax_h, v1_grid, v2_grid_h, F_log; colormap=:viridis)
    Colorbar(fig[2, 1, Right()], hm)

    neg_mask = map(x -> x < 0.0 ? 1.0 : NaN, F)
    ax_n = Axis(fig[2, 2];
        xlabel="v₁", ylabel="v₂",
        title="negative-region mask (red = f_s < 0)", aspect=DataAspect())
    hm2 = heatmap!(ax_n, v1_grid, v2_grid_h, F_log; colormap=:viridis)
    Colorbar(fig[2, 2, Right()], hm2)
    heatmap!(ax_n, v1_grid, v2_grid_h, neg_mask;
        colormap=[:transparent, :red], colorrange=(0.0, 1.0))

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
    E_err = [abs(energy_history[n+1] - E0) / abs(E0) for n in steps]
    P_err = [hypot(momentum_history[n+1][1] - P0[1],
                   momentum_history[n+1][2] - P0[2]) /
             max(hypot(P0[1], P0[2]), 1e-30) for n in steps]

    fig = Figure(; size=(1200, 1500))

    ax_S = Axis(fig[1, 1]; xlabel="step", ylabel="H_h",
        title="Entropy H_h (monotone increase expected)")
    lines!(ax_S, collect(steps), entropy_history; color=:red, linewidth=2)

    ax_E = Axis(fig[2, 1]; xlabel="step", ylabel="rel. error",
        title="Energy conservation error", yscale=log10)
    lines!(ax_E, collect(steps), max.(E_err, 1e-18); color=:blue, linewidth=2)

    ax_P = Axis(fig[3, 1]; xlabel="step", ylabel="rel. error",
        title="Momentum conservation error", yscale=log10)
    lines!(ax_P, collect(steps), max.(P_err, 1e-18); color=:green, linewidth=2)

    ax_I = Axis(fig[4, 1]; xlabel="step", ylabel="iter",
        title="Inner-iteration count")
    lines!(ax_I, 1:p.N_STEPS, iter_history; color=:black, linewidth=2)

    ax_R = Axis(fig[5, 1]; xlabel="step", ylabel="‖r‖",
        title="Picard fixed-point residual ‖G(v) − v‖₂", yscale=log10)
    lines!(ax_R, 1:p.N_STEPS, max.(res_history, 1e-30); color=:purple, linewidth=2)

    ax_F = Axis(fig[6, 1]; xlabel="step", ylabel="‖f_s − f_p‖₂",
        title="Histogram-based projection error  ‖f_s − f_p‖₂")
    lines!(ax_F, 1:p.N_STEPS, fp_l2_history; color=:orange, linewidth=2)

    ax_N = Axis(fig[7, 1]; xlabel="step", ylabel="∫max(−f_s,0)",
        title="Negative-part L¹ of f_s  (Gibbs oscillation indicator)")
    lines!(ax_N, 1:p.N_STEPS, neg_history; color=:darkred, linewidth=2)

    png_name = "dashboard_$(suffix).png"
    save(png_name, fig)
    println("Saved $png_name")
    return png_name
end


function run_simulation(p::SimParameters)
    print_summary(p)
    Random.seed!(p.seed)

    ws = build_workspace(p)
    println("Workspace: n_dofs=$(ws.n_dofs)  n_elements=$(ws.n_elements)")

    v_particles = zeros(p.N_PARTICLES, 2)
    v_particles[:, 1] .= p.σ1 * randn(p.N_PARTICLES)
    v_particles[:, 2] .= p.σ2 * randn(p.N_PARTICLES)
    w_particles = fill(1.0 / p.N_PARTICLES, p.N_PARTICLES)
    f_coeffs    = zeros(ws.n_dofs)

    l2_project!(ws, f_coeffs, v_particles, w_particles)
    f_s = build_field(ws, f_coeffs)

    entropy_history  = Float64[]
    energy_history   = Float64[]
    momentum_history = NTuple{2, Float64}[]
    iter_history     = Int[]
    res_history      = Float64[]
    fp_l2_history    = Float64[]
    neg_history      = Float64[]

    push!(entropy_history,  compute_entropy(ws, f_s))
    push!(energy_history,   compute_energy(v_particles, w_particles))
    push!(momentum_history, compute_momentum(v_particles, w_particles))
    println("Initial  S_h = $(entropy_history[end])")
    println("Initial  E   = $(energy_history[end])")
    println("Initial  P   = $(momentum_history[end])")
    println("Initial  ‖f_s − f_p‖₂ = " *
            string(compute_fs_minus_fp_l2(ws, f_s, v_particles, w_particles)))
    println("Initial  ∫max(−f_s,0) = " *
            string(compute_negative_part_l1(ws, f_s)))

    r_vec = zeros(ws.n_dofs)
    L_vec = zeros(ws.n_dofs)
    G     = zeros(p.N_PARTICLES, 2)
    dot_v = zeros(p.N_PARTICLES, 2)
    v1    = copy(v_particles)

    v_mid  = similar(v_particles)
    dv     = similar(v_particles)
    dS_mid = zeros(p.N_PARTICLES, 2)
    G_eff  = zeros(p.N_PARTICLES, 2)
    f_buf  = zeros(ws.n_dofs)

    Gv         = zeros(p.N_PARTICLES, 2)
    r_curr     = zeros(p.N_PARTICLES, 2)
    r_prev     = zeros(p.N_PARTICLES, 2)
    Gv_prev    = zeros(p.N_PARTICLES, 2)
    v_old_buf  = zeros(p.N_PARTICLES, 2)
    ΔF         = zeros(2 * p.N_PARTICLES, p.m_anderson)
    ΔG         = zeros(2 * p.N_PARTICLES, p.m_anderson)

    snapshot_steps = Set([0, p.N_STEPS ÷ 4, p.N_STEPS ÷ 2,
                          (3 * p.N_STEPS) ÷ 4, p.N_STEPS])
    snapshots_v = Dict{Int, Matrix{Float64}}()
    snapshots_v[0] = copy(v_particles)
    save_fs_snapshot(ws, p.suffix, 0, f_coeffs)
    plot_fs_diagnostics(ws, f_coeffs, p.suffix, 0)

    for step in 1:p.N_STEPS
        S0 = entropy_history[end]

        compute_r!(ws, r_vec, f_s)
        ldiv!(L_vec, ws.M_lu, r_vec)
        compute_G!(ws, G, v_particles, L_vec)
        compute_collision!(ws, dot_v, v_particles, w_particles, G)
        @. v1 = v_particles + p.DT * dot_v

        iter, res_final, n_rs = step_anderson!(ws,
                              v1, v_particles, w_particles, S0, p.DT,
                              v_mid, dv, dS_mid, G_eff, dot_v, f_buf,
                              r_vec, L_vec, G,
                              Gv, r_curr, r_prev, Gv_prev, v_old_buf, ΔF, ΔG;
                              m=p.m_anderson, max_iter=p.max_iter, tol=p.tol,
                              damping=p.damping, use_anderson=p.use_anderson,
                              verbose=(step <= 3))
        v_particles .= v1

        l2_project!(ws, f_coeffs, v_particles, w_particles)
        f_s = build_field(ws, f_coeffs)

        push!(entropy_history,  compute_entropy(ws, f_s))
        push!(energy_history,   compute_energy(v_particles, w_particles))
        push!(momentum_history, compute_momentum(v_particles, w_particles))
        push!(iter_history,     iter)
        push!(res_history,      res_final)
        push!(fp_l2_history,    compute_fs_minus_fp_l2(ws, f_s, v_particles, w_particles))
        push!(neg_history,      compute_negative_part_l1(ws, f_s))

        if step in snapshot_steps
            snapshots_v[step] = copy(v_particles)
            save_fs_snapshot(ws, p.suffix, step, f_coeffs)
            plot_fs_diagnostics(ws, f_coeffs, p.suffix, step)
        end

        step % 25 == 0 &&
            println("Step $step/$(p.N_STEPS)  iter=$iter  rs=$n_rs" *
                    "  ‖r‖=$(round(res_final; sigdigits=3))" *
                    "  ‖f_s−f_p‖=$(round(fp_l2_history[end]; sigdigits=4))" *
                    "  neg=$(round(neg_history[end]; sigdigits=4))" *
                    "  S=$(round(entropy_history[end]; digits=6))" *
                    "  E=$(round(energy_history[end]; digits=8))")
    end

    cons_csv = "conservation_history_$(p.suffix).csv"
    open(cons_csv, "w") do io
        println(io, "step,time,entropy,energy,momentum_1,momentum_2," *
                    "iter,residual,fp_minus_fs,neg_part")
        for n in 0:p.N_STEPS
            t  = n * p.DT
            S  = entropy_history[n+1]
            E  = energy_history[n+1]
            P1 = momentum_history[n+1][1]
            P2 = momentum_history[n+1][2]
            it = n == 0 ? 0   : iter_history[n]
            rs = n == 0 ? 0.0 : res_history[n]
            fl = n == 0 ? 0.0 : fp_l2_history[n]
            nv = n == 0 ? 0.0 : neg_history[n]
            println(io, "$n,$t,$S,$E,$P1,$P2,$it,$rs,$fl,$nv")
        end
    end
    println("Saved $cons_csv")

    snap_csv = "particle_snapshots_$(p.suffix).csv"
    open(snap_csv, "w") do io
        println(io, "step,time,particle_idx,v1,v2")
        for s in sort(collect(keys(snapshots_v)))
            pts = snapshots_v[s]
            t   = s * p.DT
            for i in axes(pts, 1)
                println(io, "$s,$t,$i,$(pts[i, 1]),$(pts[i, 2])")
            end
        end
    end
    println("Saved $snap_csv")

    plot_run_dashboard(ws,
                       entropy_history, energy_history, momentum_history,
                       iter_history, res_history, fp_l2_history,
                       neg_history, p.suffix)

    return (; entropy_history, energy_history, momentum_history,
              iter_history, res_history, fp_l2_history, neg_history,
              snapshots_v,
              label=(p.use_anderson ? "Anderson(m=$(p.m_anderson))" : "Picard"))
end


function main(args=ARGS)
    if isempty(args)
        preset = "parameters_default.jl"
        overrides = String[]
    else
        preset = args[1]
        overrides = collect(String, args[2:end])
    end
    isfile(preset) || error("Preset file not found: $preset")

    println("Loading preset: $preset")
    # The preset file ends by binding PARAMS = SimParameters(...). `include`
    # returns the last expression's value, so we capture PARAMS without
    # relying on module globals.
    params_loaded = include(joinpath(@__DIR__, preset))
    p = parse_overrides(params_loaded::SimParameters, overrides)

    res = run_simulation(p)
    a = sum(res.iter_history) / length(res.iter_history)
    println("\n--- Inner-iter summary ---")
    println("$(res.label):  avg=$(round(a; digits=2))  max=$(maximum(res.iter_history))" *
            "  steps=$(length(res.iter_history))")
end

main()
