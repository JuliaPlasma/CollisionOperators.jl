#! /usr/bin/env -S julia --color=yes --startup-file=no
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Time integration via the Gonzalez discrete gradient method.
# Implements Equations (59)–(61) of Jeyakumar et al.:
#
#   (v^{n+1} - v^n)/Δt = G̃_γσ(v_mid) ∇̄_σ S_h(v^n, v^{n+1})          (60)
#
# where the midpoint approximation for the dissipation matrix is (Eq. 61):
#   G̃_γσ(v_n, v_{n+1}) = G_γσ((v_n + v_{n+1})/2)
#
# and the Gonzalez discrete gradient (Eq. 59) for S_h is:
#   ∇̄S_h(z_n, z_{n+1}) = ∇S_h(z_{mid})
#       + (z_{n+1} - z_n) * [S_h(z_{n+1}) - S_h(z_n) - (z_{n+1}-z_n)·∇S_h(z_mid)]
#                            / |z_{n+1} - z_n|²
#
# This guarantees exact discrete conservation of momentum and energy, and
# monotone dissipation of entropy (discrete H-theorem), Sections 4.1–4.2.
#
# The implicit equation is solved by Anderson-accelerated fixed-point
# iteration: a windowed least-squares mixing of the most recent `m` Picard
# residuals. The first iteration is plain Picard; subsequent iterations
# blend history to achieve superlinear convergence at the cost of one small
# dense LSQ solve per iteration. Conservation is exact only at the implicit
# solution, so a tight tolerance (≈1e-13) is required to expose the
# structure-preserving property of the discrete gradient.

include("MantisWrappers.jl")
using .MantisWrappers
using GLMakie
using Random
using LinearAlgebra

include("parameters.jl")


# Compute ∂S_h/∂v_α = -w_α G_α for every particle at positions v_parts.
# Uses Eq. (38): ∂S_h/∂v_α = -Σ_k w_α L_k ∇φ_k(v_α).
# All scratch (`f_coeffs_buf`, `r_vec`, `L_vec`, `G_buf`) is caller-owned.
function compute_entropy_gradient!(dS, v_parts, w_parts,
                                    f_coeffs_buf, r_vec, L_vec, G_buf)
    l2_project!(f_coeffs_buf, v_parts, w_parts)
    f_s = build_field(f_coeffs_buf)
    compute_r!(r_vec, f_s)
    ldiv!(L_vec, M_lu, r_vec)
    compute_G!(G_buf, v_parts, L_vec)
    @inbounds for α in axes(v_parts, 1)
        dS[α, 1] = -w_parts[α] * G_buf[α, 1]
        dS[α, 2] = -w_parts[α] * G_buf[α, 2]
    end
    return nothing
end


# One Picard map: given current iterate v_in, write v_out = v0 + dt·G̃(v_mid)·∇̄S
# implementing the right-hand side of Eq. (60) with the Gonzalez discrete
# gradient (Eq. 59) and the midpoint dissipation matrix (Eq. 61).
function picard_map!(v_out, v_in, v0, w_parts, S0, dt,
                     v_mid, dv, dS_mid, G_eff, dot_v_buf, f_buf,
                     r_vec, L_vec, G_buf)
    N = size(v0, 1)
    @. v_mid = 0.5 * (v0 + v_in)
    @. dv    = v_in - v0

    # 1. ∂S_h/∂v_α at the midpoint
    compute_entropy_gradient!(dS_mid, v_mid, w_parts,
                               f_buf, r_vec, L_vec, G_buf)

    # 2. S_h(v_in) for the Gonzalez correction scalar
    l2_project!(f_buf, v_in, w_parts)
    S1 = compute_entropy(build_field(f_buf))

    # 3. Gonzalez scalar correction (Eq. 59)
    dot_dv_dS = 0.0
    nrm2_dv   = 0.0
    @inbounds for α in 1:N
        dot_dv_dS += dv[α, 1] * dS_mid[α, 1] + dv[α, 2] * dS_mid[α, 2]
        nrm2_dv   += dv[α, 1]^2               + dv[α, 2]^2
    end
    correction = nrm2_dv > 1e-30 ? (S1 - S0 - dot_dv_dS) / nrm2_dv : 0.0

    # 4. Discrete gradient mapped to the G-convention used by compute_collision!:
    #    G_eff[α] = -(∇̄S_h,α) / w_α
    @inbounds for α in 1:N
        inv_w = 1.0 / w_parts[α]
        G_eff[α, 1] = -(dS_mid[α, 1] + correction * dv[α, 1]) * inv_w
        G_eff[α, 2] = -(dS_mid[α, 2] + correction * dv[α, 2]) * inv_w
    end

    # 5. Apply midpoint dissipation matrix G̃(v_mid) and form the update
    compute_collision!(dot_v_buf, v_mid, w_parts, G_eff)
    @. v_out = v0 + dt * dot_v_buf
    return nothing
end


# Anderson-accelerated fixed-point iteration for one Gonzalez time step.
#
# Let G(v) be the Picard map (Eq. 60 RHS) and r(v) = G(v) - v its residual.
# At iteration k we form
#     ΔF_k = [r_{k-m+1}-r_{k-m}, …, r_k-r_{k-1}]   (2N × m_k)
#     ΔG_k = [G_{k-m+1}-G_{k-m}, …, G_k-G_{k-1}]   (2N × m_k)
# solve the small LSQ
#     γ_k = argmin_γ ‖r_k − ΔF_k γ‖₂
# and update
#     v_{k+1} = G(v_k) − ΔG_k γ_k.
# k=1 reduces to plain Picard (no history yet).
#
# Convergence criterion: ‖r_k‖ < tol·‖v_k‖. A strict tol is needed because
# Gonzalez conservation is exact only at the fixed point.
#
# Returns the iteration count actually used.
function step_anderson!(v1, v0, w_parts, S0, dt,
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
        vold_v .= v1_v                     # save v_in for damped mixing
        picard_map!(Gv, v1, v0, w_parts, S0, dt,
                    v_mid, dv, dS_mid, G_eff, dot_v_buf, f_buf,
                    r_vec, L_vec, G_buf)
        @. r_v = Gv_v - v1_v
        nrm_r = norm(r_v)
        k == 1 && (nrm_r0 = nrm_r)

        if nrm_r < tol * (norm(v1_v) + 1e-30)
            v1 .= Gv
            verbose && println("    Anderson k=$k  ‖r‖=$nrm_r  history=$history  [converged]")
            return k, nrm_r, n_restart
        end

        # Restart safeguard: if residual blew past best by safety factor, drop
        # the history and take one plain Picard step on this iteration. The
        # next iteration starts rebuilding the window from a clean (rp_v,Gp_v)
        # pair saved at the bottom of this loop body.
        just_restarted = false
        if k > 1 && nrm_r > restart_factor * nrm_best
            history = 0
            n_restart += 1
            just_restarted = true
        end
        nrm_r < nrm_best && (nrm_best = nrm_r)

        verbose && println("    Anderson k=$k  ‖r‖=$nrm_r  history=$history" *
                           (just_restarted ? "  [restart]" : ""))

        if k == 1 || just_restarted || !use_anderson
            # (damped) Picard: v1 ← β·G(v_old) + (1−β)·v_old
            # damping=1.0 ⇒ plain Picard. Used when use_anderson=false.
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
            # Tikhonov-regularized normal equations:
            #   γ = (ΔFᵀΔF + λ²I)⁻¹ ΔFᵀ r
            # λ² is scaled to the mean diagonal of ΔFᵀΔF so the regularizer
            # tracks the column scale of ΔF (not ‖r‖, which can underflow
            # the regularizer to zero as Anderson converges).
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
            # u_anderson = G(v_old) − ΔG·γ;  put into v1 first
            v1 .= Gv
            mul!(v1_v, ΔGv, γ, -1.0, 1.0)  # v1 ← Gv − ΔG·γ = u_anderson
            # damped mixing: v1 ← β·u_anderson + (1−β)·v_old
            @. v1_v = damping * v1_v + (1 - damping) * vold_v
        end

        rp_v .= r_v
        Gp_v .= Gv_v
    end

    @warn "Anderson did not converge" max_iter tol nrm_r0 nrm_r nrm_best n_restart
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


function run_simulation(; use_anderson::Bool, suffix::String,
                          damping::Float64, m_anderson::Int=5,
                          tol::Float64=1e-12, max_iter::Int=1000)
    label = use_anderson ? "Anderson(m=$m_anderson)" : "plain Picard"
    println("\n========================================")
    println("=== Run: $label  damping=$damping  tol=$tol  max_iter=$max_iter")
    println("========================================")
    Random.seed!(42)
    v_particles = zeros(N_PARTICLES, 2)
    # Anisotropic Gaussian; σ₁,σ₂ chosen so 3σ ≈ I_MAX, i.e. ~99.7% of
    # particles sit inside the fine-mesh interior [I_MIN, I_MAX]², the
    # remaining ~0.3% tail spills into the coarse outer buffer that
    # extends to V_MIN/V_MAX.
    v_particles[:, 1] .= σ₁ * randn(N_PARTICLES)
    v_particles[:, 2] .= σ₂ * randn(N_PARTICLES)
    w_particles = fill(1.0 / N_PARTICLES, N_PARTICLES)
    f_coeffs    = zeros(n_dofs)

    l2_project!(f_coeffs, v_particles, w_particles)
    f_s = build_field(f_coeffs)

    entropy_history  = Float64[]
    energy_history   = Float64[]
    momentum_history = NTuple{2, Float64}[]
    iter_history     = Int[]
    res_history      = Float64[]

    push!(entropy_history,  compute_entropy(f_s))
    push!(energy_history,   compute_energy(v_particles, w_particles))
    push!(momentum_history, compute_momentum(v_particles, w_particles))
    println("Initial  S_h = $(entropy_history[end])")
    println("Initial  E   = $(energy_history[end])")
    println("Initial  P   = $(momentum_history[end])")

    # ------------------------------------------------------------------
    # Preallocate every per-step / per-iteration buffer once.
    # ------------------------------------------------------------------
    r_vec  = zeros(n_dofs)
    L_vec  = zeros(n_dofs)
    G      = zeros(N_PARTICLES, 2)
    dot_v  = zeros(N_PARTICLES, 2)
    v1     = copy(v_particles)

    v_mid  = similar(v_particles)
    dv     = similar(v_particles)
    dS_mid = zeros(N_PARTICLES, 2)
    G_eff  = zeros(N_PARTICLES, 2)
    f_buf  = zeros(n_dofs)

    # Anderson workspace: history window of m residuals
    Gv         = zeros(N_PARTICLES, 2)
    r_curr     = zeros(N_PARTICLES, 2)
    r_prev     = zeros(N_PARTICLES, 2)
    Gv_prev    = zeros(N_PARTICLES, 2)
    v_old_buf  = zeros(N_PARTICLES, 2)
    ΔF         = zeros(2 * N_PARTICLES, m_anderson)
    ΔG         = zeros(2 * N_PARTICLES, m_anderson)

    snapshot_steps = Set([0, N_STEPS ÷ 4, N_STEPS ÷ 2, N_STEPS])
    snapshots = Dict{Int,Matrix{Float64}}()
    snapshots[0] = copy(v_particles)

    for step in 1:N_STEPS
        S0 = entropy_history[end]

        # --------------------------------------------------------------
        # Initial guess: explicit Euler step at v^n to seed Anderson.
        # --------------------------------------------------------------
        compute_r!(r_vec, f_s)
        ldiv!(L_vec, M_lu, r_vec)
        compute_G!(G, v_particles, L_vec)
        compute_collision!(dot_v, v_particles, w_particles, G)
        @. v1 = v_particles + DT * dot_v

        # --------------------------------------------------------------
        # Anderson-accelerated fixed-point iteration (Eqs. 59–61)
        # --------------------------------------------------------------
        iter, res_final, n_rs = step_anderson!(v1, v_particles, w_particles, S0, DT,
                              v_mid, dv, dS_mid, G_eff, dot_v, f_buf,
                              r_vec, L_vec, G,
                              Gv, r_curr, r_prev, Gv_prev, v_old_buf, ΔF, ΔG;
                              m=m_anderson, max_iter=max_iter, tol=tol,
                              damping=damping, use_anderson=use_anderson,
                              verbose=(step <= 3))
        v_particles .= v1

        l2_project!(f_coeffs, v_particles, w_particles)
        f_s = build_field(f_coeffs)
        push!(entropy_history,  compute_entropy(f_s))
        push!(energy_history,   compute_energy(v_particles, w_particles))
        push!(momentum_history, compute_momentum(v_particles, w_particles))
        push!(iter_history,     iter)
        push!(res_history,      res_final)

        step in snapshot_steps && (snapshots[step] = copy(v_particles))
        step % 25 == 0 &&
            println("Step $step/$N_STEPS  iter=$iter  rs=$n_rs  ‖r‖=$(round(res_final; sigdigits=3))" *
                    "  S = $(round(entropy_history[end]; digits=6))" *
                    "  E = $(round(energy_history[end]; digits=8))" *
                    "  P = ($(round(momentum_history[end][1]; digits=8))," *
                    " $(round(momentum_history[end][2]; digits=8)))")
    end

    # ------------------------------------------------------------------
    # Save conservation histories to CSV
    # ------------------------------------------------------------------
    cons_csv = "conservation_history_$(suffix).csv"
    open(cons_csv, "w") do io
        println(io, "step,time,entropy,energy,momentum_1,momentum_2,iter,residual")
        for n in 0:N_STEPS
            t  = n * DT
            S  = entropy_history[n+1]
            E  = energy_history[n+1]
            P1 = momentum_history[n+1][1]
            P2 = momentum_history[n+1][2]
            it = n == 0 ? 0   : iter_history[n]
            rs = n == 0 ? 0.0 : res_history[n]
            println(io, "$n,$t,$S,$E,$P1,$P2,$it,$rs")
        end
    end
    println("Saved $cons_csv")

    # ------------------------------------------------------------------
    # Save particle snapshots (v_1, v_2 per particle) at each quarter
    # ------------------------------------------------------------------
    snap_csv = "particle_snapshots_$(suffix).csv"
    open(snap_csv, "w") do io
        println(io, "step,time,particle_idx,v1,v2")
        for s in sort(collect(keys(snapshots)))
            pts = snapshots[s]
            t   = s * DT
            for i in axes(pts, 1)
                println(io, "$s,$t,$i,$(pts[i, 1]),$(pts[i, 2])")
            end
        end
    end
    println("Saved $snap_csv")

    begin # Visualization
        E0    = energy_history[1]
        P0    = momentum_history[1]
        steps = 0:N_STEPS

        E_err = [abs(energy_history[n+1] - E0) / abs(E0) for n in steps]
        P_err = [hypot(momentum_history[n+1][1] - P0[1],
                       momentum_history[n+1][2] - P0[2]) / hypot(P0[1], P0[2])
                 for n in steps]

        snap_keys = sort(collect(keys(snapshots)))
        n_snap_rows = cld(length(snap_keys), 2)

        fig = Figure(; size=(1200, 200 * (n_snap_rows + 3)))

        # Sample standard deviations σ_d = sqrt(⟨v_d²⟩ − ⟨v_d⟩²) per snapshot;
        # annotate the initial and final scatters with σ₁, σ₂ to visualize
        # anisotropy collapse toward the isotropic Maxwellian.
        sample_std(v) = (μ = sum(v)/length(v); sqrt(sum((v .- μ).^2)/length(v)))
        s_init  = snap_keys[1]
        s_final = snap_keys[end]
        for (idx, s) in enumerate(snap_keys)
            row, col = fldmod1(idx, 2)
            t_str = round(s * DT; digits=4)
            pts = snapshots[s]
            title_str = if s == s_init || s == s_final
                σ1 = sample_std(@view pts[:, 1])
                σ2 = sample_std(@view pts[:, 2])
                "t = $t_str   σ₁=$(round(σ1; digits=4))   σ₂=$(round(σ2; digits=4))"
            else
                "t = $t_str"
            end
            ax = Axis(fig[row, col];
                title=title_str,
                xlabel="v₁", ylabel="v₂", aspect=DataAspect())
            scatter!(ax, pts[:, 1], pts[:, 2]; markersize=2, color=:blue, alpha=0.3)
            xlims!(ax, V_MIN, V_MAX)
            ylims!(ax, V_MIN, V_MAX)
        end

        # H-function evolution (sign convention: H_h is the entropy functional
        # we expect to grow monotonically; the standard Boltzmann H = ∫ f log f
        # has the opposite sign and would decrease).
        H_history = entropy_history
        ax_H = Axis(fig[n_snap_rows+1, 1:2];
            xlabel="time step", ylabel="H_h",
            title="Boltzmann H-function (monotone increase expected)")
        lines!(ax_H, steps, H_history; color=:red, linewidth=2)

        # Energy conservation error (log scale)
        ax_E = Axis(fig[n_snap_rows+2, 1:2];
            xlabel="time step", ylabel="relative error",
            title="Energy conservation error  |E_n − E_0| / E_0",
            yscale=log10)
        lines!(ax_E, steps, max.(E_err, 1e-18); color=:blue, linewidth=2)

        # Momentum conservation error (log scale)
        ax_P = Axis(fig[n_snap_rows+3, 1:2];
            xlabel="time step", ylabel="relative error",
            title="Momentum conservation error  ‖P_n − P_0‖ / ‖P_0‖",
            yscale=log10)
        lines!(ax_P, steps, max.(P_err, 1e-18); color=:green, linewidth=2)

        png_name = "landau_collision_$(suffix)_2d.png"
        save(png_name, fig)
        println("Saved $png_name")
    end

    return (; entropy_history, energy_history, momentum_history,
              iter_history, res_history, snapshots, label)
end


function plot_comparison(res_picard, res_anderson)
    steps   = 0:N_STEPS
    E0_p    = res_picard.energy_history[1]
    P0_p    = res_picard.momentum_history[1]
    E0_a    = res_anderson.energy_history[1]
    P0_a    = res_anderson.momentum_history[1]

    Eerr_p = [abs(res_picard.energy_history[n+1]   - E0_p) / abs(E0_p) for n in steps]
    Eerr_a = [abs(res_anderson.energy_history[n+1] - E0_a) / abs(E0_a) for n in steps]
    Perr_p = [hypot(res_picard.momentum_history[n+1][1]   - P0_p[1],
                    res_picard.momentum_history[n+1][2]   - P0_p[2]) /
              max(hypot(P0_p[1], P0_p[2]), 1e-30) for n in steps]
    Perr_a = [hypot(res_anderson.momentum_history[n+1][1] - P0_a[1],
                    res_anderson.momentum_history[n+1][2] - P0_a[2]) /
              max(hypot(P0_a[1], P0_a[2]), 1e-30) for n in steps]

    fig = Figure(; size=(1200, 1000))

    ax_S = Axis(fig[1, 1]; xlabel="time step", ylabel="H_h",
        title="Boltzmann H-function (monotone increase expected)")
    lines!(ax_S, steps, res_picard.entropy_history;   color=:blue, linewidth=2,
        label=res_picard.label)
    lines!(ax_S, steps, res_anderson.entropy_history; color=:red,  linewidth=2,
        label=res_anderson.label, linestyle=:dash)
    axislegend(ax_S; position=:rb)

    ax_E = Axis(fig[2, 1]; xlabel="time step", ylabel="relative error",
        title="Energy conservation error  |E_n − E_0| / E_0", yscale=log10)
    lines!(ax_E, steps, max.(Eerr_p, 1e-18); color=:blue, linewidth=2,
        label=res_picard.label)
    lines!(ax_E, steps, max.(Eerr_a, 1e-18); color=:red,  linewidth=2,
        label=res_anderson.label, linestyle=:dash)
    axislegend(ax_E; position=:rb)

    ax_P = Axis(fig[3, 1]; xlabel="time step", ylabel="relative error",
        title="Momentum conservation error  ‖P_n − P_0‖ / ‖P_0‖", yscale=log10)
    lines!(ax_P, steps, max.(Perr_p, 1e-18); color=:blue, linewidth=2,
        label=res_picard.label)
    lines!(ax_P, steps, max.(Perr_a, 1e-18); color=:red,  linewidth=2,
        label=res_anderson.label, linestyle=:dash)
    axislegend(ax_P; position=:rb)

    ax_I = Axis(fig[4, 1]; xlabel="time step", ylabel="iterations / step",
        title="Inner-iteration count per Gonzalez step")
    lines!(ax_I, 1:N_STEPS, res_picard.iter_history;   color=:blue, linewidth=2,
        label=res_picard.label)
    lines!(ax_I, 1:N_STEPS, res_anderson.iter_history; color=:red,  linewidth=2,
        label=res_anderson.label, linestyle=:dash)
    axislegend(ax_I; position=:rt)

    save("landau_collision_compare_2d.png", fig)
    println("Saved landau_collision_compare_2d.png")
end


function main()
    println("V ∈ [$V_MIN, $V_MAX],  I ∈ [$I_MIN, $I_MAX]")
    println("p=$P_DEG, k=$K_REG,  inner $N_ELEM×$N_ELEM elements (+ 2 outer per dim)")
    println("N_particles=$N_PARTICLES,  σ₁=$σ₁, σ₂=$σ₂ (~99.7% in [I_MIN,I_MAX]²),  Δt=$DT,  N_steps=$N_STEPS")

    # Path C (long-time + fine mesh): finer mesh pushes Picard's Lipschitz
    # past 1 and damped Picard cannot recover, so use Anderson with a larger
    # history window and looser damping to stay within max_iter.
    res_anderson = run_simulation(use_anderson=true, suffix="anderson", damping=0.7,
                                   m_anderson=8, max_iter=2000)

    avg_a = sum(res_anderson.iter_history) / length(res_anderson.iter_history)
    max_a = maximum(res_anderson.iter_history)
    println("\n--- Inner-iter summary ---")
    println("Anderson(m=8, β=0.7): avg=$(round(avg_a; digits=2))  max=$max_a")
end

main()
