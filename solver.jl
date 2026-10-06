# solver.jl — the implicit time step: one Picard map of the Gonzalez
# discrete-gradient (or Lenard–Bernstein) update, and the Anderson-accelerated
# fixed-point iteration that solves it. Pure compute: no file I/O, no plotting.
#
# Needs `Workspace` plus the physics in functions.jl, so main.jl includes this
# after MantisWrappers. The CPU/GPU hot-loop hooks live here because this is
# their only hot call site; main() repoints them at startup.

using LinearAlgebra: norm, mul!, ldiv!

# Hot-loop implementations, swappable at startup: `--use_gpu=true` loads
# collision_gpu.jl / projection_gpu.jl (CUDA) and repoints these Refs. Call
# sites go through invokelatest so the swap survives world-age.
const COLLISION_FN = Ref{Any}(compute_collision!)
const L2PROJ_FN = Ref{Any}(l2_project!)
const COMPG_FN = Ref{Any}(compute_G!)
const LOGGRAD_FN = Ref{Any}(eval_loggrad_at_particles!)   # LB base gradient

# Compute ∂S_h/∂v_α = -w_α G_α for every particle. Workspace-aware.
function compute_entropy_gradient!(ws::Workspace, dS, v_parts, w_parts,
        f_coeffs_buf, r_vec, L_vec, G_buf)
    Base.invokelatest(L2PROJ_FN[], ws, f_coeffs_buf, v_parts, w_parts)
    f_s = build_field(ws, f_coeffs_buf)
    compute_r!(ws, r_vec, f_s)
    ldiv!(L_vec, ws.M_lu, r_vec)
    Base.invokelatest(COMPG_FN[], ws, G_buf, v_parts, L_vec)
    @inbounds for α in axes(v_parts, 1)
        dS[α, 1] = -w_parts[α] * G_buf[α, 1]
        dS[α, 2] = -w_parts[α] * G_buf[α, 2]
    end
    return nothing
end

# One Picard map: v_out = v0 + dt · G̃(v_mid) · ∇̄S
# `use_gonzalez=false` drops the discrete-gradient correction term, leaving the
# plain implicit-midpoint rule ∇̄S = ∇S(v_mid). This avoids the Gonzalez |Δv|²
# denominator that blows up when started at (or near) equilibrium (Δv → 0).
function picard_map!(ws::Workspace, v_out, v_in, v0, w_parts, S0, dt,
        v_mid, dv, dS_mid, G_eff, dot_v_buf, f_buf,
        r_vec, L_vec, G_buf; use_gonzalez::Bool = true)
    N = size(v0, 1)
    is_lb = ws.p.collision_model == :lb
    @. v_mid = 0.5 * (v0 + v_in)
    @. dv = v_in - v0

    # --- log-density gradient g (≡ G_eff) at v_mid -----------------------------
    if is_lb
        # LB uses the direct clamped ∇f_s/f_s as base gradient in BOTH modes.
        # The FEM-projected ∇(M⁻¹r) seed used by Landau injects Gibbs ringing
        # into the LB drift in low-density cells — Landau's f-weighted mobility
        # suppresses those regions, LB's does not, feeding a runaway negativity
        # loop (see notes_LB_gonzalez_port.md §3). The discrete-gradient entropy
        # identity is exact for ANY base gradient, so Gonzalez adds only the
        # rank-1 λ correction on top of the well-behaved direct gradient.
        Base.invokelatest(L2PROJ_FN[], ws, f_buf, v_mid, w_parts)
        Base.invokelatest(LOGGRAD_FN[], ws, G_eff, v_mid, f_buf)
        if use_gonzalez
            Base.invokelatest(L2PROJ_FN[], ws, f_buf, v_in, w_parts)
            S1 = compute_entropy(ws, build_field(ws, f_buf))
            dot_dv_dS = 0.0
            nrm2_dv = 0.0
            @inbounds for α in 1:N   # ∂S/∂v_α = -w_α g_α
                dot_dv_dS -= w_parts[α] * (dv[α, 1] * G_eff[α, 1] + dv[α, 2] * G_eff[α, 2])
                nrm2_dv += dv[α, 1]^2 + dv[α, 2]^2
            end
            λ = nrm2_dv > 1e-30 ? (S1 - S0 - dot_dv_dS) / nrm2_dv : 0.0
            @inbounds for α in 1:N
                inv_w = 1.0 / w_parts[α]
                G_eff[α, 1] -= λ * dv[α, 1] * inv_w
                G_eff[α, 2] -= λ * dv[α, 2] * inv_w
            end
        end
    else
        # Landau: FEM-projected entropy gradient + Gonzalez discrete-grad correction.
        compute_entropy_gradient!(ws, dS_mid, v_mid, w_parts,
            f_buf, r_vec, L_vec, G_buf)
        correction = 0.0
        if use_gonzalez
            Base.invokelatest(L2PROJ_FN[], ws, f_buf, v_in, w_parts)
            S1 = compute_entropy(ws, build_field(ws, f_buf))

            dot_dv_dS = 0.0
            nrm2_dv = 0.0
            @inbounds for α in 1:N
                dot_dv_dS += dv[α, 1] * dS_mid[α, 1] + dv[α, 2] * dS_mid[α, 2]
                nrm2_dv += dv[α, 1]^2 + dv[α, 2]^2
            end
            correction = nrm2_dv > 1e-30 ? (S1 - S0 - dot_dv_dS) / nrm2_dv : 0.0
        end

        @inbounds for α in 1:N
            inv_w = 1.0 / w_parts[α]
            G_eff[α, 1] = -(dS_mid[α, 1] + correction * dv[α, 1]) * inv_w
            G_eff[α, 2] = -(dS_mid[α, 2] + correction * dv[α, 2]) * inv_w
        end
    end

    # --- RHS assembly ----------------------------------------------------------
    if is_lb
        n, U1, U2, Q = compute_moments(v_mid, w_parts)
        A1, A2, B = compute_drift_multipliers(v_mid, w_parts, G_eff, n, U1, U2, Q)
        compute_LB_velocity!(dot_v_buf, v_mid, G_eff, A1, A2, B, ws.p.nu)
    else
        Base.invokelatest(COLLISION_FN[], ws, dot_v_buf, v_mid, w_parts, G_eff)
    end
    @. v_out = v0 + dt * dot_v_buf
    return nothing
end

# Anderson-accelerated fixed-point iteration. `use_anderson=false` falls back
# to plain damped Picard.
#
# Convergence rules (any one triggers a successful exit; returned `v1` is the
# best Gv seen across all iterations):
#   1. Relative+floor:   ‖r‖ < max(tol * ‖v‖, abs_floor)
#      `abs_floor` caps how tight we ask for — past the numerical noise floor
#      of the Picard map, asking for less is pointless and burns wall time.
#   2. Stagnation:       every `stag_window` iter, compare `nrm_best` against
#      its value `stag_window` iters ago; relative drop < `stag_rel_tol` ⇒ exit.
#      Catches the late-step plateau where Anderson can't push below 1e-7.
#
# Adaptive damping: once past `damp_decay_start` iterations without exit,
# multiply damping by `damp_decay_factor` (more conservative step) to stabilize
# stiff late-time fixed-point maps.
function step_anderson!(ws::Workspace,
        v1, v0, w_parts, S0, dt,
        v_mid, dv, dS_mid, G_eff, dot_v_buf, f_buf,
        r_vec, L_vec, G_buf,
        Gv, r_curr, r_prev, Gv_prev, v_old, ΔF, ΔG;
        m = 5, max_iter = 1000, tol = 1e-12,
        abs_floor = 1e-7,
        stag_window = 50, stag_rel_tol = 0.01,
        damp_decay_start = 200, damp_decay_factor = 0.5,
        restart_factor = Inf, damping = 0.5,
        reg_factor = 1e-10, verbose = false,
        use_anderson::Bool = true, use_gonzalez::Bool = true)
    v1_v = vec(v1)
    Gv_v = vec(Gv)
    r_v = vec(r_curr)
    rp_v = vec(r_prev)
    Gp_v = vec(Gv_prev)
    vold_v = vec(v_old)

    history = 0
    slot = 0               # ring cursor into the ΔF/ΔG columns
    nrm_r0 = 0.0
    nrm_r = 0.0
    nrm_best = Inf
    n_restart = 0

    # Track best iterate so stagnation / max_iter exits return the best Gv
    # rather than the latest (which may be worse on a non-monotone trajectory).
    Gv_best = copy(Gv)
    nrm_best_window = Inf  # nrm_best snapshot from `stag_window` iters ago

    for k in 1:max_iter
        vold_v .= v1_v
        picard_map!(ws, Gv, v1, v0, w_parts, S0, dt,
            v_mid, dv, dS_mid, G_eff, dot_v_buf, f_buf,
            r_vec, L_vec, G_buf; use_gonzalez = use_gonzalez)
        @. r_v = Gv_v - v1_v
        nrm_r = norm(r_v)
        k == 1 && (nrm_r0 = nrm_r)

        if nrm_r < nrm_best
            nrm_best = nrm_r
            Gv_best .= Gv
        end

        # Convergence: relative tol but never tighter than the absolute floor.
        eff_tol = max(tol * (norm(v1_v) + 1e-30), abs_floor)
        if nrm_r < eff_tol
            v1 .= Gv
            verbose && println("    k=$k  ‖r‖=$nrm_r  history=$history  [converged]")
            return k, nrm_r, n_restart, nrm_r0
        end

        # Stagnation early-exit: insufficient progress over a window of iters.
        if k > stag_window && k % stag_window == 0
            rel_improve = (nrm_best_window - nrm_best) /
                          (nrm_best_window + 1e-30)
            if rel_improve < stag_rel_tol
                v1 .= Gv_best
                verbose && println("    k=$k  stagnated  nrm_best=$nrm_best  " *
                        "Δ_rel=$rel_improve")
                return k, nrm_best, n_restart, nrm_r0
            end
            nrm_best_window = nrm_best
        end

        just_restarted = false
        if k > 1 && nrm_r > restart_factor * nrm_best
            history = 0
            slot = 0
            n_restart += 1
            just_restarted = true
        end

        verbose && println("    k=$k  ‖r‖=$nrm_r  history=$history" *
                (just_restarted ? "  [restart]" : ""))

        # Adaptive damping kicks in once the fast phase has clearly missed.
        damping_eff = k > damp_decay_start ? damping * damp_decay_factor : damping

        if k == 1 || just_restarted || !use_anderson
            @. v1_v = damping_eff * Gv_v + (1 - damping_eff) * vold_v
        else
            # The least-squares problem is invariant under a common column
            # permutation of ΔF and ΔG, so the newest difference just
            # overwrites the oldest column: a ring cursor instead of shifting
            # the whole window left by one, which recopied 2·N·(m-1) entries
            # on every iteration once the window was full.
            slot = mod1(slot + 1, m)
            history = min(history + 1, m)
            @views ΔF[:, slot] .= r_v .- rp_v
            @views ΔG[:, slot] .= Gv_v .- Gp_v

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
            @. v1_v = damping_eff * v1_v + (1 - damping_eff) * vold_v
        end

        rp_v .= r_v
        Gp_v .= Gv_v
    end

    # max_iter exhausted: return best Gv (not the latest, which may be worse).
    v1 .= Gv_best
    @warn "Solver did not converge" max_iter tol abs_floor nrm_r0 nrm_r nrm_best n_restart
    return max_iter, nrm_best, n_restart, nrm_r0
end
