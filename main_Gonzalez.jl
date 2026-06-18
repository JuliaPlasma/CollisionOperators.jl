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
using Serialization
using LinearAlgebra
using LinearAlgebra: ldiv!, mul!

# ---- Run history ------------------------------------------------------------
# Bundle the 7 per-step diagnostic series so they travel as one value instead of
# 7 parallel arrays threaded through every push!/checkpoint/return site.
struct RunHistory
    entropy::Vector{Float64}
    energy::Vector{Float64}
    momentum::Vector{NTuple{2,Float64}}
    iter::Vector{Int}
    res::Vector{Float64}
    fp_l2::Vector{Float64}
    neg::Vector{Float64}
end
RunHistory() = RunHistory(Float64[], Float64[], NTuple{2,Float64}[], Int[], Float64[], Float64[], Float64[])

"""
    push_step!(h, entropy, energy, momentum, iter, res, fp_l2, neg) -> h

Append one evolution step's diagnostics to the [`RunHistory`](@ref) `h`.
"""
function push_step!(h::RunHistory, entropy, energy, momentum, iter, res, fp_l2, neg)
    push!(h.entropy, entropy)
    push!(h.energy, energy)
    push!(h.momentum, momentum)
    push!(h.iter, iter)
    push!(h.res, res)
    push!(h.fp_l2, fp_l2)
    push!(h.neg, neg)
    return h
end


# ---- Checkpoint / resume ----------------------------------------------------
# Full simulation state serialized via stdlib `Serialization` (no extra deps).
# Written at every snapshot step (≡ every 25 steps + final). On resume, the
# main loop reloads state and appends to the existing CSVs.

"""
    checkpoint_path(suffix, step) -> String

Filename for the checkpoint of `suffix` at `step` (zero-padded to 4 digits).
"""
function checkpoint_path(suffix::String, step::Int)
    return "checkpoint_$(suffix)_step$(lpad(step, 4, '0')).jls"
end

"""
    save_checkpoint(suffix, step, v_particles, w_particles, f_coeffs, h, rng_state) -> String

Serialize the full simulation state to `checkpoint_path(suffix, step)`. State
captured:

  - `step`          : last completed step (resume continues at step+1)
  - `v_particles`   : N×2 velocity matrix
  - `w_particles`   : N weights (constant 1/N, saved for sanity)
  - `f_coeffs`      : spline coefficient vector (size = ws.n_dofs)
  - `*_history`     : the seven diagnostic vectors from `h::RunHistory`
  - `rng_state`     : copy of Random.default_rng() so any post-resume sampling
                      is reproducible

The flat `*_history` NamedTuple shape is written (not RunHistory) so old
checkpoints stay loadable and `load_checkpoint` needs no change.
"""
function save_checkpoint(
    suffix::String, step::Int, v_particles, w_particles, f_coeffs, h::RunHistory, rng_state
)
    fname = checkpoint_path(suffix, step)
    open(fname, "w") do io
        serialize(
            io,
            (;
                step,
                v_particles,
                w_particles,
                f_coeffs,
                entropy_history=h.entropy,
                energy_history=h.energy,
                momentum_history=h.momentum,
                iter_history=h.iter,
                res_history=h.res,
                fp_l2_history=h.fp_l2,
                neg_history=h.neg,
                rng_state,
            ),
        )
    end
    println("Saved $fname")
    return fname
end

"""
    history_from_checkpoint(ckpt) -> RunHistory

Rebuild a [`RunHistory`](@ref) from a deserialized checkpoint NamedTuple.
"""
function history_from_checkpoint(ckpt)
    return RunHistory(
        ckpt.entropy_history,
        ckpt.energy_history,
        ckpt.momentum_history,
        ckpt.iter_history,
        ckpt.res_history,
        ckpt.fp_l2_history,
        ckpt.neg_history,
    )
end

"""
    load_checkpoint(suffix, step) -> NamedTuple

Deserialize a checkpoint. `step=:auto` (or any non-positive Int) picks the
highest-step checkpoint matching `suffix` from the cwd; otherwise the exact
`checkpoint_path(suffix, step)` is loaded.
"""
function load_checkpoint(suffix::String, step)
    fname = if step === :auto || (step isa Integer && step <= 0)
        files = filter(readdir()) do f
            startswith(f, "checkpoint_$(suffix)_step") && endswith(f, ".jls")
        end
        isempty(files) && error("No checkpoint files for suffix=$suffix")
        sort(files)[end]
    else
        checkpoint_path(suffix, Int(step))
    end
    isfile(fname) || error("Checkpoint not found: $fname")
    println("Loading checkpoint: $fname")
    return open(deserialize, fname)
end


"""
    compute_entropy_gradient!(ws, dS, v_parts, w_parts, f_coeffs_buf, r_vec, L_vec, G_buf)

Compute the entropy gradient `∂S_h/∂v_α = -w_α G_α` for every particle, where
`G = ∇L`, `L = M⁻¹ r`, and `r` is the entropy-gradient seed of the L²-projected
field `f_s`. Workspace-aware: writes the N×2 result into `dS` (overwritten) and
uses `f_coeffs_buf`, `r_vec`, `L_vec`, `G_buf` as scratch.
"""
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

"""
    picard_map!(ws, v_out, v_in, v0, w_parts, S0, dt, v_mid, dv, dS_mid, G_eff,
                dot_v_buf, f_buf, r_vec, L_vec, G_buf)

One Picard map of the Gonzalez discrete-gradient scheme:
`v_out = v0 + dt · G̃(v_mid) · ∇̄S`. The midpoint `v_mid = ½(v0 + v_in)` is used
for the collision kernel; the discrete-gradient correction enforces exact
entropy production (`S1 − S0 = ∇̄S · Δv`). Writes the N×2 result into `v_out`;
`v_mid`, `dv`, `dS_mid`, `G_eff`, `dot_v_buf`, `f_buf`, `r_vec`, `L_vec`,
`G_buf` are scratch (overwritten). `S0` is the entropy at `v0`.
"""
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


"""
    step_anderson!(ws, v1, v0, w_parts, S0, dt, v_mid, dv, dS_mid, G_eff,
                   dot_v_buf, f_buf, r_vec, L_vec, G_buf,
                   Gv, r_curr, r_prev, Gv_prev, v_old, ΔF, ΔG; kwargs...)

Anderson-accelerated fixed-point iteration for the Gonzalez Picard map. State is
the N×2 velocity matrix; all linear-algebra ops use its flat `vec()` view
(length 2N), so the solver is dimension-agnostic. `use_anderson=false` ⇒ damped
Picard. Returns `(iters, ‖r‖, n_restarts)`; the returned `v1` is the best Gv
seen across all iterations.

Convergence rules (any one triggers a successful exit):
  1. Relative+floor:   ‖r‖ < max(tol · ‖v‖, abs_floor). `abs_floor` caps how
     tight we ask for — past the Picard map's numerical noise floor, asking for
     less is pointless and burns wall time.
  2. Stagnation:       every `stag_window` iter, compare `nrm_best` against its
     value `stag_window` iters ago; relative drop < `stag_rel_tol` ⇒ exit.

Adaptive damping: once past `damp_decay_start` iterations without exit, multiply
damping by `damp_decay_factor` (more conservative step) to stabilize stiff
late-time fixed-point maps.

Mutated in place (output + scratch; pass distinct, preallocated arrays):
  `v1`     — output iterate (best Gv on return)
  `v_mid`, `dv`, `dS_mid`, `G_eff`, `dot_v_buf`, `f_buf`, `r_vec`, `L_vec`,
  `G_buf` — Picard-map scratch
  `Gv`     — fixed-point map output G(v1)
  `r_curr`, `r_prev` — current / previous residual G(v)−v
  `Gv_prev`          — previous map output
  `v_old`            — previous iterate
  `ΔF`, `ΔG`         — Anderson difference matrices (2N × m)

Read-only: `v0` (base velocities), `w_parts`, `S0`, `dt`, and all scalar kwargs.
"""
function step_anderson!(ws::Workspace,
                        v1, v0, w_parts, S0, dt,
                        v_mid, dv, dS_mid, G_eff, dot_v_buf, f_buf,
                        r_vec, L_vec, G_buf,
                        Gv, r_curr, r_prev, Gv_prev, v_old, ΔF, ΔG;
                        m=5, max_iter=1000, tol=1e-12,
                        abs_floor=1e-7,
                        stag_window=50, stag_rel_tol=0.01,
                        damp_decay_start=200, damp_decay_factor=0.5,
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

    # Track best iterate so stagnation / max_iter exits return the best Gv
    # rather than the latest (which may be worse on a non-monotone trajectory).
    Gv_best = copy(Gv)
    nrm_best_window = Inf  # nrm_best snapshot from `stag_window` iters ago

    for k in 1:max_iter
        vold_v .= v1_v
        picard_map!(ws, Gv, v1, v0, w_parts, S0, dt,
                    v_mid, dv, dS_mid, G_eff, dot_v_buf, f_buf,
                    r_vec, L_vec, G_buf)
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
            return k, nrm_r, n_restart
        end

        # Stagnation early-exit: insufficient progress over a window of iters.
        if k > stag_window && k % stag_window == 0
            rel_improve = (nrm_best_window - nrm_best) /
                          (nrm_best_window + 1e-30)
            if rel_improve < stag_rel_tol
                v1 .= Gv_best
                verbose && println("    k=$k  stagnated  nrm_best=$nrm_best  " *
                                   "Δ_rel=$rel_improve")
                return k, nrm_best, n_restart
            end
            nrm_best_window = nrm_best
        end

        just_restarted = false
        if k > 1 && nrm_r > restart_factor * nrm_best
            history = 0
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
            @. v1_v = damping_eff * v1_v + (1 - damping_eff) * vold_v
        end

        rp_v .= r_v
        Gp_v .= Gv_v
    end

    # max_iter exhausted: return best Gv (not the latest, which may be worse).
    v1 .= Gv_best
    @warn "Solver did not converge" max_iter tol abs_floor nrm_r0 nrm_r nrm_best n_restart
    return max_iter, nrm_best, n_restart
end


"""
    compute_momentum(v_parts, w_parts) -> (p1, p2)

Total weighted momentum `Σ w_α v_α`, by component. Read-only.
"""
function compute_momentum(v_parts, w_parts)
    p1 = sum(w_parts[α] * v_parts[α, 1] for α in axes(v_parts, 1))
    p2 = sum(w_parts[α] * v_parts[α, 2] for α in axes(v_parts, 1))
    return (p1, p2)
end

"""
    compute_energy(v_parts, w_parts) -> E

Total weighted kinetic energy `½ Σ w_α |v_α|²`. Read-only.
"""
function compute_energy(v_parts, w_parts)
    return 0.5 * sum(w_parts[α] * (v_parts[α, 1]^2 + v_parts[α, 2]^2)
                     for α in axes(v_parts, 1))
end


"""
    save_fs_snapshot(ws, suffix, step, f_coeffs) -> String

Save the f_s coefficient vector + breakpoints so post-run scripts can rebuild
the spline. CSV chosen for diff-friendliness with the conservation CSV.
"""
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


"""
    plot_fs_diagnostics(ws, f_coeffs, suffix, step; kwargs...) -> String

Per-snapshot dashboard: 1D slices `f_s(v₁_fixed, v₂)` along v₂ at three fixed v₁
values, plus the `log10|f_s|` heatmap with a negative-region (`f_s < 0`) red
mask overlay (the Gibbs-oscillation probe). Saves a PNG and returns its name.
"""
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


# ---- Streaming I/O helpers --------------------------------------------------

"""
    write_cons_row(io, step, t, entropy, energy, momentum, iter, res, fp_l2, neg)

Single source for the conservation-CSV row schema (init row + per-step row).
Writes one row and flushes (crash-safe).
"""
function write_cons_row(io, step, t, entropy, energy, momentum, iter, res, fp_l2, neg)
    println(io, "$step,$t,$entropy,$energy,$(momentum[1]),$(momentum[2])," *
                "$iter,$res,$fp_l2,$neg")
    flush(io)
    return nothing
end

"""
    dump_particles(io, step, t, v_particles)

Append every particle's `(step, t, idx, v1, v2)` row to `io` and flush.
"""
function dump_particles(io, step, t, v_particles)
    for i in axes(v_particles, 1)
        println(io, "$step,$t,$i,$(v_particles[i, 1]),$(v_particles[i, 2])")
    end
    flush(io)
    return nothing
end

"""
    take_snapshot(ws, p, step, f_coeffs, v_particles, w_particles, h, snap_io)

Everything written at a snapshot step: f_s CSV + diagnostic PNG + particle dump
+ checkpoint. Runs identically at step 0 and inside the loop.
"""
function take_snapshot(ws, p, step, f_coeffs, v_particles, w_particles, h::RunHistory, snap_io)
    save_fs_snapshot(ws, p.suffix, step, f_coeffs)
    plot_fs_diagnostics(ws, f_coeffs, p.suffix, step)
    dump_particles(snap_io, step, step * p.DT, v_particles)
    save_checkpoint(p.suffix, step, v_particles, w_particles, f_coeffs, h, copy(Random.default_rng()))
    return nothing
end


"""
    plot_run_dashboard(ws, entropy_history, energy_history, momentum_history,
                       iter_history, res_history, fp_l2_history, neg_history, suffix) -> String

Per-run quick-look dashboard: entropy, conservation errors (energy/momentum),
inner-iteration count, fixed-point residual, projection error, and negative
part. Main analytical payload is the conservation CSV. Saves a PNG.
"""
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


"""
    run_simulation(p; resume=nothing) -> NamedTuple

Drive the full Gonzalez discrete-gradient time integration for parameters `p`.
Either samples a fresh particle ensemble or resumes from a checkpoint
(`resume = step | :auto`). Streams the conservation + particle-snapshot CSVs
per step / per snapshot (crash-safe), writes per-snapshot f_s/PNG/checkpoint,
plots the run dashboard, and returns the diagnostic histories + a `label`.
"""
function run_simulation(p::SimParameters; resume=nothing)
    print_summary(p)
    Random.seed!(p.seed)

    ws = build_workspace(p)
    println("Workspace: n_dofs=$(ws.n_dofs)  n_elements=$(ws.n_elements)")

    # ---- State init: either fresh sample or resume from checkpoint ----
    start_step = 0
    local v_particles, w_particles, f_coeffs, hist

    if resume !== nothing
        ckpt = load_checkpoint(p.suffix, resume)
        start_step = ckpt.step
        v_particles = ckpt.v_particles
        w_particles = ckpt.w_particles
        f_coeffs    = ckpt.f_coeffs
        hist = history_from_checkpoint(ckpt)
        copy!(Random.default_rng(), ckpt.rng_state)

        size(v_particles, 1) == p.N_PARTICLES || error(
            "Checkpoint N_PARTICLES=$(size(v_particles,1)) ≠ preset N_PARTICLES=$(p.N_PARTICLES)")
        length(f_coeffs) == ws.n_dofs || error(
            "Checkpoint n_dofs=$(length(f_coeffs)) ≠ workspace n_dofs=$(ws.n_dofs); mesh changed?")
        start_step < p.N_STEPS || error(
            "Checkpoint step=$start_step ≥ N_STEPS=$(p.N_STEPS); nothing to do")

        println("Resuming from step $start_step (running through $(p.N_STEPS))")
    else
        v_particles = zeros(p.N_PARTICLES, 2)
        v_particles[:, 1] .= p.σ1 * randn(p.N_PARTICLES)
        v_particles[:, 2] .= p.σ2 * randn(p.N_PARTICLES)
        w_particles = fill(1.0 / p.N_PARTICLES, p.N_PARTICLES)
        f_coeffs    = zeros(ws.n_dofs)

        l2_project!(ws, f_coeffs, v_particles, w_particles)

        hist = RunHistory()
        f_s0 = build_field(ws, f_coeffs)
        push!(hist.entropy,  compute_entropy(ws, f_s0))
        push!(hist.energy,   compute_energy(v_particles, w_particles))
        push!(hist.momentum, compute_momentum(v_particles, w_particles))
        println("Initial  S_h = $(hist.entropy[end])")
        println("Initial  E   = $(hist.energy[end])")
        println("Initial  P   = $(hist.momentum[end])")
        fs_minus_fp_l2 = compute_fs_minus_fp_l2(ws, f_s0, v_particles, w_particles)
        println("Initial  ‖f_s − f_p‖₂ = " * string(fs_minus_fp_l2))
        neg_part = compute_negative_part_l1(ws, f_s0)
        println("Initial  ∫max(−f_s,0) = " * string(neg_part))
    end

    f_s = build_field(ws, f_coeffs)

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

    # Snapshot every 25 steps (plus final step if not already a multiple of 25).
    # Crash-safe: conservation + particles appended per-step / per-snapshot so a
    # killed run still leaves usable data through the last completed step.
    snapshot_steps = Set(0:25:p.N_STEPS)
    push!(snapshot_steps, p.N_STEPS)

    cons_csv = "conservation_history_$(p.suffix).csv"
    snap_csv = "particle_snapshots_$(p.suffix).csv"

    if start_step == 0
        cons_io = open(cons_csv, "w")
        println(cons_io, "step,time,entropy,energy,momentum_1,momentum_2," *
                         "iter,residual,fp_minus_fs,neg_part")
        write_cons_row(cons_io, 0, 0.0, hist.entropy[1], hist.energy[1],
                       hist.momentum[1], 0, 0.0, fs_minus_fp_l2, neg_part)

        snap_io = open(snap_csv, "w")
        println(snap_io, "step,time,particle_idx,v1,v2")

        # Step-0 snapshot (f_s CSV + PNG + particle dump + checkpoint) so future
        # runs can resume even before any time-stepping completed.
        take_snapshot(ws, p, 0, f_coeffs, v_particles, w_particles, hist, snap_io)
    else
        # Resume: keep existing CSV rows ≤ start_step, append from now on.
        cons_io = open(cons_csv, "a")
        snap_io = open(snap_csv, "a")
    end

    for step in (start_step+1):p.N_STEPS
        S0 = hist.entropy[end]

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
                              abs_floor=p.abs_floor,
                              stag_window=p.stag_window,
                              stag_rel_tol=p.stag_rel_tol,
                              damp_decay_start=p.damp_decay_start,
                              damp_decay_factor=p.damp_decay_factor,
                              damping=p.damping, use_anderson=p.use_anderson,
                              verbose=(step <= 3))
        v_particles .= v1

        l2_project!(ws, f_coeffs, v_particles, w_particles)
        f_s = build_field(ws, f_coeffs)

        push_step!(hist,
                   compute_entropy(ws, f_s),
                   compute_energy(v_particles, w_particles),
                   compute_momentum(v_particles, w_particles),
                   iter, res_final,
                   compute_fs_minus_fp_l2(ws, f_s, v_particles, w_particles),
                   compute_negative_part_l1(ws, f_s))

        # Append this step's conservation row (crash-safe).
        write_cons_row(cons_io, step, step * p.DT, hist.entropy[end],
                       hist.energy[end], hist.momentum[end], iter, res_final,
                       hist.fp_l2[end], hist.neg[end])

        if step in snapshot_steps
            take_snapshot(ws, p, step, f_coeffs, v_particles, w_particles, hist, snap_io)
        end

        step % 25 == 0 &&
            println("Step $step/$(p.N_STEPS)  iter=$iter  rs=$n_rs" *
                    "  ‖r‖=$(round(res_final; sigdigits=3))" *
                    "  ‖f_s−f_p‖=$(round(hist.fp_l2[end]; sigdigits=4))" *
                    "  neg=$(round(hist.neg[end]; sigdigits=4))" *
                    "  S=$(round(hist.entropy[end]; digits=6))" *
                    "  E=$(round(hist.energy[end]; digits=8))")
    end

    # CSVs already streamed per-step / per-snapshot above. Just close.
    close(cons_io)
    close(snap_io)
    println("Saved $cons_csv")
    println("Saved $snap_csv")

    plot_run_dashboard(ws,
                       hist.entropy, hist.energy, hist.momentum,
                       hist.iter, hist.res, hist.fp_l2,
                       hist.neg, p.suffix)

    return (; entropy_history=hist.entropy, energy_history=hist.energy,
              momentum_history=hist.momentum, iter_history=hist.iter,
              res_history=hist.res, fp_l2_history=hist.fp_l2,
              neg_history=hist.neg,
              label=(p.use_anderson ? "Anderson(m=$(p.m_anderson))" : "Picard"))
end


"""
    main(args=ARGS)

CLI entry point. Parses `ARGS[1]` as the preset file and `ARGS[2:]` as
`--key=value` overrides (plus a `--resume=<step|auto>` flag stripped before
override parsing), then runs the simulation and prints an inner-iter summary.
"""
function main(args=ARGS)
    if isempty(args)
        preset = "parameters_default.jl"
        overrides = String[]
    else
        preset = args[1]
        overrides = collect(String, args[2:end])
    end
    isfile(preset) || error("Preset file not found: $preset")

    # Strip --resume=<step|auto> out of overrides before parse_overrides sees it
    # (it's not a SimParameters field). Accepts an integer step number or the
    # literal `auto` to pick the highest-step checkpoint for this suffix.
    resume = nothing
    overrides = filter(overrides) do tok
        if startswith(tok, "--resume=")
            val = tok[length("--resume=")+1:end]
            resume = (val == "auto") ? :auto : parse(Int, val)
            return false
        end
        return true
    end

    println("Loading preset: $preset")
    # The preset file ends by binding PARAMS = SimParameters(...). `include`
    # returns the last expression's value, so we capture PARAMS without
    # relying on module globals.
    params_loaded = include(joinpath(@__DIR__, preset))
    p = parse_overrides(params_loaded::SimParameters, overrides)

    res = run_simulation(p; resume=resume)
    if isempty(res.iter_history)
        println("\n--- No new steps run (already at N_STEPS) ---")
    else
        a = sum(res.iter_history) / length(res.iter_history)
        println("\n--- Inner-iter summary ---")
        println("$(res.label):  avg=$(round(a; digits=2))  max=$(maximum(res.iter_history))" *
                "  steps=$(length(res.iter_history))")
    end
end

main()
