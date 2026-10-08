# solver.jl — the implicit time step: one Picard map of the Gonzalez
# discrete-gradient (or Lenard–Bernstein) update, and the two solvers for its
# fixed point: Anderson-accelerated Picard and Jacobian-free Newton–Krylov.
# Pure compute: no file I/O, no plotting.
#
# Needs `Workspace` plus the physics in functions.jl, so main.jl includes this
# after MantisWrappers. The CPU/GPU hot-loop hooks live here because this is
# their only hot call site; main() repoints them at startup.

using LinearAlgebra: norm, mul!, ldiv!

include("newton_krylov.jl")   # generic GMRES + JFNK, wired up by step_newton!

"""
    COLLISION_FN, L2PROJ_FN, COMPG_FN, LOGGRAD_FN

Swappable implementations of the four hot-loop kernels, held in `Ref`s so the
backend can be chosen at startup rather than through the type system.

They default to the CPU routines from `functions.jl`. Passing `--use_gpu=true`
makes `main` load `collision_gpu.jl` and `projection_gpu.jl` and repoint these at
the CUDA kernels: the ``O(N^2)`` Landau pair sum (`COLLISION_FN`, FP64 or FP32),
the particle-to-spline L2 projection (`L2PROJ_FN`), the spline gradient gather
(`COMPG_FN`) and, for Lenard–Bernstein, the direct log-density gradient
(`LOGGRAD_FN`).

Every call site goes through `Base.invokelatest` so the swap is visible to code
that was already compiled, which a plain call would miss under world-age.
"""
const COLLISION_FN = Ref{Any}(compute_collision!)
const L2PROJ_FN = Ref{Any}(l2_project!)
const COMPG_FN = Ref{Any}(compute_G!)
const LOGGRAD_FN = Ref{Any}(eval_loggrad_at_particles!)   # LB base gradient

# A docstring before `const` attaches to that binding alone, so alias the shared
# one onto the other three; otherwise `@ref` to them has nothing to resolve.
@doc (@doc COLLISION_FN) L2PROJ_FN
@doc (@doc COLLISION_FN) COMPG_FN
@doc (@doc COLLISION_FN) LOGGRAD_FN

@doc raw"""
    compute_entropy_gradient!(ws, dS, v_parts, w_parts, f_coeffs_buf, r_vec, L_vec, G_buf)

Entropy gradient with respect to every particle velocity,

```math
\frac{\partial S_h}{\partial v_\alpha} = -w_\alpha G_\alpha ,
```

where ``G_\alpha = \nabla (M^{-1} r)(v_\alpha)`` is the finite-element projection of
``\nabla \log f_s`` evaluated at particle ``\alpha`` and ``w_\alpha`` is its weight.

The projection chain is ``v \mapsto f_s`` (L2 projection), ``f_s \mapsto r``, then
``r \mapsto M^{-1} r`` through the cached mass-matrix factorisation `ws.M_lu`. Both
the projection and the gradient evaluation go through [`L2PROJ_FN`](@ref) and
[`COMPG_FN`](@ref), so this routine runs on the GPU when those are repointed.

Writes `dS` in place; `f_coeffs_buf`, `r_vec`, `L_vec` and `G_buf` are scratch.
"""
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

@doc raw"""
    picard_map!(ws, v_out, v_in, v0, w_parts, S0, dt, v_mid, dv, dS_mid, G_eff,
                dot_v_buf, f_buf, r_vec, L_vec, G_buf; use_gonzalez = true)

One Picard map of the implicit step. Given a trial end-of-step state ``v_\text{in}``
it returns the next iterate

```math
v_\text{out} = v_0 + \Delta t \, \dot{v}(v_\text{mid}, \overline{\nabla} S) ,
\qquad
v_\text{mid} = \tfrac{1}{2}\,(v_0 + v_\text{in}) ,
\qquad
\Delta v = v_\text{in} - v_0 ,
```

so a fixed point ``v_\text{out} = v_\text{in}`` is a solution of the implicit step.
[`step_anderson!`](@ref) is what drives it to that fixed point.

# Discrete gradient

With `use_gonzalez = true` the entropy gradient carries the Gonzalez correction

```math
\overline{\nabla} S = \nabla S(v_\text{mid}) + \lambda \, \Delta v ,
\qquad
\lambda = \frac{S(v_\text{in}) - S_0 - \Delta v \cdot \nabla S(v_\text{mid})}
               {\lVert \Delta v \rVert^2} ,
```

with ``S_0 = S(v_0)`` passed in as `S0`. The rank-one term is chosen so that the
discrete chain rule

```math
\overline{\nabla} S \cdot \Delta v = S(v_\text{in}) - S_0
```

holds *exactly* rather than to within truncation error, which is what makes the
scheme's entropy production discrete-exact.

`use_gonzalez = false` drops ``\lambda`` and leaves the plain implicit-midpoint rule
``\overline{\nabla} S = \nabla S(v_\text{mid})``. That removes the
``\lVert \Delta v \rVert^2`` denominator, which is ill-conditioned when the run starts
at or near equilibrium (``\Delta v \to 0``); in both branches the denominator is
guarded by a ``10^{-30}`` floor.

# Collision model

`ws.p.collision_model` selects how ``\dot{v}`` is assembled from ``\overline{\nabla} S``:

- `:landau` — the ``O(N^2)`` pair sum through [`COLLISION_FN`](@ref), on a
  finite-element entropy gradient.
- `:lb` — the ``O(N)`` Lenard–Bernstein drift, built from `compute_moments` and
  `compute_drift_multipliers` so momentum and energy are conserved exactly. LB
  uses the direct clamped ``\nabla f_s / f_s`` as its base gradient in both modes:
  the FEM-projected seed injects Gibbs ringing into low-density cells, which
  Landau's ``f``-weighted mobility suppresses and LB's does not. The
  discrete-gradient identity holds for *any* base gradient, so Gonzalez only adds
  the rank-one ``\lambda`` term on top.

Writes `v_out` in place; every other array argument is scratch.
"""
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

@doc raw"""
    step_anderson!(ws, v1, v0, w_parts, S0, dt, ...; m = 5, max_iter = 1000,
                   tol = 1e-12, abs_floor = 1e-7, damping = 0.5, ...)
        -> (iterations, residual, n_restarts, initial_residual)

Drive [`picard_map!`](@ref) to its fixed point with Anderson acceleration. `v1`
enters as the initial guess and leaves holding the solution. With
`use_anderson = false` this degrades to plain damped Picard.

# Anderson acceleration

Writing the Picard map as ``G`` and the residual as ``r_k = G(v_k) - v_k``, the last
``m`` differences are kept in ``\Delta F`` and ``\Delta G``. Each iteration solves the
small least-squares problem

```math
\gamma = \arg\min_\gamma \lVert r_k - \Delta F \gamma \rVert_2^2 ,
\qquad
v_{k+1} = (1 - \beta)\, v_k + \beta \left( G(v_k) - \Delta G \gamma \right) ,
```

with `damping` ``\beta`` blending the accelerated step back toward the raw Picard
update. The normal equations are regularised with
``\lambda^2 = \texttt{reg\_factor} \cdot \operatorname{mean}(\operatorname{diag}(\Delta F^\top \Delta F))``
so the ``m \times m`` system stays solvable when ``\Delta F`` is near rank-deficient.

The window is a ring buffer: the newest difference overwrites the oldest column
rather than shifting the whole window left, which is valid because the
least-squares problem is invariant under a common column permutation of
``\Delta F`` and ``\Delta G``. See [The Anderson window update](@ref) for the measured cost.

# Exit conditions

Any one of these ends the solve successfully. With `exit_picard_step = true`
(the default) `v1` leaves holding ``G(v)`` for the converged iterate ``v``, or
for the best one seen on a stagnation or cap exit. That final Picard update is
what keeps the step conservative. With ``m = \tfrac12(v_0 + v)`` and
``F = v - G(v)``, the collision operator conserves momentum and energy at ``m``,
so

```math
P(G(v)) - P(v_0) = 0 , \qquad
E(G(v)) - E(v_0) = -\tfrac{\Delta t}{2} \textstyle\sum_\alpha w_\alpha F_\alpha \cdot \dot v_\alpha(m) ,
```

whereas returning ``v`` itself leaves ``\sum_\alpha w_\alpha F_\alpha`` and
``\sum_\alpha w_\alpha m_\alpha \cdot F_\alpha``, first order in the residual. The
returned residual is ``\lVert F(v) \rVert``, not that of ``G(v)``. Along the stiff
direction of the Landau map the update amplifies the residual (a median of about
2x over 30 FP64 steps, `scripts/exit_update_check.jl`), which is the price of
exact momentum conservation. `exit_picard_step = false` returns ``v`` instead;
an FP32 A/B showed no cost benefit from it, and 3–10x worse energy drift
(`docs/src/newton_krylov.md`).

1. **Residual.** ``\lVert r \rVert < \max(\texttt{tol} \cdot \lVert v \rVert, \texttt{abs\_floor})``.
   `abs_floor` caps how tight the solve is asked to be: below the numerical noise
   floor of the Picard map the extra iterations buy nothing. In FP32 the floor
   matters a great deal: `abs_floor = 1e-8` cuts the iteration count by over 3x
   against the `1e-10` the presets carry, at no cost in accuracy.
2. **Stagnation.** Every `stag_window` iterations the best residual is compared
   with its value one window earlier; a relative drop below `stag_rel_tol` exits.
   This catches the late-step plateau where Anderson cannot push further. An
   iteration count that is an exact multiple of `stag_window` is the signature of
   a step that left by this route rather than by converging.
3. **Iteration cap.** `max_iter`, which warns.

Past `damp_decay_start` iterations without an exit, `damping` is multiplied by
`damp_decay_factor` for a more conservative step, which stabilises stiff
late-time maps. If `restart_factor` is finite, the window is cleared whenever the
residual exceeds that multiple of the best residual so far.
"""
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
        use_anderson::Bool = true, use_gonzalez::Bool = true,
        exit_picard_step::Bool = true)
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

    # Track the best iterate so stagnation / max_iter exits return it rather than
    # the latest (which may be worse on a non-monotone trajectory). By default
    # that is G(v), whose final Picard update restores momentum and energy
    # conservation; `exit_picard_step = false` keeps the iterate v instead.
    v_best = copy(v1)
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
            v_best .= exit_picard_step ? Gv : v1
        end

        # Convergence: relative tol but never tighter than the absolute floor.
        eff_tol = max(tol * (norm(v1_v) + 1e-30), abs_floor)
        if nrm_r < eff_tol
            exit_picard_step && (v1 .= Gv)
            verbose && println("    k=$k  ‖r‖=$nrm_r  history=$history  [converged]")
            return k, nrm_r, n_restart, nrm_r0
        end

        # Stagnation early-exit: insufficient progress over a window of iters.
        if k > stag_window && k % stag_window == 0
            rel_improve = (nrm_best_window - nrm_best) /
                          (nrm_best_window + 1e-30)
            if rel_improve < stag_rel_tol
                v1 .= v_best
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

    # max_iter exhausted: return the best iterate, not the latest.
    v1 .= v_best
    @warn "Solver did not converge" max_iter tol abs_floor nrm_r0 nrm_r nrm_best n_restart
    return max_iter, nrm_best, n_restart, nrm_r0
end

@doc raw"""
    step_newton!(ws, v1, v0, w_parts, S0, dt, ..., Gv, nk::NKWorkspace;
                 max_iter = 1000, tol = 1e-12, abs_floor = 1e-7, fd_h = 1e-5,
                 fd_rel = 0.0, eta_max = 0.9, verbose = false,
                 use_gonzalez = true, exit_picard_step = true)

Solve the same implicit step as [`step_anderson!`](@ref) by Jacobian-free
Newton–Krylov ([`newton_krylov!`](@ref)) on the residual of the Picard map,

```math
F(v) = v - \mathcal{G}(v), \qquad \mathcal{G} = \texttt{picard\_map!} ,
```

from the predictor already in `v1`. The Jacobian ``I - \partial\mathcal{G}/\partial v``
is ``I - O(\Delta t)``, so GMRES needs few iterations per Newton step.

The stopping rule and the cost unit match `step_anderson!`, so the two are
directly comparable: the target is
``\max(\texttt{tol}\cdot\lVert v \rVert, \texttt{abs\_floor})``, `max_iter`
caps the number of Picard-map evaluations, and the returned count is the number
of evaluations — each finite-difference Jacobian product and each line-search
trial is one. On exit `v1` takes the final Picard update
``\mathcal{G}(v) = v - F(v)``, as in `step_anderson!`, which restores momentum and
energy conservation; `exit_picard_step = false` returns ``v`` itself.

The finite-difference step is the absolute `fd_h` (see [`newton_krylov!`](@ref)).
`fd_rel > 0` overrides it with the earlier relative scaling
``h = \texttt{fd\_rel}\cdot\max(\lVert v \rVert, 1)``, for reproducing old runs.

Returns `(n_evals, ‖F‖, n_newton, ‖F₀‖)`; the third slot is the restart count in
`step_anderson!`.
"""
function step_newton!(ws::Workspace,
        v1, v0, w_parts, S0, dt,
        v_mid, dv, dS_mid, G_eff, dot_v_buf, f_buf,
        r_vec, L_vec, G_buf, Gv, nk::NKWorkspace;
        max_iter = 1000, tol = 1e-12, abs_floor = 1e-7, fd_h = 1e-5,
        fd_rel = 0.0, eta_max = 0.9, verbose = false, use_gonzalez::Bool = true,
        exit_picard_step::Bool = true)
    N = size(v0, 1)
    Gv_v = vec(Gv)
    function residual!(F, v)
        picard_map!(ws, Gv, reshape(v, N, 2), v0, w_parts, S0, dt,
            v_mid, dv, dS_mid, G_eff, dot_v_buf, f_buf,
            r_vec, L_vec, G_buf; use_gonzalez = use_gonzalez)
        @. F = v - Gv_v
        return F
    end

    v1_v = vec(v1)
    eff_tol = max(tol * (norm(v1_v) + 1e-30), abs_floor)
    h = fd_rel > 0 ? fd_rel * max(norm(v1_v), 1.0) : fd_h
    n_evals, nrm, n_newton, nrm0, status = newton_krylov!(v1_v, residual!, nk;
        tol = eff_tol, max_evals = max_iter, fd_h = h,
        eta_max = eta_max, verbose = verbose)
    exit_picard_step && (@. v1_v -= nk.F)
    verbose && println("    [$status]  evals=$n_evals  newton=$n_newton  ‖F‖=$nrm")
    return n_evals, nrm, n_newton, nrm0
end
