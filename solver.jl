# solver.jl — the implicit time step: one Picard map of the Gonzalez
# discrete-gradient (or Lenard–Bernstein) update, and the solvers for its fixed
# point: Anderson-accelerated Picard from GeometricSolvers, Jacobian-free
# Newton–Krylov and defect correction. Pure compute: no file I/O, no plotting.
#
# Needs `Workspace` plus the physics in functions.jl, so main.jl includes this
# after MantisWrappers. The CPU/GPU hot-loop hooks live here because this is
# their only hot call site; main() repoints them at startup.

using LinearAlgebra: norm, mul!, ldiv!
using GeometricSolvers: GeometricSolvers, Anderson, NonlinearProblem, Picard, SUCCESS

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
const COLLMETRIC_FN = Ref{Any}(compute_collision_metric!)  # F and A, for step_defect!

# A docstring before `const` attaches to that binding alone, so alias the shared
# one onto the other three; otherwise `@ref` to them has nothing to resolve.
@doc (@doc COLLISION_FN) L2PROJ_FN
@doc (@doc COLLISION_FN) COMPG_FN
@doc (@doc COLLISION_FN) LOGGRAD_FN
@doc (@doc COLLISION_FN) COLLMETRIC_FN

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
    landau_geff!(ws, G_eff, v_in, v0, w_parts, S0, v_mid, dv, dS_mid, f_buf, r_vec,
                 L_vec, G_buf; use_gonzalez = true)

The Landau discrete gradient ``\overline{\nabla} S`` of [`picard_map!`](@ref), as
`G_eff = -(∇S(v_mid) + λ Δv) / w`: the FEM entropy gradient at the midpoint of `v0` and
`v_in` plus the Gonzalez rank-one term (`λ = 0` with `use_gonzalez = false`). Sets
`v_mid` and `dv`; everything up to the collision sum, so [`step_defect!`](@ref) can
re-evaluate it without the pair sum.
"""
function landau_geff!(ws::Workspace, G_eff, v_in, v0, w_parts, S0, v_mid, dv, dS_mid,
        f_buf, r_vec, L_vec, G_buf; use_gonzalez::Bool = true)
    N = size(v0, 1)
    @. v_mid = 0.5 * (v0 + v_in)
    @. dv = v_in - v0
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
    return G_eff
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
[`step_fixed_point!`](@ref) is what drives it to that fixed point.

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
        landau_geff!(ws, G_eff, v_in, v0, w_parts, S0, v_mid, dv, dS_mid,
            f_buf, r_vec, L_vec, G_buf; use_gonzalez)
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

"""
    fixed_point_method(; m = 8, damping = 0.7, use_anderson = true)

The `GeometricSolvers` method of [`fixed_point_solve!`](@ref):
`Anderson(Picard(; damping), m; restart_factor = 1e12)`, or `Picard(; damping)` with
`use_anderson = false`.
"""
function fixed_point_method(; m = 8, damping = 0.7, use_anderson::Bool = true)
    return use_anderson ? Anderson(Picard(; damping), m; restart_factor = 1e12) :
           Picard(; damping)
end

@doc raw"""
    fixed_point_solve!(mapf!, v1, Gv; m = 8, max_iter = 2000, tol = 1e-12,
                       abs_floor = 1e-7, damping = 0.7, use_anderson = true,
                       exit_picard_step = true, verbose = false)
        -> (evaluations, residual, iterations, initial_residual)

Solve the fixed point ``v = \mathcal{G}(v)`` of the map `mapf!(Gv, v)`, which writes
``\mathcal{G}(v)`` into `Gv`, by `GeometricSolvers.solve` on the residual

```math
F(v) = v - \mathcal{G}(v) .
```

`v1` enters as the initial guess ``v_0`` and leaves holding the solution; `Gv` has the
shape of `v1` and is scratch. `GeometricSolvers`' `Picard` steps ``v - \omega F(v)``,
which for this sign of ``F`` is the damped Picard update
``(1 - \omega)\, v + \omega\, \mathcal{G}(v)`` with `damping` ``\omega``. With
`use_anderson = true` the method is `Anderson(Picard(; damping), m)`, type-II Anderson
acceleration of history depth `m` over that update, with `restart_factor = 1e12`, which
keeps its restart on residual growth off in practice; with `use_anderson = false` it is
`Picard(; damping)` alone.

The solve succeeds when
``\lVert F(v) \rVert \le \max(\texttt{tol} \cdot \lVert v_0 \rVert, \texttt{abs\_floor})``,
and stops after `max_iter` iterations of the method. A solve that ends with a code other
than `SUCCESS` keeps the iterate with the smallest ``\lVert F \rVert`` seen, warns, and
returns. `abs_floor` caps how tight the solve is asked to be: below the numerical noise
floor of the Picard map the extra iterations buy nothing.

With `exit_picard_step = true` (the default) `v1` leaves holding ``\mathcal{G}(v)`` for
the returned iterate ``v``, one more evaluation of the map. That final Picard update is
what keeps the step conservative. With ``\bar v = \tfrac12(v_0 + v)``, the collision
operator conserves momentum and energy at ``\bar v``, so

```math
P(\mathcal{G}(v)) - P(v_0) = 0 , \qquad
E(\mathcal{G}(v)) - E(v_0) = -\tfrac{\Delta t}{2} \textstyle\sum_\alpha w_\alpha F_\alpha \cdot \dot v_\alpha(\bar v) ,
```

whereas returning ``v`` itself leaves ``\sum_\alpha w_\alpha F_\alpha`` and
``\sum_\alpha w_\alpha \bar v_\alpha \cdot F_\alpha``, first order in the residual. The
returned residual is ``\lVert F(v) \rVert``, not that of ``\mathcal{G}(v)``.
`exit_picard_step = false` returns ``v`` instead; an FP32 A/B showed no cost benefit
from it, and 3–10x worse energy drift (`docs/src/newton_krylov.md`).

Returns the number of evaluations of `mapf!`, the final ``\lVert F(v) \rVert``, the
iterations of the method and ``\lVert F(v_0) \rVert``.
"""
function fixed_point_solve!(mapf!, v1, Gv; m = 8, max_iter = 2000, tol = 1e-12,
        abs_floor = 1e-7, damping = 0.7, use_anderson::Bool = true,
        exit_picard_step::Bool = true, verbose = false)
    v1_v = vec(v1)
    Gv_v = vec(Gv)
    n_evals = Ref(0)
    nrm0 = Ref(NaN)
    function residual!(F, v, _)
        mapf!(Gv, reshape(v, size(v1)))
        @. F = v - Gv_v
        n_evals[] += 1
        n_evals[] == 1 && (nrm0[] = norm(F))
        return F
    end
    method = fixed_point_method(; m, damping, use_anderson)
    v, status = GeometricSolvers.solve(NonlinearProblem(residual!), v1_v, method;
        f_abstol = max(tol * norm(v1_v), abs_floor), f_reltol = 0,
        max_iterations = max_iter)
    if exit_picard_step
        mapf!(Gv, reshape(v, size(v1)))
        n_evals[] += 1
        v1_v .= Gv_v
    else
        v1_v .= v
    end
    code, iterations, nrm = status.code, Int(status.iterations), status.fnorm
    code == SUCCESS ||
        @warn "Solver did not converge" code max_iter tol abs_floor nrm0[] nrm iterations
    verbose && println("    [$code]  iterations=$iterations  evals=$(n_evals[])  ‖F‖=$nrm")
    return n_evals[], nrm, iterations, nrm0[]
end

@doc raw"""
    step_fixed_point!(ws, v1, v0, w_parts, S0, dt, v_mid, dv, dS_mid, G_eff, dot_v_buf,
                      f_buf, r_vec, L_vec, G_buf, Gv; use_gonzalez = true, kwargs...)
        -> (evaluations, residual, iterations, initial_residual)

Drive [`picard_map!`](@ref) to its fixed point with [`fixed_point_solve!`](@ref), from
the predictor already in `v1`, which leaves holding the solution. The keywords other than
`use_gonzalez` are those of `fixed_point_solve!`; every array argument but `v1`, `v0` and
`w_parts` is scratch.
"""
function step_fixed_point!(ws::Workspace,
        v1, v0, w_parts, S0, dt,
        v_mid, dv, dS_mid, G_eff, dot_v_buf, f_buf,
        r_vec, L_vec, G_buf, Gv; use_gonzalez::Bool = true, kwargs...)
    function mapf!(out, v)
        picard_map!(ws, out, v, v0, w_parts, S0, dt,
            v_mid, dv, dS_mid, G_eff, dot_v_buf, f_buf,
            r_vec, L_vec, G_buf; use_gonzalez = use_gonzalez)
        return out
    end
    return fixed_point_solve!(mapf!, v1, Gv; kwargs...)
end

@doc raw"""
    step_newton!(ws, v1, v0, w_parts, S0, dt, ..., Gv, nk::NKWorkspace;
                 max_iter = 1000, tol = 1e-12, abs_floor = 1e-7, fd_h = 1e-5,
                 fd_rel = 0.0, eta_max = 0.9, verbose = false,
                 use_gonzalez = true, exit_picard_step = true)

Solve the same implicit step as [`step_fixed_point!`](@ref) by Jacobian-free
Newton–Krylov ([`newton_krylov!`](@ref)) on the residual of the Picard map,

```math
F(v) = v - \mathcal{G}(v), \qquad \mathcal{G} = \texttt{picard\_map!} ,
```

from the predictor already in `v1`. The Jacobian ``I - \partial\mathcal{G}/\partial v``
is ``I - O(\Delta t)``, so GMRES needs few iterations per Newton step.

The stopping rule and the cost unit match `step_fixed_point!`, so the two are
directly comparable: the target is
``\max(\texttt{tol}\cdot\lVert v \rVert, \texttt{abs\_floor})``, with
``\lVert v \rVert`` taken once at the predictor, and the returned count is the
number of Picard-map evaluations — each finite-difference Jacobian product and
each line-search trial is one. `max_iter` caps that count here, and the iterations
of the method in `step_fixed_point!`. On exit `v1` takes the final Picard update
``\mathcal{G}(v) = v - F(v)``, as in `step_fixed_point!`, which restores momentum
and energy conservation; `exit_picard_step = false` returns ``v`` itself.

The finite-difference step is the absolute `fd_h` (see [`newton_krylov!`](@ref)).
`fd_rel > 0` overrides it with the earlier relative scaling
``h = \texttt{fd\_rel}\cdot\max(\lVert v \rVert, 1)``, for reproducing old runs.

Returns `(n_evals, ‖F‖, n_newton, ‖F₀‖)`; the third slot is the iteration count of
the method in `step_fixed_point!`.
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

# Anderson-accelerated fixed-point iteration u = map(u), with a ring window of
# the last m differences, damping and a regularised least-squares solve, but no
# stagnation logic: it stops at
# ‖map(u) − u‖ < tol or after `maxit` maps, and leaves `u` holding map(u).
# Returns the number of map evaluations.
function _anderson_inner!(mapf!, u, Fu, r, rp, Fp, uold, ΔF, ΔG, tol, maxit, m, damping;
        reg_factor = 1e-10)
    u_v, Fu_v, r_v = vec(u), vec(Fu), vec(r)
    hist = 0
    slot = 0
    for k in 1:maxit
        uold .= u
        mapf!(Fu, u)
        @. r = Fu - u
        if norm(r_v) < tol
            u .= Fu
            return k
        end
        if k == 1
            @. u = damping * Fu + (1 - damping) * uold
        else
            slot = mod1(slot + 1, m)
            hist = min(hist + 1, m)
            @views ΔF[:, slot] .= r_v .- vec(rp)
            @views ΔG[:, slot] .= Fu_v .- vec(Fp)
            ΔFv = @view ΔF[:, 1:hist]
            ATA = ΔFv' * ΔFv
            λ2 = reg_factor * sum(ATA[j, j] for j in 1:hist) / hist + 1e-30
            for j in 1:hist
                ATA[j, j] += λ2
            end
            u .= Fu
            mul!(u_v, view(ΔG, :, 1:hist), ATA \ (ΔFv' * r_v), -1.0, 1.0)
            @. u = damping * u + (1 - damping) * uold
        end
        rp .= r
        Fp .= Fu
    end
    return maxit
end

@doc raw"""
    step_defect!(ws, v1, v0, w_parts, S0, dt, v_mid, dv, dS_mid, G_eff, dot_v_buf,
                 f_buf, r_vec, L_vec, G_buf, Gv, r_curr, r_prev, Gv_prev, v_old, ΔF, ΔG,
                 A, Gk, Φu; m = 8, max_iter = 2000, tol = 1e-12, abs_floor = 1e-7,
                 damping = 0.7, eta = 0.05, max_inner = 60, stag_window = 5,
                 stag_rel_tol = 0.1, use_gonzalez = true, verbose = false)
        -> (outer_iterations, residual, inner_iterations, initial_residual)

Solve the Landau implicit step of [`step_fixed_point!`](@ref) by defect correction around
a frozen metric. The collision velocity splits as

```math
\dot v_\gamma = \sum_\alpha w_\alpha U(v_\gamma - v_\alpha)(G_\alpha - G_\gamma)
             = B_\gamma - A_\gamma G_\gamma ,
\qquad A_\gamma = \sum_\alpha w_\alpha U(v_\gamma - v_\alpha) ,
```

where the pair sums ``A`` and ``B`` change slowly with the particle positions while the
entropy gradient ``G`` (an ``O(N)`` projection) changes fast. Outer iteration ``k``
evaluates the full Picard map once, ``\mathcal{G}(v_k)``, together with ``A_k`` in the
same pair pass ([`COLLMETRIC_FN`](@ref)), and then solves the cheap map

```math
\Phi_k(u) = \mathcal{G}(v_k) - \Delta t\, A_k \bigl(G_\text{eff}(u) - G_\text{eff}(v_k)\bigr)
```

by Anderson to ``\lVert \Phi_k(u) - u \rVert < \max(\eta \lVert \mathcal{G}(v_k) - v_k \rVert,
\texttt{abs\_floor}/2)``. ``\Phi_k(v_k) = \mathcal{G}(v_k)``, so a fixed point of the outer
iteration is a fixed point of ``\mathcal{G}``: the root is the same as `step_fixed_point!`'s.
Each ``\Phi_k`` evaluation needs ``G_\text{eff}`` ([`landau_geff!`](@ref)) but no pair sum.

The solve stops when the outer residual ``\lVert \mathcal{G}(v) - v \rVert`` falls below
``\max(\texttt{tol} \cdot \lVert v \rVert, \texttt{abs\_floor})``, and the exit returns
``\mathcal{G}(v)`` computed with the antisymmetric pair sum, so momentum and energy are
conserved exactly as in `step_fixed_point!`. Every `stag_window` outer
iterations, a best residual that fell by less than `stag_rel_tol` ends the solve with
the best ``\mathcal{G}(v)``. `max_iter` caps the outer iterations.

Measured (N = 40k, FP64, RTX 3080 probe): 6–11 outer iterations per step against
10–120 Anderson iterations, each ``\Phi_k`` costing about 2 % of a full map. In FP32 the
full map is cheap and this does not pay.
"""
function step_defect!(ws::Workspace,
        v1, v0, w_parts, S0, dt,
        v_mid, dv, dS_mid, G_eff, dot_v_buf, f_buf,
        r_vec, L_vec, G_buf,
        Gv, r_curr, r_prev, Gv_prev, v_old, ΔF, ΔG, A, Gk, Φu;
        m = 8, max_iter = 2000, tol = 1e-12, abs_floor = 1e-7, damping = 0.7,
        eta = 0.05, max_inner = 60, stag_window = 5, stag_rel_tol = 0.1,
        use_gonzalez::Bool = true, verbose = false)
    N = size(v0, 1)
    nrm0 = 0.0
    nrm_best = Inf
    nrm_best_window = Inf
    v_best = copy(v1)
    inner = 0

    # Φ_k(u) = 𝒢(v_k) − Δt A_k (G_eff(u) − G_k); Gv holds 𝒢(v_k), Gk holds G_k.
    function frozen_map!(out, u)
        landau_geff!(ws, G_eff, u, v0, w_parts, S0, v_mid, dv, dS_mid,
            f_buf, r_vec, L_vec, G_buf; use_gonzalez)
        @inbounds for a in 1:N
            d1 = G_eff[a, 1] - Gk[a, 1]
            d2 = G_eff[a, 2] - Gk[a, 2]
            out[a, 1] = Gv[a, 1] - dt * (A[a, 1] * d1 + A[a, 2] * d2)
            out[a, 2] = Gv[a, 2] - dt * (A[a, 2] * d1 + A[a, 3] * d2)
        end
        return out
    end

    for k in 1:max_iter
        landau_geff!(ws, G_eff, v1, v0, w_parts, S0, v_mid, dv, dS_mid,
            f_buf, r_vec, L_vec, G_buf; use_gonzalez)
        Base.invokelatest(COLLMETRIC_FN[], ws, dot_v_buf, A, v_mid, w_parts, G_eff)
        @. Gv = v0 + dt * dot_v_buf
        @. r_curr = Gv - v1
        nrm = norm(r_curr)
        k == 1 && (nrm0 = nrm)
        if nrm < nrm_best
            nrm_best = nrm
            v_best .= Gv
        end

        if nrm < max(tol * (norm(v1) + 1e-30), abs_floor)
            v1 .= Gv
            verbose && println("    outer k=$k  ‖r‖=$nrm  inner=$inner  [converged]")
            return k, nrm, inner, nrm0
        end
        if k > stag_window && k % stag_window == 0
            if (nrm_best_window - nrm_best) / (nrm_best_window + 1e-30) < stag_rel_tol
                v1 .= v_best
                verbose && println("    outer k=$k  stagnated  nrm_best=$nrm_best")
                return k, nrm_best, inner, nrm0
            end
            nrm_best_window = nrm_best
        end

        Gk .= G_eff
        n_in = _anderson_inner!(frozen_map!, v1, Φu, r_curr, r_prev, Gv_prev, v_old,
            ΔF, ΔG, max(eta * nrm, 0.5 * abs_floor), max_inner, m, damping)
        inner += n_in
        verbose && println("    outer k=$k  ‖r‖=$nrm  inner +$n_in")
    end

    v1 .= v_best
    @warn "Defect correction did not converge" max_iter tol abs_floor nrm0 nrm_best inner
    return max_iter, nrm_best, inner, nrm0
end
