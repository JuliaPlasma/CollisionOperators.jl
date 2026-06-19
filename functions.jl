# Physics + diagnostics routines. All take `ws::Workspace` as first argument.

# Toggle for the log-square identity  log f = ½ log f².  When true, the entropy
# and entropy-gradient-seed integrands use ½·log(f²), so quadrature points where
# the projected f_s is slightly NEGATIVE still contribute (guard on |f|, i.e.
# f² > floor) instead of being clamped to zero by the `f_val > 1e-30` positivity
# test. Set once per run from `PARAMS.use_logsq` in `run_simulation`.
const USE_LOGSQ = Ref(false)

"""
    l2_project!(ws, f_coeffs, v_parts, w_parts)

L² projection of the weighted Dirac measure Σ w_α δ(v − v_α) onto X⁰: solves
`M c = b` with `b_k = Σ_α w_α φ_k(v_α)`, giving `f_s(v) = Σ c_k φ_k(v)`. The mass
system is solved via the prefactored `ws.M_lu`. Writes the result into
`f_coeffs` (overwritten); out-of-domain particles are skipped.
"""
function l2_project!(ws::Workspace, f_coeffs, v_parts, w_parts)
    rhs = zeros(ws.n_dofs)
    nloc = (ws.p.P_DEG + 1)^2
    for α in axes(v_parts, 1)
        loc = locate_particle(ws, v_parts[α, 1], v_parts[α, 2])
        isnothing(loc) && continue
        fast_eval_particle!(ws, ws.lp_vals, ws.lp_gids, loc)
        @inbounds for j in 1:nloc
            rhs[ws.lp_gids[j]] += w_parts[α] * ws.lp_vals[j]
        end
    end
    f_coeffs .= ws.M_lu \ rhs
    return nothing
end

"""
    compute_entropy(ws, field) -> S

Discrete entropy `S_h = -∫ f_s log f_s dv` (sign convention: dS/dt ≥ 0), via
quadrature over all elements. Cells where `f_s ≤ 1e-30` are skipped (log
singularity / negative undershoots excluded). Read-only.
"""
function compute_entropy(ws::Workspace, field::Forms.FormField)
    S = 0.0
    for e in 1:ws.n_elements
        jac = element_measure(ws, e)
        fv, _ = evaluate(ws, field, e)
        for q in eachindex(ws.qrule_integrate.weights)
            f_val = fv[1][q]
            if USE_LOGSQ[]
                # log f = ½ log f²: keep f<0 points via |f| guard.
                if f_val^2 > 1e-60
                    S -= f_val * 0.5 * log(f_val^2) * ws.qrule_integrate.weights[q] * jac
                end
            elseif f_val > 1e-30
                S -= f_val * log(f_val) * ws.qrule_integrate.weights[q] * jac
            end
        end
    end
    return S
end

"""
    compute_r!(ws, r, field)

Entropy-gradient seed `r_i = ∫ φ_i (1 + log f_s) dv`, assembled by quadrature
over all elements. Cells where `f_s ≤ 1e-30` contribute 0. Writes the result
into `r` (overwritten); `L = M⁻¹ r` is the field-side entropy gradient.
"""
function compute_r!(ws::Workspace, r, field::Forms.FormField)
    fill!(r, 0.0)
    for e in 1:ws.n_elements
        jac = element_measure(ws, e)
        fv, _ = evaluate(ws, field, e)
        evals, indices = evaluate(ws, e)
        for q in eachindex(ws.qrule_integrate.weights)
            f_val = fv[1][q]
            integrand = if USE_LOGSQ[]
                # 1 + log f = 1 + ½ log f²: f<0 points contribute via |f| guard.
                f_val^2 > 1e-60 ? (1 + 0.5 * log(f_val^2)) : 0.0
            else
                f_val > 1e-30 ? (1 + log(f_val)) : 0.0
            end
            for (j, gidx) in enumerate(indices[1])
                r[gidx] += integrand * evals[1][q, j] *
                          ws.qrule_integrate.weights[q] * jac
            end
        end
    end
    return nothing
end

"""
    compute_G!(ws, G, v_parts, L_vec)

Particle-side entropy gradient `G_α = ∇L(v_α)` where `L = M⁻¹ r` is the FE
coefficient vector `L_vec`. Evaluates the spline gradient at each particle and
maps reference-cell derivatives to physical space via the cell sizes `h1, h2`.
Writes the N×2 result into `G` (overwritten); out-of-domain particles → 0.
"""
function compute_G!(ws::Workspace, G, v_parts, L_vec)
    fill!(G, 0.0)
    nloc = (ws.p.P_DEG + 1)^2
    for α in axes(v_parts, 1)
        loc = locate_particle(ws, v_parts[α, 1], v_parts[α, 2])
        isnothing(loc) && continue
        fast_eval_particle_grad!(ws, ws.G_vals, ws.G_dxi1, ws.G_dxi2, ws.G_gids, loc)
        inv_h1 = 1.0 / loc.h1
        inv_h2 = 1.0 / loc.h2
        acc1 = 0.0; acc2 = 0.0
        @inbounds for j in 1:nloc
            L = L_vec[ws.G_gids[j]]
            acc1 += L * ws.G_dxi1[j] * inv_h1
            acc2 += L * ws.G_dxi2[j] * inv_h2
        end
        G[α, 1] = acc1
        G[α, 2] = acc2
    end
    return nothing
end

"""
    compute_collision!(ws, dot_v, v_parts, w_parts, G)

Landau collision-operator velocity update: for each particle γ accumulate the
projected pairwise interaction `Σ_α w_α (g − d (d·g)/|d|²)/|d|` with
`d = v_γ − v_α`, `g = G_α − G_γ`. The `(I − dd̂ᵀ)/|d|` projection is the 2D
Landau kernel. Writes the N×2 result into `dot_v` (overwritten); particles on or
outside the domain boundary are skipped. Threaded over γ.
"""
function compute_collision!(ws::Workspace, dot_v, v_parts, w_parts, G)
    v1_lo, v1_hi = ws.bp1[1], ws.bp1[end]
    v2_lo, v2_hi = ws.bp2[1], ws.bp2[end]
    fill!(dot_v, 0.0)
    N = size(v_parts, 1)
    Threads.@threads for γ in 1:N
        vγ1, vγ2 = v_parts[γ, 1], v_parts[γ, 2]
        (vγ1 <= v1_lo || vγ1 >= v1_hi ||
         vγ2 <= v2_lo || vγ2 >= v2_hi) && continue
        Gγ1, Gγ2 = G[γ, 1], G[γ, 2]
        acc1, acc2 = 0.0, 0.0
        for α in 1:N
            γ == α && continue
            vα1, vα2 = v_parts[α, 1], v_parts[α, 2]
            (vα1 <= v1_lo || vα1 >= v1_hi ||
             vα2 <= v2_lo || vα2 >= v2_hi) && continue
            d1 = vγ1 - vα1
            d2 = vγ2 - vα2
            dist2 = d1^2 + d2^2
            dist2 < 1e-24 && continue
            dist = sqrt(dist2)
            g1 = G[α, 1] - Gγ1
            g2 = G[α, 2] - Gγ2
            dv_dot_g = (d1 * g1 + d2 * g2) / dist2
            inv_dist = 1.0 / dist
            acc1 += w_parts[α] * (g1 - d1 * dv_dot_g) * inv_dist
            acc2 += w_parts[α] * (g2 - d2 * dv_dot_g) * inv_dist
        end
        dot_v[γ, 1] = acc1
        dot_v[γ, 2] = acc2
    end
    return nothing
end

# ##############################################################################
# Diagnostics
# ##############################################################################

"""
    compute_negative_part_l1(ws, field) -> neg

Diagnostic (1): negative-part L¹ norm of f_s, `∫ max(-f_s, 0) dv`, on the same
Gauss–Legendre grid used elsewhere. A direct probe of L²-projection Gibbs
oscillations — the empirical density is non-negative everywhere, so any negative
lobe in f_s is a projection artifact. Read-only.
"""
function compute_negative_part_l1(ws::Workspace, field::Forms.FormField)
    neg = 0.0
    for e in 1:ws.n_elements
        jac = element_measure(ws, e)
        fv, _ = evaluate(ws, field, e)
        for q in eachindex(ws.qrule_integrate.weights)
            f_val = fv[1][q]
            if f_val < 0.0
                neg += (-f_val) * ws.qrule_integrate.weights[q] * jac
            end
        end
    end
    return neg
end

"""
    compute_fs_minus_fp_l2(ws, field, v_parts, w_parts) -> ‖f_s − f_p‖₂

Diagnostic (2): L² projection error `‖f_s − f_p‖₂` with `f_p` the
element-constant histogram density `f_p(v) = (Σ_{α: v_α ∈ e(v)} w_α) / |e(v)|`.
The discrepancy between the smooth B-spline f_s and the piecewise-constant
histogram captures both the Gibbs oscillation amplitude and the cell-to-cell
mass-distribution mismatch that drives spurious gradients via `L = M⁻¹ r`.
Read-only; out-of-domain particles are skipped in the histogram.
"""
function compute_fs_minus_fp_l2(ws::Workspace, field::Forms.FormField,
                                 v_parts::AbstractMatrix, w_parts::AbstractVector)
    # Per-element particle mass:  m_e = Σ_{α ∈ e} w_α
    elem_mass = zeros(ws.n_elements)
    for α in axes(v_parts, 1)
        loc = locate_particle(ws, v_parts[α, 1], v_parts[α, 2])
        isnothing(loc) && continue
        elem_mass[loc.elem_id] += w_parts[α]
    end

    sumsq = 0.0
    for e in 1:ws.n_elements
        jac = element_measure(ws, e)            # |element|
        fp_e = elem_mass[e] / jac                # histogram density on element e
        fv, _ = evaluate(ws, field, e)
        for q in eachindex(ws.qrule_integrate.weights)
            f_val = fv[1][q]
            d = f_val - fp_e
            sumsq += d^2 * ws.qrule_integrate.weights[q] * jac
        end
    end
    return sqrt(sumsq)
end
