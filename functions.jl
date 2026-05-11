# Physics + diagnostics routines. All take `ws::Workspace` as first argument.

# ## L² projection of weighted Dirac measure onto X⁰
#
# Solves M c = b where b_k = Σ_α w_α φ_k(v_α). The result f_s(v) = Σ c_k φ_k(v).
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

# ## Entropy S_h = -∫ f_s log f_s dv (so dS/dt ≥ 0 with our sign convention)
function compute_entropy(ws::Workspace, field::Forms.FormField)
    S = 0.0
    for e in 1:ws.n_elements
        jac = element_measure(ws, e)
        fv, _ = evaluate(ws, field, e)
        for q in eachindex(ws.qrule_integrate.weights)
            f_val = fv[1][q]
            if f_val > 1e-30
                S -= f_val * log(f_val) * ws.qrule_integrate.weights[q] * jac
            end
        end
    end
    return S
end

# ## r_i = ∫ φ_i (1 + log f_s) dv  → entropy gradient seed
function compute_r!(ws::Workspace, r, field::Forms.FormField)
    fill!(r, 0.0)
    for e in 1:ws.n_elements
        jac = element_measure(ws, e)
        fv, _ = evaluate(ws, field, e)
        evals, indices = evaluate(ws, e)
        for q in eachindex(ws.qrule_integrate.weights)
            f_val = fv[1][q]
            integrand = f_val > 1e-30 ? (1 + log(f_val)) : 0.0
            for (j, gidx) in enumerate(indices[1])
                r[gidx] += integrand * evals[1][q, j] *
                          ws.qrule_integrate.weights[q] * jac
            end
        end
    end
    return nothing
end

# ## G_α = ∇L(v_α) where L = M⁻¹ r — particle-side entropy gradient
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

# ## Landau collision-operator velocity update
function compute_collision!(ws::Workspace, dot_v, v_parts, w_parts, G)
    p = ws.p
    fill!(dot_v, 0.0)
    N = size(v_parts, 1)
    Threads.@threads for γ in 1:N
        vγ1, vγ2 = v_parts[γ, 1], v_parts[γ, 2]
        (vγ1 <= p.V_MIN || vγ1 >= p.V_MAX ||
         vγ2 <= p.V_MIN || vγ2 >= p.V_MAX) && continue
        Gγ1, Gγ2 = G[γ, 1], G[γ, 2]
        acc1, acc2 = 0.0, 0.0
        for α in 1:N
            γ == α && continue
            vα1, vα2 = v_parts[α, 1], v_parts[α, 2]
            (vα1 <= p.V_MIN || vα1 >= p.V_MAX ||
             vα2 <= p.V_MIN || vα2 >= p.V_MAX) && continue
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
#
# (1) Negative-part L¹ norm of f_s:    ∫ max(-f_s, 0) dv
#     A direct probe of L²-projection Gibbs oscillations: the empirical density
#     is non-negative everywhere, so any negative lobe in f_s is a projection
#     artifact. Computed on the same Gauss–Legendre grid used elsewhere.
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

# (2) ‖f_s − f_p‖₂  with f_p the *element-constant histogram density*:
#       f_p(v) = (Σ_{α: v_α ∈ e(v)} w_α) / |e(v)|
#     where e(v) is the element containing v. Discrepancy between the smooth
#     B-spline f_s and the piecewise-constant histogram f_p captures both the
#     Gibbs oscillation amplitude *and* the cell-to-cell mass-distribution
#     mismatch that drives spurious gradients via L = M⁻¹ r.
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
