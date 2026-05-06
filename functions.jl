function l2_project!(f_coeffs, v_parts, w_parts)
    rhs = zeros(n_dofs)
    nloc = (P_DEG + 1)^2
    for α in axes(v_parts, 1)
        loc = locate_particle(v_parts[α, 1], v_parts[α, 2])
        isnothing(loc) && continue

        # φ_k(v_α) + global DOF indices into preallocated buffers (zero heap alloc)
        fast_eval_particle!(_lp_vals, _lp_gids, loc)

        # b_k += w_α · φ_k(v_α)
        @inbounds for j in 1:nloc
            rhs[_lp_gids[j]] += w_parts[α] * _lp_vals[j]
        end
    end
    # M f = b
    f_coeffs .= M_lu \ rhs
end
function compute_entropy(field::Forms.FormField)
    S = 0.0
    for e in 1:n_elements
        jac = element_measure(e)       # |J_e|
        fv, _ = evaluate(field, e)
        for q in eachindex(_qrule_integrate.weights)
            f_val = fv[1][q]           # f_s(v_q^e)
            if f_val > 1e-30
                # S -= f_s log(f_s) · w_q · |J_e|
                S -= f_val * log(f_val) * _qrule_integrate.weights[q] * jac
            end
        end
    end
    return S
end
function compute_r!(r, field::Forms.FormField)
    fill!(r, 0.0)
    for e in 1:n_elements
        jac = element_measure(e)
        fv, _ = evaluate(field, e)                         # f_s(v_q^e)
        evals, indices = evaluate(e)                  # φ_i(v_q^e)
        for q in eachindex(_qrule_integrate.weights)
            f_val = fv[1][q]
            integrand = f_val > 1e-30 ? (1 + log(f_val)) : 0.0
            for (j, gidx) in enumerate(indices[1])
                # r_i += φ_i(v_q) · (1 + log f_s(v_q)) · w_q · |J_e|
                r[gidx] += integrand * evals[1][q, j] * _qrule_integrate.weights[q] * jac
            end
        end
    end
end
function compute_G!(G, v_parts, L_vec)
    fill!(G, 0.0)
    nloc = (P_DEG + 1)^2
    for α in axes(v_parts, 1)
        loc = locate_particle(v_parts[α, 1], v_parts[α, 2])
        isnothing(loc) && continue

        # values + ∂/∂ξ_d gradients into preallocated buffers (zero heap alloc)
        fast_eval_particle_grad!(_G_vals, _G_dxi1, _G_dxi2, _G_gids, loc)

        inv_h1 = 1.0 / loc.h1
        inv_h2 = 1.0 / loc.h2
        acc1 = 0.0
        acc2 = 0.0
        @inbounds for j in 1:nloc
            L = L_vec[_G_gids[j]]
            # chain rule: ∂φ/∂v_d = (∂φ/∂ξ_d) / h_d
            acc1 += L * _G_dxi1[j] * inv_h1
            acc2 += L * _G_dxi2[j] * inv_h2
        end
        G[α, 1] = acc1
        G[α, 2] = acc2
    end
end
function compute_collision!(dot_v, v_parts, w_parts, G)
    fill!(dot_v, 0.0)
    N = size(v_parts, 1)
    Threads.@threads for γ in 1:N
        vγ1, vγ2 = v_parts[γ, 1], v_parts[γ, 2]
        (vγ1 <= V_MIN || vγ1 >= V_MAX || vγ2 <= V_MIN || vγ2 >= V_MAX) && continue
        Gγ1, Gγ2 = G[γ, 1], G[γ, 2]
        acc1, acc2 = 0.0, 0.0
        for α in 1:N
            γ == α && continue
            vα1, vα2 = v_parts[α, 1], v_parts[α, 2]
            (vα1 <= V_MIN || vα1 >= V_MAX || vα2 <= V_MIN || vα2 >= V_MAX) && continue

            # Δv = v_γ - v_α
            d1 = vγ1 - vα1
            d2 = vγ2 - vα2
            dist2 = d1^2 + d2^2
            dist2 < 1e-24 && continue
            dist = sqrt(dist2)

            # g = G_α - G_γ.  Landau metriplectic ODE:
            #   dv_γ/dt = ∑_α w_α A(v_γ−v_α) (G_α − G_γ),
            # equivalently −∑_α w_α A(v_γ−v_α) (G_γ − G_α). The minus sign is
            # required by the H-theorem dS/dt ≥ 0 since ∂S/∂v_γ = −w_γ G_γ.
            g1 = G[α, 1] - Gγ1
            g2 = G[α, 2] - Gγ2

            # [U·g]_i = (g_i - Δv̂_i (Δv̂·g)) / |Δv|
            dv_dot_g = (d1 * g1 + d2 * g2) / dist2    # (Δv·g) / |Δv|²
            inv_dist = 1.0 / dist                       # 1/|Δv|
            acc1 += w_parts[α] * (g1 - d1 * dv_dot_g) * inv_dist
            acc2 += w_parts[α] * (g2 - d2 * dv_dot_g) * inv_dist
        end
        dot_v[γ, 1] = acc1
        dot_v[γ, 2] = acc2
    end
end
