# # MantisWrappers
#
# This module wraps the Mantis FEM library to provide a convenient interface for
# particle-to-B-spline projection, entropy computation, and Landau collision operators
# on a 2D Cartesian velocity domain.

module MantisWrappers

using Mantis
using LinearAlgebra
using LinearAlgebra: mul!

include("parameters.jl")

# ## Geometry and function spaces
#
# We build a 2D tensor-product B-spline space on the velocity domain [V_MIN, V_MAX]².
# The 0-form space X⁰ represents scalar fields f(v₁, v₂) = Σᵢ fᵢ φᵢ(v).

begin # Geometry and function spaces
    # Non-uniform mesh: N_ELEM uniform inner elements covering [I_MIN, I_MAX]
    # plus one outer (coarse) element on each side reaching out to V_MIN / V_MAX.
    # Total elements per dim = N_ELEM + 2.
    const bp = collect(Float64, vcat(V_MIN, range(I_MIN, I_MAX; length=N_ELEM + 1), V_MAX))

    const geo_1d = Geometry.CartesianGeometry((bp,))
    const B_1d = FunctionSpaces.BSplineSpace(geo_1d, P_DEG, K_REG)
    const TP = FunctionSpaces.TensorProductSpace((B_1d, B_1d), Geometry.CartesianGeometry)
    const X⁰ = Forms.FormSpace(0, TP, "f")

    const n_dofs = Forms.get_num_basis(X⁰) # number of basis functions (degrees of freedom)
    const geo_2d = Forms.get_geometry(X⁰)
    const n_elements = Geometry.get_num_elements(geo_2d)
    const lin_indices = LinearIndices((length(bp)-1, length(bp)-1))
end

# ## Mass matrix assembly
#
# The mass matrix M_{ij} = ∫ φᵢ φⱼ dv is assembled using the Mantis weak-form pipeline.
# We use a zero forcing term (AnalyticalFormField) as a dummy — only the bilinear form
# ∫ v⁰ ∧ ★(u⁰) is needed. The LU factorization is stored for repeated solves.

const M_lu = let
    qrule = Quadrature.tensor_product_rule((P_DEG + 1, P_DEG + 1), Quadrature.gauss_legendre)
    dΩ = Quadrature.StandardQuadrature(qrule, n_elements)
    f_zero = Forms.AnalyticalFormField(0, x -> [zeros(size(x, 1))], geo_2d, "0")
    wfi = Assemblers.WeakFormInputs(X⁰, f_zero)
    v⁰ = Assemblers.get_test_form(wfi)
    u⁰ = Assemblers.get_trial_form(wfi)
    M_expr = ∫(v⁰ ∧ ★(u⁰), dΩ)
    M_wf = Assemblers.WeakForm(((M_expr,),), ((0,),), wfi)
    M, _ = Assemblers.assemble(M_wf)
    lu(M)
end

# ## Quadrature rule for integration
#
# A separate quadrature rule used for numerical integration of entropy and the r-vector.

const _qrule_integrate = Quadrature.tensor_product_rule((N_QUAD, N_QUAD), Quadrature.gauss_legendre)

# ## Particle location
#
# Given a particle at (v₁, v₂), find which element it belongs to and compute the
# reference (canonical) coordinates ξ ∈ [0,1]² within that element.

struct ParticleLocation
    elem_id::Int
    xi::Points.CartesianPoints
    h1::Float64   # element width in v₁ direction
    h2::Float64   # element width in v₂ direction
end

function locate_particle(v1, v2)
    (v1 <= V_MIN || v1 >= V_MAX || v2 <= V_MIN || v2 >= V_MAX) && return nothing # TODO: type instable
    i = searchsortedlast(bp, v1)
    j = searchsortedlast(bp, v2)
    h1 = bp[i+1] - bp[i]
    h2 = bp[j+1] - bp[j]
    ξ1 = (v1 - bp[i]) / h1
    ξ2 = (v2 - bp[j]) / h2
    return ParticleLocation(lin_indices[i, j], Points.CartesianPoints(([ξ1], [ξ2])), h1, h2)
end

"""
    evaluate(field_or_space, location)

Thin wrappers around Mantis evaluation routines.

# Methods

- `evaluate(ff::AbstractFormField, elem_id::Int)` — field values at quadrature nodes of element `elem_id`.
- `evaluate(fs::AbstractFormSpace, elem_id::Int)` — basis values at quadrature nodes of element `elem_id`.
- `evaluate(ff::AbstractFormField, loc::ParticleLocation)` — field values at particle position `loc`.
- `evaluate(fs::AbstractFormSpace, loc::ParticleLocation)` — basis values at particle position `loc`.

# Convenience (default to global coordinate space `X⁰`)

- `evaluate(elem_id::Int)` — equivalent to `evaluate(X⁰, elem_id)`.
- `evaluate(loc::ParticleLocation)` — equivalent to `evaluate(X⁰, loc)`.
"""
function evaluate end

evaluate(ff::Forms.AbstractFormField, e::Int) = Forms.evaluate(ff, e, _qrule_integrate.nodes)

evaluate(fs::Forms.AbstractFormSpace, elem_id::Int) = Forms.evaluate(fs, elem_id, _qrule_integrate.nodes)
evaluate(e::Int) = evaluate(X⁰, e)

evaluate(fs::Forms.AbstractFormSpace, loc::ParticleLocation) = Forms.evaluate(fs, loc.elem_id, loc.xi)
evaluate(ff::Forms.AbstractFormField, loc::ParticleLocation) = Forms.evaluate(ff, loc.elem_id, loc.xi)
evaluate(loc::ParticleLocation) = evaluate(X⁰, loc)

# Evaluate basis function derivatives in canonical coordinates.
# Returns local_basis[deriv_order+1][deriv_index][component][point, basis].
# For a 2D 0-form: local_basis[2][1][1] = ∂φ/∂ξ₁, local_basis[2][2][1] = ∂φ/∂ξ₂.

evaluate_basis_derivatives(elem_id::Int, xi, nderivatives) =
    Forms._evaluate_form_in_canonical_coordinates(X⁰, elem_id, xi, nderivatives)
evaluate_basis_derivatives(loc::ParticleLocation, nderivatives) =
    Forms._evaluate_form_in_canonical_coordinates(X⁰, loc.elem_id, loc.xi, nderivatives)

# Build a FormField from coefficient vector: f_s(v) = Σᵢ fᵢ φᵢ(v).

build_field(coeffs) = Forms.build_form_field(X⁰, coeffs)

# Element Jacobian determinant (uniform Cartesian grid → same for all elements).

element_measure(e) = Geometry.get_element_measure(geo_2d, e)
const jac = element_measure(1)

# ## Bézier extraction cache
#
# 1D Bézier extraction matrices (one per 1D element) are precomputed once.
# On each particle evaluation: φ¹(ξ) = B_bernstein(ξ) · Cᵉ (row-vector form),
# so mul!(phi, transpose(C), B) gives the B-spline values with zero allocation.
#
# 2D basis = tensor product of two 1D evaluations, assembled by a manual double
# loop (equivalent to kron) directly into the caller's preallocated buffer.

const n_elem_1d = length(bp) - 1
const _ext1d = [FunctionSpaces.get_extraction(B_1d, e, 1)[1] for e in 1:n_elem_1d]
const _basis_start_1d = [first(FunctionSpaces.get_basis_indices(B_1d, e)) for e in 1:n_elem_1d]
const _n_dofs_1d = FunctionSpaces.get_num_basis(B_1d)
const _lin_dofs_2d = LinearIndices((_n_dofs_1d, _n_dofs_1d))

# Evaluate Bernstein basis and first derivative into preallocated vectors.
#   B[i+1]  = B_{i,p}(ξ)                 for i = 0..p
#   dB[i+1] = d/dξ B_{i,p}(ξ) = p · (B_{i-1,p-1}(ξ) − B_{i,p-1}(ξ))
function _bernstein_eval!(B::AbstractVector, dB::AbstractVector, p::Int, ξ::Float64)
    one_minus = 1.0 - ξ
    @inbounds for i in 0:p
        B[i+1] = binomial(p, i) * ξ^i * one_minus^(p-i)
    end
    if p == 0
        fill!(dB, 0.0)
    else
        @inbounds for i in 0:p
            bm1_left  = i > 0 ? binomial(p-1, i-1) * ξ^(i-1) * one_minus^(p-i)   : 0.0
            bm1_right = i < p ? binomial(p-1, i)   * ξ^i     * one_minus^(p-1-i) : 0.0
            dB[i+1] = p * (bm1_left - bm1_right)
        end
    end
    return nothing
end

# Per-call scratch for the 1D Bernstein / B-spline values & derivatives.
# Single-threaded; switch to per-thread buffers if particle loops go @threads.
struct _ParticleBuf
    B1::Vector{Float64}
    dB1::Vector{Float64}
    phi1::Vector{Float64}
    dphi1::Vector{Float64}
    B2::Vector{Float64}
    dB2::Vector{Float64}
    phi2::Vector{Float64}
    dphi2::Vector{Float64}
end
_ParticleBuf() = _ParticleBuf(ntuple(_ -> zeros(P_DEG + 1), 8)...)

const _pbuf = _ParticleBuf()

# Fill `vals` with φ_{j1,j2}(ξ₁,ξ₂) and `gids` with their global DOF indices for
# the (p+1)² basis functions supported on `loc`. Zero heap allocation.
function fast_eval_particle!(vals::AbstractVector, gids::AbstractVector,
                              loc::ParticleLocation, buf::_ParticleBuf = _pbuf)
    # CartesianPoints getindex(1) → tuple (ξ₁, ξ₂)
    ξ = loc.xi[1]
    ξ1 = ξ[1]
    ξ2 = ξ[2]
    # Inverse map elem_id → (i, j) 1D element indices
    ci = CartesianIndices(lin_indices)[loc.elem_id]
    i = ci[1]
    j = ci[2]

    C1 = _ext1d[i];  C2 = _ext1d[j]
    s1 = _basis_start_1d[i];  s2 = _basis_start_1d[j]

    _bernstein_eval!(buf.B1, buf.dB1, P_DEG, ξ1)
    _bernstein_eval!(buf.B2, buf.dB2, P_DEG, ξ2)
    mul!(buf.phi1, transpose(C1), buf.B1)
    mul!(buf.phi2, transpose(C2), buf.B2)

    p1 = P_DEG + 1
    k = 0
    @inbounds for j2 in 1:p1, j1 in 1:p1
        k += 1
        vals[k] = buf.phi1[j1] * buf.phi2[j2]
        gids[k] = _lin_dofs_2d[s1 + j1 - 1, s2 + j2 - 1]
    end
    return k
end

# Values + canonical-coord gradients. Physical gradient: ∂/∂v_d = (∂/∂ξ_d) / h_d.
function fast_eval_particle_grad!(vals::AbstractVector,
                                   dvals_dxi1::AbstractVector,
                                   dvals_dxi2::AbstractVector,
                                   gids::AbstractVector,
                                   loc::ParticleLocation, buf::_ParticleBuf = _pbuf)
    ξ = loc.xi[1]
    ξ1 = ξ[1]
    ξ2 = ξ[2]
    ci = CartesianIndices(lin_indices)[loc.elem_id]
    i = ci[1]
    j = ci[2]

    C1 = _ext1d[i];  C2 = _ext1d[j]
    s1 = _basis_start_1d[i];  s2 = _basis_start_1d[j]

    _bernstein_eval!(buf.B1, buf.dB1, P_DEG, ξ1)
    _bernstein_eval!(buf.B2, buf.dB2, P_DEG, ξ2)
    mul!(buf.phi1,  transpose(C1), buf.B1)
    mul!(buf.dphi1, transpose(C1), buf.dB1)
    mul!(buf.phi2,  transpose(C2), buf.B2)
    mul!(buf.dphi2, transpose(C2), buf.dB2)

    p1 = P_DEG + 1
    k = 0
    @inbounds for j2 in 1:p1, j1 in 1:p1
        k += 1
        vals[k]       = buf.phi1[j1]  * buf.phi2[j2]
        dvals_dxi1[k] = buf.dphi1[j1] * buf.phi2[j2]
        dvals_dxi2[k] = buf.phi1[j1]  * buf.dphi2[j2]
        gids[k]       = _lin_dofs_2d[s1 + j1 - 1, s2 + j2 - 1]
    end
    return k
end

# Preallocated per-particle scratch for l2_project! and compute_G!.
const _NLOC = (P_DEG + 1)^2
const _lp_vals = zeros(_NLOC)
const _lp_gids = zeros(Int, _NLOC)
const _G_vals  = zeros(_NLOC)
const _G_dxi1  = zeros(_NLOC)
const _G_dxi2  = zeros(_NLOC)
const _G_gids  = zeros(Int, _NLOC)

export n_dofs, n_elements, M_lu
export locate_particle, evaluate, build_field
export fast_eval_particle!, fast_eval_particle_grad!

# ## Physics routines
#
# The actual computation functions (L² projection, entropy, entropy gradient, collision
# operator) are defined in functions.jl.

include("functions.jl")

function evaluate_on_grid(field::Forms.FormField, v_grid)
    nv = length(v_grid)
    F = zeros(nv, nv)
    for (j, v2) in enumerate(v_grid)
        for (i, v1) in enumerate(v_grid)
            loc = locate_particle(v1, v2)
            isnothing(loc) && continue
            fv, _ = evaluate(field, loc::ParticleLocation)
            F[i, j] = fv[1][1]
        end
    end
    return F
end

export compute_entropy, compute_r!, compute_G!, compute_collision!, l2_project!, evaluate_on_grid

end # module
