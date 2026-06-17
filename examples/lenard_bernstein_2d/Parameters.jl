# Parameters.jl — 2D Lenard-Bernstein configuration.
#
# Workflow:
#   1. Preset files (parameters_LB2D_*.jl) build a `SimParameters` bound to
#      `PARAMS::SimParameters`. Each preset directly supplies the anisotropic
#      breakpoint vectors `bp1`, `bp2`.
#   2. main_LB.jl picks ARGS[1] as preset, --key=val tokens override scalars.
#   3. Vector fields (`bp1`, `bp2`) not CLI-overridable — edit the preset.
#
# Initial condition: anisotropic centered Gaussian  v ~ N(0, diag(σ1², σ2²)).
# Defaults match the 2D Landau "v3 baseline" mesh + IC that exhibited the
# honeycomb artifact, so an LB run on the identical mesh isolates whether the
# artifact is mesh/projection-driven (appears here too) or Landau-kernel-
# specific (absent here).

using Random

Base.@kwdef struct SimParameters
    # Velocity-space breakpoints (anisotropic, possibly non-uniform).
    bp1::Vector{Float64} = [-6.0; -5.0; LinRange(-4.0, 4.0, 17); 5.0; 6.0]
    bp2::Vector{Float64} = [-6.0; LinRange(-2.5, 2.5, 26); 6.0]

    # B-spline space
    P_DEG::Int = 2
    K_REG::Int = 1
    N_QUAD::Int = 6

    # Particles + anisotropic Gaussian IC (centered, untruncated)
    N_PARTICLES::Int = 40_000
    σ1::Float64 = 4 / 3
    σ2::Float64 = 0.5

    # Collision frequency
    nu::Float64 = 1.0

    # Time integration (implicit midpoint)
    DT::Float64 = 0.001
    N_STEPS::Int = 800

    # Implicit-solver knobs
    use_anderson::Bool = true
    damping::Float64 = 0.7
    m_anderson::Int = 8
    tol::Float64 = 1e-12
    max_iter::Int = 2000
    abs_floor::Float64 = 1e-10
    stag_window::Int = 30
    stag_rel_tol::Float64 = 0.1
    damp_decay_start::Int = 200
    damp_decay_factor::Float64 = 0.5

    # Snapshot cadence (every snap_every steps + final)
    snap_every::Int = 50

    # Run identifier
    suffix::String = "LB2D_v3"

    # RNG seed
    seed::Int = 42
end

function _parse_override_token(tok::AbstractString)
    startswith(tok, "--") || return nothing
    eq = findfirst('=', tok)
    eq === nothing && error("CLI override $tok must use --key=value form")
    key = Symbol(tok[3:(eq - 1)])
    val_str = tok[(eq + 1):end]
    return key, val_str
end

function _coerce(::Type{T}, s::AbstractString) where {T}
    if T === Bool
        s in ("true", "1") && return true
        s in ("false", "0") && return false
        error("Cannot parse $s as Bool")
    elseif T <: AbstractString
        return String(s)
    elseif T <: AbstractVector
        error("Vector parameters (`bp1`, `bp2`) are not CLI-overridable; edit the preset file")
    else
        return parse(T, s)
    end
end

function parse_overrides(p::SimParameters, args)
    isempty(args) && return p
    fields = fieldnames(SimParameters)
    field_type = Dict(f => fieldtype(SimParameters, f) for f in fields)
    overrides = Dict{Symbol,Any}()
    for tok in args
        parsed = _parse_override_token(tok)
        parsed === nothing && continue
        key, val_str = parsed
        haskey(field_type, key) || error("Unknown parameter: $key")
        overrides[key] = _coerce(field_type[key], val_str)
    end
    return SimParameters(; (f => get(overrides, f, getfield(p, f)) for f in fields)...)
end

function print_summary(p::SimParameters)
    n1, n2 = length(p.bp1) - 1, length(p.bp2) - 1
    dv1_min, dv1_max = extrema(diff(p.bp1))
    dv2_min, dv2_max = extrema(diff(p.bp2))
    println("==== SimParameters (LB 2D) ====")
    println("v₁ ∈ [$(p.bp1[1]), $(p.bp1[end])]  cells=$n1  Δv₁ ∈ [$dv1_min, $dv1_max]")
    println("v₂ ∈ [$(p.bp2[1]), $(p.bp2[end])]  cells=$n2  Δv₂ ∈ [$dv2_min, $dv2_max]")
    println("P_DEG=$(p.P_DEG)  K_REG=$(p.K_REG)  N_QUAD=$(p.N_QUAD)")
    println("N_PARTICLES=$(p.N_PARTICLES)  σ=($(p.σ1), $(p.σ2))  seed=$(p.seed)")
    println("ν=$(p.nu)  DT=$(p.DT)  N_STEPS=$(p.N_STEPS)  T_final=$(p.DT * p.N_STEPS)")
    println(
        "solver=$(p.use_anderson ? "Anderson(m=$(p.m_anderson))" : "Picard")" *
        "  damping=$(p.damping)  tol=$(p.tol)  max_iter=$(p.max_iter)",
    )
    println("suffix=$(p.suffix)")
    return println("===============================")
end

# Sample initial particle velocities: anisotropic centered Gaussian. Returns
# an N×2 matrix (rows = particles, cols = (v₁, v₂)).
function sample_initial_velocities(p::SimParameters, rng=Random.default_rng())
    σ1 = p.σ1
    σ2 = p.σ2
    N = p.N_PARTICLES

    v1 = σ1 * randn(rng, N)
    v2 = σ2 * randn(rng, N)
    return [v1 v2]
end
