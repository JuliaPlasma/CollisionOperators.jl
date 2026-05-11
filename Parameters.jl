# Parameters.jl — simulation configuration carried via a single immutable struct.
#
# Workflow:
#   1. Concrete preset files (parameters_default.jl, parameters_finemesh.jl, …)
#      construct a `SimParameters` value bound to the global symbol `PARAMS`.
#   2. The CLI entry point in main_Gonzalez.jl picks the preset file from
#      ARGS[1] (defaulting to "parameters_default.jl") and `include`s it.
#   3. Remaining ARGS of the form --key=val override individual fields via
#      `parse_overrides`, returning a new SimParameters with the changes.
#
# Every downstream module (MantisWrappers, functions, main loop) accepts an
# instance of `SimParameters` rather than reading globals; this is the only
# state shared across the call chain besides the preallocated `Workspace`.

Base.@kwdef struct SimParameters
    # Velocity domain
    V_MIN::Float64 = -6.0
    V_MAX::Float64 =  6.0
    I_MIN::Float64 = -4.0
    I_MAX::Float64 =  4.0

    # B-spline space
    P_DEG::Int = 2
    K_REG::Int = 1
    N_ELEM_1::Int = 10           # v₁ inner element count (uniform in [I_MIN,I_MAX])
    N_ELEM_2::Int = 25           # v₂ inner element count
    N_QUAD::Int   = 6            # 1D Gauss–Legendre points per element

    # Particle initial condition (anisotropic Gaussian, untruncated)
    N_PARTICLES::Int = 10_000
    σ1::Float64 = 4/3
    σ2::Float64 = 0.5

    # Time integration
    DT::Float64 = 0.001
    N_STEPS::Int = 400

    # Implicit-solver knobs
    use_anderson::Bool = true
    damping::Float64   = 0.7
    m_anderson::Int    = 8
    tol::Float64       = 1e-12
    max_iter::Int      = 2000

    # Run identifier (used in output file names)
    suffix::String = "anderson"

    # Random seed (so independent runs use the same particle IC)
    seed::Int = 42
end

# Parse a single CLI token of the form "--key=val" or "--key val" (the latter
# would require sequential scanning; we only support the equals form to keep
# the parser stateless and easy to read).
function _parse_override_token(tok::AbstractString)
    startswith(tok, "--") || return nothing
    eq = findfirst('=', tok)
    eq === nothing && error("CLI override $tok must use --key=value form")
    key = Symbol(tok[3:eq-1])
    val_str = tok[eq+1:end]
    return key, val_str
end

# Coerce a string into the field's declared type. Bool accepts true/false/1/0;
# strings pass through; numeric types use parse(T, s).
function _coerce(::Type{T}, s::AbstractString) where {T}
    if T === Bool
        s in ("true", "1")  && return true
        s in ("false", "0") && return false
        error("Cannot parse $s as Bool")
    elseif T <: AbstractString
        return String(s)
    else
        return parse(T, s)
    end
end

"""
    parse_overrides(p::SimParameters, args)

Apply every `--key=value` token in `args` on top of `p`, returning a new
`SimParameters`. Unknown keys raise. Empty `args` returns `p` unchanged.
"""
function parse_overrides(p::SimParameters, args)
    isempty(args) && return p
    fields = fieldnames(SimParameters)
    types  = (fieldtype(SimParameters, f) for f in fields)
    field_type = Dict(f => fieldtype(SimParameters, f) for f in fields)
    overrides = Dict{Symbol, Any}()
    for tok in args
        parsed = _parse_override_token(tok)
        parsed === nothing && continue
        key, val_str = parsed
        haskey(field_type, key) || error("Unknown parameter: $key")
        overrides[key] = _coerce(field_type[key], val_str)
    end
    return SimParameters(; (f => get(overrides, f, getfield(p, f)) for f in fields)...)
end

"""
    print_summary(p::SimParameters)

One-shot human-readable dump of the active configuration. Used at the top of
each run so the .log captures exactly what was run.
"""
function print_summary(p::SimParameters)
    println("==== SimParameters ====")
    println("V ∈ [$(p.V_MIN), $(p.V_MAX)],  I ∈ [$(p.I_MIN), $(p.I_MAX)]")
    println("P_DEG=$(p.P_DEG)  K_REG=$(p.K_REG)  N_ELEM=$(p.N_ELEM_1)×$(p.N_ELEM_2)" *
            "  N_QUAD=$(p.N_QUAD)")
    println("N_PARTICLES=$(p.N_PARTICLES)  σ=($(p.σ1), $(p.σ2))  seed=$(p.seed)")
    println("DT=$(p.DT)  N_STEPS=$(p.N_STEPS)")
    println("solver=$(p.use_anderson ? "Anderson(m=$(p.m_anderson))" : "Picard")" *
            "  damping=$(p.damping)  tol=$(p.tol)  max_iter=$(p.max_iter)")
    println("suffix=$(p.suffix)")
    println("=======================")
end
