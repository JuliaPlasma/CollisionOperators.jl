# Parameters.jl — simulation configuration carried via a single immutable struct.
#
# Workflow:
#   1. Concrete preset files (parameters_default.jl, parameters_finemesh.jl, …)
#      construct a `SimParameters` value bound to the global symbol `PARAMS`.
#      Each preset directly provides the breakpoint vectors `bp1`, `bp2`
#      (anisotropic, possibly non-uniform).
#   2. The CLI entry point in main_Gonzalez.jl picks the preset file from
#      ARGS[1] (defaulting to "parameters_default.jl") and `include`s it.
#   3. Remaining ARGS of the form --key=val override individual *scalar* fields
#      via `parse_overrides` (vector fields like `bp1`/`bp2` are not CLI-
#      overridable — re-edit the preset file instead).
#
# Every downstream module (MantisWrappers, functions, main loop) accepts an
# instance of `SimParameters` rather than reading globals; this is the only
# state shared across the call chain besides the preallocated `Workspace`.

Base.@kwdef struct SimParameters
    # Velocity-space breakpoints (anisotropic, possibly non-uniform).
    # `bp1[1]`/`bp1[end]` define the v₁ domain; cell count = `length(bp1) - 1`.
    bp1::Vector{Float64} = [-6.0; -5.0; LinRange(-4.0, 4.0, 17); 5.0; 6.0]
    bp2::Vector{Float64} = [-6.0; LinRange(-2.5, 2.5, 13); 6.0]

    # B-spline space
    P_DEG::Int = 2
    K_REG::Int = 1
    N_QUAD::Int = 6              # 1D Gauss–Legendre points per element

    # Particle initial condition (anisotropic Gaussian, untruncated)
    N_PARTICLES::Int = 10_000
    σ1::Float64 = 4/3
    σ2::Float64 = 0.5

    # Time integration
    DT::Float64 = 0.001
    N_STEPS::Int = 400

    # Discrete-gradient variant: true = Gonzalez (entropy-exact, has |Δv|²
    # denominator that blows up near equilibrium); false = plain implicit
    # midpoint (drop the Gonzalez correction term).
    use_gonzalez::Bool = true

    # Entropy/seed integrand variant: false = positivity-clamped log f (drop
    # f_s<0 quadrature points); true = log f = ½ log f² identity, so Gibbs
    # undershoot points contribute via |f| guard. Probe for spike sensitivity.
    use_logsq::Bool = false

    # Implicit-solver knobs
    use_anderson::Bool = true
    damping::Float64   = 0.7
    m_anderson::Int    = 8
    tol::Float64       = 1e-12   # relative tol on ‖r‖ / ‖v‖
    max_iter::Int      = 2000
    # Anderson convergence safety net (see `step_anderson!` doc-comment):
    abs_floor::Float64       = 1e-7   # cap on effective tol — past this, asking
                                       # for less is pointless (Picard noise floor)
    stag_window::Int         = 50     # iters between stagnation checks
    stag_rel_tol::Float64    = 0.01   # < 1% drop in `nrm_best` over window ⇒ exit
    damp_decay_start::Int    = 200    # iter index after which damping is decayed
    damp_decay_factor::Float64 = 0.5  # damping multiplier once decay starts

    # Run identifier (used in output file names)
    suffix::String = "anderson"

    # Random seed (so independent runs use the same particle IC)
    seed::Int = 42
end

# Parse a single CLI token of the form "--key=val". The space-separated form
# would require sequential scanning; we only support the equals form to keep
# the parser stateless and easy to read.
function _parse_override_token(tok::AbstractString)
    startswith(tok, "--") || return nothing
    eq = findfirst('=', tok)
    eq === nothing && error("CLI override $tok must use --key=value form")
    key = Symbol(tok[3:eq-1])
    val_str = tok[eq+1:end]
    return key, val_str
end

# Coerce a string into the field's declared type. Bool accepts true/false/1/0;
# strings pass through; numeric types use parse(T, s). Vector fields are
# rejected at the call site — too clumsy to express on the CLI.
function _coerce(::Type{T}, s::AbstractString) where {T}
    if T === Bool
        s in ("true", "1")  && return true
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

"""
    parse_overrides(p::SimParameters, args)

Apply every `--key=value` token in `args` on top of `p`, returning a new
`SimParameters`. Unknown keys raise. Empty `args` returns `p` unchanged.
"""
function parse_overrides(p::SimParameters, args)
    isempty(args) && return p
    fields = fieldnames(SimParameters)
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
    n1, n2 = length(p.bp1) - 1, length(p.bp2) - 1
    dv1_min, dv1_max = extrema(diff(p.bp1))
    dv2_min, dv2_max = extrema(diff(p.bp2))
    println("==== SimParameters ====")
    println("v₁ ∈ [$(p.bp1[1]), $(p.bp1[end])]  cells=$n1  Δv₁ ∈ [$dv1_min, $dv1_max]")
    println("v₂ ∈ [$(p.bp2[1]), $(p.bp2[end])]  cells=$n2  Δv₂ ∈ [$dv2_min, $dv2_max]")
    println("P_DEG=$(p.P_DEG)  K_REG=$(p.K_REG)  N_QUAD=$(p.N_QUAD)")
    println("N_PARTICLES=$(p.N_PARTICLES)  σ=($(p.σ1), $(p.σ2))  seed=$(p.seed)")
    println("DT=$(p.DT)  N_STEPS=$(p.N_STEPS)")
    println("solver=$(p.use_anderson ? "Anderson(m=$(p.m_anderson))" : "Picard")" *
            "  damping=$(p.damping)  tol=$(p.tol)  max_iter=$(p.max_iter)")
    println("disc_grad=$(p.use_gonzalez ? "Gonzalez" : "plain-midpoint")" *
            "  entropy_integrand=$(p.use_logsq ? "½log f²" : "clamped log f")")
    println("suffix=$(p.suffix)")
    println("=======================")
end
