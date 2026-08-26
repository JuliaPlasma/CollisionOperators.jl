# LB_sq_d04 — Lenard-Bernstein counterpart of parameters_sq_d04.jl. Identical
# mesh / IC / solver knobs as the Landau sq_d04 run (square mesh bp1=bp2, inner
# Δ=0.4, anisotropic IC σ1=4/3 σ2=0.5 → isotropic, Gonzalez + ½·log f²), so an
# LB run isolates whether behavior is operator-specific vs mesh/projection-
# driven. Only the collision model differs: collision_model=:lb selects the O(N)
# conservative LB drift v̇ = -ν(∇f/f + A + B v) instead of the O(N²) Landau sum.
# ν=1.0. GPU does not apply to LB (runs on CPU).
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 21); 5.0; 6.0],   # Δinner=0.4
    bp2 = [-6.0; -5.0; LinRange(-4.0, 4.0, 21); 5.0; 6.0],   # = bp1
    P_DEG=2, K_REG=1, N_QUAD=6,
    N_PARTICLES=40_000,
    σ1=4/3, σ2=0.5,
    DT=0.001, N_STEPS=2000,
    collision_model=:lb,
    nu=1.0,
    use_anderson=true,
    use_gonzalez=true,
    use_logsq=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    abs_floor=1e-10,
    stag_window=30, stag_rel_tol=0.1,
    damp_decay_start=200, damp_decay_factor=0.5,
    suffix="LB_sq_d04",
    seed=42,
)
