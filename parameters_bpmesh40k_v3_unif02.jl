# v3_unif02 — full-domain uniform mesh with v3 inner spacing.
# Δv₁=0.5, Δv₂=0.2 everywhere (no outer wide slab).
# Stripe pitch reference Δv₂ unchanged vs v3 → isolates outer-mesh effect.
PARAMS = SimParameters(
    bp1 = collect(LinRange(-6.0, 6.0, 25)),   # Δv₁=0.5, 24 cells
    bp2 = collect(LinRange(-6.0, 6.0, 61)),   # Δv₂=0.2, 60 cells
    P_DEG=2, K_REG=1, N_QUAD=6,
    N_PARTICLES=40_000,
    σ1=4/3, σ2=0.5,
    DT=0.001, N_STEPS=100,
    use_anderson=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=100,
    abs_floor=1e-10,
    stag_window=30, stag_rel_tol=0.1,
    damp_decay_start=200, damp_decay_factor=0.5,
    suffix="bpmesh40k_v3_unif02",
    seed=42,
)
