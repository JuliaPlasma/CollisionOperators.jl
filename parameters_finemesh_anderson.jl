# Fine v₂ mesh: 25 inner v₂ cells over [-2.5, 2.5] (Δv₂_inner = 0.2 ≈ σ₂/2.5)
# in the narrow direction. Anderson config (`use_anderson=true`).
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 17); 5.0; 6.0],
    bp2 = [-6.0; LinRange(-2.5, 2.5, 26); 6.0],
    P_DEG=2, K_REG=1, N_QUAD=6,
    N_PARTICLES=10_000,
    σ1=4/3, σ2=0.5,
    DT=0.001, N_STEPS=400,
    use_anderson=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    suffix="finemesh_anderson",
    seed=42,
)
