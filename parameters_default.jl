# Default Anderson configuration. Inner-refined anisotropic mesh:
#   v₁: 16 inner cells over [-4, 4]   (Δv₁ = 0.5  ≈ σ₁/2.67)
#   v₂: 12 inner cells over [-2.5, 2.5]  (Δv₂ ≈ 0.42 ≈ σ₂/1.2)
# plus 1–2 outer padding cells per side reaching support boundary [-6, 6].
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 17); 5.0; 6.0],
    bp2 = [-6.0; LinRange(-2.5, 2.5, 13); 6.0],
    P_DEG=2, K_REG=1, N_QUAD=6,
    N_PARTICLES=10_000,
    σ1=4/3, σ2=0.5,
    DT=0.001, N_STEPS=400,
    use_anderson=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    suffix="default",
    seed=42,
)
