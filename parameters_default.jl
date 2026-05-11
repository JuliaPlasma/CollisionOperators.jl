# Default Anderson configuration for diagnostics runs.
# Anisotropic mesh: dense in v₂ to resolve narrow σ₂=0.5 distribution,
# coarser in v₁ where σ₁=4/3 spreads broadly.
PARAMS = SimParameters(
    V_MIN=-6.0, V_MAX=6.0,
    I_MIN=-4.0, I_MAX=4.0,
    P_DEG=2, K_REG=1,
    N_ELEM_1=10, N_ELEM_2=25,
    N_QUAD=6,
    N_PARTICLES=10_000,
    σ1=4/3, σ2=0.5,
    DT=0.001, N_STEPS=400,
    use_anderson=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    suffix="anderson",
    seed=42,
)
