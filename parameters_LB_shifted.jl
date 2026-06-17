# 1D LB shifted-normal IC (paper §5.2, eq 45).
#   f(v,0) = N(μ=2, σ=1)
# Final time t = 1.0  (1250 × 8e-4).
PARAMS = SimParameters(
    bp = collect(LinRange(-10.0, 10.0, 42)),
    P_DEG=3, K_REG=2, N_QUAD=6,
    N_PARTICLES=1000,
    ic_type="shifted",
    ic_mu=2.0, ic_sigma=1.0,
    nu=1.0,
    DT=8e-4, N_STEPS=1250,
    snap_every=125,
    use_anderson=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    suffix="LB_shifted",
    seed=42,
)
