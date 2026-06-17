# 1D LB uniform IC on [-2, 2] (paper §5.4, eq 47).
#   f(v,0) = 1/4 on |v| ≤ 2 else 0
# Final time t = 1.0  (10000 × 1e-4).  paper uses h = 1e-4 here, N=200.
PARAMS = SimParameters(
    bp = collect(LinRange(-10.0, 10.0, 42)),
    P_DEG=3, K_REG=2, N_QUAD=6,
    N_PARTICLES=200,
    ic_type="uniform",
    ic_L=2.0,
    nu=1.0,
    DT=1e-4, N_STEPS=10_000,
    snap_every=500,
    use_anderson=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    suffix="LB_uniform",
    seed=42,
)
