# 1D LB default: bi-Maxwellian initial condition (paper §5.3, eq 46).
#   f(v,0) = (1/√(2π)) * (exp(-(v-2)²/2) + exp(-(v+2)²/2))
# Final time t = 5.0  (6250 × 8e-4).
# Cubic spline, 41 cells uniform on [-10, 10] (paper §5.2 setup).
PARAMS = SimParameters(
    bp = collect(LinRange(-10.0, 10.0, 42)),
    P_DEG=3, K_REG=2, N_QUAD=6,
    N_PARTICLES=1000,
    ic_type="bimax",
    ic_sigma=1.0, ic_sep=2.0,
    nu=1.0,
    DT=8e-4, N_STEPS=6250,
    snap_every=250,
    use_anderson=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    suffix="LB_bimax",
    seed=42,
)
