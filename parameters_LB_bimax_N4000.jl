# bi-Max T=20, N=4000. Noise-floor scaling point. Expect floor+trough ~1/2 of N=1000.
PARAMS = SimParameters(
    bp = collect(LinRange(-10.0, 10.0, 42)),
    P_DEG=3, K_REG=2, N_QUAD=6,
    N_PARTICLES=4000,
    ic_type="bimax",
    ic_sigma=1.0, ic_sep=2.0,
    nu=1.0,
    DT=8e-4, N_STEPS=25_000,
    snap_every=1000,
    use_anderson=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    suffix="LB_bimax_N4000",
    seed=42,
)
