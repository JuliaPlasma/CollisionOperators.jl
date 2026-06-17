# bi-Max align mesh (h=0.5, peaks ±2 on knot), T=100 (125k steps).
# Worst-case clustering test: particles sit where f_s spline has knot wiggles.
PARAMS = SimParameters(
    bp = collect(-10.0:0.5:10.0),
    P_DEG=3, K_REG=2, N_QUAD=6,
    N_PARTICLES=1000,
    ic_type="bimax",
    ic_sigma=1.0, ic_sep=2.0,
    nu=1.0,
    DT=8e-4, N_STEPS=125_000,
    snap_every=2500,
    use_anderson=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    suffix="LB_align_T100",
    seed=42,
)
