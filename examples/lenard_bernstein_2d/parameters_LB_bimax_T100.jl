# bi-Max baseline mesh, T=100 (125k steps, 20× past equilibrium).
# Goal: detect secular particle clustering onto grid knots (1D honeycomb-analog).
PARAMS = SimParameters(
    bp = collect(LinRange(-10.0, 10.0, 42)),
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
    suffix="LB_bimax_T100",
    seed=42,
)
