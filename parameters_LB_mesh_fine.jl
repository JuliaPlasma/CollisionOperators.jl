# Mesh sweep, bi-Max IC. 81 knots h=0.25, peaks ±2 on knot. 2× refinement.
# Honeycomb test: finer mesh should suppress mesh artifacts.
PARAMS = SimParameters(
    bp = collect(-10.0:0.25:10.0),  # 81 knots, h=0.25, ±2 ∈ knots
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
    suffix="LB_mesh_fine",
    seed=42,
)
