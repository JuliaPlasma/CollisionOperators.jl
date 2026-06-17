# LB2D_v3_bp2eqbp1 — v3 baseline but with an isotropic mesh: bp2 == bp1.
# Probe whether the dense anisotropic bp2 (26-pt LinRange(-2.5,2.5)) was masking
# the tail negative-region artifact rather than fixing it. Same IC/solver as
# parameters_LB2D_v3.jl; only the v2 breakpoints change.
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 17); 5.0; 6.0],
    bp2 = [-6.0; -5.0; LinRange(-4.0, 4.0, 17); 5.0; 6.0],
    P_DEG=2, K_REG=1, N_QUAD=6,
    N_PARTICLES=40_000,
    σ1=4/3, σ2=0.5,
    nu=1.0,
    DT=0.001, N_STEPS=800,
    use_anderson=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    abs_floor=1e-10,
    stag_window=30, stag_rel_tol=0.1,
    damp_decay_start=200, damp_decay_factor=0.5,
    snap_every=50,
    suffix="LB2D_v3_bp2eqbp1",
    seed=42,
)
