# LB2D_v3_shift — LB mesh-shift probe. Identical to LB2D_v3 baseline except the
# bp2 inner grid shifted +0.05 (matches Landau parameters_bpmesh40k_v3_shift.jl).
# Goal: test whether the lateral (v₁) banding in the LB scatter tracks the mesh
# phase the way the Landau stripe artifact does.
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 17); 5.0; 6.0],
    bp2 = [-6.0; LinRange(-2.45, 2.55, 26); 6.0],   # +0.05 shift vs v3
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
    suffix="LB2D_v3_shift",
    seed=42,
)
