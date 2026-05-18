# v3_800 — 800-step σ relaxation probe. Identical to v3 mesh + criteria.
# Goal: measure anisotropy decay σ₁/σ₂ over longer horizon (t = 0.8).
# Stripe artifact ignored — σ from particle stats is mesh/N independent
# (verified v4/v5/v6 trajectories overlap on σ plot).
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 17); 5.0; 6.0],
    bp2 = [-6.0; LinRange(-2.5, 2.5, 26); 6.0],
    P_DEG=2, K_REG=1, N_QUAD=6,
    N_PARTICLES=40_000,
    σ1=4/3, σ2=0.5,
    DT=0.001, N_STEPS=800,
    use_anderson=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    abs_floor=1e-10,
    stag_window=30, stag_rel_tol=0.1,
    damp_decay_start=200, damp_decay_factor=0.5,
    suffix="bpmesh40k_v3_800",
    seed=42,
)
