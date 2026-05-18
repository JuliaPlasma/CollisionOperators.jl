# v6_800 — 800-step σ relaxation probe on dense mesh.
# Mesh identical to v6 (bp2 inner Δv₂=0.125, 60k particles).
# v3_800 with sparse mesh blew up: Anderson iter 15→154 by step 325 due to
# stagnation in projection-noise plateau on coarse bp2. Dense mesh should
# stay in fast-convergence regime longer.
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 25); 5.0; 6.0],
    bp2 = [-6.0; LinRange(-2.5, 2.5, 41); 6.0],
    P_DEG=2, K_REG=1, N_QUAD=6,
    N_PARTICLES=60_000,
    σ1=4/3, σ2=0.5,
    DT=0.001, N_STEPS=800,
    use_anderson=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    abs_floor=1e-10,
    stag_window=30, stag_rel_tol=0.1,
    damp_decay_start=200, damp_decay_factor=0.5,
    suffix="bpmesh60k_v6_800",
    seed=42,
)
