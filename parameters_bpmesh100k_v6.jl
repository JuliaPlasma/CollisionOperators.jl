# v6 — 100-step probe: test 1/√N scaling of projection-noise amplitude.
# Identical to v5 except N_PARTICLES = 40k → 100k.
# Prediction: ACF peak at lag=Δv₂=0.125 should drop from v5's ~0.5 by factor
# √(40/100) ≈ 0.63 → expected ~0.31 if mechanism is projection variance.
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 25); 5.0; 6.0],
    bp2 = [-6.0; LinRange(-2.5, 2.5, 41); 6.0],
    P_DEG=2, K_REG=1, N_QUAD=6,
    N_PARTICLES=100_000,
    σ1=4/3, σ2=0.5,
    DT=0.001, N_STEPS=100,
    use_anderson=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    abs_floor=1e-10,
    stag_window=30, stag_rel_tol=0.1,
    damp_decay_start=200, damp_decay_factor=0.5,
    suffix="bpmesh100k_v6",
    seed=42,
)
