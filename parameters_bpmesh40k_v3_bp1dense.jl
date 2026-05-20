# v3_bp1dense — test bp1 dependence of horizontal stripe length.
# Identical to v3 baseline (parameters_bpmesh40k_v3_800.jl @ 100 step) except
# inner bp1 doubled: LinRange(-4,4,17) → LinRange(-4,4,33) → Δv₁ 0.5 → 0.25.
# Prediction: stripe horizontal segments ∝ Δv₁ (Greville midpoints in v₁).
# Expect ACF along v₁ to peak at lag = 0.25 instead of 0.5.
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 33); 5.0; 6.0],   # 2× denser
    bp2 = [-6.0; LinRange(-2.5, 2.5, 26); 6.0],              # unchanged
    P_DEG=2, K_REG=1, N_QUAD=6,
    N_PARTICLES=40_000,
    σ1=4/3, σ2=0.5,
    DT=0.001, N_STEPS=100,
    use_anderson=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=100,
    abs_floor=1e-10,
    stag_window=30, stag_rel_tol=0.1,
    damp_decay_start=200, damp_decay_factor=0.5,
    suffix="bpmesh40k_v3_bp1dense",
    seed=42,
)
