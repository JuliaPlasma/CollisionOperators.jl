# v5 — 100-step probe: refine bp2 inner to kill cell-locking stripes.
# Goal: test cell-locking hypothesis (see docs/cell_locking_analysis.md).
#
# bp2 inner LinRange(26) → LinRange(41): Δv₂ 0.2 → 0.125
# Per-σ resolution:
#   v₁: 4 cells/σ (unchanged from v4)
#   v₂: 4 cells/σ (was 2.5)
# bp1 identical to v4. All Anderson criteria identical to v3/v4.
# Prediction:
#   - stripe ACF peak should shift to lag = 0.125 (or vanish if amplitude < noise)
#   - neg_part further drop from v4's 0.0092
#   - ‖f_p − f_s‖₂ further drop from v4's 0.0328
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 25); 5.0; 6.0],
    bp2 = [-6.0; LinRange(-2.5, 2.5, 41); 6.0],
    P_DEG=2, K_REG=1, N_QUAD=6,
    N_PARTICLES=40_000,
    σ1=4/3, σ2=0.5,
    DT=0.001, N_STEPS=100,
    use_anderson=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    abs_floor=1e-10,
    stag_window=30, stag_rel_tol=0.1,
    damp_decay_start=200, damp_decay_factor=0.5,
    suffix="bpmesh40k_v5",
    seed=42,
)
