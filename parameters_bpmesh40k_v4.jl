# v4 — 100-step probe: only change bp1 inner refinement vs v3 baseline.
# Goal: isolate effect of denser v₁ mesh on (a) Anderson residual, (b) spike growth.
#
# bp1 inner LinRange(17) → LinRange(25): Δv₁ 0.5 → 0.333, matches bp2 inner pt count
# Per-σ resolution rebalanced:
#   v₁: 4 cells/σ (was 2.7)
#   v₂: 2.5 cells/σ (unchanged)
# All Anderson criteria identical to v3 — only the mesh changes.
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 25); 5.0; 6.0],
    bp2 = [-6.0; LinRange(-2.5, 2.5, 26); 6.0],
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
    suffix="bpmesh40k_v4",
    seed=42,
)
