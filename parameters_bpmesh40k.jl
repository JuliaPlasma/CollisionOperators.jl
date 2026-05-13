# 40k-particle Anderson run over the inner-refined bp mesh
# (bp2 has 25 inner cells = finemesh-style v₂, bp1 = default inner refinement).
#
# Uses the new `step_anderson!` safety net (commit 4c3cdac):
#   abs_floor=1e-7  caps the effective tol so we never demand below the
#                   Picard noise floor (old run stalled at nrm_best≈1e-7
#                   while tol*‖v‖ ≈ 2e-10 demanded ~1e-9 absolute → max_iter
#                   exhaustion, wall-time exploded to >10 min/step by step 325).
#   stag_window=50, stag_rel_tol=0.01  early-exit on plateau.
#   damp_decay_start=200, damp_decay_factor=0.5  stabilize stiff late steps.
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 17); 5.0; 6.0],
    bp2 = [-6.0; LinRange(-2.5, 2.5, 26); 6.0],
    P_DEG=2, K_REG=1, N_QUAD=6,
    N_PARTICLES=40_000,
    σ1=4/3, σ2=0.5,
    DT=0.001, N_STEPS=400,
    use_anderson=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    abs_floor=1e-7,
    stag_window=50, stag_rel_tol=0.01,
    damp_decay_start=200, damp_decay_factor=0.5,
    suffix="bpmesh40k_fix",
    seed=42,
)
