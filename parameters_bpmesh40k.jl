# 40k-particle Anderson run over the inner-refined bp mesh
# (bp2 has 25 inner cells = finemesh-style v₂, bp1 = default inner refinement).
#
# Tol relaxed from 1e-12 → 1e-10 to avoid late-step max_iter exhaustion:
# previous bpmesh40k run plateaued at nrm_best ≈ 1e-7 (already 1e-3 relative drop)
# but tol*‖v‖ ≈ 2e-10 demanded ≈ 1e-9 absolute → 5+ non-converged steps near
# step 325–350 each burning the full max_iter=2000, run wall-time exploded
# from <30 s/step to >10 min/step before manual kill at step 350.
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 17); 5.0; 6.0],
    bp2 = [-6.0; LinRange(-2.5, 2.5, 26); 6.0],
    P_DEG=2, K_REG=1, N_QUAD=6,
    N_PARTICLES=40_000,
    σ1=4/3, σ2=0.5,
    DT=0.001, N_STEPS=400,
    use_anderson=true,
    damping=0.7, m_anderson=8,
    tol=1e-10, max_iter=2000,
    suffix="bpmesh40k",
    seed=42,
)
