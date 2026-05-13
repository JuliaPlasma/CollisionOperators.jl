# 40k-particle Anderson run — v2: finer v1 mesh + tighter Anderson safety net.
#
# v1 from v1-archive analysis:
#   - v1 direction showed Gibbs spikes in spline projection (user obs.)
#   - bp1 inner spacing was 0.5 ([-4,4] LinRange(17)) → refine to 0.25 (33 pts)
#   - bp2 unchanged at LinRange(-2.5,2.5, 26) = 0.2 spacing (fine enough)
#
# Anderson safety net (commit 4c3cdac):
#   - abs_floor 1e-7 → 5e-8     (tighter; old run already cleared 1e-7 cleanly)
#   - stag_rel_tol 0.01 → 0.05  (more aggressive early-exit on plateau)
#   - keep damping decay at iter=200
#
# Cell count: bp1=37pts→36 cells, bp2=28pts→27 cells, ⇒ 36×27=972 cells
# (vs prev 20×27=540). DOF ≈ (36+P_DEG)*(27+P_DEG) = 38*29 ≈ 1102.
# Anticipate ~75% slower per Picard map but fewer iters from cleaner Gibbs.
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 33); 5.0; 6.0],
    bp2 = [-6.0; LinRange(-2.5, 2.5, 26); 6.0],
    P_DEG=2, K_REG=1, N_QUAD=6,
    N_PARTICLES=40_000,
    σ1=4/3, σ2=0.5,
    DT=0.001, N_STEPS=400,
    use_anderson=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    abs_floor=5e-8,
    stag_window=50, stag_rel_tol=0.05,
    damp_decay_start=200, damp_decay_factor=0.5,
    suffix="bpmesh40k_v2",
    seed=42,
)
