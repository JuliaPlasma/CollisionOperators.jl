# 40k-particle Anderson run — v3: isolate Anderson criteria changes.
#
# Sandra's request: rerun the original failing case (same mesh as bpmesh40k v1)
# but only change the Anderson convergence criteria, so we can attribute any
# improvement to criteria vs mesh.
#
# Mesh: identical to v1 (bp1 inner LinRange(17), bp2 inner LinRange(26))
# Criteria changes vs v1:
#   tol         : 1e-10 → 1e-12       (was loose tol; let abs_floor bind)
#   abs_floor   : (n/a)  → 1e-10      (probe actual Picard noise floor)
#   stag_window : (n/a)  → 30         (catch failing steps fast)
#   stag_rel_tol: (n/a)  → 0.1        (>10% drop needed in window or bail)
#   damp_decay  : (n/a)  → 0.5 after iter 200
#   max_iter    : 2000 (unchanged — let stagnation be the only fast-exit;
#                       isolates the effect from prior max_iter cap)
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
    abs_floor=1e-10,
    stag_window=30, stag_rel_tol=0.1,
    damp_decay_start=200, damp_decay_factor=0.5,
    suffix="bpmesh40k_v3",
    seed=42,
)
