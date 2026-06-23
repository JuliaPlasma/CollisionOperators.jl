# logsq_off100 — 100-step baseline: clamped log f (use_logsq=false).
# Same mesh/IC/solver as iso300_plainmid (dense bp2, isotropic σ≈1.007,
# plain midpoint). Short run to compare the log f = ½ log f² trick's effect
# on the Gibbs spike / negative-f_s checkerboard.
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 17); 5.0; 6.0],
    bp2 = [-6.0; LinRange(-2.5, 2.5, 26); 6.0],
    P_DEG=2, K_REG=1, N_QUAD=6,
    N_PARTICLES=40_000,
    σ1=sqrt((( 4/3)^2 + 0.5^2) / 2), σ2=sqrt((( 4/3)^2 + 0.5^2) / 2),
    DT=0.001, N_STEPS=100,
    use_anderson=true,
    use_gonzalez=false,
    use_logsq=false,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    abs_floor=1e-10,
    stag_window=30, stag_rel_tol=0.1,
    damp_decay_start=200, damp_decay_factor=0.5,
    suffix="logsq_off100",
    seed=42,
)
