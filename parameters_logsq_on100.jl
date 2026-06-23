# logsq_on100 — 100-step trick: log f = ½ log f² (use_logsq=true).
# Identical to parameters_logsq_off100.jl except use_logsq=true, so f_s<0
# Gibbs-undershoot quadrature points now contribute to entropy S and seed r
# (via |f| guard) instead of being clamped to zero. Compare spike vs off run.
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 17); 5.0; 6.0],
    bp2 = [-6.0; LinRange(-2.5, 2.5, 26); 6.0],
    P_DEG=2, K_REG=1, N_QUAD=6,
    N_PARTICLES=40_000,
    σ1=sqrt((( 4/3)^2 + 0.5^2) / 2), σ2=sqrt((( 4/3)^2 + 0.5^2) / 2),
    DT=0.001, N_STEPS=100,
    use_anderson=true,
    use_gonzalez=false,
    use_logsq=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    abs_floor=1e-10,
    stag_window=30, stag_rel_tol=0.1,
    damp_decay_start=200, damp_decay_factor=0.5,
    suffix="logsq_on100",
    seed=42,
)
