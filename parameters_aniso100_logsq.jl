# aniso100_logsq — anisotropic v3 IC + log f = ½ log f² trick, 100 steps.
# Original honeycomb scenario (σ1=4/3, σ2=0.5, dense bp2, Gonzalez discrete
# gradient — |Δv|² denominator fine off-equilibrium) with use_logsq=true to
# test whether the ½ log f² seed kills the bulk Gibbs spike in the real
# anisotropic→isotropic relaxation, as it did for the isotropic probe.
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 17); 5.0; 6.0],
    bp2 = [-6.0; LinRange(-2.5, 2.5, 26); 6.0],
    P_DEG=2, K_REG=1, N_QUAD=6,
    N_PARTICLES=40_000,
    σ1=4/3, σ2=0.5,
    DT=0.001, N_STEPS=100,
    use_anderson=true,
    use_gonzalez=true,
    use_logsq=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    abs_floor=1e-10,
    stag_window=30, stag_rel_tol=0.1,
    damp_decay_start=200, damp_decay_factor=0.5,
    suffix="aniso100_logsq",
    seed=42,
)
