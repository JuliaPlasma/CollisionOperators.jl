# aniso_bp2eq_plainmid — square mesh (bp2=bp1) + ½log f², PLAIN midpoint.
# Continuation of aniso1000_bp2eqbp1_logsq past step 1000 with use_gonzalez=false
# to avoid Gonzalez |Δv|² denominator blowup as the distribution isotropizes
# (Δv→0 near equilibrium). Resume from the step-1000 checkpoint of the Gonzalez
# square-mesh run (copied to checkpoint_aniso_bp2eq_plainmid_step1000.jls).
# Plain implicit midpoint: conserves energy/momentum, not entropy-exact.
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 17); 5.0; 6.0],
    bp2 = [-6.0; -5.0; LinRange(-4.0, 4.0, 17); 5.0; 6.0],
    P_DEG=2, K_REG=1, N_QUAD=6,
    N_PARTICLES=40_000,
    σ1=4/3, σ2=0.5,
    DT=0.001, N_STEPS=2000,
    use_anderson=true,
    use_gonzalez=false,
    use_logsq=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    abs_floor=1e-10,
    stag_window=30, stag_rel_tol=0.1,
    damp_decay_start=200, damp_decay_factor=0.5,
    suffix="aniso_bp2eq_plainmid",
    seed=42,
)
