# aniso_rect_plainmid — plain implicit midpoint twin of aniso100_logsq.
# Same dense non-uniform rect mesh (dense bp2), same anisotropic IC
# (σ1=4/3, σ2=0.5), same ½log f² trick, same seed=42 → identical initial
# particles as the Gonzalez headline run, so the two can be compared step-by-step.
# Only difference: use_gonzalez=false (drop the discrete-gradient correction →
# plain midpoint: conserves energy/momentum, NOT entropy-exact). Fresh 0→2000.
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 17); 5.0; 6.0],
    bp2 = [-6.0; LinRange(-2.5, 2.5, 26); 6.0],
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
    suffix="aniso_rect_plainmid",
    seed=42,
)
