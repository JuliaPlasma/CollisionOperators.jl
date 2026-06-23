# aniso1000_bp2eqbp1_logsq — square mesh (bp2=bp1) + ½ log f² trick, 1000 steps.
# Same anisotropic relaxation as aniso100_logsq (σ1=4/3, σ2=0.5, Gonzalez,
# use_logsq=true) but with a symmetric square mesh (bp2 set equal to bp1)
# instead of the dense non-uniform bp2. Isolates whether the residual negative
# ring is mesh-anisotropy driven (honeycomb tied to dense bp2 / stripe pitch)
# vs Landau-kernel intrinsic: ring gone → mesh artifact; ring persists → solver.
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 17); 5.0; 6.0],
    bp2 = [-6.0; -5.0; LinRange(-4.0, 4.0, 17); 5.0; 6.0],
    P_DEG=2, K_REG=1, N_QUAD=6,
    N_PARTICLES=40_000,
    σ1=4/3, σ2=0.5,
    DT=0.001, N_STEPS=1000,
    use_anderson=true,
    use_gonzalez=true,
    use_logsq=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    abs_floor=1e-10,
    stag_window=30, stag_rel_tol=0.1,
    damp_decay_start=200, damp_decay_factor=0.5,
    suffix="aniso1000_bp2eqbp1_logsq",
    seed=42,
)
