# iso100_logsq_unif — isotropic IC + ½ log f² trick on a UNIFORM mesh, 100 steps.
# Isotropic equilibrium start (σ≈1.007, energy-matched to v3), plain implicit
# midpoint (Gonzalez |Δv|² denominator blows up at equilibrium), use_logsq=true.
# Uniform breakpoints (Δv₁=0.5, Δv₂=0.25) — uniform-mesh counterpart of the
# isotropic logsq probe.
PARAMS = SimParameters(
    bp1 = collect(LinRange(-6.0, 6.0, 25)),   # Δv₁=0.5, 24 cells
    bp2 = collect(LinRange(-6.0, 6.0, 49)),   # Δv₂=0.25, 48 cells
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
    suffix="iso100_logsq_unif",
    seed=42,
)
