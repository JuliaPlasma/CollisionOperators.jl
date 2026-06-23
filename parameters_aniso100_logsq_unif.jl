# aniso100_logsq_unif — anisotropic v3 IC + ½ log f² trick on a UNIFORM mesh.
# Same as aniso100_logsq (σ1=4/3, σ2=0.5, Gonzalez, use_logsq=true, 100 steps)
# but full-domain uniform breakpoints (Δv₁=0.5, Δv₂=0.25) instead of the dense
# non-uniform bp2. Tests whether the trick still suppresses the spike when the
# mesh is uniform (no dense-region projection floor).
PARAMS = SimParameters(
    bp1 = collect(LinRange(-6.0, 6.0, 25)),   # Δv₁=0.5, 24 cells
    bp2 = collect(LinRange(-6.0, 6.0, 49)),   # Δv₂=0.25, 48 cells
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
    suffix="aniso100_logsq_unif",
    seed=42,
)
