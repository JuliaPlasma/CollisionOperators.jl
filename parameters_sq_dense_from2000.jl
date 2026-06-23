# sq_dense_from2000 — densified SQUARE mesh (bp1 = bp2), started from the
# aniso100_logsq step-2000 particle state (see make_sq_dense_checkpoint.jl).
# Mirrors the current bp1 layout ([-6,-5, inner[-4,4], 5,6]) but with the inner
# region densified to Δ=0.2 (matching the rect dense-core pitch) and applied to
# BOTH directions — a symmetric square mesh with no giant coarse outer cell.
# Tests whether the neg-ring lobes (which sit in rect's giant Δv2=3.5 outer cells)
# vanish once the mesh is square + uniformly dense in the bulk.
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 41); 5.0; 6.0],   # Δinner=0.2
    bp2 = [-6.0; -5.0; LinRange(-4.0, 4.0, 41); 5.0; 6.0],   # = bp1
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
    suffix="sq_dense_from2000",
    seed=42,
)
