# Plain Picard counterpart of parameters_default.jl. Same mesh, particle
# count, dt, steps; only the solver knobs change (`use_anderson=false`,
# `damping=1.0`). `max_iter` aligned to 2000 so the comparison is symmetric.
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 17); 5.0; 6.0],
    bp2 = [-6.0; LinRange(-2.5, 2.5, 13); 6.0],
    P_DEG=2, K_REG=1, N_QUAD=6,
    N_PARTICLES=10_000,
    σ1=4/3, σ2=0.5,
    DT=0.001, N_STEPS=400,
    use_anderson=false,
    damping=1.0, m_anderson=8,    # m_anderson unused but kept for buffer sizing
    tol=1e-12, max_iter=2000,
    suffix="picard",
    seed=42,
)
