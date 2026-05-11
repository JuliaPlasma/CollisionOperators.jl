# Plain Picard counterpart of parameters_default.jl. Same fixed setting
# (mesh, particles, dt, steps) but use_anderson=false, damping=1.0,
# max_iter aligned to 2000 so the comparison is symmetric.
PARAMS = SimParameters(
    V_MIN=-6.0, V_MAX=6.0,
    I_MIN=-4.0, I_MAX=4.0,
    P_DEG=2, K_REG=1,
    N_ELEM_1=10, N_ELEM_2=25,
    N_QUAD=6,
    N_PARTICLES=10_000,
    σ1=4/3, σ2=0.5,
    DT=0.001, N_STEPS=400,
    use_anderson=false,
    damping=1.0, m_anderson=8,    # m_anderson unused but kept for buffer sizing
    tol=1e-12, max_iter=2000,
    suffix="picard",
    seed=42,
)
