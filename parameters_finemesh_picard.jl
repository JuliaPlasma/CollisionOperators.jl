# Fine v₂ mesh, plain Picard counterpart of parameters_finemesh_anderson.jl.
PARAMS = SimParameters(
    V_MIN=-6.0, V_MAX=6.0,
    I_MIN=-4.0, I_MAX=4.0,
    P_DEG=2, K_REG=1,
    N_ELEM_1=10, N_ELEM_2=50,
    N_QUAD=6,
    N_PARTICLES=10_000,
    σ1=4/3, σ2=0.5,
    DT=0.001, N_STEPS=400,
    use_anderson=false,
    damping=1.0, m_anderson=8,
    tol=1e-12, max_iter=2000,
    suffix="finemesh_picard",
    seed=42,
)
