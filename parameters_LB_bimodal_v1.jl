# LB_bimodal_v1 — Lenard-Bernstein on the identical mesh + IC + timestep as the
# 2D Landau bimodal run (parameters_bimodal.jl / suffix=bimodal_v1_rclone), for
# the dS/dt operator comparison. IC: bimodal in v₁ (balanced 50/50 mixture of
# N(±2, 1)) with v₂ ~ N(0, 1). ν=1, 25000 steps. Same seed as the standalone
# LB/parameters_LB_bimodal_v1.jl so the particle IC is bitwise identical.
# Runs on GPU with --use_gpu=true (projection + log-gradient on device, O(N)
# drift on CPU); collision_model=:lb.
PARAMS = SimParameters(
    bp1 = [-6.0; LinRange(-5.0, 5.0, 26); 6.0],          # Δ=0.4 over [-5,5]
    bp2 = [-6.0; -4.5:0.4:4.5; 6.0],
    P_DEG = 2, K_REG = 1, N_QUAD = 6,
    N_PARTICLES = 40_000,
    σ1 = 1.0, σ2 = 1.0,
    v1_peak = 2.0,
    collision_model = :lb,
    nu = 1.0,
    DT = 0.001, N_STEPS = 25000,
    use_anderson = true,
    use_gonzalez = true,
    damping = 0.7, m_anderson = 8,
    tol = 1e-12, max_iter = 2000,
    abs_floor = 1e-10,
    stag_window = 30, stag_rel_tol = 0.1,
    damp_decay_start = 200, damp_decay_factor = 0.5,
    snap_every = 250,
    suffix = "LB_bimodal_v1_gpu",
    seed = 42
)
