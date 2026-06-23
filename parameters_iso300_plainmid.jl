# iso300_plainmid — isotropic start probe, plain implicit midpoint.
# Goal: v3 (anisotropic→isotropic relaxation) shows honeycomb pseudo-physics.
# Here we START at the isotropic equilibrium itself. If honeycomb still appears
# with no relaxation driving it, the bulk pattern is pure mesh/projection;
# if it stays clean, the kernel needs the relaxation flux to pump it.
#
# σ = √((σ1²+σ2²)/2) = energy-matched isotropic equilibrium of the v3 IC
# (σ1=4/3, σ2=0.5) → ≈ 1.0069, so this is v3's true Landau fixed point.
#
# use_gonzalez=false: Gonzalez |Δv|² denominator → 0 at equilibrium (Δv ≈ 0),
# so the discrete-gradient correction is undefined/ill-conditioned here. Plain
# implicit midpoint (∇̄S = ∇S(v_mid)) drops it and runs.
#
# Identical mesh + B-spline + solver knobs to v3_800.
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 17); 5.0; 6.0],
    bp2 = [-6.0; LinRange(-2.5, 2.5, 26); 6.0],
    P_DEG=2, K_REG=1, N_QUAD=6,
    N_PARTICLES=40_000,
    σ1=sqrt((( 4/3)^2 + 0.5^2) / 2), σ2=sqrt((( 4/3)^2 + 0.5^2) / 2),
    DT=0.001, N_STEPS=300,
    use_anderson=true,
    use_gonzalez=false,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    abs_floor=1e-10,
    stag_window=30, stag_rel_tol=0.1,
    damp_decay_start=200, damp_decay_factor=0.5,
    suffix="iso300_plainmid",
    seed=42,
)
