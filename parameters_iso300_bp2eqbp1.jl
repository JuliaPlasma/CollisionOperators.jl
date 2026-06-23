# iso300_bp2eqbp1 — isotropic start, plain midpoint, bp2 == bp1.
# Same as iso300_plainmid but the v₂ mesh is set identical to v₁ (square,
# symmetric breakpoints) per Sandra's request: bp2's extra-dense [-2.5,2.5]/26
# layout was tuned long ago to suppress v₂-tail negative-f_s blocks; test
# whether a coarser symmetric bp2=bp1 still keeps neg-mask under control.
#
# σ = √((4/3)²+0.5²)/2) ≈ 1.0069 (energy-matched isotropic equilibrium of v3).
# use_gonzalez=false: plain implicit midpoint (no |Δv|² denominator).
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 17); 5.0; 6.0],
    bp2 = [-6.0; -5.0; LinRange(-4.0, 4.0, 17); 5.0; 6.0],
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
    suffix="iso300_bp2eqbp1",
    seed=42,
)
