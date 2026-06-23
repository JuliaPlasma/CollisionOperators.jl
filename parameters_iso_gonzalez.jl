# iso_gonzalez — DIRECT test of the Gonzalez |Δv|² denominator at equilibrium.
# Twin of iso300_plainmid (isotropic equilibrium start, σ1=σ2=√((σ1²+σ2²)/2)
# ≈1.0069, dense rect mesh) but with use_gonzalez=TRUE. Starting AT the Landau
# fixed point, the per-step displacement Δv = v^{n+1}-v^n is pure thermal noise
# (≈0) from step 1 — the worst case for the discrete-gradient correction
# (S1-S0-Δv·∇S)/|Δv|². If the denominator truly blows up, iter→max_iter / NaN /
# residual stuck should appear immediately. If it stays healthy, the original
# "can't run Gonzalez at equilibrium" assumption is refuted. 2000 steps.
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 17); 5.0; 6.0],
    bp2 = [-6.0; LinRange(-2.5, 2.5, 26); 6.0],
    P_DEG=2, K_REG=1, N_QUAD=6,
    N_PARTICLES=40_000,
    σ1=sqrt((( 4/3)^2 + 0.5^2) / 2), σ2=sqrt((( 4/3)^2 + 0.5^2) / 2),
    DT=0.001, N_STEPS=2000,
    use_anderson=true,
    use_gonzalez=true,
    use_logsq=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    abs_floor=1e-10,
    stag_window=30, stag_rel_tol=0.1,
    damp_decay_start=200, damp_decay_factor=0.5,
    suffix="iso_gonzalez",
    seed=42,
)
