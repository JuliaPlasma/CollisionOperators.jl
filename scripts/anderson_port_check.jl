# Short-run check of the `:anderson` solve of the implicit step.
#
# Runs the time loop of `run_simulation` (main.jl) for a short run: the same
# initial condition, the same explicit predictor and the same `:anderson` solve
# call with the preset's knobs. It writes one row per step to
# `scripts/anderson_port_check_before.csv`:
#
#   step        : time step
#   iterations  : Picard-map evaluations of the solve
#   residual    : the final ‖r‖ that the solver returns
#   target      : max(tol·‖v‖, abs_floor), the solve's stop test
#   exit        : residual, stagnation or cap
#   dP1, dP2    : momentum drift P(step) − P(0), per component
#   dE_rel      : energy drift (E(step) − E(0)) / E(0)
#   over_200    : whether the step took more than 200 iterations
#
# and the solution of every step to `<outdir>/anderson_port_check_before.jls`.
#
# Preset: parameters_sq_d04.jl (2D Landau, CPU FP64), with N_PARTICLES and
# N_STEPS overridden for a run of about a minute.
#
#   julia --startup-file=no -t auto --project=. scripts/anderson_port_check.jl <outdir>

const REPO = abspath(joinpath(@__DIR__, ".."))
const OUTDIR = abspath(ARGS[1])
const PRESET = "parameters_sq_d04.jl"
const OVERRIDES = ["--N_PARTICLES=2000", "--N_STEPS=50"]

include(joinpath(REPO, "main.jl"))
using Serialization, Printf, LinearAlgebra, Random

p = parse_overrides(include(joinpath(REPO, PRESET)), OVERRIDES)
USE_LOGSQ[] = p.use_logsq
Random.seed!(p.seed)
ws = build_workspace(p)
N = p.N_PARTICLES

# Initial condition, as in run_simulation.
v = zeros(N, 2)
v[:, 1] .= p.σ1 * randn(N)
v[:, 2] .= p.σ2 * randn(N)
p.v1_peak == 0.0 || error("the check covers the single-Gaussian initial condition")
w = fill(1.0 / N, N)
f_coeffs = zeros(ws.n_dofs)
l2_project!(ws, f_coeffs, v, w)
f_s = build_field(ws, f_coeffs)
S = compute_entropy(ws, f_s)
P0 = compute_momentum(v, w)
E0 = compute_energy(v, w)

r_vec = zeros(ws.n_dofs)
L_vec = zeros(ws.n_dofs)
G = zeros(N, 2)
dot_v = zeros(N, 2)
v1 = copy(v)
v_mid = similar(v)
dv = similar(v)
dS_mid = zeros(N, 2)
G_eff = zeros(N, 2)
f_buf = zeros(ws.n_dofs)
Gv = zeros(N, 2)
r_curr = zeros(N, 2)
r_prev = zeros(N, 2)
Gv_prev = zeros(N, 2)
v_old_buf = zeros(N, 2)
ΔF = zeros(2N, p.m_anderson)
ΔG = zeros(2N, p.m_anderson)

rows = String[]
solutions = Matrix{Float64}[]
@printf("%5s %6s %11s %11s %11s %11s %11s %11s %5s\n", "step", "iter", "‖r‖", "target",
    "exit", "dP1", "dP2", "dE_rel", ">200")
for step in 1:(p.N_STEPS)
    global f_s, S
    # Explicit predictor, as in run_simulation.
    if p.collision_model == :lb
        Base.invokelatest(LOGGRAD_FN[], ws, G, v, f_coeffs)
        n_lb, U1_lb, U2_lb, Q_lb = compute_moments(v, w)
        A1_lb, A2_lb, B_lb = compute_drift_multipliers(v, w, G, n_lb, U1_lb, U2_lb, Q_lb)
        compute_LB_velocity!(dot_v, v, G, A1_lb, A2_lb, B_lb, p.nu)
    else
        compute_r!(ws, r_vec, f_s)
        ldiv!(L_vec, ws.M_lu, r_vec)
        Base.invokelatest(COMPG_FN[], ws, G, v, L_vec)
        Base.invokelatest(COLLISION_FN[], ws, dot_v, v, w, G)
    end
    @. v1 = v + p.DT * dot_v

    # The :anderson solve call of main.jl.
    iter, res, _, _ = step_anderson!(ws,
        v1, v, w, S, p.DT,
        v_mid, dv, dS_mid, G_eff, dot_v, f_buf,
        r_vec, L_vec, G,
        Gv, r_curr, r_prev, Gv_prev, v_old_buf, ΔF, ΔG;
        m = p.m_anderson, max_iter = p.max_iter, tol = p.tol,
        abs_floor = p.abs_floor,
        stag_window = p.stag_window,
        stag_rel_tol = p.stag_rel_tol,
        damp_decay_start = p.damp_decay_start,
        damp_decay_factor = p.damp_decay_factor,
        damping = p.damping, use_anderson = p.use_anderson,
        use_gonzalez = p.use_gonzalez,
        exit_picard_step = p.exit_picard_step)

    # The stop test of the solve, with ‖v‖ at the returned state.
    target = max(p.tol * norm(v1), p.abs_floor)
    route = res < target ? "residual" : iter == p.max_iter ? "cap" : "stagnation"

    v .= v1
    l2_project!(ws, f_coeffs, v, w)
    f_s = build_field(ws, f_coeffs)
    S = compute_entropy(ws, f_s)
    P = compute_momentum(v, w)
    E = compute_energy(v, w)
    dP1, dP2, dE = P[1] - P0[1], P[2] - P0[2], (E - E0) / E0
    push!(solutions, copy(v))
    push!(rows, join((step, iter, res, target, route, dP1, dP2, dE, iter > 200), ','))
    @printf("%5d %6d %11.3e %11.3e %11s %11.3e %11.3e %11.3e %5s\n", step, iter, res,
        target, route, dP1, dP2, dE, iter > 200)
    flush(stdout)
end

mkpath(OUTDIR)
open(joinpath(@__DIR__, "anderson_port_check_before.csv"), "w") do io
    println(io, "step,iterations,residual,target,exit,dP1,dP2,dE_rel,over_200")
    foreach(r -> println(io, r), rows)
end
serialize(joinpath(OUTDIR, "anderson_port_check_before.jls"), solutions)
