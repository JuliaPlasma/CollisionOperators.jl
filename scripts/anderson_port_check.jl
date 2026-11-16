# Short-run check of the `:anderson` solve of the implicit step.
#
# Runs the time loop of `run_simulation` (main.jl) for a short run: the same
# initial condition, the same explicit predictor and the same `:anderson` solve
# call with the preset's knobs. It writes one row per step to
# `scripts/anderson_port_check_after.csv`:
#
#   step        : time step
#   iterations  : Picard-map evaluations of the solve
#   outer       : iterations of the method (Anderson or Picard)
#   residual    : the final ‖r‖ that the solver returns
#   target      : max(tol·‖v‖, abs_floor), the solve's stop test
#   exit        : residual, stalled or cap
#   dP1, dP2    : momentum drift P(step) − P(0), per component
#   dE_rel      : energy drift (E(step) − E(0)) / E(0)
#   over_200    : whether the step took more than 200 Picard-map evaluations
#   rel_diff    : ‖v − v_before‖ / ‖v_before‖ against the solution of the same
#                 step in `<outdir>/anderson_port_check_before.jls`
#   rel_step    : the same, for the step solved from the before-run's state at
#                 the start of the step rather than from this run's
#
# `scripts/anderson_port_check_before.csv` and that file are the output of the
# same run with the solver that `step_fixed_point!` replaces, `step_anderson!`:
# the version of this script in the commit that adds it writes them.
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
before = deserialize(joinpath(OUTDIR, "anderson_port_check_before.jls"))

# Initial condition, as in run_simulation.
v = zeros(N, 2)
v[:, 1] .= p.σ1 * randn(N)
v[:, 2] .= p.σ2 * randn(N)
p.v1_peak == 0.0 || error("the check covers the single-Gaussian initial condition")
w = fill(1.0 / N, N)
P0 = compute_momentum(v, w)
E0 = compute_energy(v, w)

f_coeffs = zeros(ws.n_dofs)
r_vec = zeros(ws.n_dofs)
L_vec = zeros(ws.n_dofs)
G = zeros(N, 2)
dot_v = zeros(N, 2)
v_mid = similar(v)
dv = similar(v)
dS_mid = zeros(N, 2)
G_eff = zeros(N, 2)
f_buf = zeros(ws.n_dofs)
Gv = zeros(N, 2)

# One time step from `v`: the explicit predictor and the :anderson solve call of
# run_simulation. Returns the new state and the solve's output.
function advance(v)
    l2_project!(ws, f_coeffs, v, w)
    f_s = build_field(ws, f_coeffs)
    S = compute_entropy(ws, f_s)
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
    v1 = v .+ p.DT .* dot_v
    f_abstol = max(p.tol * norm(v1), p.abs_floor)
    iter, res, outer, _ = step_fixed_point!(ws,
        v1, v, w, S, p.DT,
        v_mid, dv, dS_mid, G_eff, dot_v, f_buf,
        r_vec, L_vec, G, Gv;
        m = p.m_anderson, max_iter = p.max_iter, tol = p.tol,
        abs_floor = p.abs_floor,
        damping = p.damping, use_anderson = p.use_anderson,
        use_gonzalez = p.use_gonzalez,
        exit_picard_step = p.exit_picard_step)
    return v1, iter, res, outer, f_abstol
end

rows = String[]
@printf("%5s %6s %6s %11s %11s %9s %11s %11s %11s %5s %11s %11s\n", "step", "iter",
    "outer", "‖r‖", "target", "exit", "dP1", "dP2", "dE_rel", ">200", "rel_diff",
    "rel_step")
v_start = copy(v)
for step in 1:(p.N_STEPS)
    global v
    v1, iter, res, outer, f_abstol = advance(v)
    # The stop test of the solve, with ‖v‖ at the returned state, as in the
    # before-run; the solve itself tests against `f_abstol`.
    target = max(p.tol * norm(v1), p.abs_floor)
    route = res <= f_abstol ? "residual" : outer == p.max_iter ? "cap" : "stalled"
    v = v1
    P = compute_momentum(v, w)
    E = compute_energy(v, w)
    dP1, dP2, dE = P[1] - P0[1], P[2] - P0[2], (E - E0) / E0
    rel = norm(v - before[step]) / norm(before[step])
    # The same step from the before-run's state at its start.
    v_step = first(advance(step == 1 ? v_start : before[step - 1]))
    rel_step = norm(v_step - before[step]) / norm(before[step])
    push!(rows,
        join(
            (step, iter, outer, res, target, route, dP1, dP2, dE, iter > 200, rel,
                rel_step), ','))
    @printf("%5d %6d %6d %11.3e %11.3e %9s %11.3e %11.3e %11.3e %5s %11.3e %11.3e\n",
        step, iter, outer, res, target, route, dP1, dP2, dE, iter > 200, rel, rel_step)
    flush(stdout)
end

open(joinpath(@__DIR__, "anderson_port_check_after.csv"), "w") do io
    println(io,
        "step,iterations,outer,residual,target,exit,dP1,dP2,dE_rel,over_200,rel_diff,rel_step")
    foreach(r -> println(io, r), rows)
end
