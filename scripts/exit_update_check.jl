# Does the solvers' exit update v1 = G(x_best) help or hurt on steps that stop
# above the target?
#
# step_anderson! (and step_newton!) return G(x_best) but report ‖F(x_best)‖.
# One extra evaluation of ‖F(G(x_best))‖ settles it:
#   ratio = ‖F(G(x_best))‖ / ‖F(x_best)‖  > 1  ⇒ the exit update makes the state worse
# On missed steps also ‖J û‖ along û = F(v1)/‖F(v1)‖ (central difference): ≈ 1 means
# a benign direction, ≫ 1 the stiff direction of scripts/taylor_test.jl.
#
# Advances the step-1500 sq_d04 state with Anderson exactly as run_simulation
# does (CPU FP64, DT = 0.005, preset knobs, abs_floor = 1e-10 — the FP64
# configuration that missed the target on 11 of 50 steps on a GPU).
#
#   julia -t auto --project=. scripts/exit_update_check.jl . <ckpt.jls> [n_steps=30]

const REPO = abspath(ARGS[1])
const CKPT = ARGS[2]
const NSTEPS = length(ARGS) ≥ 3 ? parse(Int, ARGS[3]) : 30

include(joinpath(REPO, "main.jl"))
using Serialization, Printf, LinearAlgebra

p = parse_overrides(include(joinpath(REPO, "parameters_LB_sq_d04.jl")),
    ["--collision_model=landau", "--DT=0.005", "--abs_floor=1e-10"])
USE_LOGSQ[] = p.use_logsq
ws = build_workspace(p)
ck = open(deserialize, CKPT)
v = copy(ck.v_particles)
w = ck.w_particles
N = size(v, 1)
dt = p.DT

f_coeffs = zeros(ws.n_dofs)
r_vec = zeros(ws.n_dofs); L_vec = zeros(ws.n_dofs)
G = zeros(N, 2); dot_v = zeros(N, 2); Gv = zeros(N, 2)
v_mid = similar(v); dvb = similar(v); dS_mid = zeros(N, 2); G_eff = zeros(N, 2)
f_buf = zeros(ws.n_dofs)
bufs = (zeros(N, 2), zeros(N, 2), zeros(N, 2), zeros(N, 2),
    zeros(2N, p.m_anderson), zeros(2N, p.m_anderson))

function F!(out, x, v0, S0)
    picard_map!(ws, Gv, reshape(x, N, 2), v0, w, S0, dt,
        v_mid, dvb, dS_mid, G_eff, dot_v, f_buf, r_vec, L_vec, G;
        use_gonzalez = p.use_gonzalez)
    @. out = x - $vec(Gv)
    return out
end

@printf("%5s %6s %11s %13s %7s %8s\n", "step", "evals", "‖F(x_best)‖", "‖F(G(x_best))‖", "ratio", "‖Jû‖")
ratios_missed = Float64[]; ratios_conv = Float64[]
for step in 1:NSTEPS
    l2_project!(ws, f_coeffs, v, w)
    f_s = build_field(ws, f_coeffs)
    S0 = compute_entropy(ws, f_s)
    compute_r!(ws, r_vec, f_s)
    ldiv!(L_vec, ws.M_lu, r_vec)
    compute_G!(ws, G, v, L_vec)
    compute_collision!(ws, dot_v, v, w, G)
    v1 = v .+ dt .* dot_v
    v0 = copy(v)

    it, nrm_best, _, _ = step_anderson!(ws, v1, v0, w, S0, dt,
        v_mid, dvb, dS_mid, G_eff, dot_v, f_buf, r_vec, L_vec, G, Gv, bufs...;
        m = p.m_anderson, max_iter = p.max_iter, tol = p.tol, abs_floor = p.abs_floor,
        stag_window = p.stag_window, stag_rel_tol = p.stag_rel_tol,
        damp_decay_start = p.damp_decay_start, damp_decay_factor = p.damp_decay_factor,
        damping = p.damping, use_gonzalez = p.use_gonzalez,
        exit_picard_step = true)   # the default: return G(x_best)

    x = vec(copy(v1))
    Fx = F!(similar(x), x, v0, S0)
    nrm_after = norm(Fx)
    ratio = nrm_after / nrm_best
    target = max(p.tol * norm(x), p.abs_floor)
    missed = nrm_best ≥ target
    Ju = NaN
    if missed
        u = Fx ./ nrm_after
        h = 1e-6
        Fp = F!(similar(x), x .+ h .* u, v0, S0)
        Fm = F!(similar(x), x .- h .* u, v0, S0)
        Ju = norm(Fp .- Fm) / (2h)
        push!(ratios_missed, ratio)
    else
        push!(ratios_conv, ratio)
    end
    @printf("%5d %6d %11.3e %13.3e %7.2f %8s%s\n", 1500 + step, it, nrm_best,
        nrm_after, ratio, missed ? @sprintf("%.2f", Ju) : "", missed ? "  missed" : "")
    flush(stdout)
    v .= v1   # carry G(x_best) forward, as the driver does
end

med(a) = isempty(a) ? NaN : sort(a)[cld(length(a), 2)]
@printf("\nmissed steps:    %d, median ratio %.2f, range %s\n", length(ratios_missed),
    med(ratios_missed), isempty(ratios_missed) ? "-" : string(extrema(round.(ratios_missed; digits = 2))))
@printf("converged steps: %d, median ratio %.2f\n", length(ratios_conv), med(ratios_conv))
