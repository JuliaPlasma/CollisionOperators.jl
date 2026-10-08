# Taylor remainder test of the implicit-step residual F(v) = v - G(v).
#
#   r(ε) = ‖F(v + ε u) - F(v) - ε J u‖
#
# smooth map  ⇒ r ∝ ε²  (slope 2 on log–log), Newton model exact to O(ε²)
# kinked map  ⇒ r ∝ ε   (slope 1), no Jacobian predicts a finite step
#
# State: sq_d04 step-1500 checkpoint (t = 2), DT = 0.005, CPU FP64, N = 40k —
# the configuration of the nkab pod runs. Jacobian products by central
# difference at two small h (their agreement bounds the error in J u).
#
#   julia -t auto --project=. scripts/taylor_test.jl . <ckpt.jls> [use_gonzalez=true]
#
# Results and interpretation: docs/src/newton_krylov.md, "Taylor test of the
# residual map".

const REPO = abspath(ARGS[1])
const CKPT = ARGS[2]
const GONZ = length(ARGS) ≥ 3 ? parse(Bool, ARGS[3]) : true

include(joinpath(REPO, "main.jl"))
using Serialization, Printf, LinearAlgebra, Random

p = parse_overrides(include(joinpath(REPO, "parameters_LB_sq_d04.jl")),
    ["--collision_model=landau", "--DT=0.005", "--use_gonzalez=$GONZ"])
USE_LOGSQ[] = p.use_logsq
ws = build_workspace(p)
ck = open(deserialize, CKPT)
v0 = ck.v_particles
w = ck.w_particles
N = size(v0, 1)
dt = p.DT

f_coeffs = zeros(ws.n_dofs)
l2_project!(ws, f_coeffs, v0, w)
f_s = build_field(ws, f_coeffs)
S0 = compute_entropy(ws, f_s)

r_vec = zeros(ws.n_dofs); L_vec = zeros(ws.n_dofs)
G = zeros(N, 2); dot_v = zeros(N, 2); Gv = zeros(N, 2)
v_mid = similar(v0); dvb = similar(v0); dS_mid = zeros(N, 2); G_eff = zeros(N, 2)
f_buf = zeros(ws.n_dofs)

const NEVAL = Ref(0)
function F!(out::Vector, v::Vector)
    NEVAL[] += 1
    picard_map!(ws, Gv, reshape(v, N, 2), v0, w, S0, dt,
        v_mid, dvb, dS_mid, G_eff, dot_v, f_buf, r_vec, L_vec, G;
        use_gonzalez = GONZ)
    @. out = v - $vec(Gv)
    return out
end

# Explicit-Euler predictor, exactly as in run_simulation.
compute_r!(ws, r_vec, f_s)
ldiv!(L_vec, ws.M_lu, r_vec)
compute_G!(ws, G, v0, L_vec)
compute_collision!(ws, dot_v, v0, w, G)
v_pred = vec(v0 .+ dt .* dot_v)

# Converged point: Anderson to the FP64 target.
v_star = reshape(copy(v_pred), N, 2)
t0 = time()
it, res, _, r0 = step_anderson!(ws, v_star, v0, w, S0, dt,
    v_mid, dvb, dS_mid, G_eff, dot_v, f_buf, r_vec, L_vec, G,
    Gv, zeros(N, 2), zeros(N, 2), zeros(N, 2), zeros(N, 2),
    zeros(2N, p.m_anderson), zeros(2N, p.m_anderson);
    m = p.m_anderson, max_iter = 300, tol = 1e-12, abs_floor = 1e-11,
    stag_window = 30, stag_rel_tol = 0.1, damping = p.damping,
    use_gonzalez = GONZ)
@printf("gonzalez=%s  Anderson: %d evals, ‖F‖ %.2e → %.2e  (%.2f s/eval)\n",
    GONZ, it, r0, res, (time() - t0) / it)
v_star = vec(v_star)

function taylor(label, v, u)
    u = u / norm(u)
    Fv = F!(similar(v), v)
    Fp = similar(v); Fm = similar(v)
    Ju = Dict{Float64, Vector{Float64}}()
    for h in (1e-4, 1e-6)
        F!(Fp, v .+ h .* u); F!(Fm, v .- h .* u)
        Ju[h] = (Fp .- Fm) ./ (2h)
    end
    J = Ju[1e-6]
    @printf("\n== %s   ‖F(v)‖ = %.3e   ‖Ju‖ = %.4f   ‖Ju(1e-4) - Ju(1e-6)‖ = %.2e\n",
        label, norm(Fv), norm(J), norm(Ju[1e-4] .- J))
    @printf("%10s %12s %12s %10s %8s\n", "ε", "r(ε)", "r/ε²", "r/(ε‖Ju‖)", "slope")
    prev = nothing
    for ε in 10.0 .^ (-1:-1:-9)
        F!(Fp, v .+ ε .* u)
        r = norm(Fp .- Fv .- ε .* J)
        slope = prev === nothing ? NaN : log10(prev / r)
        @printf("%10.0e %12.3e %12.3e %10.2e %8.2f\n", ε, r, r / ε^2, r / (ε * norm(J)), slope)
        prev = r
    end
end

rng = MersenneTwister(1)
taylor("predictor, u = F/‖F‖", v_pred, F!(similar(v_pred), v_pred))
taylor("predictor, u random", v_pred, randn(rng, 2N))
taylor("solution,  u = F/‖F‖", v_star, F!(similar(v_star), v_star))
taylor("solution,  u random", v_star, randn(rng, 2N))
println("\ntotal evaluations: ", NEVAL[])
