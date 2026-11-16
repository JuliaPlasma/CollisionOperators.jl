# Unit tests for the physics kernels the driver depends on: the particle moments
# and the LB drift multipliers, plus CLI override parsing.
#
# main.jl is a script, not a module, so it is included once here and every
# testset below shares that load — it pulls CairoMakie and Mantis, which are
# slow. That include doubles as a smoke test that the driver's include chain
# (io.jl / solver.jl / plots.jl) still loads. The dependency-free CSV and
# checkpoint helpers are covered separately in integration/io_helpers.jl.
using Test

include(joinpath(@__DIR__, "..", "..", "main.jl"))

@testset "compute_momentum / compute_energy" begin
    v = [1.0 2.0; -3.0 0.5]
    w = [0.25, 0.75]
    @test all(compute_momentum(v, w) .≈
              (0.25 * 1.0 + 0.75 * -3.0, 0.25 * 2.0 + 0.75 * 0.5))
    @test compute_energy(v, w) ≈
          0.5 * (0.25 * (1.0^2 + 2.0^2) + 0.75 * ((-3.0)^2 + 0.5^2))

    # Uniform weights summing to 1: energy is the mean of ½|v|².
    n = 100
    vr = randn(n, 2)
    wr = fill(1 / n, n)
    @test compute_energy(vr, wr) ≈ sum(0.5 .* (vr[:, 1] .^ 2 .+ vr[:, 2] .^ 2)) / n
end

@testset "compute_moments" begin
    v = [1.0 2.0; -3.0 0.5; 0.0 -1.0]
    w = [0.2, 0.3, 0.5]
    n, U1, U2, Q = MantisWrappers.compute_moments(v, w)
    @test n ≈ 1.0
    @test U1 ≈ 0.2 * 1.0 + 0.3 * -3.0
    @test U2 ≈ 0.2 * 2.0 + 0.3 * 0.5 + 0.5 * -1.0
    @test Q ≈ 0.2 * 5.0 + 0.3 * 9.25 + 0.5 * 1.0
    # Energy is half the second moment.
    @test compute_energy(v, w) ≈ 0.5 * Q
end

@testset "LB drift conserves momentum and energy" begin
    # compute_drift_multipliers solves the 3x3 system precisely so that the LB
    # velocity has zero net momentum and zero net energy change. That invariant
    # is the operator's reason for existing, so assert it directly.
    rng = Random.MersenneTwister(2024)
    for N in (16, 257)
        v = randn(rng, N, 2)
        w = rand(rng, N)
        w ./= sum(w)
        g = randn(rng, N, 2)
        n, U1, U2, Q = MantisWrappers.compute_moments(v, w)
        A1, A2, B = MantisWrappers.compute_drift_multipliers(v, w, g, n, U1, U2, Q)
        dot_v = similar(v)
        MantisWrappers.compute_LB_velocity!(dot_v, v, g, A1, A2, B, 1.0)
        p1 = sum(w[i] * dot_v[i, 1] for i in 1:N)
        p2 = sum(w[i] * dot_v[i, 2] for i in 1:N)
        de = sum(w[i] * (v[i, 1] * dot_v[i, 1] + v[i, 2] * dot_v[i, 2]) for i in 1:N)
        scale = maximum(abs, dot_v)
        @test abs(p1) < 1e-12 * scale
        @test abs(p2) < 1e-12 * scale
        @test abs(de) < 1e-12 * scale * maximum(abs, v)
    end
end

@testset "parse_overrides" begin
    base = SimParameters(suffix = "base")
    p = parse_overrides(base, ["--suffix=run2", "--DT=0.004", "--N_STEPS=300"])
    @test p.suffix == "run2"
    @test p.DT == 0.004
    @test p.N_STEPS == 300
    @test p.N_PARTICLES == base.N_PARTICLES          # untouched fields survive
    @test parse_overrides(base, String[]) == base
    @test_throws Exception parse_overrides(base, ["--no_such_field=1"])
end

using LinearAlgebra: Diagonal, I, norm, qr
using Logging: Logging

@testset "fixed_point_method" begin
    for (m, damping) in ((8, 0.7), (3, 0.4))
        a = fixed_point_method(; m, damping)
        @test a isa GeometricSolvers.Anderson{m}
        @test a.restart_factor == 1e12
        @test a.member isa GeometricSolvers.Picard
        @test a.member.damping == damping
        q = fixed_point_method(; m, damping, use_anderson = false)
        @test q isa GeometricSolvers.Picard
        @test q.damping == damping
    end
end

# fixed_point_solve! on linear maps G(v) = A v + b with the known fixed point
# v* = (I - A) \ b.
linear_map(A, b) = (out, v) -> (out .= A * v .+ b; out)

@testset "fixed_point_solve!: sign of the residual and damping" begin
    # Damped Picard v ← (1 - ω) v + ω G(v) scales the error along an eigenvector
    # of A with eigenvalue λ by 1 - ω + ωλ. For λ = -1.5 that is -0.25 at ω = 0.5
    # but -1.5 undamped, and the step v + ωF(v) of the opposite sign scales by
    # 1 + ω(1 - λ) > 1 for every λ < 1: both diverge where the solve converges.
    A = Diagonal([0.5, -0.3, -1.5, 0.2])
    b = [1.0, -2.0, 0.5, 3.0]
    vstar = (I - A) \ b
    v1 = zeros(4)
    evals, res, iterations, res0 = @test_logs min_level = Logging.Warn fixed_point_solve!(
        linear_map(A, b), v1, zeros(4); damping = 0.5, use_anderson = false,
        tol = 1e-12, abs_floor = 1e-12, max_iter = 500)
    @test isapprox(v1, vstar; rtol = 1e-10)
    @test res <= 1e-12
    @test res0 ≈ norm(b)                    # ‖F(v₀)‖ = ‖v₀ − G(v₀)‖ at v₀ = 0
    @test evals == iterations + 2           # F(v₀), one per step, the exit G(v)
    @test 1 < iterations < 500

    # One Anderson step is the member step v₀ − ωF(v₀) = v₀ + ω(G(v₀) − v₀): at
    # the cap it is the best iterate, returned with a warning.
    v0 = [1.0, 1.0, 1.0, 1.0]
    v1 = copy(v0)
    @test_logs (:warn, "Solver did not converge") fixed_point_solve!(
        linear_map(A, b), v1, zeros(4); damping = 0.5, max_iter = 1,
        exit_picard_step = false)
    @test v1 ≈ v0 .+ 0.5 .* ((A * v0 .+ b) .- v0)
end

@testset "fixed_point_solve!: Anderson converges, history depth m" begin
    rng = Random.MersenneTwister(7)
    n = 6
    Q = Matrix(qr(randn(rng, n, n)).Q)
    A = Q * Diagonal([0.9, 0.8, -0.7, 0.6, -0.5, 0.95]) * Q'
    b = randn(rng, n)
    vstar = (I - A) \ b
    its = map((1, 2)) do m
        v1 = zeros(n)
        _, res, iterations, _ = fixed_point_solve!(linear_map(A, b), v1, zeros(n);
            m, damping = 0.7, tol = 1e-12, abs_floor = 1e-12)
        @test isapprox(v1, vstar; rtol = 1e-10)
        @test res <= 1e-12
        iterations
    end
    # The depth reaches the solve: here m = 2 takes 96 iterations and m = 1 takes
    # 340, against 772 for Picard alone.
    @test its[2] < its[1]
end

@testset "fixed_point_solve!: stop test max(tol·‖v₀‖, abs_floor)" begin
    A = Diagonal([0.5, -0.3, 0.75, 0.2])
    b = [1.0, -2.0, 0.5, 3.0]
    v0 = fill(100.0, 4)                      # ‖v₀‖ = 200
    # Undamped Picard scales each error component by its eigenvalue, so ‖F‖ falls
    # by a factor between 0.2 and 0.75 per step and the solve stops with ‖F‖ in
    # (0.2, 1] times the target.
    for (tol, abs_floor) in ((1e-11, 1e-14), (1e-16, 1e-8))
        target = max(tol * 200, abs_floor)
        v1 = copy(v0)
        _, res, _, _ = fixed_point_solve!(linear_map(A, b), v1, zeros(4);
            damping = 1.0, use_anderson = false, tol, abs_floor, max_iter = 500)
        @test 0.2 * target < res <= target
    end
end

@testset "fixed_point_solve!: the final Picard step" begin
    # The map records its last argument and value. A Picard solve that succeeds
    # returns the last iterate it evaluated, v; the exit step then evaluates G(v).
    A = Diagonal([0.5, -0.3, 0.75, 0.2])
    b = [1.0, -2.0, 0.5, 3.0]
    for exit_picard_step in (true, false)
        last_in, last_out = zeros(4), zeros(4)
        mapf! = (out, v) -> (out .= A * v .+ b; last_in .= v; last_out .= out; out)
        v1 = zeros(4)
        evals, res, iterations, _ = fixed_point_solve!(mapf!, v1, zeros(4);
            damping = 0.7, use_anderson = false, tol = 0.0, abs_floor = 1e-6,
            exit_picard_step)
        @test res <= 1e-6
        if exit_picard_step
            @test evals == iterations + 2           # F(v₀), one per step, G(v)
            @test v1 == last_out
            @test norm(last_in - last_out) ≈ res    # last_in is v: ‖v − G(v)‖
        else
            @test evals == iterations + 1
            @test v1 == last_in
            @test norm(v1 - (A * v1 .+ b)) ≈ res
        end
    end
end

@testset "step_fixed_point! on the Landau map" begin
    p = SimParameters(N_PARTICLES = 200, DT = 0.01)
    USE_LOGSQ[] = p.use_logsq
    ws = build_workspace(p)
    rng = Random.MersenneTwister(3)
    N = p.N_PARTICLES
    v0 = [p.σ1 .* randn(rng, N) p.σ2 .* randn(rng, N)]
    w = fill(1 / N, N)
    f = zeros(ws.n_dofs)
    l2_project!(ws, f, v0, w)
    S0 = compute_entropy(ws, build_field(ws, f))
    bufs = (similar(v0), similar(v0), zeros(N, 2), zeros(N, 2), zeros(N, 2),
        zeros(ws.n_dofs), zeros(ws.n_dofs), zeros(ws.n_dofs), zeros(N, 2))
    function gmap(v, use_gonzalez)
        out = zeros(N, 2)
        picard_map!(ws, out, v, v0, w, S0, p.DT, bufs...; use_gonzalez)
        return out
    end
    # The fixed points with and without the Gonzalez term are about 3e-3 apart.
    for use_gonzalez in (true, false)
        v1 = copy(v0)
        _, res, _, res0 = step_fixed_point!(ws, v1, v0, w, S0, p.DT, bufs..., zeros(N, 2);
            m = 8, tol = 1e-12, abs_floor = 1e-11, damping = 0.7, use_gonzalez,
            exit_picard_step = false)
        @test res <= 1e-11
        @test res0 ≈ norm(v0 - gmap(v0, use_gonzalez))
        @test norm(v1 - gmap(v1, use_gonzalez)) ≈ res
        @test norm(v1 - gmap(v1, !use_gonzalez)) > 1e-4
    end
end
