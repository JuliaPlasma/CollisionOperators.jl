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
        w = rand(rng, N); w ./= sum(w)
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
