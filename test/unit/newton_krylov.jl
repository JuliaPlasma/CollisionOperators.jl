# Unit tests for the generic GMRES and Jacobian-free Newton–Krylov solver.
# newton_krylov.jl depends on LinearAlgebra alone, so these run on every core
# pass; the wiring to the Picard map (`step_newton!`) is exercised by the runs.
using Test
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "..", "..", "newton_krylov.jl"))

@testset "gmres! matches a direct solve" begin
    rng = MersenneTwister(7)
    n = 60
    # Nonsymmetric, well conditioned: identity plus a small perturbation, the
    # same shape as the implicit-step Jacobian I - O(Δt).
    A = I + 0.3 * randn(rng, n, n) / sqrt(n)
    b = randn(rng, n)
    x = zeros(n)
    gw = GMRESWorkspace(n, n)
    k, rnorm = gmres!(x, (y, u) -> mul!(y, A, u), b, gw; rtol = 1e-12)
    @test norm(A * x - b) ≤ 1e-10 * norm(b)
    @test rnorm ≈ norm(A * x - b) atol = 1e-10 * norm(b)
    @test k ≤ n

    # A loose tolerance stops early, and the reported residual is honest.
    k2, rnorm2 = gmres!(x, (y, u) -> mul!(y, A, u), b, gw; rtol = 1e-2)
    @test k2 < k
    @test rnorm2 ≤ 1e-2 * norm(b)
    @test norm(A * x - b) ≈ rnorm2 rtol = 1e-6

    # The iteration cap is respected.
    k3, _ = gmres!(x, (y, u) -> mul!(y, A, u), b, gw; rtol = 1e-14, itmax = 3)
    @test k3 == 3

    # Zero right-hand side: no products, zero solution.
    k0, r0 = gmres!(x, (y, u) -> mul!(y, A, u), zeros(n), gw; rtol = 1e-8)
    @test (k0, r0) == (0, 0.0)
    @test iszero(x)
end

@testset "newton_krylov! on an implicit-midpoint-like fixed point" begin
    # F(x) = x - x0 - dt * f((x0 + x)/2) with a nonlinear, non-gradient f.
    rng = MersenneTwister(11)
    n = 40
    B = randn(rng, n, n) / sqrt(n)
    x0 = randn(rng, n)
    dt = 0.2
    f(y) = tanh.(B * y) .- 0.5 .* y .^ 3
    F!(out, x) = (out .= x .- x0 .- dt .* f(0.5 .* (x0 .+ x)); out)

    nk = NKWorkspace(n, 20)
    x = x0 .+ dt .* f(x0)                 # explicit Euler predictor
    n_evals, rnorm, n_newton, rnorm0, status = newton_krylov!(x, F!, nk;
        tol = 1e-11, max_evals = 200, fd_rel = sqrt(eps()))
    @test status === :converged
    @test rnorm < 1e-11
    @test norm(F!(similar(x), x)) ≈ rnorm atol = 1e-12
    @test nk.F ≈ F!(similar(x), x)
    @test rnorm0 > rnorm
    # Superlinear: a handful of Newton steps, far fewer evaluations than the
    # budget.
    @test n_newton ≤ 6
    @test n_evals < 60

    # Budget exhaustion returns the best iterate and says so.
    x = x0 .+ dt .* f(x0)
    n_evals, rnorm_b, _, _, status = newton_krylov!(x, F!, nk;
        tol = 1e-14, max_evals = 4, fd_rel = sqrt(eps()))
    @test status === :budget
    @test n_evals ≤ 4
    @test norm(F!(similar(x), x)) ≈ rnorm_b
end

@testset "newton_krylov! stalls cleanly at a noise floor" begin
    # Additive noise of size 1e-6 makes ‖F‖ < 1e-10 unreachable; the line search
    # must fail and stop instead of burning the whole budget.
    rng = MersenneTwister(3)
    n = 30
    x0 = randn(rng, n)
    F!(out, x) = (out .= x .- x0 .- 0.1 .* sin.(x) .+ 1e-6 .* randn(rng, n); out)
    nk = NKWorkspace(n, 20)
    x = copy(x0)
    n_evals, rnorm, _, _, status = newton_krylov!(x, F!, nk;
        tol = 1e-10, max_evals = 10_000, fd_rel = 1e-3)
    @test status === :stalled
    @test n_evals < 200
    @test rnorm < 1e-4
end
