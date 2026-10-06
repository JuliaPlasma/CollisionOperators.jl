# Unit tests for the dependency-free helpers in main.jl: the conservation-history
# CSV format (including migration of files written before the `dt` column) and the
# particle moment / drift-multiplier kernels that the LB operator is built on.
#
# main.jl is a script, not a module, so it is included once here and every testset
# below shares that load — it pulls CairoMakie and Mantis, which are slow.
using Test

include(joinpath(@__DIR__, "..", "..", "main.jl"))

# A conservation CSV in the pre-`dt` layout: `time` is `step * DT_then`, so a
# resume that changed DT rewrote the whole history's timeline.
function legacy_csv(path; steps_dt, with_r0 = true, repeat_steps = ())
    cols = ["step", "time", "entropy", "energy", "momentum_1", "momentum_2",
        "iter", "residual", "fp_minus_fs", "neg_part"]
    with_r0 && push!(cols, "r0")
    open(path, "w") do io
        println(io, join(cols, ','))
        for (step, dt) in steps_dt
            row = [string(step), string(step * dt), "1.0", "2.0", "0.1", "0.2",
                "7", "1e-12", "0.5", "0.25"]
            with_r0 && push!(row, "0.75")
            println(io, join(row, ','))
            step in repeat_steps && println(io, join(row, ','))
        end
    end
    return path
end

read_col(path, name) = begin
    lines = readlines(path)
    idx = findfirst(==(name), split(first(lines), ','))
    [parse(Float64, split(l, ',')[idx]) for l in Iterators.drop(lines, 1)]
end

@testset "CONS_COLS layout" begin
    @test CONS_COLS[1] == "step"
    @test CONS_COLS[2] == "time"
    @test CONS_COLS[end] == "dt"
    @test allunique(CONS_COLS)
    # Positional readers (plot_dashboard_LB.jl) index columns 1..10 by number.
    @test CONS_COLS[1:10] == ["step", "time", "entropy", "energy", "momentum_1",
        "momentum_2", "iter", "residual", "fp_minus_fs", "neg_part"]
end

@testset "migrate_cons_csv!" begin
    mktempdir() do dir
        # Constant DT: the accumulated time must match step * DT.
        f = legacy_csv(joinpath(dir, "const.csv");
            steps_dt = [(s, 0.001) for s in 0:10])
        migrate_cons_csv!(f)
        hdr = split(first(readlines(f)), ',')
        @test hdr == CONS_COLS
        @test all(l -> length(split(l, ',')) == length(CONS_COLS),
            Iterators.drop(readlines(f), 1))
        @test read_col(f, "time") ≈ [s * 0.001 for s in 0:10]
        @test read_col(f, "dt") ≈ [0.0; fill(0.001, 10)]

        # cumsum(dt) == time is the invariant the column exists to provide.
        @test cumsum(read_col(f, "dt")) ≈ read_col(f, "time")

        # Idempotent: a second call must not re-migrate an already-migrated file.
        before = read(f, String)
        migrate_cons_csv!(f)
        @test read(f, String) == before
    end
end

@testset "migrate_cons_csv! recovers a DT change" begin
    mktempdir() do dir
        # What a resume at a new DT used to produce: step 1000 at DT=0.001 reads
        # t=1.0, then step 1001 jumps to 1001*0.002 = 2.002 and step 1500 to 3.0,
        # although the run only reached t = 1.0 + 500*0.002 = 2.0.
        steps = [(s, 0.001) for s in 0:1000]
        append!(steps, [(s, 0.002) for s in 1001:1500])
        f = legacy_csv(joinpath(dir, "dtchange.csv"); steps_dt = steps)
        migrate_cons_csv!(f)
        t = read_col(f, "time")
        dt = read_col(f, "dt")
        @test t[findfirst(==(1000.0), read_col(f, "step"))] ≈ 1.0
        @test t[end] ≈ 2.0                      # not the 3.0 the old layout stored
        @test dt[end] ≈ 0.002
        @test cumsum(dt) ≈ t
    end
end

@testset "migrate_cons_csv! on a file without r0" begin
    mktempdir() do dir
        # The June bimodal runs predate the r0 column; migration must pad it so
        # appended rows do not make the file ragged.
        f = legacy_csv(joinpath(dir, "nor0.csv");
            steps_dt = [(s, 0.001) for s in 0:5], with_r0 = false)
        migrate_cons_csv!(f)
        @test split(first(readlines(f)), ',') == CONS_COLS
        @test all(l -> length(split(l, ',')) == length(CONS_COLS),
            Iterators.drop(readlines(f), 1))
        @test read_col(f, "r0") == zeros(6)
    end
end

@testset "migrate_cons_csv! with repeated steps" begin
    mktempdir() do dir
        # An older resume appended rather than truncating, so a step can appear
        # twice; the duplicate must not be counted twice in the accumulated time.
        f = legacy_csv(joinpath(dir, "dup.csv");
            steps_dt = [(s, 0.001) for s in 0:10], repeat_steps = (3, 4, 5))
        migrate_cons_csv!(f)
        step = read_col(f, "step")
        t = read_col(f, "time")
        @test length(step) == 14                      # 11 steps + 3 duplicates
        # Every row for a given step carries that step's time, and it is step*DT.
        for (s, ti) in zip(step, t)
            @test ti ≈ s * 0.001
        end
    end
end

@testset "migrate_cons_csv! edge cases" begin
    mktempdir() do dir
        @test migrate_cons_csv!(joinpath(dir, "absent.csv")) === nothing
        empty = joinpath(dir, "empty.csv"); write(empty, "")
        @test migrate_cons_csv!(empty) === nothing
        hdr_only = joinpath(dir, "hdr.csv")
        write(hdr_only, join(CONS_COLS[1:11], ',') * "\n")
        @test migrate_cons_csv!(hdr_only) === nothing
    end
end

@testset "cons_time_at" begin
    mktempdir() do dir
        f = legacy_csv(joinpath(dir, "t.csv");
            steps_dt = [(s, 0.004) for s in 0:20])
        migrate_cons_csv!(f)
        @test cons_time_at(f, 0) ≈ 0.0
        @test cons_time_at(f, 20) ≈ 0.08
        @test cons_time_at(f, 999) === nothing         # step not in the file
        @test cons_time_at(joinpath(dir, "absent.csv"), 1) === nothing
    end
end

@testset "truncate_csv_after" begin
    mktempdir() do dir
        f = legacy_csv(joinpath(dir, "trunc.csv");
            steps_dt = [(s, 0.001) for s in 0:20])
        truncate_csv_after(f, 12)
        @test read_col(f, "step") == collect(0.0:12.0)
        @test first(readlines(f)) == join(CONS_COLS[1:11], ',')   # header kept
        truncate_csv_after(f, 100)                                # no-op past the end
        @test read_col(f, "step") == collect(0.0:12.0)
    end
end

@testset "checkpoint_path" begin
    @test checkpoint_path("run", 0) == "checkpoint_run_step0000.jls"
    @test checkpoint_path("run", 25) == "checkpoint_run_step0025.jls"
    @test checkpoint_path("run", 20233) == "checkpoint_run_step20233.jls"
    # sort() on these names must order by step for load_checkpoint(:auto); that
    # holds only while the step count stays at or below the padded width.
    @test sort([checkpoint_path("r", s) for s in (0, 25, 1500)]) ==
          [checkpoint_path("r", s) for s in (0, 25, 1500)]
end

@testset "rclone_remote" begin
    withenv("RCLONE_REMOTE" => nothing) do
        # suffix underscores become dashes in the bucket path
        @test rclone_remote("sq_d04_gpu1k32") ==
              "mpcdf-s3://collision-operators/sq-d04-gpu1k32"
    end
    withenv("RCLONE_REMOTE" => "other:bucket/dir") do
        @test rclone_remote("anything") == "other:bucket/dir"
    end
    withenv("RCLONE_UPLOAD" => "0") do
        @test rclone_enabled() == false
    end
    withenv("RCLONE_UPLOAD" => nothing) do
        @test rclone_enabled() == true
    end
end

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
