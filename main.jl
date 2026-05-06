#! /usr/bin/env -S julia --color=yes --startup-file=no
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

include("MantisWrappers.jl")
using .MantisWrappers
using GLMakie
using Random
using LinearAlgebra

include("parameters.jl");
println("V ∈ [$V_MIN, $V_MAX],  p=$P_DEG, k=$K_REG,  $N_ELEM×$N_ELEM elements")
println("N_particles=$N_PARTICLES,  σ₁=$σ₁, σ₂=$σ₂,  Δt=$DT,  N_steps=$N_STEPS")


function compute_momentum(v_parts, w_parts)
    p1 = sum(w_parts[α] * v_parts[α, 1] for α in axes(v_parts, 1))
    p2 = sum(w_parts[α] * v_parts[α, 2] for α in axes(v_parts, 1))
    return (p1, p2)
end

function compute_energy(v_parts, w_parts)
    return 0.5 * sum(w_parts[α] * (v_parts[α, 1]^2 + v_parts[α, 2]^2)
                     for α in axes(v_parts, 1))
end


function main()

    # Initialize particles from a Maxwellian distribution
    Random.seed!(42)
    v_particles = zeros(N_PARTICLES, 2)
    v_particles[:, 1] .= σ₁ * randn(N_PARTICLES)
    v_particles[:, 2] .= σ₂ * randn(N_PARTICLES)
    w_particles = fill(1.0 / N_PARTICLES, N_PARTICLES)
    f_coeffs = zeros(n_dofs)

    l2_project!(f_coeffs, v_particles, w_particles)
    f_s = build_field(f_coeffs)        # f_s(v) = Σ_i f_i φ_i(v)

    entropy_history  = Float64[]
    energy_history   = Float64[]
    momentum_history = NTuple{2, Float64}[]

    push!(entropy_history,  compute_entropy(f_s))
    push!(energy_history,   compute_energy(v_particles, w_particles))
    push!(momentum_history, compute_momentum(v_particles, w_particles))
    println("Initial  S_h = $(entropy_history[end])")
    println("Initial  E   = $(energy_history[end])")
    println("Initial  P   = $(momentum_history[end])")
    r_vec = zeros(n_dofs)
    L_vec = zeros(n_dofs)
    G = zeros(N_PARTICLES, 2)
    dot_v = zeros(N_PARTICLES, 2)

    snapshot_steps = [0, N_STEPS ÷ 4, N_STEPS ÷ 2, N_STEPS]
    snapshots = Dict{Int,Matrix{Float64}}()
    snapshots[0] = copy(v_particles)



    # Time-stepping loop
    for step in 1:N_STEPS
        compute_r!(r_vec, f_s)
        L_vec .= M_lu \ r_vec
        compute_G!(G, v_particles, L_vec)
        compute_collision!(dot_v, v_particles, w_particles, G)
        v_particles .+= DT .* dot_v

        l2_project!(f_coeffs, v_particles, w_particles)
        f_s = build_field(f_coeffs)        # f_s(v) = Σ_i f_i φ_i(v)

        push!(entropy_history,  compute_entropy(f_s))
        push!(energy_history,   compute_energy(v_particles, w_particles))
        push!(momentum_history, compute_momentum(v_particles, w_particles))

        if step in snapshot_steps
            snapshots[step] = copy(v_particles)
        end

        step % 25 == 0 &&
            println("Step $step/$N_STEPS  S = $(round(entropy_history[end]; digits=6))" *
                    "  E = $(round(energy_history[end]; digits=8))" *
                    "  P = ($(round(momentum_history[end][1]; digits=8))," *
                    " $(round(momentum_history[end][2]; digits=8)))")
    end

    # ------------------------------------------------------------------
    # Save conservation histories to CSV
    # ------------------------------------------------------------------
    open("conservation_history_FE.csv", "w") do io
        println(io, "step,time,entropy,energy,momentum_1,momentum_2")
        for n in 0:N_STEPS
            t  = n * DT
            S  = entropy_history[n+1]
            E  = energy_history[n+1]
            P1 = momentum_history[n+1][1]
            P2 = momentum_history[n+1][2]
            println(io, "$n,$t,$S,$E,$P1,$P2")
        end
    end
    println("Saved conservation_history_FE.csv")

    # ------------------------------------------------------------------
    # Save particle snapshots (v_1, v_2 per particle) at each quarter
    # ------------------------------------------------------------------
    open("particle_snapshots_FE.csv", "w") do io
        println(io, "step,time,particle_idx,v1,v2")
        for s in sort(collect(keys(snapshots)))
            pts = snapshots[s]
            t   = s * DT
            for i in axes(pts, 1)
                println(io, "$s,$t,$i,$(pts[i, 1]),$(pts[i, 2])")
            end
        end
    end
    println("Saved particle_snapshots_FE.csv")

    begin # Visualization
        E0    = energy_history[1]
        P0    = momentum_history[1]
        steps = 0:N_STEPS

        E_err = [abs(energy_history[n+1] - E0) / abs(E0) for n in steps]
        P_err = [hypot(momentum_history[n+1][1] - P0[1],
                       momentum_history[n+1][2] - P0[2]) / hypot(P0[1], P0[2])
                 for n in steps]

        snap_keys = sort(collect(keys(snapshots)))
        n_snap_rows = cld(length(snap_keys), 2)

        fig = Figure(; size=(1200, 200 * (n_snap_rows + 3)))

        # Sample standard deviations σ_d = sqrt(⟨v_d²⟩ − ⟨v_d⟩²) per snapshot;
        # annotate the initial and final scatters with σ₁, σ₂ to visualize
        # anisotropy collapse toward the isotropic Maxwellian.
        sample_std(v) = (μ = sum(v)/length(v); sqrt(sum((v .- μ).^2)/length(v)))
        s_init  = snap_keys[1]
        s_final = snap_keys[end]
        for (idx, s) in enumerate(snap_keys)
            row, col = fldmod1(idx, 2)
            t_str = round(s * DT; digits=4)
            pts = snapshots[s]
            title_str = if s == s_init || s == s_final
                σ1 = sample_std(@view pts[:, 1])
                σ2 = sample_std(@view pts[:, 2])
                "t = $t_str   σ₁=$(round(σ1; digits=4))   σ₂=$(round(σ2; digits=4))"
            else
                "t = $t_str"
            end
            ax = Axis(fig[row, col];
                title=title_str,
                xlabel="v₁", ylabel="v₂", aspect=DataAspect())
            scatter!(ax, pts[:, 1], pts[:, 2]; markersize=2, color=:blue, alpha=0.3)
            xlims!(ax, V_MIN, V_MAX)
            ylims!(ax, V_MIN, V_MAX)
        end

        # H-function evolution (sign convention: H_h is the entropy functional
        # we expect to grow monotonically; the standard Boltzmann H = ∫ f log f
        # has the opposite sign and would decrease).
        H_history = entropy_history
        ax_H = Axis(fig[n_snap_rows+1, 1:2];
            xlabel="time step", ylabel="H_h",
            title="Boltzmann H-function (monotone increase expected)")
        lines!(ax_H, steps, H_history; color=:red, linewidth=2)

        # Energy conservation error (log scale)
        ax_E = Axis(fig[n_snap_rows+2, 1:2];
            xlabel="time step", ylabel="relative error",
            title="Energy conservation error  |E_n − E_0| / E_0",
            yscale=log10)
        lines!(ax_E, steps, max.(E_err, 1e-18); color=:blue, linewidth=2)

        # Momentum conservation error (log scale)
        ax_P = Axis(fig[n_snap_rows+3, 1:2];
            xlabel="time step", ylabel="relative error",
            title="Momentum conservation error  ‖P_n − P_0‖ / ‖P_0‖",
            yscale=log10)
        lines!(ax_P, steps, max.(P_err, 1e-18); color=:green, linewidth=2)

        save("landau_collision_2d.png", fig)
        println("Saved landau_collision_2d.png")
    end
end

main()
