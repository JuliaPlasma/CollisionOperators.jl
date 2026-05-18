# σ₁(t), σ₂(t) from particle snapshots — v4 vs v5 vs v6.
# Expected: anisotropic relaxation toward σ_eq = √((σ₁²+σ₂²)/2) ≈ 1.007.
using GLMakie, DelimitedFiles
using Statistics: std, mean
GLMakie.activate!()

function sigma_series(csv)
    raw, _ = readdlm(csv, ',', Any, '\n'; header=true)
    steps = sort(unique(Int.(raw[:, 1])))
    s1 = Float64[]; s2 = Float64[]; t = Float64[]
    for s in steps
        m = Int.(raw[:, 1]) .== s
        push!(s1, std(Float64.(raw[m, 4])))
        push!(s2, std(Float64.(raw[m, 5])))
        push!(t, Float64(raw[findfirst(m), 2]))
    end
    return steps, t, s1, s2
end

s_v4, t_v4, σ1_v4, σ2_v4 = sigma_series("particle_snapshots_bpmesh40k_v4.csv")
s_v5, t_v5, σ1_v5, σ2_v5 = sigma_series("particle_snapshots_bpmesh40k_v5.csv")
s_v6, t_v6, σ1_v6, σ2_v6 = sigma_series("particle_snapshots_bpmesh60k_v6.csv")

# Theory: T_eq = (σ₁₀² + σ₂₀²)/2, σ_eq = √T_eq
σ10 = 4/3; σ20 = 0.5
T_eq = (σ10^2 + σ20^2)/2
σ_eq = sqrt(T_eq)
println("Initial σ₁ = $σ10  σ₂ = $σ20")
println("Equilibrium σ_eq = √((σ₁²+σ₂²)/2) = $(round(σ_eq; digits=4))")

# Print step 0, 50, 100 for each run
function report(tag, s, σ1, σ2)
    for i in [1, findfirst(==(50), s), length(s)]
        i === nothing && continue
        T = (σ1[i]^2 + σ2[i]^2)/2
        anis = σ1[i]/σ2[i]
        println("$tag step=$(s[i])  σ₁=$(round(σ1[i]; digits=4))  σ₂=$(round(σ2[i]; digits=4))  T=$(round(T; digits=4))  σ₁/σ₂=$(round(anis; digits=4))")
    end
end
report("v4", s_v4, σ1_v4, σ2_v4)
report("v5", s_v5, σ1_v5, σ2_v5)
report("v6", s_v6, σ1_v6, σ2_v6)

fig = Figure(; size=(1500, 900))

# σ₁
ax1 = Axis(fig[1, 1]; xlabel="step", ylabel="σ₁",
    title="σ₁(t) — should decay toward σ_eq=$(round(σ_eq; digits=3))")
lines!(ax1, s_v4, σ1_v4; color=:steelblue,  linewidth=2, label="v4 (40k, Δv₂=0.2)")
lines!(ax1, s_v5, σ1_v5; color=:crimson,    linewidth=2, label="v5 (40k, Δv₂=0.125)")
lines!(ax1, s_v6, σ1_v6; color=:darkgreen,  linewidth=2, label="v6 (60k, Δv₂=0.125)")
hlines!(ax1, [σ_eq]; color=:black, linestyle=:dash, label="σ_eq")
hlines!(ax1, [σ10]; color=:gray, linestyle=:dot, label="σ₁₀=4/3")
axislegend(ax1; position=:rt)

# σ₂
ax2 = Axis(fig[1, 2]; xlabel="step", ylabel="σ₂",
    title="σ₂(t) — should grow toward σ_eq")
lines!(ax2, s_v4, σ2_v4; color=:steelblue,  linewidth=2, label="v4")
lines!(ax2, s_v5, σ2_v5; color=:crimson,    linewidth=2, label="v5")
lines!(ax2, s_v6, σ2_v6; color=:darkgreen,  linewidth=2, label="v6")
hlines!(ax2, [σ_eq]; color=:black, linestyle=:dash, label="σ_eq")
hlines!(ax2, [σ20]; color=:gray, linestyle=:dot, label="σ₂₀=0.5")
axislegend(ax2; position=:rb)

# Anisotropy ratio σ₁/σ₂
ax3 = Axis(fig[2, 1]; xlabel="step", ylabel="σ₁/σ₂",
    title="Anisotropy ratio (should decay 8/3 → 1)")
lines!(ax3, s_v4, σ1_v4 ./ σ2_v4; color=:steelblue,  linewidth=2, label="v4")
lines!(ax3, s_v5, σ1_v5 ./ σ2_v5; color=:crimson,    linewidth=2, label="v5")
lines!(ax3, s_v6, σ1_v6 ./ σ2_v6; color=:darkgreen,  linewidth=2, label="v6")
hlines!(ax3, [1.0]; color=:black, linestyle=:dash, label="isotropic")
hlines!(ax3, [σ10/σ20]; color=:gray, linestyle=:dot, label="initial 8/3")
axislegend(ax3; position=:rt)

# Total kinetic energy proxy: T(t) = (σ₁² + σ₂²)/2 — should conserve
ax4 = Axis(fig[2, 2]; xlabel="step", ylabel="T = (σ₁²+σ₂²)/2",
    title="Energy proxy — should be flat")
lines!(ax4, s_v4, (σ1_v4.^2 .+ σ2_v4.^2)/2; color=:steelblue,  linewidth=2, label="v4")
lines!(ax4, s_v5, (σ1_v5.^2 .+ σ2_v5.^2)/2; color=:crimson,    linewidth=2, label="v5")
lines!(ax4, s_v6, (σ1_v6.^2 .+ σ2_v6.^2)/2; color=:darkgreen,  linewidth=2, label="v6")
hlines!(ax4, [T_eq]; color=:black, linestyle=:dash, label="T_eq")
axislegend(ax4; position=:rb)

Label(fig[0, 1:2], "Sigma evolution — v4 vs v5 vs v6 (100 steps, dt=0.001 → t∈[0, 0.1])";
    fontsize=18, tellwidth=false)

save("sigma_evolution_v4v5v6.png", fig)
println("Saved sigma_evolution_v4v5v6.png")
