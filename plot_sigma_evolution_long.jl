# σ₁(t), σ₂(t), σ₁/σ₂, T(t) across short (100-step) + long (800-step) runs.
# Compare anisotropy relaxation horizon: 100 step (t=0.1) too short to see decay,
# 800-step v3_800/v6_800 should reveal exponential decay toward σ_eq=1.007.
using GLMakie, DelimitedFiles
using Statistics: std
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

runs = [
    ("v4 100step 40k Δv₂=0.2",   "particle_snapshots_bpmesh40k_v4.csv",      :steelblue),
    ("v5 100step 40k Δv₂=0.125", "particle_snapshots_bpmesh40k_v5.csv",      :crimson),
    ("v6 100step 60k Δv₂=0.125", "particle_snapshots_bpmesh60k_v6.csv",      :darkgreen),
    ("v3_800 325step 40k sparse","particle_snapshots_bpmesh40k_v3_800.csv",  :darkorange),
    ("v6_800 275step 60k dense", "particle_snapshots_bpmesh60k_v6_800.csv",  :purple),
    ("v3_unif025 400step uniform Δv₂=0.25", "particle_snapshots_bpmesh40k_v3_unif025.csv", :black),
]

data = Dict()
for (lbl, csv, _) in runs
    isfile(csv) || (println("skip $csv"); continue)
    data[lbl] = sigma_series(csv)
end

σ10 = 4/3; σ20 = 0.5
T_eq = (σ10^2 + σ20^2)/2
σ_eq = sqrt(T_eq)
println("σ_eq = $(round(σ_eq; digits=4))  T_eq = $(round(T_eq; digits=4))")

# Endpoint summary
for (lbl, _, _) in runs
    haskey(data, lbl) || continue
    s, t, σ1, σ2 = data[lbl]
    i = length(s)
    T = (σ1[i]^2 + σ2[i]^2)/2
    println("$lbl  step=$(s[i]) t=$(round(t[i]; digits=3))  σ₁=$(round(σ1[i]; digits=4))  σ₂=$(round(σ2[i]; digits=4))  σ₁/σ₂=$(round(σ1[i]/σ2[i]; digits=4))  T=$(round(T; digits=4))")
end

fig = Figure(; size=(1600, 1000))

ax1 = Axis(fig[1, 1]; xlabel="t", ylabel="σ₁",
    title="σ₁(t) — decay toward σ_eq")
for (lbl, _, col) in runs
    haskey(data, lbl) || continue
    _, t, σ1, _ = data[lbl]
    lines!(ax1, t, σ1; color=col, linewidth=2, label=lbl)
end
hlines!(ax1, [σ_eq]; color=:black, linestyle=:dash, label="σ_eq=$(round(σ_eq;digits=3))")
hlines!(ax1, [σ10]; color=:gray, linestyle=:dot, label="σ₁₀=4/3")
axislegend(ax1; position=:rt, labelsize=10)

ax2 = Axis(fig[1, 2]; xlabel="t", ylabel="σ₂",
    title="σ₂(t) — grow toward σ_eq")
for (lbl, _, col) in runs
    haskey(data, lbl) || continue
    _, t, _, σ2 = data[lbl]
    lines!(ax2, t, σ2; color=col, linewidth=2, label=lbl)
end
hlines!(ax2, [σ_eq]; color=:black, linestyle=:dash)
hlines!(ax2, [σ20]; color=:gray, linestyle=:dot, label="σ₂₀=0.5")
axislegend(ax2; position=:rb, labelsize=10)

ax3 = Axis(fig[2, 1]; xlabel="t", ylabel="σ₁/σ₂",
    title="Anisotropy ratio (8/3 → 1)")
for (lbl, _, col) in runs
    haskey(data, lbl) || continue
    _, t, σ1, σ2 = data[lbl]
    lines!(ax3, t, σ1 ./ σ2; color=col, linewidth=2, label=lbl)
end
hlines!(ax3, [1.0]; color=:black, linestyle=:dash, label="isotropic")
hlines!(ax3, [σ10/σ20]; color=:gray, linestyle=:dot, label="initial 8/3")
axislegend(ax3; position=:rt, labelsize=10)

ax4 = Axis(fig[2, 2]; xlabel="t", ylabel="T=(σ₁²+σ₂²)/2",
    title="Energy proxy — flat conservation check")
for (lbl, _, col) in runs
    haskey(data, lbl) || continue
    _, t, σ1, σ2 = data[lbl]
    lines!(ax4, t, (σ1.^2 .+ σ2.^2) ./ 2; color=col, linewidth=2, label=lbl)
end
hlines!(ax4, [T_eq]; color=:black, linestyle=:dash, label="T_eq")
axislegend(ax4; position=:rb, labelsize=10)

Label(fig[0, 1:2],
    "σ evolution — short (100 step) + long (800 step) runs   σ_eq=$(round(σ_eq;digits=3))";
    fontsize=18, tellwidth=false)

save("sigma_evolution_long.png", fig)
println("Saved sigma_evolution_long.png")
