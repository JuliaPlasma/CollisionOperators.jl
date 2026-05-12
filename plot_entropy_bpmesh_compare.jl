# Plot S(t) for 10k vs 40k on same bp mesh (reconstructed from fs_snapshot when
# conservation CSV missing). Sparse-point overlay since 40k only has 4 dumps.
using GLMakie, DelimitedFiles
GLMakie.activate!()

# columns: step,time,entropy
function load(path)
    raw, _ = readdlm(path, ',', Any, '\n'; header=true)
    Float64.(raw[:, 2]), Float64.(raw[:, 3])
end

t10, S10 = load("entropy_from_fs_bpmesh10k_anderson.csv")
t40, S40 = load("entropy_from_fs_bpmesh40k_anderson.csv")
# Dense 10k from conservation CSV for smooth reference curve
raw, _ = readdlm("conservation_history_bpmesh10k_anderson.csv", ',', Any, '\n'; header=true)
t10_dense = Float64.(raw[:, 2])
S10_dense = Float64.(raw[:, 3])

ΔS10 = S10 .- S10[1]
ΔS40 = S40 .- S40[1]
ΔS10_dense = S10_dense .- S10_dense[1]

fig = Figure(; size=(1100, 800))

ax1 = Axis(fig[1, 1]; xlabel="t", ylabel="S(t)",
    title="Entropy S(t) — same bp mesh (n_dofs=638)")
lines!(ax1, t10_dense, S10_dense; label="10k (dense)", color=:steelblue, linewidth=1.5)
scatter!(ax1, t10, S10; label="10k (fs_snapshot pts)", color=:steelblue, markersize=14)
scatter!(ax1, t40, S40; label="40k (fs_snapshot pts)", color=:crimson, markersize=14, marker=:diamond)
lines!(ax1, t40, S40; color=:crimson, linewidth=1.5, linestyle=:dash)
axislegend(ax1; position=:rb)

ax2 = Axis(fig[1, 2]; xlabel="t", ylabel="ΔS = S(t) − S(0)",
    title="Entropy production ΔS")
lines!(ax2, t10_dense, ΔS10_dense; label="10k (dense)", color=:steelblue, linewidth=1.5)
scatter!(ax2, t10, ΔS10; color=:steelblue, markersize=14)
scatter!(ax2, t40, ΔS40; label="40k (fs_snapshot pts)", color=:crimson, markersize=14, marker=:diamond)
lines!(ax2, t40, ΔS40; color=:crimson, linewidth=1.5, linestyle=:dash)
hlines!(ax2, [0.0]; color=:gray, linestyle=:dash)
axislegend(ax2; position=:rb)

# Difference S40 - S10 at common steps (0/100/200/300)
mask = [findfirst(==(t), t10) for t in t40]
diff_S = S40 .- S10[mask]
ax3 = Axis(fig[2, 1]; xlabel="t", ylabel="S₄₀ₖ − S₁₀ₖ",
    title="Bias 40k vs 10k at matched steps")
lines!(ax3, t40, diff_S; color=:black, linewidth=2)
scatter!(ax3, t40, diff_S; color=:black, markersize=12)
hlines!(ax3, [0.0]; color=:gray, linestyle=:dash)

# dS/dt finite diff between snapshot pts (Δt=0.1)
dSdt10 = diff(S10) ./ diff(t10)
dSdt40 = diff(S40) ./ diff(t40)
t10_mid = 0.5 .* (t10[1:end-1] .+ t10[2:end])
t40_mid = 0.5 .* (t40[1:end-1] .+ t40[2:end])
ax4 = Axis(fig[2, 2]; xlabel="t", ylabel="ΔS/Δt over Δt=0.1",
    title="Coarse entropy production rate")
scatterlines!(ax4, t10_mid, dSdt10; label="10k", color=:steelblue, markersize=10, linewidth=2)
scatterlines!(ax4, t40_mid, dSdt40; label="40k", color=:crimson, markersize=10, linewidth=2, marker=:diamond)
hlines!(ax4, [0.0]; color=:gray, linestyle=:dash)
axislegend(ax4; position=:rt)

println("=== bpmesh 10k (steps 0,100,200,300,400)")
for i in eachindex(t10); println("  t=$(t10[i])  S=$(S10[i])  ΔS=$(ΔS10[i])"); end
println("=== bpmesh 40k (steps 0,100,200,300) — RUN STALLED, no step 400")
for i in eachindex(t40); println("  t=$(t40[i])  S=$(S40[i])  ΔS=$(ΔS40[i])"); end
println("=== bias S40 − S10 at matched t")
for i in eachindex(t40); println("  t=$(t40[i])  ΔS_bias=$(diff_S[i])"); end

save("entropy_bpmesh_10k_vs_40k.png", fig)
println("Saved entropy_bpmesh_10k_vs_40k.png")
