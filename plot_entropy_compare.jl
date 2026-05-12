# Compare entropy increase between two conservation_history CSVs.
# Run as:
#   julia --project=. plot_entropy_compare.jl <csv1> <label1> <csv2> <label2> [out.png]
using GLMakie, DelimitedFiles
GLMakie.activate!()

length(ARGS) >= 4 || error("usage: plot_entropy_compare.jl csv1 label1 csv2 label2 [out]")
csv1, lab1, csv2, lab2 = ARGS[1], ARGS[2], ARGS[3], ARGS[4]
out = length(ARGS) >= 5 ? ARGS[5] : "entropy_compare.png"

function load(csv)
    raw, _ = readdlm(csv, ',', Any, '\n'; header=true)
    t = Float64.(raw[:, 2])
    S = Float64.(raw[:, 3])
    return t, S
end

t1, S1 = load(csv1)
t2, S2 = load(csv2)
ΔS1 = S1 .- S1[1]
ΔS2 = S2 .- S2[1]

fig = Figure(; size=(1100, 800))

ax1 = Axis(fig[1, 1];
    xlabel="t", ylabel="S(t)",
    title="Entropy S(t)")
lines!(ax1, t1, S1; label=lab1, linewidth=2, color=:steelblue)
lines!(ax1, t2, S2; label=lab2, linewidth=2, color=:crimson)
axislegend(ax1; position=:rb)

ax2 = Axis(fig[1, 2];
    xlabel="t", ylabel="ΔS = S(t) − S(0)",
    title="Entropy production ΔS")
lines!(ax2, t1, ΔS1; label=lab1, linewidth=2, color=:steelblue)
lines!(ax2, t2, ΔS2; label=lab2, linewidth=2, color=:crimson)
hlines!(ax2, [0.0]; color=:gray, linestyle=:dash)
axislegend(ax2; position=:rb)

# dS/dt via central diff (should be ≥ 0 for H-theorem)
dSdt1 = (S1[3:end] .- S1[1:end-2]) ./ (t1[3:end] .- t1[1:end-2])
dSdt2 = (S2[3:end] .- S2[1:end-2]) ./ (t2[3:end] .- t2[1:end-2])
ax3 = Axis(fig[2, 1];
    xlabel="t", ylabel="dS/dt",
    title="Entropy production rate dS/dt  (H-theorem ⇒ ≥ 0)")
lines!(ax3, t1[2:end-1], dSdt1; label=lab1, linewidth=1.5, color=:steelblue)
lines!(ax3, t2[2:end-1], dSdt2; label=lab2, linewidth=1.5, color=:crimson)
hlines!(ax3, [0.0]; color=:gray, linestyle=:dash)
axislegend(ax3; position=:rt)

# relative ΔS aligned at t=0
ax4 = Axis(fig[2, 2];
    xlabel="t", ylabel="ΔS / |ΔS_final|",
    title="Normalized ΔS (shape comparison)")
n1 = abs(ΔS1[end]) > 0 ? abs(ΔS1[end]) : 1.0
n2 = abs(ΔS2[end]) > 0 ? abs(ΔS2[end]) : 1.0
lines!(ax4, t1, ΔS1 ./ n1; label=lab1, linewidth=2, color=:steelblue)
lines!(ax4, t2, ΔS2 ./ n2; label=lab2, linewidth=2, color=:crimson)
axislegend(ax4; position=:rb)

# summary
println("=== $lab1  ($csv1)")
println("  S(0)=$(S1[1])  S(end)=$(S1[end])  ΔS_final=$(ΔS1[end])")
println("  min dS/dt = $(minimum(dSdt1))   max dS/dt = $(maximum(dSdt1))")
println("=== $lab2  ($csv2)")
println("  S(0)=$(S2[1])  S(end)=$(S2[end])  ΔS_final=$(ΔS2[end])")
println("  min dS/dt = $(minimum(dSdt2))   max dS/dt = $(maximum(dSdt2))")

save(out, fig)
println("Saved $out")
