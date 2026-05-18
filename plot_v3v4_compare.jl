# Compare v3 (bp1 inner 17) vs v4 (bp1 inner 25) at 100-step horizon.
# Two-panel: (a) per-step residual + iter, (b) neg_part growth.
# Run: julia --project=. plot_v3v4_compare.jl
using GLMakie, DelimitedFiles
GLMakie.activate!()

function load_csv(tag)
    raw, _ = readdlm("conservation_history_$(tag).csv", ',', Any, '\n'; header=true)
    n = size(raw, 1)
    step   = Int.(raw[:, 1])
    iter   = Int.(raw[:, 7])
    res    = Float64.(raw[:, 8])
    fpfs   = Float64.(raw[:, 9])
    neg    = Float64.(raw[:, 10])
    return step, iter, res, fpfs, neg
end

s3, it3, r3, fp3, ng3 = load_csv("bpmesh40k_v3")
s4, it4, r4, fp4, ng4 = load_csv("bpmesh40k_v4")

# Truncate v3 to first 101 rows (steps 0..100)
n4 = length(s4)
keep3 = 1:n4

fig = Figure(; size=(1400, 900))

ax_r = Axis(fig[1, 1]; xlabel="step", ylabel="‖r‖",
    title="Final residual per step (log)", yscale=log10)
scatter!(ax_r, s3[keep3], max.(r3[keep3], 1e-15);
    color=:steelblue, markersize=5, label="v3 (bp1=17, DOF=638)")
scatter!(ax_r, s4, max.(r4, 1e-15);
    color=:crimson, markersize=5, label="v4 (bp1=25, DOF=870)")
axislegend(ax_r; position=:rb)

ax_i = Axis(fig[1, 2]; xlabel="step", ylabel="Anderson iter",
    title="Anderson iter count per step")
scatter!(ax_i, s3[keep3], it3[keep3];
    color=:steelblue, markersize=5, label="v3")
scatter!(ax_i, s4, it4;
    color=:crimson, markersize=5, label="v4")
axislegend(ax_i; position=:lt)

ax_n = Axis(fig[2, 1]; xlabel="step", ylabel="∫max(−f_s, 0)",
    title="Negative-part L¹ — Gibbs spike growth")
lines!(ax_n, s3[keep3], ng3[keep3]; color=:steelblue, linewidth=2, label="v3")
lines!(ax_n, s4, ng4; color=:crimson, linewidth=2, label="v4")
axislegend(ax_n; position=:lt)

ax_f = Axis(fig[2, 2]; xlabel="step", ylabel="‖f_p − f_s‖₂",
    title="Projection error")
lines!(ax_f, s3[keep3], fp3[keep3]; color=:steelblue, linewidth=2, label="v3")
lines!(ax_f, s4, fp4; color=:crimson, linewidth=2, label="v4")
axislegend(ax_f; position=:rb)

Label(fig[0, 1:2],
    "v3 vs v4 — bp1 inner refinement (17 → 25 points), first 100 steps";
    fontsize=18, tellwidth=false)

save("v3v4_compare.png", fig)
println("Saved v3v4_compare.png")
