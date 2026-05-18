# Side-by-side comparison of two Anderson runs with the SAME mesh but
# different convergence criteria. Designed to show whether the residual
# floor reported in run A was a real Picard-map noise floor or just an
# early-exit artifact of the tol setting.
#
# Run as:
#   julia --project=. compare_convergence.jl <tag_A> <tag_B> [label_A] [label_B]
#
# Reads conservation_history_<tag>.csv (also tries archive_*/) for each tag.
using GLMakie, DelimitedFiles, Statistics
GLMakie.activate!()

length(ARGS) >= 2 || error("usage: compare_convergence.jl <tag_A> <tag_B> [label_A] [label_B]")
tag_A = ARGS[1]
tag_B = ARGS[2]
lab_A = length(ARGS) >= 3 ? ARGS[3] : tag_A
lab_B = length(ARGS) >= 4 ? ARGS[4] : tag_B

function find_csv(tag)
    cands = ["conservation_history_$(tag).csv"]
    for d in readdir(".")
        isdir(d) && occursin("archive", d) || continue
        push!(cands, joinpath(d, "conservation_history_$(tag).csv"))
    end
    for c in cands
        isfile(c) && return c
    end
    error("no conservation CSV for tag $tag in . or archive_*")
end

function load_run(tag)
    path = find_csv(tag)
    raw, _ = readdlm(path, ',', Any, '\n'; header=true)
    (step  = Int.(raw[:, 1]),
     iter  = Int.(raw[:, 7]),
     nrm_r = Float64.(raw[:, 8]),
     fpL2  = Float64.(raw[:, 9]),
     neg   = Float64.(raw[:, 10]),
     path  = path)
end

A = load_run(tag_A)
B = load_run(tag_B)
println("A: $(A.path)  rows=$(length(A.step))")
println("B: $(B.path)  rows=$(length(B.step))")

# Detect non-converged steps per run
max_iter_A = maximum(A.iter)
max_iter_B = maximum(B.iter)
ncA = A.iter .== max_iter_A
ncB = B.iter .== max_iter_B
println("$lab_A: max_iter=$max_iter_A  non_conv=$(count(ncA))/$(length(A.step))")
println("$lab_B: max_iter=$max_iter_B  non_conv=$(count(ncB))/$(length(B.step))")

# ‖r‖ stats on converged steps
for (lab, R, nc) in ((lab_A, A, ncA), (lab_B, B, ncB))
    r = R.nrm_r[.!nc]
    isempty(r) && continue
    rr = sort(r)
    println("$lab  ‖r‖  min=$(rr[1])  median=$(rr[end÷2+1])  max=$(rr[end])")
end

# Overlap step range (B usually shorter)
xmax = min(maximum(A.step), maximum(B.step))
println("Common step range: 0..$xmax")

# ───────── plot ─────────
fig = Figure(; size=(1300, 1000))
Label(fig[0, 1:2],
    "Anderson criteria comparison — same mesh\n$lab_A  vs  $lab_B";
    fontsize=16, tellwidth=false)

colA = :steelblue; colB = :crimson

# (1) iter per step (truncated to overlap)
ax1 = Axis(fig[1, 1]; xlabel="step", ylabel="iter",
           title="Anderson iter count per step  (red = max_iter)")
scatter!(ax1, A.step, A.iter; color=ifelse.(ncA, :red, colA),
         markersize=4, label=lab_A)
scatter!(ax1, B.step, B.iter; color=ifelse.(ncB, :red, colB),
         markersize=4, label=lab_B, marker=:diamond)
axislegend(ax1; position=:lt)

# (2) ‖r‖ per step (log) — the headline panel
ax2 = Axis(fig[1, 2]; xlabel="step", ylabel="‖r‖",
           title="Final residual ‖r‖ per step (log)", yscale=log10)
scatter!(ax2, A.step, max.(A.nrm_r, 1e-16);
         color=ifelse.(ncA, :red, colA), markersize=4, label=lab_A)
scatter!(ax2, B.step, max.(B.nrm_r, 1e-16);
         color=ifelse.(ncB, :red, colB), markersize=4, label=lab_B,
         marker=:diamond)
axislegend(ax2; position=:lt)

# (3) negative-part L¹ — same physics, should overlap
ax3 = Axis(fig[2, 1]; xlabel="step", ylabel="∫max(-f_s,0)",
           title="Negative-part L¹  (Gibbs growth)")
lines!(ax3, A.step, A.neg; color=colA, linewidth=1.5, label=lab_A)
lines!(ax3, B.step, B.neg; color=colB, linewidth=1.5, label=lab_B,
       linestyle=:dash)
axislegend(ax3; position=:lt)

# (4) ‖f_p − f_s‖ — same physics
ax4 = Axis(fig[2, 2]; xlabel="step", ylabel="‖f_p − f_s‖",
           title="Projection error")
lines!(ax4, A.step, A.fpL2; color=colA, linewidth=1.5, label=lab_A)
lines!(ax4, B.step, B.fpL2; color=colB, linewidth=1.5, label=lab_B,
       linestyle=:dash)
axislegend(ax4; position=:rt)

# (5) iter histograms (overlap region only)
ax5 = Axis(fig[3, 1]; xlabel="iter", ylabel="count",
           title="Iter count distribution (overlap steps)")
maskA = A.step .<= xmax
maskB = B.step .<= xmax
hist!(ax5, A.iter[maskA]; bins=0:5:max(maximum(A.iter[maskA]), 100),
      color=(colA, 0.5), label=lab_A, strokewidth=0.5)
hist!(ax5, B.iter[maskB]; bins=0:5:max(maximum(B.iter[maskB]), 100),
      color=(colB, 0.5), label=lab_B, strokewidth=0.5)
axislegend(ax5; position=:rt)

# (6) ‖r‖ vs iter scatter — does more iter always mean lower ‖r‖?
ax6 = Axis(fig[3, 2]; xlabel="iter", ylabel="‖r‖",
           title="‖r‖ vs iter  (overlap steps)", yscale=log10)
scatter!(ax6, A.iter[maskA], max.(A.nrm_r[maskA], 1e-16);
         color=(colA, 0.6), markersize=4, label=lab_A)
scatter!(ax6, B.iter[maskB], max.(B.nrm_r[maskB], 1e-16);
         color=(colB, 0.6), markersize=4, label=lab_B, marker=:diamond)
axislegend(ax6; position=:lt)

out = "compare_convergence_$(tag_A)_vs_$(tag_B).png"
save(out, fig)
println("Saved $out")
