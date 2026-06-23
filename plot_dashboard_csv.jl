# Standalone run-dashboard from a live conservation_history CSV (works mid-run).
# Replicates main_Gonzalez.jl plot_run_dashboard 7 panels.
#   julia --project=. plot_dashboard_csv.jl <suffix> [out_suffix]
# CSV cols: step,time,entropy,energy,momentum_1,momentum_2,iter,residual,fp_minus_fs,neg_part
using GLMakie, DelimitedFiles
GLMakie.activate!()

length(ARGS) >= 1 || error("usage: plot_dashboard_csv.jl <suffix> [out_suffix]")
suffix = ARGS[1]
out    = length(ARGS) >= 2 ? ARGS[2] : "$(suffix)_live"
csv = "conservation_history_$(suffix).csv"

raw, _ = readdlm(csv, ',', Any, '\n'; header=true)
step_all = Int.(raw[:, 1])
# dedup: keep LAST occurrence per step (resume rewrites overlap)
keep = Dict{Int,Int}()
for (i, s) in enumerate(step_all); keep[s] = i; end
idx = [keep[s] for s in sort(collect(keys(keep)))]
M = raw[idx, :]
step = Int.(M[:, 1])
S    = Float64.(M[:, 3])
E    = Float64.(M[:, 4])
m1   = Float64.(M[:, 5]); m2 = Float64.(M[:, 6])
iter = Float64.(M[:, 7])
res  = Float64.(M[:, 8])
fp   = Float64.(M[:, 9])
neg  = Float64.(M[:, 10])

E0 = E[1]; P10 = m1[1]; P20 = m2[1]
E_err = abs.(E .- E0) ./ abs(E0)
P_err = hypot.(m1 .- P10, m2 .- P20) ./ max(hypot(P10, P20), 1e-30)

fig = Figure(; size=(1200, 1500))
lines!(Axis(fig[1,1]; xlabel="step", ylabel="H_h",
    title="Entropy H_h (monotone increase expected)"), step, S; color=:red, linewidth=2)
lines!(Axis(fig[2,1]; xlabel="step", ylabel="rel. error",
    title="Energy conservation error", yscale=log10), step, max.(E_err,1e-18); color=:blue, linewidth=2)
lines!(Axis(fig[3,1]; xlabel="step", ylabel="rel. error",
    title="Momentum conservation error", yscale=log10), step, max.(P_err,1e-18); color=:green, linewidth=2)
lines!(Axis(fig[4,1]; xlabel="step", ylabel="iter",
    title="Inner-iteration count"), step, iter; color=:black, linewidth=2)
lines!(Axis(fig[5,1]; xlabel="step", ylabel="‖r‖",
    title="Picard fixed-point residual ‖G(v) − v‖₂", yscale=log10), step, max.(res,1e-30); color=:purple, linewidth=2)
lines!(Axis(fig[6,1]; xlabel="step", ylabel="‖f_s − f_p‖₂",
    title="Histogram-based projection error  ‖f_s − f_p‖₂"), step, fp; color=:orange, linewidth=2)
lines!(Axis(fig[7,1]; xlabel="step", ylabel="∫max(−f_s,0)",
    title="Negative-part L¹ of f_s  (Gibbs oscillation indicator)"), step, neg; color=:darkred, linewidth=2)

png = "dashboard_$(out).png"
save(png, fig)
println("Saved $png  (steps $(step[1])..$(step[end]), $(length(step)) pts)")
