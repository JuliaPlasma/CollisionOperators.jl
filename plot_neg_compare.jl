# Overlay neg-part L¹(step) for two conservation CSVs (same IC/mesh, differ only
# in integrator). Usage: julia --project=. plot_neg_compare.jl <out.png> lbl1=csv1 lbl2=csv2 ...
using GLMakie, DelimitedFiles
GLMakie.activate!()

out = ARGS[1]
series = ARGS[2:end]

fig = Figure(; size=(1000, 640))
ax = Axis(fig[1,1]; xlabel="step", ylabel="∫max(−f_s,0)",
          title="Negative-part L¹ — integrator comparison (same IC/mesh/seed)")
cols = [:crimson, :steelblue, :darkgreen, :darkorange]
for (k, s) in enumerate(series)
    lbl, csv = split(s, "="; limit=2)
    raw, _ = readdlm(csv, ',', Any, '\n'; header=true)
    step = Int.(raw[:,1]); neg = Float64.(raw[:,10])
    # dedup keep-last per step
    d = Dict{Int,Float64}(); for i in eachindex(step); d[step[i]]=neg[i]; end
    ks = sort(collect(keys(d)))
    lines!(ax, ks, [d[s] for s in ks]; color=cols[k], linewidth=2.2, label=lbl)
end
axislegend(ax; position=:lt)
save(out, fig); println("Saved $out")
