# Compare stripe at step=100, seed=42: P=2 (v3_800) vs P=3 (v3_P3).
# Test prediction: deterministic projection residual A ∝ h^(P+1).
# P=2 → h^3 = 0.008 baseline. P=3 → h^4 = 0.0016 → 5× drop if det-dominated.
using GLMakie, DelimitedFiles
using Statistics: mean
GLMakie.activate!()
include("Parameters.jl")
include("parameters_bpmesh40k_v3_800.jl"); p = PARAMS

Δv2 = p.bp2[4] - p.bp2[3]
println("Δv₂ = $(round(Δv2; digits=4))")

function load_slab(csv, step)
    raw, _ = readdlm(csv, ',', Any, '\n'; header=true)
    mask = Int.(raw[:, 1]) .== step
    v1s = Float64.(raw[mask, 4]); v2s = Float64.(raw[mask, 5])
    band = (v1s .>= 3.0) .& (v1s .<= 4.5)
    return v1s[band], v2s[band]
end

function analyze(v2b; bin_div=4)
    edges = collect(-2.5:Δv2/bin_div:2.5)
    counts = zeros(Int, length(edges)-1)
    for v in v2b
        idx = searchsortedlast(edges, v)
        1 <= idx <= length(counts) && (counts[idx] += 1)
    end
    w = 30
    smooth = [mean(counts[max(1,i-w):min(length(counts),i+w)]) for i in eachindex(counts)]
    resid = counts .- smooth
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
    maxlag = length(resid) ÷ 4
    acf = zeros(maxlag)
    for lag in 0:maxlag-1
        n = length(resid) - lag
        acf[lag+1] = sum(resid[1:n] .* resid[1+lag:end]) / n
    end
    dx = edges[2] - edges[1]
    lags = (0:maxlag-1) .* dx
    # Phase fit: r ≈ A cos(k v) + B sin(k v)
    k = 2π / Δv2
    A = sum(resid .* cos.(k .* centers)) * 2 / length(centers)
    B = sum(resid .* sin.(k .* centers)) * 2 / length(centers)
    amp = hypot(A, B)
    φ = atan(B, A)
    v₀ = φ / k
    return centers, counts, resid, lags, acf ./ max(acf[1], 1e-30), amp, v₀
end

v1_p2, v2_p2 = load_slab("particle_snapshots_bpmesh40k_v3_800.csv",  100)
v1_p3, v2_p3 = load_slab("particle_snapshots_bpmesh40k_v3_P3.csv",   100)
println("P=2  N_band=$(length(v1_p2))")
println("P=3  N_band=$(length(v1_p3))")

c2, h2, r2, lag2, acf2, A2, v0_2 = analyze(v2_p2)
c3, h3, r3, lag3, acf3, A3, v0_3 = analyze(v2_p3)

println("P=2  amp(cos fit)=$(round(A2; digits=3))   v₀=$(round(v0_2; digits=4))")
println("P=3  amp(cos fit)=$(round(A3; digits=3))   v₀=$(round(v0_3; digits=4))")
println("amplitude ratio P=3 / P=2 = $(round(A3/A2; digits=3))")
println("Naive det prediction  h^(P+1):  P=3 / P=2 = (h)^(4-3) = $(round(Δv2; digits=3))")

function acf_peak(lags, acf)
    win = findall(l -> 0.5*Δv2 <= l <= 1.5*Δv2, lags)
    i = win[argmax(acf[win])]
    return lags[i], acf[i]
end
pl2, ph_h2 = acf_peak(lag2, acf2)
pl3, ph_h3 = acf_peak(lag3, acf3)
println("P=2 ACF peak lag=$(round(pl2;digits=3))  h=$(round(ph_h2;digits=3))")
println("P=3 ACF peak lag=$(round(pl3;digits=3))  h=$(round(ph_h3;digits=3))")

fig = Figure(; size=(1600, 1200))

# Row 1: scatter slab
for (k, (lbl, v1b, v2b)) in enumerate([("P=2  v3_800", v1_p2, v2_p2), ("P=3  v3_P3", v1_p3, v2_p3)])
    ax = Axis(fig[1, k]; xlabel="v₁", ylabel = k==1 ? "v₂" : "",
        title="$lbl  step=100  N_band=$(length(v1b))", aspect=DataAspect())
    scatter!(ax, v1b, v2b; markersize=4.0, color=(:steelblue, 0.6))
    hlines!(ax, p.bp2; color=:red, linewidth=0.4, alpha=0.5)
    xlims!(ax, 2.5, 5.0); ylims!(ax, -2.5, 2.5)
end

# Row 2: detrended residual overlay
ax_h = Axis(fig[2, 1:2]; xlabel="v₂", ylabel="detrended count",
    title="Stripe residual: P=2 (blue, amp=$(round(A2;digits=2)))  vs  P=3 (red, amp=$(round(A3;digits=2)))   ratio=$(round(A3/A2;digits=2))")
lines!(ax_h, c2, r2; color=:steelblue, linewidth=2, label="P=2")
lines!(ax_h, c3, r3; color=:crimson,   linewidth=2, label="P=3")
vlines!(ax_h, p.bp2; color=(:gray, 0.4), linewidth=0.4)
hlines!(ax_h, [0.0]; color=:black, linewidth=0.3)
axislegend(ax_h; position=:rt)
xlims!(ax_h, -2.0, 2.0)

# Row 3: ACF overlay
ax_a = Axis(fig[3, 1:2]; xlabel="lag (v₂)", ylabel="ACF (norm)",
    title="ACF: pitch tracks Δv₂=$(round(Δv2;digits=3)) regardless of P_DEG")
lines!(ax_a, lag2, acf2; color=:steelblue, linewidth=2, label="P=2  lag=$(round(pl2;digits=3))  h=$(round(ph_h2;digits=3))")
lines!(ax_a, lag3, acf3; color=:crimson,   linewidth=2, label="P=3  lag=$(round(pl3;digits=3))  h=$(round(ph_h3;digits=3))")
vlines!(ax_a, [Δv2]; color=:black, linestyle=:dash, label="Δv₂")
for m in 2:5
    vlines!(ax_a, [m*Δv2]; color=(:black, 0.3), linewidth=0.4)
end
hlines!(ax_a, [0.0]; color=:black, linewidth=0.3)
axislegend(ax_a; position=:rt)
xlims!(ax_a, 0, lag2[end])

Label(fig[0, 1:2],
    "P_DEG comparison stripe analysis (v3 mesh, step=100, seed=42, slab 3.0≤v₁≤4.5)";
    fontsize=18, tellwidth=false)

save("stripe_P2_vs_P3.png", fig)
println("Saved stripe_P2_vs_P3.png")
