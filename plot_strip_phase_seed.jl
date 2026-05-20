# Compare stripe phase: seed=42 vs seed=43 at step=100, v3 mesh.
# Test: if pure mesh effect, stripes locked to bp2 lines (seed-invariant).
# If projection-variance, stripes shift phase but keep pitch.
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
    v1s = Float64.(raw[mask, 4])
    v2s = Float64.(raw[mask, 5])
    band = (v1s .>= 3.0) .& (v1s .<= 4.5)
    return v1s[band], v2s[band]
end

function acf_resid(v2b; bin_div=4)
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
    return centers, counts, resid, lags, acf ./ max(acf[1], 1e-30)
end

# Estimate stripe phase: locate strongest minimum in residual (v₂ between -1, 1)
function stripe_phase(centers, resid)
    win = findall(c -> -1.5 <= c <= 1.5, centers)
    # Pick deepest minimum (most-locked dip — i.e. between two stable peaks)
    return centers[win[argmin(resid[win])]]
end

v1_42, v2_42 = load_slab("particle_snapshots_bpmesh40k_v3_800.csv", 100)
v1_43, v2_43 = load_slab("particle_snapshots_bpmesh40k_v3_seed43.csv", 100)
println("seed=42  N_band=$(length(v1_42))")
println("seed=43  N_band=$(length(v1_43))")

c42, h42, r42, lag42, acf42 = acf_resid(v2_42)
c43, h43, r43, lag43, acf43 = acf_resid(v2_43)

ph42 = stripe_phase(c42, r42)
ph43 = stripe_phase(c43, r43)
println("seed=42 stripe min near v₂ = $(round(ph42; digits=4))")
println("seed=43 stripe min near v₂ = $(round(ph43; digits=4))")
println("shift Δv₂_phase = $(round(ph43 - ph42; digits=4))   (mod Δv₂=$Δv2)")

# ACF peak
function acf_peak(lags, acf)
    win = findall(l -> 0.5*Δv2 <= l <= 1.5*Δv2, lags)
    i = win[argmax(acf[win])]
    return lags[i], acf[i]
end
pl42, ph_h42 = acf_peak(lag42, acf42)
pl43, ph_h43 = acf_peak(lag43, acf43)
println("seed=42 ACF peak lag=$(round(pl42;digits=3))  h=$(round(ph_h42;digits=3))")
println("seed=43 ACF peak lag=$(round(pl43;digits=3))  h=$(round(ph_h43;digits=3))")

fig = Figure(; size=(1600, 1100))

# Row 1: scatter
for (k, (lbl, v1b, v2b)) in enumerate([("seed=42", v1_42, v2_42), ("seed=43", v1_43, v2_43)])
    ax = Axis(fig[1, k]; xlabel="v₁", ylabel = k==1 ? "v₂" : "",
        title="$lbl  step=100  N=$(length(v1b))", aspect=DataAspect())
    scatter!(ax, v1b, v2b; markersize=4.0, color=(:steelblue, 0.6))
    hlines!(ax, p.bp2; color=:red, linewidth=0.4, alpha=0.5)
    xlims!(ax, 2.5, 5.0); ylims!(ax, -2.5, 2.5)
end

# Row 2: overlay histograms (detrended residuals) — shows phase
ax_h = Axis(fig[2, 1:2]; xlabel="v₂", ylabel="detrended count",
    title="v₂ stripe phase: seed=42 (blue) vs seed=43 (red)  — bp2 edges (gray)")
lines!(ax_h, c42, r42; color=:steelblue, linewidth=2, label="seed=42")
lines!(ax_h, c43, r43; color=:crimson,   linewidth=2, label="seed=43")
vlines!(ax_h, p.bp2; color=(:gray, 0.5), linewidth=0.5)
hlines!(ax_h, [0.0]; color=:black, linewidth=0.3)
# mark detected stripe minimum
vlines!(ax_h, [ph42]; color=:steelblue, linestyle=:dash, linewidth=1.2)
vlines!(ax_h, [ph43]; color=:crimson,   linestyle=:dash, linewidth=1.2)
axislegend(ax_h; position=:rt)
xlims!(ax_h, -2.0, 2.0)

# Row 3: ACF overlay
ax_a = Axis(fig[3, 1:2]; xlabel="lag (v₂)", ylabel="ACF (norm)",
    title="ACF — pitch should be identical Δv₂=$(round(Δv2;digits=3)) regardless of seed")
lines!(ax_a, lag42, acf42; color=:steelblue, linewidth=2, label="seed=42  lag=$(round(pl42;digits=3))  h=$(round(ph_h42;digits=3))")
lines!(ax_a, lag43, acf43; color=:crimson,   linewidth=2, label="seed=43  lag=$(round(pl43;digits=3))  h=$(round(ph_h43;digits=3))")
vlines!(ax_a, [Δv2]; color=:black, linestyle=:dash, label="Δv₂")
for m in 2:5
    vlines!(ax_a, [m*Δv2]; color=(:black, 0.3), linewidth=0.4)
end
hlines!(ax_a, [0.0]; color=:black, linewidth=0.3)
axislegend(ax_a; position=:rt)
xlims!(ax_a, 0, lag42[end])

Label(fig[0, 1:2],
    "Stripe phase vs seed (v3 mesh, step=100, slab 3.0≤v₁≤4.5)";
    fontsize=18, tellwidth=false)

save("stripe_phase_seed42_vs_seed43.png", fig)
println("Saved stripe_phase_seed42_vs_seed43.png")
