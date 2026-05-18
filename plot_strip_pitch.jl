# Measure strip pitch — v4 step=100, zoom inner edge band, v₂ hist fine bins.
using GLMakie, DelimitedFiles
using Statistics: mean
GLMakie.activate!()

include("Parameters.jl")
include("parameters_bpmesh60k_v6.jl"); p = PARAMS

raw, _ = readdlm("particle_snapshots_bpmesh60k_v6.csv", ',', Any, '\n'; header=true)
mask = Int.(raw[:, 1]) .== 100
v1s = Float64.(raw[mask, 4])
v2s = Float64.(raw[mask, 5])

# Strip band: thin slab around v₁ = 3.5..4 (outer edge ring)
band = (v1s .>= 3.0) .& (v1s .<= 4.5)
v1b = v1s[band]; v2b = v2s[band]
println("band N = $(length(v1b))  (slab 3.0 ≤ v₁ ≤ 4.5)")
println("bp2 inner Δv₂ = $(round(p.bp2[4]-p.bp2[3]; digits=4))")
println("bp1 inner Δv₁ = $(round(p.bp1[4]-p.bp1[3]; digits=4))")
println("N_QUAD = $(p.N_QUAD) → sub-cell node spacing ≈ Δ/N_QUAD")

fig = Figure(; size=(1700, 1100))

# Panel 1: zoom scatter strip band, bigger markers, v₂ full range
ax1 = Axis(fig[1, 1]; xlabel="v₁", ylabel="v₂",
    title="v6 step=100  slab 3.0 ≤ v₁ ≤ 4.5  N=$(length(v1b))",
    aspect=DataAspect())
scatter!(ax1, v1b, v2b; markersize=6.0, color=(:steelblue, 0.8))
hlines!(ax1, p.bp2; color=:red, linewidth=0.6, alpha=0.7)
vlines!(ax1, p.bp1; color=:red, linewidth=0.4, alpha=0.5)
xlims!(ax1, 2.5, 5.0); ylims!(ax1, -2.5, 2.5)

# Panel 2: v₂ hist of strip band, ultra-fine bin = Δv₂/20 = 0.01
Δv2 = p.bp2[4] - p.bp2[3]
edges_fine = collect(-2.5:Δv2/4:2.5)  # 4 sub-bins per cell, ~5% shot noise
ax2 = Axis(fig[1, 2]; xlabel="v₂", ylabel="count",
    title="v₂ hist (strip band only)  bin=Δv₂/4=$(round(Δv2/4; digits=4))")
hist!(ax2, v2b; bins=edges_fine, color=:steelblue)
vlines!(ax2, p.bp2; color=:red, linewidth=0.5, label="bp2 cell edges")
# Quad nodes inside one cell — Gauss-Legendre N_QUAD=6, scaled to [bp2[i], bp2[i+1]]
gl_nodes, _ = let
    # Gauss-Legendre nodes on [-1,1] for N=6
    nodes = [-0.9324695142031521, -0.6612093864662645, -0.2386191860831969,
              0.2386191860831969,  0.6612093864662645,  0.9324695142031521]
    nodes, nothing
end
quad_lines = Float64[]
for i in 1:length(p.bp2)-1
    a, b = p.bp2[i], p.bp2[i+1]
    if a >= -2.5 && b <= 2.5
        for n in gl_nodes
            push!(quad_lines, (a+b)/2 + (b-a)/2 * n)
        end
    end
end
vlines!(ax2, quad_lines; color=(:green, 0.4), linewidth=0.3, label="quad nodes")
axislegend(ax2; position=:rt)
xlims!(ax2, -2.5, 2.5)

# Panel 3: zoom v₂ hist [-2.5, 2.5] full bp2 inner
ax3 = Axis(fig[2, 1]; xlabel="v₂", ylabel="count",
    title="v₂ hist full bp2 inner [-2.5, 2.5]  bp2(red)  quad(green)")
hist!(ax3, v2b; bins=edges_fine, color=:steelblue)
vlines!(ax3, p.bp2; color=:red, linewidth=0.8, alpha=0.7)
vlines!(ax3, quad_lines; color=(:green, 0.5), linewidth=0.4)
xlims!(ax3, -2.5, 2.5)

# Panel 4: power spectrum of v₂ hist to extract dominant pitch
bin_centers = (edges_fine[1:end-1] .+ edges_fine[2:end]) ./ 2
h, _ = let
    counts = zeros(Int, length(edges_fine)-1)
    for v in v2b
        idx = searchsortedlast(edges_fine, v)
        if 1 <= idx <= length(counts)
            counts[idx] += 1
        end
    end
    counts, nothing
end
# Detrend by subtracting smoothed envelope (moving avg width = 30 bins ~ 0.3 in v₂)
w = 30
smooth = [mean(h[max(1,i-w):min(length(h),i+w)]) for i in eachindex(h)]
resid = h .- smooth
# Autocorrelation of detrended residual — peak lag = strip pitch
maxlag = length(resid) ÷ 4
acf = zeros(maxlag)
for lag in 0:maxlag-1
    n = length(resid) - lag
    acf[lag+1] = sum(resid[1:n] .* resid[1+lag:end]) / n
end
dx = edges_fine[2] - edges_fine[1]
lags_v2 = (0:maxlag-1) .* dx
ax4 = Axis(fig[2, 2]; xlabel="lag (v₂)", ylabel="autocorr",
    title="ACF of detrended v₂ hist — peak lag = strip pitch")
lines!(ax4, lags_v2, acf ./ acf[1]; color=:steelblue)
vlines!(ax4, [Δv2]; color=:red, linewidth=1, label="Δv₂ = $(round(Δv2; digits=3))")
vlines!(ax4, [Δv2/p.N_QUAD]; color=:green, linewidth=1, label="Δv₂/N_QUAD = $(round(Δv2/p.N_QUAD; digits=3))")
axislegend(ax4; position=:rt)
# Mark first 5 multiples of Δv₂
for m in 1:5
    vlines!(ax4, [m*Δv2]; color=(:red, 0.3), linewidth=0.5)
end

Label(fig[0, 1:2], "Strip pitch measurement — v6 step=100, slab 3.0≤v₁≤4.5";
    fontsize=18, tellwidth=false)

save("v6_strip_pitch.png", fig)
println("Saved v6_strip_pitch.png")
