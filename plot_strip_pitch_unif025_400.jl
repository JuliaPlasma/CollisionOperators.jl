# Strip pitch ACF grid across unif025 long-horizon snapshots.
# Track stripe amplitude growth/saturation. Δv₂=0.25 (uniform sparse).
# Slab 3.0 ≤ v₁ ≤ 4.5 (outer band where stripes show up).
using GLMakie, DelimitedFiles
using Statistics: mean
GLMakie.activate!()
include("Parameters.jl")
include("parameters_bpmesh40k_v3_unif025.jl"); p = PARAMS

raw, _ = readdlm("particle_snapshots_bpmesh40k_v3_unif025.csv", ',', Any, '\n'; header=true)
available = sort(unique(Int.(raw[:, 1])))
println("Available steps: $available")
steps_plot = filter(s -> s in available, [100, 200, 300, 400])
Δv2 = p.bp2[4] - p.bp2[3]
println("Δv₂ = $(round(Δv2; digits=4))")

function acf_slab(v2b; bin_div=4)
    edges = collect(-2.5:Δv2/bin_div:2.5)
    counts = zeros(Int, length(edges)-1)
    for v in v2b
        idx = searchsortedlast(edges, v)
        1 <= idx <= length(counts) && (counts[idx] += 1)
    end
    w = 30
    smooth = [mean(counts[max(1,i-w):min(length(counts),i+w)]) for i in eachindex(counts)]
    resid = counts .- smooth
    maxlag = length(resid) ÷ 4
    acf = zeros(maxlag)
    for lag in 0:maxlag-1
        n = length(resid) - lag
        acf[lag+1] = sum(resid[1:n] .* resid[1+lag:end]) / n
    end
    dx = edges[2] - edges[1]
    lags = (0:maxlag-1) .* dx
    return lags, acf ./ max(acf[1], 1e-30), edges, counts, resid
end

# cos-fit amplitude
function cos_amp(resid, edges)
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
    k = 2π / Δv2
    A = sum(resid .* cos.(k .* centers)) * 2 / length(centers)
    B = sum(resid .* sin.(k .* centers)) * 2 / length(centers)
    return hypot(A, B), atan(B, A) / k
end

ncol = length(steps_plot)
fig = Figure(; size=(380 * ncol, 800))

peak_hist = NamedTuple{(:step,:t,:lag,:height,:amp,:v0), Tuple{Int,Float64,Float64,Float64,Float64,Float64}}[]

for (k, s) in enumerate(steps_plot)
    mask = Int.(raw[:, 1]) .== s
    sum(mask) == 0 && (println("step $s missing"); continue)
    v1s = Float64.(raw[mask, 4])
    v2s = Float64.(raw[mask, 5])
    band = (v1s .>= 3.0) .& (v1s .<= 4.5)
    v1b = v1s[band]; v2b = v2s[band]
    t = Float64(raw[findfirst(mask), 2])

    lags, acf_n, edges, _, resid = acf_slab(v2b)
    lo, hi = 0.5*Δv2, 1.5*Δv2
    win = findall(l -> lo <= l <= hi, lags)
    peak_idx = win[argmax(acf_n[win])]
    peak_lag = lags[peak_idx]
    peak_h   = acf_n[peak_idx]
    amp, v0 = cos_amp(resid, edges)
    push!(peak_hist, (step=s, t=t, lag=peak_lag, height=peak_h, amp=amp, v0=v0))
    println("step=$s  t=$(round(t;digits=3))  N=$(length(v1b))  ACF peak lag=$(round(peak_lag;digits=4))  h=$(round(peak_h;digits=4))  amp=$(round(amp;digits=3))  v₀=$(round(v0;digits=4))")

    ax_s = Axis(fig[1, k]; xlabel="v₁", ylabel = k==1 ? "v₂" : "",
        title="step=$s  t=$(round(t;digits=3))  N=$(length(v1b))",
        aspect=DataAspect())
    scatter!(ax_s, v1b, v2b; markersize=4.0, color=(:steelblue, 0.6))
    hlines!(ax_s, p.bp2; color=:red, linewidth=0.5, alpha=0.6)
    xlims!(ax_s, 2.5, 5.0); ylims!(ax_s, -2.5, 2.5)

    ax_a = Axis(fig[2, k]; xlabel="lag (v₂)", ylabel = k==1 ? "ACF (norm)" : "",
        title="ACF  peak lag=$(round(peak_lag;digits=3))  h=$(round(peak_h;digits=3))  amp=$(round(amp;digits=2))")
    lines!(ax_a, lags, acf_n; color=:steelblue, linewidth=1.5)
    vlines!(ax_a, [Δv2]; color=:red, linewidth=1, label="Δv₂=$(round(Δv2;digits=3))")
    for m in 2:5
        vlines!(ax_a, [m*Δv2]; color=(:red, 0.3), linewidth=0.4)
    end
    hlines!(ax_a, [0.0]; color=:black, linewidth=0.3)
    axislegend(ax_a; position=:rt, labelsize=9)
    xlims!(ax_a, 0, lags[end])
end

println("\n--- ACF / cos-fit summary (unif025 400-step) ---")
for r in peak_hist
    println("step=$(r.step)  t=$(round(r.t;digits=3))  lag=$(round(r.lag;digits=3))  h=$(round(r.height;digits=3))  amp=$(round(r.amp;digits=3))  v₀=$(round(r.v0;digits=4))")
end

Label(fig[0, 1:ncol],
    "v3_unif025 stripe ACF evolution — uniform mesh Δv₂=0.25  slab 3.0≤v₁≤4.5";
    fontsize=18, tellwidth=false)

save("unif025_400_strip_pitch_grid.png", fig)
println("Saved unif025_400_strip_pitch_grid.png")
