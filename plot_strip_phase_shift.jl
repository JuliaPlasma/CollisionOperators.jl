# Compare stripe phase: v3 baseline vs v3_shift (+0.05 mesh shift) at step=100.
# Prediction: if stripes lock to mesh+basis, phase v₀ shifts by +0.05.
using GLMakie, DelimitedFiles
using Statistics: mean
GLMakie.activate!()
include("Parameters.jl")
include("parameters_bpmesh40k_v3_800.jl"); p_base = PARAMS
include("parameters_bpmesh40k_v3_shift.jl"); p_shift = PARAMS

Δv2 = p_base.bp2[4] - p_base.bp2[3]
mesh_shift = p_shift.bp2[4] - p_base.bp2[4]
println("Δv₂ = $(round(Δv2; digits=4))")
println("mesh shift Δ(bp2 inner) = $(round(mesh_shift; digits=4))")

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

# Cosine fit r(v₂) ≈ A cos(2π v₂/Δv₂) + B sin(2π v₂/Δv₂); phase v₀ = (Δv₂/2π) atan2(B,A)
function phase_fit(centers, resid; v2lo=-1.5, v2hi=1.5)
    sel = findall(c -> v2lo <= c <= v2hi, centers)
    cs = centers[sel]; rs = resid[sel]
    ω = 2π / Δv2
    A = 2 * mean(rs .* cos.(ω .* cs))
    B = 2 * mean(rs .* sin.(ω .* cs))
    amp = sqrt(A^2 + B^2)
    v0 = (atan(B, A) / ω)            # phase offset in v₂ units (period = Δv₂)
    return v0, amp, A, B
end

v1_b, v2_b = load_slab("particle_snapshots_bpmesh40k_v3_800.csv", 100)
v1_s, v2_s = load_slab("particle_snapshots_bpmesh40k_v3_shift.csv", 100)
println("baseline   N_band=$(length(v1_b))")
println("shift      N_band=$(length(v1_s))")

cb, hb, rb, lagb, acfb = acf_resid(v2_b)
cs, hs, rs, lags, acfs = acf_resid(v2_s)

v0_b, amp_b, _, _ = phase_fit(cb, rb)
v0_s, amp_s, _, _ = phase_fit(cs, rs)

# canonical residue (folded into single period for clean comparison)
foldmod(x) = mod(x + Δv2/2, Δv2) - Δv2/2
v0_b_f = foldmod(v0_b)
v0_s_f = foldmod(v0_s)
shift_obs = foldmod(v0_s - v0_b)

println("baseline cosine fit  v₀ = $(round(v0_b; digits=4))  (folded $(round(v0_b_f; digits=4)))  A=$(round(amp_b; digits=2))")
println("shift    cosine fit  v₀ = $(round(v0_s; digits=4))  (folded $(round(v0_s_f; digits=4)))  A=$(round(amp_s; digits=2))")
println("observed Δv₀ (folded) = $(round(shift_obs; digits=4))")
println("predicted mesh shift  = $(round(mesh_shift; digits=4))")
println("ratio Δv₀/Δmesh        = $(round(shift_obs / mesh_shift; digits=3))")

# ACF peak
function acf_peak(lags, acf)
    win = findall(l -> 0.5*Δv2 <= l <= 1.5*Δv2, lags)
    i = win[argmax(acf[win])]
    return lags[i], acf[i]
end
plb, phb = acf_peak(lagb, acfb)
pls, phs = acf_peak(lags, acfs)
println("baseline ACF peak lag=$(round(plb;digits=3))  h=$(round(phb;digits=3))")
println("shift    ACF peak lag=$(round(pls;digits=3))  h=$(round(phs;digits=3))")

fig = Figure(; size=(1600, 1200))

# Row 1: scatter w/ corresponding bp2 mesh overlay
for (k, (lbl, v1b, v2b, p)) in enumerate([
        ("v3 baseline  bp2 inner ∈ [-2.5,2.5]", v1_b, v2_b, p_base),
        ("v3 shift  bp2 inner ∈ [-2.45,2.55]",  v1_s, v2_s, p_shift)])
    ax = Axis(fig[1, k]; xlabel="v₁", ylabel = k==1 ? "v₂" : "",
        title="$lbl  step=100  N=$(length(v1b))", aspect=DataAspect())
    scatter!(ax, v1b, v2b; markersize=4.0, color=(:steelblue, 0.6))
    hlines!(ax, p.bp2; color=:red, linewidth=0.4, alpha=0.5)
    xlims!(ax, 2.5, 5.0); ylims!(ax, -2.5, 2.5)
end

# Row 2: detrended residuals overlay
ax_h = Axis(fig[2, 1:2]; xlabel="v₂", ylabel="detrended count",
    title="v₂ stripe phase: baseline (blue) vs shift (red)   bp2 lines dashed in matching color")
lines!(ax_h, cb, rb; color=:steelblue, linewidth=2, label="baseline   v₀=$(round(v0_b_f;digits=3))")
lines!(ax_h, cs, rs; color=:crimson,   linewidth=2, label="shift +0.05   v₀=$(round(v0_s_f;digits=3))")
vlines!(ax_h, p_base.bp2;  color=(:steelblue, 0.4), linewidth=0.5)
vlines!(ax_h, p_shift.bp2; color=(:crimson, 0.4), linewidth=0.5, linestyle=:dash)
hlines!(ax_h, [0.0]; color=:black, linewidth=0.3)
vlines!(ax_h, [v0_b_f]; color=:steelblue, linestyle=:dash, linewidth=1.2)
vlines!(ax_h, [v0_s_f]; color=:crimson,   linestyle=:dash, linewidth=1.2)
axislegend(ax_h; position=:rt)
xlims!(ax_h, -2.0, 2.0)

# Row 3: ACF
ax_a = Axis(fig[3, 1:2]; xlabel="lag (v₂)", ylabel="ACF (norm)",
    title="ACF — pitch Δv₂=$(round(Δv2;digits=3)) (mesh shift shouldn't change pitch)")
lines!(ax_a, lagb, acfb; color=:steelblue, linewidth=2, label="baseline lag=$(round(plb;digits=3))  h=$(round(phb;digits=3))")
lines!(ax_a, lags, acfs; color=:crimson,   linewidth=2, label="shift    lag=$(round(pls;digits=3))  h=$(round(phs;digits=3))")
vlines!(ax_a, [Δv2]; color=:black, linestyle=:dash, label="Δv₂")
for m in 2:5
    vlines!(ax_a, [m*Δv2]; color=(:black, 0.3), linewidth=0.4)
end
hlines!(ax_a, [0.0]; color=:black, linewidth=0.3)
axislegend(ax_a; position=:rt)
xlims!(ax_a, 0, lagb[end])

Label(fig[0, 1:2],
    "Mesh-shift test (step=100)   predicted Δv₀=$(round(mesh_shift;digits=3))   observed Δv₀=$(round(shift_obs;digits=3))   ratio=$(round(shift_obs/mesh_shift;digits=2))";
    fontsize=18, tellwidth=false)

save("stripe_phase_mesh_shift.png", fig)
println("Saved stripe_phase_mesh_shift.png")
