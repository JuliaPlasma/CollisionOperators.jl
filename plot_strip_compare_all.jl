# Combined stripe-shape comparison at step=100 for every v3 variant probe.
# One scatter panel per variant: v₂ in [-2.5,2.5], v₁ in [2.5,5.0] outer slab where
# stripes are most visible. Overlay bp1 (vertical) and bp2 (horizontal) lines.
using GLMakie, DelimitedFiles
GLMakie.activate!()

bp1_v3 = [-6.0; -5.0; collect(LinRange(-4.0, 4.0, 17)); 5.0; 6.0]
bp2_v3 = [-6.0; collect(LinRange(-2.5,  2.5, 26)); 6.0]
bp1_bp1dense = [-6.0; -5.0; collect(LinRange(-4.0, 4.0, 33)); 5.0; 6.0]
bp2_shift    = [-6.0; collect(LinRange(-2.45, 2.55, 26)); 6.0]
bp1_unif     = collect(LinRange(-6.0, 6.0, 25))
bp2_unif02   = collect(LinRange(-6.0, 6.0, 61))
bp2_unif025  = collect(LinRange(-6.0, 6.0, 49))

variants = [
    (id="v3_800",      bp1=bp1_v3,        bp2=bp2_v3,       title="v3 baseline\nP=2  non-unif  Δv₂=0.2"),
    (id="v3_shift",    bp1=bp1_v3,        bp2=bp2_shift,    title="v3 shift\nP=2  bp2+0.05"),
    (id="v3_seed43",   bp1=bp1_v3,        bp2=bp2_v3,       title="v3 seed=43\nP=2  same mesh"),
    (id="v3_P3",       bp1=bp1_v3,        bp2=bp2_v3,       title="v3 P=3\nbp1=17 bp2=26"),
    (id="v3_bp1dense", bp1=bp1_bp1dense,  bp2=bp2_v3,       title="v3 bp1-dense\nΔv₁=0.25 Δv₂=0.2"),
    (id="v3_unif02",   bp1=bp1_unif,      bp2=bp2_unif02,   title="v3 uniform 0.2\nΔv=(0.5,0.2)"),
    (id="v3_unif025",  bp1=bp1_unif,      bp2=bp2_unif025,  title="v3 uniform 0.25\nΔv=(0.5,0.25)"),
]

function load_slab(csv, step; v1_lo=2.5, v1_hi=5.0)
    raw, _ = readdlm(csv, ',', Any, '\n'; header=true)
    mask = Int.(raw[:, 1]) .== step
    v1s = Float64.(raw[mask, 4])
    v2s = Float64.(raw[mask, 5])
    band = (v1s .>= v1_lo) .& (v1s .<= v1_hi)
    return v1s[band], v2s[band]
end

ncol = length(variants)
fig = Figure(; size=(420 * ncol, 1200))

Label(fig[0, 1:ncol],
      "Stripe comparison at step=100 — slab 2.5 ≤ v₁ ≤ 5.0   (red lines = mesh breakpoints)";
      fontsize=18, tellwidth=false)

for (k, v) in pairs(variants)
    csv = "particle_snapshots_bpmesh40k_$(v.id).csv"
    v1b, v2b = load_slab(csv, 100)
    ax = Axis(fig[1, k];
              xlabel="v₁", ylabel = k==1 ? "v₂" : "",
              title="$(v.title)\nN_slab=$(length(v1b))",
              aspect=DataAspect())
    scatter!(ax, v1b, v2b; markersize=3.5, color=(:steelblue, 0.55))
    vlines!(ax, v.bp1; color=(:red, 0.6), linewidth=0.5)
    hlines!(ax, v.bp2; color=(:red, 0.6), linewidth=0.5)
    xlims!(ax, 2.5, 5.0); ylims!(ax, -2.5, 2.5)
end

save("strip_compare_all.png", fig)
println("Saved strip_compare_all.png")
