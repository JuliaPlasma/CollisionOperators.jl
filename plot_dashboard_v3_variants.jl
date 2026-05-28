# Dashboard comparing v3 baseline + four parameter-variant probes.
# Rows:
#   1) ‖f_s − f_p‖₂ (projection error) vs step
#   2) ‖G(v) − v‖₂ (fixed-point residual, Eq.104-105 of Jeyakumar et al.) vs step
#   3) fs_diag PNG at step=100 (per-variant slice)
# Columns: one per variant, header labels (P_DEG, bp1_inner, bp2_inner, seed, …).
using GLMakie, DelimitedFiles, PNGFiles
GLMakie.activate!()

variants = [
    (id="v3_800",      P=2, bp1=17, bp2=26, seed=42, note=""),
    (id="v3_P3",       P=3, bp1=17, bp2=26, seed=42, note=""),
    (id="v3_seed43",   P=2, bp1=17, bp2=26, seed=43, note=""),
    (id="v3_shift",    P=2, bp1=17, bp2=26, seed=42, note="bp2+0.05"),
    (id="v3_bp1dense", P=2, bp1=33, bp2=26, seed=42, note="Δv₁=0.25"),
]

function load_hist(id; max_step=100)
    csv = "conservation_history_bpmesh40k_$(id).csv"
    raw, _ = readdlm(csv, ',', Any, '\n'; header=true)
    step = Int.(raw[:, 1])
    keep = step .<= max_step
    step = step[keep]
    fp_l2 = Float64.(raw[keep, 9])    # column 9 = fp_minus_fs
    res   = Float64.(raw[keep, 8])    # column 8 = residual ‖G(v) − v‖₂
    return step, fp_l2, res
end

function fs_diag_path(id)
    "fs_diag_bpmesh40k_$(id)_step0100.png"
end

ncol = length(variants)
fig  = Figure(; size=(360 * ncol, 1200))

colors = [:steelblue, :crimson, :darkgreen, :darkorange, :purple]

# Top global caption: paper-form definition of G.
Label(fig[0, 1:ncol],
    L"G(\hat{v}^{n}, \hat{v}^{n+1}) = \hat{v}^{n} + \Delta t \, K^{+}(\hat{v}^{n+1/2}) \, L(\hat{f}) \, J(\hat{f})";
    fontsize=18, tellwidth=false)

# Per-variant column header inside the figure
for (k, v) in pairs(variants)
    Label(fig[1, k],
          "$(v.id)\nP_DEG=$(v.P)  bp1_inner=$(v.bp1)  bp2_inner=$(v.bp2)  seed=$(v.seed)" *
          (isempty(v.note) ? "" : "\n($(v.note))");
          fontsize=12, tellwidth=false)
end

# Row 1: projection error ‖f_s − f_p‖₂(t)
for (k, v) in pairs(variants)
    step, fp_l2, _ = load_hist(v.id)
    ax = Axis(fig[2, k];
              xlabel="step",
              ylabel = k==1 ? L"\Vert f_s - f_p \Vert_2" : "",
              title  = k==1 ? L"\Vert f_s - f_p \Vert_2" : "")
    lines!(ax, step, fp_l2; color=colors[k], linewidth=2)
    xlims!(ax, 0, 100)
end

# Row 2: residual ‖G(v) − v‖₂(t), log scale
for (k, v) in pairs(variants)
    step, _, res = load_hist(v.id)
    res_safe = max.(res, 1e-30)
    ax = Axis(fig[3, k];
              xlabel="step",
              ylabel = k==1 ? L"\Vert G(v) - v \Vert_2" : "",
              title  = k==1 ? L"\Vert G(\hat{v}^{n}, \hat{v}^{n+1}) - \hat{v}^{n+1} \Vert_2" : "",
              yscale = log10)
    lines!(ax, step, res_safe; color=colors[k], linewidth=2)
    xlims!(ax, 0, 100)
end

# Row 3: fs_diag PNG (step=100) embed as image
for (k, v) in pairs(variants)
    path = fs_diag_path(v.id)
    ax = Axis(fig[4, k];
              title = "fs_diag step=100",
              aspect = DataAspect())
    hidedecorations!(ax); hidespines!(ax)
    if isfile(path)
        img = PNGFiles.load(path)
        image!(ax, rotr90(img))
    else
        text!(ax, "missing\n$path"; position=(0.0, 0.0), align=(:center,:center))
    end
end

# Equation legend bar bottom (definitions). Plain text + a couple inline math chunks.
Label(fig[5, 1:ncol],
      L"\hat{v}^{n+1/2} = (\hat{v}^{n} + \hat{v}^{n+1})/2";
      fontsize=14, tellwidth=false)
Label(fig[6, 1:ncol],
      "fhat = L²-projection of Σ_α w_α δ(v − v_α) onto B-spline basis;   K⁺ = Landau dissipation kernel   (Eq. 104-105)";
      fontsize=11, tellwidth=false)

rowsize!(fig.layout, 0, Fixed(40))
rowsize!(fig.layout, 1, Fixed(60))
rowsize!(fig.layout, 5, Fixed(30))
rowsize!(fig.layout, 6, Fixed(25))

save("dashboard_v3_variants.png", fig)
println("Saved dashboard_v3_variants.png")
