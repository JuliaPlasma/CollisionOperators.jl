# Visualize the breakpoint grid (bp1 vertical, bp2 horizontal) overlaid on the
# initial Gaussian density. Run as:
#   julia --project=. plot_mesh.jl parameters_finemesh_anderson.jl
# Outputs mesh_<suffix>.png.
using GLMakie
GLMakie.activate!()
include("Parameters.jl")

preset = isempty(ARGS) ? "parameters_finemesh_anderson.jl" : ARGS[1]
include(preset)
p = PARAMS
print_summary(p)

# Analytical initial density f₀(v₁,v₂) = N(0,σ₁²) ⊗ N(0,σ₂²)
f0(v1, v2) = exp(-v1^2/(2p.σ1^2) - v2^2/(2p.σ2^2)) / (2π*p.σ1*p.σ2)

# Background contour grid (high-resolution evaluation of analytic Gaussian).
v1_axis = range(p.bp1[1], p.bp1[end]; length=400)
v2_axis = range(p.bp2[1], p.bp2[end]; length=400)
F = [f0(v1, v2) for v1 in v1_axis, v2 in v2_axis]

fig = Figure(; size=(1100, 900))
ax = Axis(fig[1, 1];
    xlabel="v₁", ylabel="v₂",
    title="Mesh vs initial Gaussian — $(p.suffix)\n" *
          "v₁ cells=$(length(p.bp1)-1)   v₂ cells=$(length(p.bp2)-1)",
    aspect=DataAspect())

# Filled heatmap of f₀
hm = heatmap!(ax, v1_axis, v2_axis, F; colormap=:viridis)
Colorbar(fig[1, 2], hm; label="f₀(v₁, v₂)")

# Contours at fractions of peak to show σ-ellipses
peak = maximum(F)
contour!(ax, v1_axis, v2_axis, F;
    levels=peak .* [exp(-0.5), exp(-2.0), exp(-4.5)],  # 1σ, 2σ, 3σ
    color=:white, linewidth=1.5)

# Breakpoint grid lines
vlines!(ax, p.bp1; color=:red, linewidth=0.7, alpha=0.7)
hlines!(ax, p.bp2; color=:red, linewidth=0.7, alpha=0.7)

# Annotate σ box
poly!(ax,
    Point2f[(-p.σ1, -p.σ2), (p.σ1, -p.σ2), (p.σ1, p.σ2), (-p.σ1, p.σ2)];
    color=(:white, 0.0), strokecolor=:yellow, strokewidth=1.5)

# Inset zoom on inner region for v₂ resolution check
ax2 = Axis(fig[2, 1];
    xlabel="v₁", ylabel="v₂",
    title="Inner zoom [-3, 3] × [-1.5, 1.5]",
    aspect=DataAspect())
xlims!(ax2, -3, 3);  ylims!(ax2, -1.5, 1.5)
heatmap!(ax2, v1_axis, v2_axis, F; colormap=:viridis)
contour!(ax2, v1_axis, v2_axis, F;
    levels=peak .* [exp(-0.5), exp(-2.0), exp(-4.5)],
    color=:white, linewidth=1.5)
vlines!(ax2, p.bp1; color=:red, linewidth=0.7, alpha=0.7)
hlines!(ax2, p.bp2; color=:red, linewidth=0.7, alpha=0.7)
poly!(ax2,
    Point2f[(-p.σ1, -p.σ2), (p.σ1, -p.σ2), (p.σ1, p.σ2), (-p.σ1, p.σ2)];
    color=(:white, 0.0), strokecolor=:yellow, strokewidth=1.5)

out = "mesh_$(p.suffix).png"
save(out, fig)
println("Saved $out")
