# Transcode an aniso100_logsq checkpoint's PARTICLE state onto the dense-square
# mesh of parameters_sq_dense_from2000.jl, writing a fresh step-0 checkpoint
# (+ step-0 fs_snapshot) so main_Gonzalez.jl can --resume=auto from it.
# Particles are mesh-independent (the truth); only the spline projection changes,
# which is exactly the mesh effect we want to isolate.
#   julia --project=. make_sq_dense_checkpoint.jl <src_ckpt.jls> <preset.jl>
using Serialization, Random
include("MantisWrappers.jl")
using .MantisWrappers

# compute_energy / compute_momentum live in main_Gonzalez.jl (not MantisWrappers);
# replicate them verbatim so step-0 values match the running loop's scale.
compute_momentum(v, w) = (sum(w[α]*v[α,1] for α in axes(v,1)),
                          sum(w[α]*v[α,2] for α in axes(v,1)))
compute_energy(v, w) = 0.5 * sum(w[α]*(v[α,1]^2 + v[α,2]^2) for α in axes(v,1))

src_ckpt = ARGS[1]
preset   = ARGS[2]
include(preset); p = PARAMS

USE_LOGSQ[] = p.use_logsq                       # match entropy integrand
src = open(deserialize, src_ckpt)
v_particles = src.v_particles
w_particles = src.w_particles
println("Loaded $(size(v_particles,1)) particles from $src_ckpt (src step=$(src.step))")

ws = build_workspace(p)
println("New square workspace: n_dofs=$(ws.n_dofs)  n_elements=$(ws.n_elements)")

f_coeffs = zeros(ws.n_dofs)
l2_project!(ws, f_coeffs, v_particles, w_particles)   # re-project onto dense square
f_s = build_field(ws, f_coeffs)

S0 = compute_entropy(ws, f_s)
E0 = compute_energy(v_particles, w_particles)
P0 = compute_momentum(v_particles, w_particles)
neg0 = compute_negative_part_l1(ws, f_s)
println("step0 on new mesh:  S=$S0  E=$E0  P=$P0  ∫(neg)=$neg0")

# Histories: step-0 entries for conserved quantities; per-step ones start empty
# (mirrors the fresh-init path in run_simulation).
entropy_history  = Float64[S0]
energy_history   = Float64[E0]
momentum_history = NTuple{2,Float64}[P0]
iter_history     = Int[]
res_history      = Float64[]
fp_l2_history    = Float64[]
neg_history      = Float64[]

ckpt0 = "checkpoint_$(p.suffix)_step0000.jls"
open(ckpt0, "w") do io
    serialize(io, (; step=0, v_particles, w_particles, f_coeffs,
                     entropy_history, energy_history, momentum_history,
                     iter_history, res_history, fp_l2_history, neg_history,
                     rng_state=copy(Random.default_rng())))
end
println("Wrote $ckpt0")

# Step-0 fs_snapshot (same format as save_fs_snapshot) for immediate diagnostics.
snap0 = "fs_snapshot_$(p.suffix)_step0000.csv"
open(snap0, "w") do io
    println(io, "# bp1=", join(ws.bp1, ","))
    println(io, "# bp2=", join(ws.bp2, ","))
    println(io, "# n_dofs=", ws.n_dofs)
    println(io, "coeff")
    for c in f_coeffs; println(io, c); end
end
println("Wrote $snap0")
