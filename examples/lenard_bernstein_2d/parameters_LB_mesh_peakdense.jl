# Mesh sweep, bi-Max IC. Anisotropic: coarse tails h=1.0, dense bulk h=0.2 in |v|≤3.
# Honeycomb test: peak-localized refinement vs uniform.
function _build_peakdense_bp()
    tail_l = collect(-10.0:1.0:-3.0)            # -10, ..., -3   (h=1)
    bulk   = collect(-3.0:0.2:3.0)[2:end-1]     # -2.8, ..., 2.8 (h=0.2, exclude endpoints)
    tail_r = collect(3.0:1.0:10.0)              #  3, ..., 10    (h=1)
    return vcat(tail_l, bulk, tail_r)
end

PARAMS = SimParameters(
    bp = _build_peakdense_bp(),
    P_DEG=3, K_REG=2, N_QUAD=6,
    N_PARTICLES=1000,
    ic_type="bimax",
    ic_sigma=1.0, ic_sep=2.0,
    nu=1.0,
    DT=8e-4, N_STEPS=6250,
    snap_every=250,
    use_anderson=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    suffix="LB_mesh_peakdense",
    seed=42,
)
