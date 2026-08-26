# GPU port of the two O(N) particle↔spline loops that dominate the CPU side
# of a Picard iteration (profile_iter_cpu.jl: l2_project! 19.2 ms +
# compute_G! 21.0 ms of the ~63 ms total; element quadrature stays on CPU).
#
# Same semantics as functions.jl:
#   l2_project_gpu!  — scatter w_α·φ_k(v_α) into rhs via FP64 atomics, then
#                      the (CPU, 676×676) M_lu solve. Atomic ordering makes
#                      rhs roundoff-nondeterministic (~1e-15 rel) — the only
#                      deviation from the CPU path.
#   compute_G_gpu!   — per-particle gather of ∇L(v_α); deterministic, same
#                      accumulation order as CPU.
#
# Specialized to P_DEG == 2 (hardcoded 3-point Bernstein); the caller must
# keep the CPU path for other degrees. Tables (breakpoints, Bézier extraction
# 3×3×n_cells, basis starts) are uploaded once and cached.

using CUDA

# Largest i with bp[i] <= x, assuming bp[1] < x < bp[end].
@inline function _cell_search(bp, n::Int, x::Float64)
    i = 1; hi = n
    while i + 1 < hi
        mid = (i + hi) >> 1
        if bp[mid] <= x
            i = mid
        else
            hi = mid
        end
    end
    return i
end

@inline _bern2(ξ::Float64) = ((1.0 - ξ)^2, 2.0 * ξ * (1.0 - ξ), ξ * ξ)
@inline _dbern2(ξ::Float64) = (-2.0 * (1.0 - ξ), 2.0 - 4.0 * ξ, 2.0 * ξ)

# phi[j] = Σ_i C[i,j,cell] · B[i]   (3×3, matches transpose(C)·B on CPU)
@inline function _extract3(C, cell::Int, B::NTuple{3, Float64})
    @inbounds begin
        p1 = C[1, 1, cell] * B[1] + C[2, 1, cell] * B[2] + C[3, 1, cell] * B[3]
        p2 = C[1, 2, cell] * B[1] + C[2, 2, cell] * B[2] + C[3, 2, cell] * B[3]
        p3 = C[1, 3, cell] * B[1] + C[2, 3, cell] * B[2] + C[3, 3, cell] * B[3]
    end
    return (p1, p2, p3)
end

function _l2_scatter_kernel!(rhs, v1, v2, w, bp1, bp2, C1, C2, bs1, bs2,
                             n1::Int, n2::Int, nd1::Int, N::Int)
    α = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    α > N && return nothing
    @inbounds begin
        x = v1[α]; y = v2[α]
        (x <= bp1[1] || x >= bp1[n1] || y <= bp2[1] || y >= bp2[n2]) && return nothing
        i = _cell_search(bp1, n1, x)
        j = _cell_search(bp2, n2, y)
        ξ1 = (x - bp1[i]) / (bp1[i+1] - bp1[i])
        ξ2 = (y - bp2[j]) / (bp2[j+1] - bp2[j])
        φ1 = _extract3(C1, i, _bern2(ξ1))
        φ2 = _extract3(C2, j, _bern2(ξ2))
        wα = w[α]
        s1 = bs1[i]; s2 = bs2[j]
        for j2 in 1:3, j1 in 1:3
            gid = (s1 + j1 - 1) + (s2 + j2 - 2) * nd1
            CUDA.@atomic rhs[gid] += wα * φ1[j1] * φ2[j2]
        end
    end
    return nothing
end

function _G_gather_kernel!(G1, G2, v1, v2, L, bp1, bp2, C1, C2, bs1, bs2,
                           n1::Int, n2::Int, nd1::Int, N::Int)
    α = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    α > N && return nothing
    @inbounds begin
        x = v1[α]; y = v2[α]
        if x <= bp1[1] || x >= bp1[n1] || y <= bp2[1] || y >= bp2[n2]
            G1[α] = 0.0
            G2[α] = 0.0
            return nothing
        end
        i = _cell_search(bp1, n1, x)
        j = _cell_search(bp2, n2, y)
        h1 = bp1[i+1] - bp1[i]
        h2 = bp2[j+1] - bp2[j]
        ξ1 = (x - bp1[i]) / h1
        ξ2 = (y - bp2[j]) / h2
        φ1  = _extract3(C1, i, _bern2(ξ1))
        φ2  = _extract3(C2, j, _bern2(ξ2))
        dφ1 = _extract3(C1, i, _dbern2(ξ1))
        dφ2 = _extract3(C2, j, _dbern2(ξ2))
        inv_h1 = 1.0 / h1
        inv_h2 = 1.0 / h2
        s1 = bs1[i]; s2 = bs2[j]
        acc1 = 0.0; acc2 = 0.0
        for j2 in 1:3, j1 in 1:3           # same order as the CPU loop
            gid = (s1 + j1 - 1) + (s2 + j2 - 2) * nd1
            Lg = L[gid]
            acc1 += Lg * dφ1[j1] * φ2[j2] * inv_h1
            acc2 += Lg * φ1[j1] * dφ2[j2] * inv_h2
        end
        G1[α] = acc1
        G2[α] = acc2
    end
    return nothing
end

# Per-particle gather of the direct log-density gradient g = ∇f_s(v_α)/f_s(v_α)
# from spline coefficients `fc` (GPU port of eval_loggrad_at_particles!, the LB
# base gradient). Accumulates f and ∇f in the same 3×3 order as the CPU loop;
# clamps each component to ±g_max after dividing by max(|f|, fs_floor). Out-of-
# domain particles → g = 0 (drift reduces to A + B v).
function _loggrad_gather_kernel!(g1, g2, v1, v2, fc, bp1, bp2, C1, C2, bs1, bs2,
                                 n1::Int, n2::Int, nd1::Int, N::Int,
                                 fs_floor::Float64, g_max::Float64)
    α = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    α > N && return nothing
    @inbounds begin
        x = v1[α]; y = v2[α]
        if x <= bp1[1] || x >= bp1[n1] || y <= bp2[1] || y >= bp2[n2]
            g1[α] = 0.0
            g2[α] = 0.0
            return nothing
        end
        i = _cell_search(bp1, n1, x)
        j = _cell_search(bp2, n2, y)
        h1 = bp1[i+1] - bp1[i]
        h2 = bp2[j+1] - bp2[j]
        ξ1 = (x - bp1[i]) / h1
        ξ2 = (y - bp2[j]) / h2
        φ1  = _extract3(C1, i, _bern2(ξ1))
        φ2  = _extract3(C2, j, _bern2(ξ2))
        dφ1 = _extract3(C1, i, _dbern2(ξ1))
        dφ2 = _extract3(C2, j, _dbern2(ξ2))
        inv_h1 = 1.0 / h1
        inv_h2 = 1.0 / h2
        s1 = bs1[i]; s2 = bs2[j]
        f = 0.0; d1 = 0.0; d2 = 0.0
        for j2 in 1:3, j1 in 1:3           # same order as the CPU loop
            gid = (s1 + j1 - 1) + (s2 + j2 - 2) * nd1
            c = fc[gid]
            f  += c * φ1[j1]  * φ2[j2]
            d1 += c * dφ1[j1] * φ2[j2]  * inv_h1
            d2 += c * φ1[j1]  * dφ2[j2] * inv_h2
        end
        invf = 1.0 / max(abs(f), fs_floor)
        g1[α] = clamp(d1 * invf, -g_max, g_max)
        g2[α] = clamp(d2 * invf, -g_max, g_max)
    end
    return nothing
end

mutable struct GpuProjBuf
    N::Int
    n_dofs::Int
    v1::CuVector{Float64}; v2::CuVector{Float64}
    w::CuVector{Float64}
    rhs::CuVector{Float64}
    L::CuVector{Float64}
    G1::CuVector{Float64}; G2::CuVector{Float64}
    bp1::CuVector{Float64}; bp2::CuVector{Float64}
    C1::CuArray{Float64, 3}; C2::CuArray{Float64, 3}
    bs1::CuVector{Int32}; bs2::CuVector{Int32}
    h::Vector{Float64}          # length-N host staging
    hd::Vector{Float64}         # length-n_dofs host staging
end

const _PROJ_BUF = Ref{Union{Nothing, GpuProjBuf}}(nothing)

function _proj_buf(ws::Workspace, N::Int)
    b = _PROJ_BUF[]
    if b === nothing || b.N != N
        ws.p.P_DEG == 2 || error("projection_gpu.jl is specialized to P_DEG=2")
        p1 = 3
        flat(exts) = begin
            A = Array{Float64, 3}(undef, p1, p1, length(exts))
            for (k, C) in enumerate(exts)
                A[:, :, k] .= Matrix(C)
            end
            A
        end
        b = GpuProjBuf(N, ws.n_dofs,
            CUDA.zeros(Float64, N), CUDA.zeros(Float64, N),
            CUDA.zeros(Float64, N),
            CUDA.zeros(Float64, ws.n_dofs),
            CUDA.zeros(Float64, ws.n_dofs),
            CUDA.zeros(Float64, N), CUDA.zeros(Float64, N),
            CuArray(ws.bp1), CuArray(ws.bp2),
            CuArray(flat(ws.ext1d_1)), CuArray(flat(ws.ext1d_2)),
            CuArray(Int32.(ws.basis_start_1d_1)), CuArray(Int32.(ws.basis_start_1d_2)),
            zeros(N), zeros(ws.n_dofs))
        _PROJ_BUF[] = b
    end
    return b
end

function l2_project_gpu!(ws::Workspace, f_coeffs, v_parts, w_parts)
    N = size(v_parts, 1)
    b = _proj_buf(ws, N)
    _upload_col!(b.v1, v_parts, 1, b.h)
    _upload_col!(b.v2, v_parts, 2, b.h)
    copyto!(b.w, w_parts)
    fill!(b.rhs, 0.0)
    @cuda threads=256 blocks=cld(N, 256) _l2_scatter_kernel!(
        b.rhs, b.v1, b.v2, b.w, b.bp1, b.bp2, b.C1, b.C2, b.bs1, b.bs2,
        length(ws.bp1), length(ws.bp2), ws.n_dofs_1d_1, N)
    copyto!(b.hd, b.rhs)
    f_coeffs .= ws.M_lu \ b.hd
    return nothing
end

function compute_G_gpu!(ws::Workspace, G, v_parts, L_vec)
    N = size(v_parts, 1)
    b = _proj_buf(ws, N)
    _upload_col!(b.v1, v_parts, 1, b.h)
    _upload_col!(b.v2, v_parts, 2, b.h)
    copyto!(b.L, L_vec)
    @cuda threads=256 blocks=cld(N, 256) _G_gather_kernel!(
        b.G1, b.G2, b.v1, b.v2, b.L, b.bp1, b.bp2, b.C1, b.C2, b.bs1, b.bs2,
        length(ws.bp1), length(ws.bp2), ws.n_dofs_1d_1, N)
    copyto!(b.h, b.G1)
    @inbounds for i in 1:N
        G[i, 1] = b.h[i]
    end
    copyto!(b.h, b.G2)
    @inbounds for i in 1:N
        G[i, 2] = b.h[i]
    end
    return nothing
end

# LB base gradient on the GPU: gather g = ∇f_s/f_s at particles from f_coeffs.
# Reuses the b.L device buffer (length n_dofs) to hold f_coeffs and b.G1/b.G2
# for the result. fs_floor/g_max mirror the FS_FLOOR/G_MAX consts in functions.jl.
function eval_loggrad_gpu!(ws::Workspace, g, v_parts, f_coeffs)
    N = size(v_parts, 1)
    b = _proj_buf(ws, N)
    _upload_col!(b.v1, v_parts, 1, b.h)
    _upload_col!(b.v2, v_parts, 2, b.h)
    copyto!(b.L, f_coeffs)
    @cuda threads=256 blocks=cld(N, 256) _loggrad_gather_kernel!(
        b.G1, b.G2, b.v1, b.v2, b.L, b.bp1, b.bp2, b.C1, b.C2, b.bs1, b.bs2,
        length(ws.bp1), length(ws.bp2), ws.n_dofs_1d_1, N, 1e-30, 100.0)
    copyto!(b.h, b.G1)
    @inbounds for i in 1:N
        g[i, 1] = b.h[i]
    end
    copyto!(b.h, b.G2)
    @inbounds for i in 1:N
        g[i, 2] = b.h[i]
    end
    return nothing
end
