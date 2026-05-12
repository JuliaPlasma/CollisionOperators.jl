# Reconstruct entropy S(t) from saved fs_snapshot_*.csv files.
# Each snapshot has header lines:
#   # bp1=v1,v2,...
#   # bp2=v1,v2,...
#   # n_dofs=N
#   coeff
#   <N coefficient floats>
# Run as:
#   julia --project=. compute_entropy_from_fs.jl <tag> [dt]
# tag = e.g. "bpmesh10k_anderson". Globs fs_snapshot_<tag>_step*.csv.
# dt  = time step (defaults to 0.001).
include("MantisWrappers.jl")
using .MantisWrappers
using DelimitedFiles

length(ARGS) >= 1 || error("usage: compute_entropy_from_fs.jl <tag> [dt]")
tag = ARGS[1]
dt  = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.001

function parse_header(path)
    bp1 = Float64[]; bp2 = Float64[]; ndof = 0; data_start = 0
    open(path) do io
        for (i, ln) in enumerate(eachline(io))
            if startswith(ln, "# bp1=")
                bp1 = parse.(Float64, split(ln[length("# bp1=")+1:end], ','))
            elseif startswith(ln, "# bp2=")
                bp2 = parse.(Float64, split(ln[length("# bp2=")+1:end], ','))
            elseif startswith(ln, "# n_dofs=")
                ndof = parse(Int, ln[length("# n_dofs=")+1:end])
            elseif ln == "coeff"
                data_start = i + 1
                break
            end
        end
    end
    return bp1, bp2, ndof, data_start
end

function load_coeffs(path, ndof, data_start)
    coeffs = Vector{Float64}(undef, ndof)
    open(path) do io
        for _ in 1:data_start-1; readline(io); end
        for k in 1:ndof
            coeffs[k] = parse(Float64, readline(io))
        end
    end
    return coeffs
end

files = sort(filter(f -> occursin(Regex("^fs_snapshot_$(tag)_step\\d+\\.csv\$"), f),
                    readdir(".")))
isempty(files) && error("No files matching fs_snapshot_$(tag)_step*.csv")
println("Found $(length(files)) snapshots")

# Use first file to build workspace (bp1/bp2 identical across snapshots)
bp1, bp2, ndof, _ = parse_header(files[1])
println("bp1 ($(length(bp1)) pts): $bp1")
println("bp2 ($(length(bp2)) pts): $bp2")
println("n_dofs=$ndof")

p = SimParameters(; bp1=bp1, bp2=bp2)
ws = build_workspace(p)
ws.n_dofs == ndof || error("n_dofs mismatch: workspace=$(ws.n_dofs) snapshot=$ndof")

println("step,time,entropy")
results = NTuple{3, Float64}[]
for f in files
    m = match(r"step(\d+)\.csv$", f)
    step = parse(Int, m.captures[1])
    _, _, _, ds = parse_header(f)
    c = load_coeffs(f, ndof, ds)
    field = build_field(ws, c)
    S = compute_entropy(ws, field)
    t = step * dt
    push!(results, (step, t, S))
    println("$step,$t,$S")
end

out = "entropy_from_fs_$(tag).csv"
open(out, "w") do io
    println(io, "step,time,entropy")
    for (s, t, S) in results
        println(io, "$s,$t,$S")
    end
end
println("Saved $out")
