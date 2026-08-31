# Verify that a JuliaFormatter reformat changed no code.
#
# `git diff -w` is not sufficient: it ignores whitespace *within* a line, but a reformat also
# moves code across line breaks, which -w reports as a real change. The property that has to
# hold is that the code *means* the same thing. This compares the parsed expressions after
# normalising away exactly the three things a reformat is allowed to do:
#
#   1. line-number metadata            — LineNumberNodes are dropped
#   2. statement grouping              — `a; b` on one line vs. two lines parses as a nested
#                                        :toplevel/:block, so nested sequence nodes are flattened
#   3. short-form method definitions   — SciML's short_to_long_function_def rewrites
#                                        `f(x) = body` as `function f(x) … end`; both are
#                                        canonicalised to the same node
#
# Anything else that differs is a real change and is reported, and the script exits non-zero.
# Run it from the repository root against the ref the reformat started from:
#
#   julia --startup-file=no scripts/verify_format_ast.jl [ref]     # ref defaults to HEAD

const REF = length(ARGS) >= 1 ? ARGS[1] : "HEAD"

is_seq(e) = e isa Expr && (e.head === :toplevel || e.head === :block)

"""Short-form `f(x) = body` and `function f(x) … end` become the same node."""
function canon_def(head, args)
    if head === :(=) && length(args) == 2 && args[1] isa Expr &&
       args[1].head in (:call, :where, :(::))
        # The right-hand side is already a :block when it spanned a line break.
        body = args[2] isa Expr && args[2].head === :block ? args[2] : Expr(:block, args[2])
        return Expr(:function, args[1], body)
    end
    return Expr(head, args...)
end

normalise(x) = x
function normalise(e::Expr)
    args = Any[]
    for a in e.args
        a isa LineNumberNode && continue
        n = normalise(a)
        # Flatten a nested sequence into its enclosing sequence: `a; b` on one line is the same
        # code as `a` and `b` on two.
        if is_seq(e) && is_seq(n)
            append!(args, n.args)
        else
            push!(args, n)
        end
    end
    return canon_def(e.head, args)
end

changed = split(read(`git diff --name-only $REF -- "*.jl"`, String))
if isempty(changed)
    println("no changed .jl files vs $REF")
    exit(0)
end

equal, differ, failed = String[], String[], String[]

for f in changed
    local a, b
    try
        a = normalise(Meta.parseall(read(`git show $REF:$f`, String); filename = f))
        b = normalise(Meta.parseall(read(f, String); filename = f))
    catch err
        push!(failed, "$f: $(sprint(showerror, err))")
        continue
    end
    # Meta.parseall captures a syntax error as Expr(:error, …) rather than throwing.
    if any(e -> e isa Expr && e.head === :error, (a, b))
        push!(failed, "$f: the file itself does not parse")
    elseif a == b
        push!(equal, f)
    else
        push!(differ, f)
    end
end

println("code-equivalent: ", length(equal), " / ", length(changed), " changed files")
isempty(differ) || println("REAL CODE CHANGE in:\n  ", join(differ, "\n  "))
isempty(failed) || println("could not compare:\n  ", join(failed, "\n  "))

exit(isempty(differ) && isempty(failed) ? 0 : 1)
