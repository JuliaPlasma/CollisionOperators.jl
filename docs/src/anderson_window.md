# The Anderson window update

The Anderson acceleration in [`step_anderson!`](https://github.com/junyixu/CollisionOperators.jl/blob/main/solver.jl)
keeps the last ``m`` residual and map differences in two dense buffers, ``\Delta F``
and ``\Delta G``, each of size ``2N \times m``. Every iteration contributes one new
column and must drop the oldest.

## Why a ring cursor

The window used to be kept in chronological order, which meant shifting the whole
thing left by one column once it was full:

```julia
@views ΔF[:, 1:(m - 1)] .= ΔF[:, 2:m]
@views ΔG[:, 1:(m - 1)] .= ΔG[:, 2:m]
```

That recopies ``2N(m-1)`` entries on every iteration — at ``N = 40{,}000`` and
``m = 8`` that is 1.12 million `Float64`, about 9 MB written and 9 MB read, for no
numerical purpose.

The order of the columns does not matter. The least-squares problem

```math
\gamma = \arg\min_\gamma \lVert r_k - \Delta F \gamma \rVert_2^2 ,
\qquad x_{k+1} = G(x_k) - \Delta G \gamma
```

is invariant under a common column permutation of ``\Delta F`` and ``\Delta G``:
permuting the columns permutes ``\gamma`` identically, and ``\Delta G \gamma`` is
unchanged. So the newest difference can simply overwrite the oldest column in
place:

```julia
slot = mod1(slot + 1, m)
history = min(history + 1, m)
@views ΔF[:, slot] .= r_v .- rp_v
@views ΔG[:, slot] .= Gv_v .- Gp_v
```

While the window is still filling, `slot == history`, so the view `ΔF[:, 1:history]`
that the solve reads stays correct. The cursor must be reset together with
`history` when the solver restarts, or the write lands outside that view.

## Measured effect

At ``N = 40{,}000``, ``m = 8`` (so ``\Delta F``, ``\Delta G`` are ``80{,}000 \times 8``):

| | shift-left | ring cursor |
|---|---|---|
| window update alone | 2.12 ms | 0.08 ms |

That is **≈ 2.03 ms saved per Anderson iteration**, which is roughly 6 % of a 32 ms
FP32 iteration, 1.5 % of an FP64 one, and negligible on CPU, where the ``O(N^2)``
collision sum dominates everything else.

An end-to-end A/B on a single RTX 3070 — four interleaved phases (old, new, old,
new), each resuming from the same step-1500 checkpoint for 400 steps at
``\Delta t = 0.005`` with `abs_floor = 1e-8` — gives:

| phase | ms / iteration (median) | uncontended minimum | Anderson iterations |
|---|---|---|---|
| shift #1 | 40.79 | 33.80 | 11,163 |
| ring #1 | 34.90 | 31.95 | 12,165 |
| shift #2 | 36.24 | 33.38 | 12,618 |
| ring #2 | 34.23 | 30.77 | 12,226 |

Both arms were run on the same pod, because the per-iteration cost of an FP32 run
is bound by host work rather than the GPU, and the spread between Community hosts
is larger than the effect being measured.

Read this as corroboration, not proof. The direction is consistent everywhere —
both medians, both minima, and ``P(\text{a random shift interval} > \text{a random
ring interval}) = 0.70`` — but with two phases per arm an exact permutation test
only reaches ``p = 1/6``. The controlled microbenchmark above is the stronger
evidence; the pod run confirms nothing contradicts it at full scale.

## What does not change

* **Iteration counts.** 11,163 / 12,618 for shift against 12,165 / 12,226 for ring.
  The between-arm difference is well inside the same-code spread, which comes from
  the non-deterministic order of the FP32 reduction on the GPU.
* **Conservation.** Energy drift and entropy curves are indistinguishable between
  the two. Momentum drift stays at ``\sim 10^{-10.6}`` with the two arms
  interleaved rather than separated.
* **The answer, to within round-off.** The permutation invariance is exact in exact
  arithmetic; the reordered summation in ``\Delta F^\top \Delta F`` differs at the
  level of machine epsilon. That only becomes visible where it flips a discrete
  branch — in practice the `stag_window` stagnation exit, on steps that had not
  converged anyway.
