# Cost of the implicit solve

The implicit midpoint step is solved by Anderson-accelerated Picard iteration
(`step_anderson!`), and every iteration re-evaluates the projection, the
log-gradient and — for Landau — the $O(N^2)$ collision sum. Wall time is
therefore set by the *iteration count*, not by the number of time steps, and the
iteration count turns out to depend strongly on the arithmetic precision of the
velocity update. This page records what was measured, because two reasonable
expectations are both wrong: raising `DT` does not reliably buy wall time, and
`gpu_fp32 = true` does not buy the full per-iteration speed-up it appears to.
Both turn out to be the same effect — a convergence tolerance set below what
fp32 can reach — and the last section measures what happens when it is raised.

All numbers below are the `sq_d04` configuration — `parameters_LB_sq_d04.jl`
with `collision_model = :landau`, $N = 40\,000$, mesh `bp1 = bp2` with inner
$\Delta = 0.4$, `warmstart = :euler`, `m_anderson = 8`, `tol = 1e-12`,
`abs_floor = 1e-10`, `stag_window = 30` — started from the same seed, so the runs
compared here follow the same trajectory and differ only in the stated knob.

## The quantity to think in: iterations per unit physical time

Comparing `iter/step` across different `DT` is misleading, and so is comparing
wall time across runs that cover different physical intervals. The invariant
cost measure is

```math
\text{iterations per unit } t \;=\; \frac{1}{\Delta t}\,\overline{\text{iter}},
```

which multiplied by the per-iteration time gives wall time per unit physical
time. The iteration count per step also drifts slowly with physical time (it
falls by roughly a quarter between $t \approx 0.75$ and $t \approx 1.25$, then
flattens), so windows are stated explicitly below.

## Precision sets the iteration count

Measured over `steps 501–1000` at `DT = 0.001` — the same physical window
$t \in [0.5, 1.0]$ for every run:

| arithmetic | run | `iter/step` | energy drift |
|:--|:--|--:|--:|
| fp64 (CPU) | `sq_d04_nnws` | 24.3 | $1.1\times10^{-11}$ |
| fp64 (GPU) | `sq_d04_gpu1k` | 25.5 | — |
| fp64 (GPU) | `sq_d04_gpu1k64_dt` | 24.5 | — |
| fp32 (GPU) | `sq_d04_gpu1k32` | 36.2 | $2.0\times10^{-8}$ |
| fp32 (GPU) | `sq_d04_gpu1k32_blas1` | 35.4 | — |
| fp32 (GPU) | `sq_d04_gpu1k32_dt_3070` | 35.6 | — |
| fp16 (GPU) | `sq_d04_fp16` | 106.5 | $6.3\times10^{-6}$ |

fp32 costs about **1.45×** the iterations of fp64 at this step size, and fp16
about **4.3×** — in the fp16 run every one of the 500 steps ends above a
$10^{-9}$ residual, i.e. none of them converges. The relative energy drift in
the last column is the cheapest way to identify the precision of an old run
whose `params_<suffix>.jl` predates the per-run parameter dump.

## The same `DT` increase pays off in fp64 and not in fp32

| iterations per unit $t$ | `DT = 0.001` | `DT = 0.002` | `DT = 0.005` |
|:--|--:|--:|--:|
| fp64 | 24.5k ($t\,0.5\!\to\!1$) | **12.7k** ($t\,1\!\to\!2$) | — |
| fp32 | 35.6k ($t\,0.5\!\to\!1$) | 23.4k ($t\,1\!\to\!2$) | **20.9k** ($t\,2\!\to\!3$) |
| fp32, `abs_floor = 1e-8` | — | — | **6.4k** ($t\,2\!\to\!3$) |

The last row is the subject of
[Matching the residual target to the precision](@ref) below: the fp32 penalty is
not intrinsic to fp32, it is the cost of chasing a residual fp32 cannot
comfortably reach.

In fp64 the per-step iteration count barely moves when the step doubles (24.5 →
25.4), so the cost per unit physical time halves — the expected behaviour. In
fp32 the per-step count grows nearly in proportion to `DT` (35.6 → 46.8 →
104.4), so the saving collapses: $1.52\times$ from `0.001` to `0.002`, and only
$1.12\times$ from `0.002` to `0.005`. **Past `DT ≈ 0.002`, a larger step buys
essentially nothing in fp32.**

## Why: the residual target and the stagnation exit

`step_anderson!` accepts an iterate when

```math
\lVert r \rVert_2 < \text{eff\_tol} = \max\big(\texttt{tol}\cdot\lVert v\rVert_2,\ \texttt{abs\_floor}\big),
```

and otherwise gives up when the stagnation check fires: every `stag_window`
iterations the best residual is compared against its value one window earlier,
and a relative improvement below `stag_rel_tol` exits with the best iterate.

For this configuration $\lVert v \rVert_2 = \sqrt{2EN} = 284.8$ (with
$E = 1.0139$ and unit total mass), so

```math
\text{eff\_tol} = \max(1\times10^{-12}\cdot 284.8,\ 1\times10^{-10}) = 2.85\times10^{-10},
```

set by `tol`, with `abs_floor` sitting below it and never binding. Observed
residuals at convergence are $\approx 2.5\times10^{-10}$, which confirms the
model. Note that the `sq_d04` preset lowers `abs_floor` from the
`step_anderson!` signature default of `1e-7` to `1e-10`, which is what disables
the floor as a safety net.

How each step actually terminates — `iter` being an exact multiple of
`stag_window = 30` identifies a stagnation exit:

| run | window | stagnation exits | $\lVert r\rVert > 10^{-9}$ | max $\lVert r\rVert$ |
|:--|:--|--:|--:|--:|
| fp64, `DT = 0.002` | `1001–1500` | 46/500 (9%) | 45/500 | $-$ |
| fp32, `DT = 0.002` | `1001–1500` | 88/500 (18%) | 60/500 | $1.4\times10^{-4}$ |
| fp32, `DT = 0.002` | `1401–1500` | 15/100 (15%) | 13/100 | $5.2\times10^{-5}$ |
| fp32, `DT = 0.005` | `1501–1700` | **200/200 (100%)** | 35/200 | $1.9\times10^{-4}$ |

At `DT = 0.005` in fp32 *no step reaches the tolerance*: the iteration counts are
exactly $\{90\times142,\ 120\times29,\ 150\times20,\ 180\times9\}$, so every step
grinds through at least three stagnation windows and exits on the check. That is
the whole of the missing speed-up — the solve is paying for iterations that no
longer reduce the residual.

## Matching the residual target to the precision

Raising `abs_floor` above `tol`$\cdot\lVert v\rVert_2$ hands the exit criterion
back to the floor. Three values, same 200 steps from the same step-1500
checkpoint, `DT = 0.005`, fp32, each validated point by point against the fp64
`DT = 0.001` trajectory:

| `abs_floor` | `iter/step` | total iterations | stagnation exits | worst $\lvert\Delta S\rvert$ | energy drift | wall |
|:--|--:|--:|--:|--:|--:|--:|
| `1e-10` (preset) | 104.4 | 20\,880 | 200/200 | $1.84\times10^{-6}$ | $3.0\times10^{-8}$ | 636.0 s |
| `1e-8` | **32.1** | **6\,430** | 36/200 | $1.85\times10^{-6}$ | $3.0\times10^{-8}$ | 298.1 s |
| `1e-7` (signature default) | 28.9 | 5\,770 | 35/200 | $1.83\times10^{-6}$ | $3.0\times10^{-8}$ | 273.5 s |

**The accuracy column does not move.** $1.84\times10^{-6}$ is the `DT = 0.005`
time-step error against the `DT = 0.001` reference; it is identical whether the
solve stops at $3\times10^{-10}$ or at $2\times10^{-4}$, so the 14\,450 extra
iterations the tight floor buys are spent entirely below the discretisation
error. The iteration histogram tells the same story from the other side: at
`1e-10` the counts are $\{90, 120, 150, 180\}$ — nothing but stagnation windows —
while at `1e-8` they fall back to 10–19, a healthy Anderson convergence.

`1e-8` is the better of the two loose values for production: it captures
$3.25\times$ of the available $3.62\times$ speed-up while keeping an order of
magnitude of margin over the signature default, which matters over the thousands
of steps a relaxation run takes rather than the 200 measured here.

With it, fp32 at `DT = 0.005` costs 6.4k iterations per unit physical time —
below fp64's 12.7k at `DT = 0.002` — so the configuration ranking in the
previous sections is a statement about the *default* `abs_floor`, not about
precision as such.

At production scale the saving is larger still, because the iteration count also
falls as the solution approaches equilibrium and a looser floor lets it: carrying
the same trajectory from $t = 3$ to $t = 30$ (5\,400 steps, `snap_every = 50`)
took 88\,205 iterations, **16.3 per step** — 8 to 16 per step beyond $t \approx 8$ —
and 3\,794 s end to end on one RTX 3070, roughly a quarter of which is snapshot
overhead rather than iteration (each snapshot step writes a checkpoint, uploads
it, dumps 40\,000 particle rows and renders an `fs_diag` PNG through CairoMakie).
The structure-preserving properties survive the looser floor over all 7\,100
steps: relative energy drift $2.9\times10^{-7}$, momentum drift
$3.4\times10^{-11}$, and the discrete entropy is monotone at every single step.

## Accuracy is not the constraint

The `DT = 0.005` fp32 run was validated point by point against the fp64
`DT = 0.001` trajectory over the same interval (200 comparisons, $t\,2\to3$):

| quantity | agreement |
|:--|:--|
| entropy, worst of 200 points | $\lvert\Delta S\rvert \le 1.8\times10^{-6}$ (0.006% of the remaining entropy gap) |
| entropy at $t = 3.0$ | $2.820856363$ vs $2.820856036$ |
| energy drift | $3.0\times10^{-8}$ relative |
| $T_1,\,T_2$ at $t = 3.0$ | agree to $2\times10^{-6}$ |

Even the fp16 run tracks the entropy to $5\times10^{-6}$ at $t = 5$. Time-step
and precision errors are far below anything the diagnostics resolve; what
degrades with a looser configuration is the *solvability* of the fixed point, not
the solution.

## Consequences

- **Do not tune `DT` upward past `0.002` in fp32** expecting wall time back. In
  fp64 the step size behaves as ordinary second-order theory suggests.
- **fp32 is still the right default on consumer GPUs, by less than it looks.**
  Same RTX 4090, `DT = 0.002`, 500 steps covering $t\,1\to2$: fp32 704.5 s over
  22\,508 iterations (31.3 ms/iteration) against fp64 1767.2 s over 12\,707
  iterations (139.1 ms/iteration). The per-iteration ratio is $4.44\times$, but
  the iteration-count advantage of fp64 pulls the ratio *per unit physical time*
  down to $2.51\times$. On hardware whose fp64 throughput is not crippled, fp64
  is expected to win outright — the crossover sits near a $2.5\times$
  per-iteration penalty, not $4.4\times$.
- **Set the residual target before touching the step size.** Chasing
  $2.85\times10^{-10}$ is what makes large steps unaffordable in fp32; with
  `abs_floor = 1e-8` the same `DT = 0.005` step costs $3.25\times$ fewer
  iterations at identical accuracy. Prefer `abs_floor` over `tol` for this:
  it is an absolute quantity, so it does not drift with $\lVert v \rVert_2$ as
  the distribution evolves, leaving `tol` untouched keeps fp64 reference runs
  bit-identical, and capping the effective tolerance past the noise floor is
  precisely what the parameter is documented to be for.
- **An fp32 run at the preset `abs_floor` spends most of its time below its own
  discretisation error.** Any configuration change should be checked against a
  trajectory computed at a tighter `DT` *and* against the iteration histogram:
  a count that is an exact multiple of `stag_window` means that step never
  converged, and a run where most steps look like that is overpaying.

## Reproducing

Each measurement above is a resume of one checkpoint with one knob changed, so a
comparison costs a few hundred steps rather than a whole run. The pattern:
rename a checkpoint and the conservation CSV to a fresh suffix (the checkpoint
carries the full state and histories, and nothing in it depends on the suffix),
then resume with the knob overridden:

```sh
julia main.jl parameters_LB_sq_d04.jl --collision_model=landau \
    --use_gpu=true --gpu_fp32=true --snap_every=25 \
    --suffix=<fresh-suffix> --N_STEPS=1700 --DT=0.005 \
    --abs_floor=1e-8 --resume=1500
```

The `iter` and `residual` columns of `conservation_history_<suffix>.csv` carry
everything needed: `iter` sums give iterations per unit $t$, and
`iter % stag_window == 0` separates stagnation exits from converged steps.
Temperature anisotropy is not in that CSV — it has to come from the second
moments of `particle_snapshots_<suffix>.csv`, which `main.jl` writes locally and
does not upload.
