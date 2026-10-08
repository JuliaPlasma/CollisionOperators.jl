# Newton–Krylov vs. Anderson

`step_newton!` solves the same implicit step as
`step_anderson!`, the fixed point
$v = \mathcal{G}(v)$ of the Picard map, but by Jacobian-free Newton–Krylov
(JFNK) on the residual

```math
F(v) = v - \mathcal{G}(v) = 0 .
```

It is selected with `--solver=newton`. This page records whether it reduces the
cost of the solve. **In FP32 at the production tolerance it does not.**

- With the default finite-difference step, Newton needs 1.4–2.0× more Picard-map
  evaluations than Anderson. A Taylor test of the residual map traces this to
  the step size: it is far too large along one stiff direction of the map (see
  [Why Newton lost](#Why-Newton-lost)).
- With a correctly sized step, Newton costs 41% less than before and ties
  Anderson at best: 1.28× at `DT = 0.005`, 0.99× at `0.01`, 1.11× at `0.02` (see
  [DT sweep](#DT-sweep-with-a-correctly-sized-step)).
- In FP64 the two come out roughly even.

Anderson with `abs_floor = 1e-8` remains the production setting. The sweep also
shows that Anderson's cost per unit physical time **falls** by 40% from
`DT = 0.005` to `DT = 0.02` without measurable loss of accuracy.

## The method

`newton_krylov!` is generic over a residual closure
and plain vectors. Each Newton step solves $J(v_k)\,\delta = F(v_k)$ with an
unrestarted GMRES (`gmres!`, modeled on Krylov.jl's
`gmres.jl` but with no dependency on it) and takes $v_{k+1} = v_k - \lambda\delta$.

- **Jacobian products** use a forward difference,
  $J u \approx \big(F(v + h u) - F(v)\big)/h$ with
  $h = \texttt{fd\_rel}\cdot\max(\lVert v\rVert, 1)/\lVert u\rVert$. Each product is
  one Picard-map evaluation. `fd_rel` defaults to $\sqrt{\varepsilon}$ of the
  collision kernel's precision: $3.45\times10^{-4}$ for FP32, which also keeps the
  probe above the Float32 cast of $v$.
- **Forcing term**: Eisenstat–Walker choice 2,
  $\eta_k = 0.9\,(\lVert F_k\rVert/\lVert F_{k-1}\rVert)^2$, safeguarded, capped at
  `nk_eta_max = 0.9`, and kept above $\tfrac12\,\texttt{tol}/\lVert F_k\rVert$.
- **Globalisation**: Armijo backtracking on $\lVert F\rVert$, halving $\lambda$. If
  8 halvings fail, the residual has reached the noise floor of $F$ and the step
  exits with the best iterate seen.

The stopping rule is the same as Anderson's,
$\lVert F\rVert < \max(\texttt{tol}\cdot\lVert v\rVert, \texttt{abs\_floor})$.
Both solvers write **Picard-map evaluations** to the `iter` column, counting every
Jacobian product and every line-search trial, so the counts compare directly.
The cost per evaluation is the same for both (one $O(N^2)$ pair sum), so wall
time follows the evaluation count.

## Setup

All runs used one RTX 3070 (Community, host suffix `-64411cb6`) and the same commit.
They resume the same checkpoint, `sq_d04_gpu1k32_dt_3070_2026-10-03` step 1500
($t = 2$), and run at `DT = 0.005`, so they cover the setup of
[Cost of the implicit solve](implicit_solve_cost.md):

- **FP32 sweep**: 200 steps (to $t = 3$) for each solver at
  `abs_floor` $\in \{10^{-7}, 10^{-8}, 10^{-9}, 10^{-10}\}$.
- **FP64 control**: 50 steps at `abs_floor = 1e-10`. It separates a method
  failure from FP32 noise in the finite-difference Jacobian.
- **Ablation**: Newton at `1e-8` with tight forcing, and with the finite-difference
  step $\times 30$ and $\div 30$.

The scripts are `logs/run_nk_ab.sh` and `logs/run_nk_ablate.sh` in the Runpod
workspace. Results are under `mpcdf-s3:collision-operators/nkab-*-2026-10-08`.

## Results

### FP32, 200 steps

| `abs_floor` | Newton evals | Anderson evals | Newton / Anderson | steps missing target (N / A) | median evals/step (N / A) |
|:--|--:|--:|--:|--:|--:|
| `1e-7`  |  8 957 |  6 271 | 1.43 | 41 / 33 | — |
| `1e-8`  | 12 441 |  6 188 | **2.01** | 48 / 28 | 44 / 14 |
| `1e-9`  | 18 495 |  9 654 | 1.92 | 64 / 46 | — |
| `1e-10` | 25 352 | 20 040 | 1.27 | 200 / 200 | — |

At `1e-10` neither solver reaches the target on any step, because it is below
the FP32 noise floor. Anderson exits on its stagnation window, and Newton's line
search fails. The Anderson counts reproduce the earlier runs on another 3070
(6 430 at `1e-8`, 5 770 at `1e-7`, 20 880 at `1e-10`) to within a few percent.

### Accuracy is unaffected

| check | result |
|:--|:--|
| $\max_t \lvert S_\text{Newton} - S_\text{Anderson}\rvert$, FP32 `1e-8` / `1e-7` | $9.1\times10^{-8}$ / $6.9\times10^{-8}$ |
| $S(t{=}3)$, FP32 `1e-8`: Newton / Anderson / FP64 `DT = 0.001` reference | $2.8208565$ / $2.8208564$ / $2.8208560$ |
| entropy decreases, any run | none |
| energy at $t = 3$, Newton vs Anderson | agree to $1.5\times10^{-10}$ relative |

The differences between the solvers are at least 5× smaller than the
`DT = 0.005` time-step error ($\le 1.8\times10^{-6}$), which is the error the
comparison is about.

### FP64 control, 50 steps, `abs_floor = 1e-10`

| | Newton | Anderson |
|:--|--:|--:|
| total evaluations | **1 996** | 2 114 |
| median evals/step | 32 | 19 |
| steps missing $3\times10^{-10}$ | 23 | 11 |
| worst residual | $2.4\times10^{-3}$ | $4.7\times10^{-5}$ |
| stagnation-window exits (68–150 evals) | — | 11 |

In FP64 the totals are even, but they come about differently. Anderson is
cheaper on a typical step and pays for a tail of stagnation exits. Newton costs
more on every step and has no such tail. It does, however, leave one step at
$2.4\times10^{-3}$, where its line search gave up early.

### Ablation, FP32, `abs_floor = 1e-8`

The finite-difference step is $h = \texttt{fd\_rel}\cdot\lVert v\rVert$ with
$\lVert v\rVert \approx 285$, so the column $h$ is the actual perturbation norm.

| variant | $h$ | evaluations |
|:--|--:|--:|
| default ($\eta_\text{max} = 0.9$, `fd_rel` $= 3.45\times10^{-4}$) | $9.8\times10^{-2}$ | 12 441 |
| tight forcing, $\eta_\text{max} = 0.1$ | $9.8\times10^{-2}$ | 13 028 |
| `fd_rel = 1e-2` | $2.9$ | 15 898 |
| `fd_rel = 1e-5` | $2.9\times10^{-3}$ | 11 452 |

Smaller steps do better, but every step tried is at least $3\times10^{-3}$.
The Taylor test below shows that this whole range is too coarse to resolve the
Jacobian along the direction that matters. Tight forcing makes Newton slightly
worse.

## Taylor test of the residual map

To see whether a Newton model of $F$ can be accurate at all, the remainder

```math
r(\varepsilon) = \bigl\lVert F(v + \varepsilon u) - F(v) - \varepsilon\, J u \bigr\rVert ,
\qquad \lVert u \rVert = 1 ,
```

was measured at two points of the step from the same step-1500 state: the
explicit-Euler predictor ($\lVert F\rVert = 5.7\times10^{-4}$) and the converged
solution ($\lVert F\rVert = 4.2\times10^{-10}$). The run used CPU FP64, `DT = 0.005`,
with $Ju$ from a central difference at $h = 10^{-6}$. A smooth map gives
$r \propto \varepsilon^2$ (slope 2), and a map with kinks at that scale gives
slope 1. Two directions were probed: a random one, and $u = F/\lVert F\rVert$,
which is where GMRES points the first Newton step.

At the solution:

| $\varepsilon$ | $r/\varepsilon^2$, random $u$ | $r/\varepsilon^2$, $u = F/\lVert F\rVert$ | relative model error $r/(\varepsilon\lVert Ju\rVert)$, $u = F/\lVert F\rVert$ |
|--:|--:|--:|--:|
| $10^{-1}$ | 0.085 | 23 | 63% |
| $10^{-2}$ | 0.095 | 216 | 61% |
| $10^{-3}$ | 0.092 | 2 092 | 59% |
| $10^{-4}$ | 0.100 | 3 392 | 9.5% |
| $10^{-5}$ | 0.098 | 7 224 | 2.0% |
| $10^{-6}$ | 0.110 | 7 826 | 0.22% |
| $10^{-7}$ | — | 8 473 | 0.024% |

The predictor gives the same picture: curvature about $10^3$ along
$F/\lVert F\rVert$, 0.09 in a random direction.

- **The map is smooth.** Both directions reach slope 2 before round-off takes
  over near $r \approx 10^{-12}$. There are no kinks.
- **One direction is stiff and strongly curved.** In a random direction
  $\lVert Ju\rVert = 1.000$ and the curvature is $\approx 0.09$, so there
  $J \approx I$, as $I - O(\Delta t)$ predicts. Along $F/\lVert F\rVert$ at the
  solution, $\lVert Ju\rVert = 3.57$ and the curvature is $\approx 8\times10^3$,
  five orders of magnitude larger. The linear model holds to 2% only for steps up
  to $\varepsilon \approx 10^{-5}$, and it is about 60% wrong from $10^{-3}$ upward.
- **The residual lines up with that direction.** The direction picked at random
  is benign, but the one Newton actually uses is the stiff one.
- **It is not the Gonzalez term.** With `use_gonzalez = false` the numbers along
  $F/\lVert F\rVert$ agree to three digits. Gonzalez only lifts the round-off
  floor, from about $10^{-13}$ to $10^{-12}$. The stiff direction comes from the
  base map. A plausible but unverified source is particles in low-density
  regions, where $\nabla f/f$ is large and sensitive to position.

The script is `scripts/taylor_test.jl`.

## Why Newton lost

The verbose traces of the first steps after resume show the same pattern in
every FP32 variant:

```text
nk=9   evals=25  gmres=1  η=0.1  λ=0.5  ‖F‖=1.12e-6
nk=10  evals=27  gmres=1  η=0.1  λ=1.0  ‖F‖=8.60e-7
nk=11  evals=30  gmres=1  η=0.1  λ=0.5  ‖F‖=3.16e-7
```

1. **The Jacobian products were wrong along the stiff direction.** Every FP32
   run used $h \ge 3\times10^{-3}$, where the table above puts the linear model
   about 60% off along $F/\lVert F\rVert$. GMRES then solves an inaccurate linear
   system. It reports a reduction of $\eta = 0.1$, but $\lVert F\rVert$ actually
   falls only 2–3× per Newton step, and half the steps need $\lambda = 0.5$.
   Convergence is linear rather than quadratic. The FP64 control used
   $h = \sqrt{\varepsilon_{64}}\,\lVert v\rVert \approx 4\times10^{-6}$, where the
   model is about 1% off, and there Newton already matched Anderson.
2. **GMRES stops after one iteration.** Because $J \approx I$ almost everywhere,
   one Krylov vector meets the forcing tolerance. A Newton step then costs
   **two** evaluations (a Jacobian product plus the trial point), where an
   Anderson iteration costs one.
3. **Anderson keeps what it learns.** Every evaluation adds a secant pair to a
   window of `m = 8`. For a linear problem Anderson acceleration is equivalent to
   GMRES (Walker & Ni, 2011), so on this nearly linear map it already behaves
   like one long GMRES over the whole solve. Newton restarts its Krylov space at
   every step.

The stiff direction also explains Anderson's own stagnation tail. Along it the
Picard map amplifies errors by roughly $\lVert Ju\rVert - 1 \approx 2.6$ instead
of shrinking them. Anderson has to learn that from secant pairs, and the strong
curvature corrupts them. Newton inverts $J$ instead of learning it.

An initial guess matters for a different reason than first assumed. The
explicit-Euler predictor starts at $\lVert F_0\rVert \approx 6\times10^{-4}$,
60× outside the $\sim10^{-5}$ range where the linear model holds. Every solve
therefore opens with a damped phase in which Newton has no advantage. Quadratic
convergence can only help between about $10^{-5}$ and the target.

*An earlier version of this page attributed point 1 to kinks in the residual map
and ruled out finite-difference error because the ablation barely depended on
`fd_rel`. The Taylor test shows the map is smooth, and that every ablated step
was too large.*

## DT sweep with a correctly sized step

A rerun with an absolute finite-difference step, `--nk_fd_rel=3.5e-8`, gives
$h \approx 10^{-5}$. All runs used FP32, `abs_floor = 1e-8`, one RTX 3070 (host
`-64411cb6`) and the same step-1500 checkpoint. Every run covers the same
physical span $t = 2 \to 3$, so the totals are cost per unit time. Accuracy is the
largest entropy deviation from the FP64 `DT = 0.001` reference
(`sq-d04-nnws`) at matching times. The checkpoint itself already sits
$2\times10^{-7}$ off that reference at $t = 2$.

| `DT` | steps | Newton evals | Anderson evals | Newton / Anderson | steps missing target (N / A) | median evals/step (N / A) | $\max_t\lvert\Delta S\rvert$ (N / A) |
|:--|--:|--:|--:|--:|--:|--:|--:|
| 0.005 | 200 | 7 355 | 5 752 | 1.28 | 55 / 25 | 30 / 15 | $1.80\times10^{-6}$ / $1.84\times10^{-6}$ |
| 0.01  | 100 | 4 438 | 4 501 | **0.99** | 36 / 30 | 39 / 23 | $2.06\times10^{-6}$ / $2.24\times10^{-6}$ |
| 0.02  |  50 | 3 820 | 3 426 | 1.11 | 35 / 21 | 61 / 66 | $1.20\times10^{-6}$ / $1.92\times10^{-6}$ |

A check run at $h \approx 10^{-6}$ (`--nk_fd_rel=3.5e-9`, `DT = 0.005`) needed
7 572 evaluations, so $h \approx 10^{-5}$ already resolves the stiff direction.
The wall times were 295 / 269 s, 185 / 189 s and 151 / 149 s (Newton / Anderson),
including Julia startup and snapshot I/O.

- **The step size was the problem.** At `DT = 0.005` the correctly sized step
  cuts Newton from 12 441 to 7 355 evaluations (−41%), and the gap to Anderson
  shrinks from 2.01× to 1.28×.
- **Newton still does not win.** It ties at `DT = 0.01` and is 11% behind at
  `0.02`. It misses the target on more steps than Anderson at every `DT`: its line
  search gives up in FP32 noise where Anderson's stagnation window keeps going.
- **Anderson does not degrade with `DT`.** The hypothesis behind the sweep,
  that more stiff directions at larger `DT` would overwhelm Anderson's window,
  is wrong at this tolerance. Anderson's cost per unit time falls from 5 752 to
  3 426 (−40%) between `DT = 0.005` and `0.02`. Its mean per step rises from 28.8
  to 68.5 (2.4×), less than the 4× reduction in steps. The median rises more
  (15 → 66), because at `DT = 0.005` a few slow steps carry much of the total.
- **Accuracy does not degrade with `DT` either.** $\max_t\lvert\Delta S\rvert$
  stays at $1.2$–$2.2\times10^{-6}$ for every `DT` and both solvers. It does not
  scale like $\Delta t^2$, so it is set by something other than the time step.
  The source is not identified; the FP32 solve at `abs_floor = 1e-8` is one
  candidate. This holds for one interval,
  $t = 2 \to 3$, and one diagnostic. It contradicts the advice in
  [Cost of the implicit solve](implicit_solve_cost.md) not to raise `DT` past
  `0.002` in FP32, which was measured at the preset `abs_floor = 1e-10`.

## Consequences

- Keep **Anderson with `abs_floor = 1e-8`** for FP32 production. A correctly
  sized Newton step brings Newton level with Anderson but not ahead of it.
- If Newton is used, set the finite-difference step in **absolute** terms,
  $h \sim 10^{-5}$, i.e. `--nk_fd_rel=3.5e-8`. The `sqrt(eps)` default is about
  $10^4$ times too large for the stiff direction. It should become the default
  before anything else is built on `--solver=newton`.
- Forward-mode AD would compute the Jacobian product exactly, but $h \approx
  10^{-5}$ is already accurate enough ($h = 10^{-6}$ changes nothing), and AD would
  cost 2–3× more per product. It is not worth building for this problem.
- **Larger `DT` is the real saving, for either solver.** `DT = 0.02` with Anderson
  costs 40% less per unit time than `DT = 0.005` at the same measured accuracy.
  Before adopting it, it needs a longer interval and a check against a tighter-`DT`
  FP64 trajectory beyond $t = 3$.
- Still untested: tight tolerances in FP64, where Newton's quadratic window is
  longest, a predictor accurate enough to start inside the $\sim10^{-5}$ linear
  range, and an Anderson-to-Newton hybrid for stagnating steps.

## Reproducing

```sh
# on a pod built from the collisionoperators-gpu template
bash /root/run_nk_ab.sh       # 10 runs, ~80 min on an RTX 3070
bash /root/run_nk_ablate.sh   # 3 runs, ~20 min, waits for the first script
bash /root/run_nk_dt.sh       # 7 runs, DT sweep, ~26 min
```

The Taylor test runs locally on CPU (about 1 s per evaluation at $N = 40\,000$
with 8 threads, 58 evaluations):

```sh
julia -t auto --project=. scripts/taylor_test.jl . <checkpoint_step1500.jls> [use_gonzalez]
```

Each run is a resume of the step-1500 checkpoint, for example:

```sh
julia main.jl parameters_LB_sq_d04.jl --collision_model=landau \
    --use_gpu=true --gpu_fp32=true --solver=newton --abs_floor=1e-8 \
    --snap_every=25 --suffix=<fresh-suffix> --N_STEPS=1700 --DT=0.005 \
    --resume=1500
```
