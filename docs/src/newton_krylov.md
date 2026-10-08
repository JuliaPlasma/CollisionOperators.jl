# Newton–Krylov vs. Anderson

`step_newton!` solves the same implicit step as
`step_anderson!`, the fixed point
$v = \mathcal{G}(v)$ of the Picard map, but by Jacobian-free Newton–Krylov
(JFNK) on the residual

```math
F(v) = v - \mathcal{G}(v) = 0 .
```

It is selected with `--solver=newton`. This page records whether it reduces the
cost of the solve. **It does not**: in FP32 it needs 1.4–2.0× more Picard-map
evaluations than Anderson at every residual target, and in FP64 it comes out
roughly even. Anderson with `abs_floor = 1e-8` remains the production setting.

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

| variant | evaluations |
|:--|--:|
| default ($\eta_\text{max} = 0.9$, `fd_rel` $= 3.45\times10^{-4}$) | 12 441 |
| tight forcing, $\eta_\text{max} = 0.1$ | 13 028 |
| `fd_rel = 1e-2` | 15 898 |
| `fd_rel = 1e-5` | 11 452 |

Changing the finite-difference step by three decades moves the count by about
±20%, and Newton stays far behind Anderson's 6 188 throughout. The tight forcing
term makes Newton slightly worse.

## Why Newton loses here

The verbose traces of the first steps after resume show the same pattern in
every variant:

```text
nk=9   evals=25  gmres=1  η=0.1  λ=0.5  ‖F‖=1.12e-6
nk=10  evals=27  gmres=1  η=0.1  λ=1.0  ‖F‖=8.60e-7
nk=11  evals=30  gmres=1  η=0.1  λ=0.5  ‖F‖=3.16e-7
```

1. **GMRES stops after one iteration.** The Jacobian is
   $I - \tfrac{\Delta t}{2}\,\partial\dot v/\partial v = I - O(\Delta t)$, so one
   Krylov vector already meets the forcing tolerance. A Newton step is then one
   scaled Picard step that costs **two** evaluations (a Jacobian product plus the
   trial point), where Anderson spends one.
2. **The linear model does not predict the nonlinear step.** GMRES reports a
   linear residual reduction of $\eta = 0.1$, yet $\lVert F\rVert$ falls only
   2–3× per Newton step, and half the steps need $\lambda = 0.5$. Convergence is
   linear, not quadratic. Because the mismatch barely depends on `fd_rel`, it is
   not finite-difference noise. The map itself responds to a finite step
   differently than its local derivative predicts. Plausible sources are the
   $1/f$ behavior of the log-entropy gradient near $f_s \approx 0$, the clamped
   or $\lvert f\rvert$-guarded integrand, and the Gonzalez quotient. The same
   effect shows in FP64, at a lower level.
3. **Anderson keeps what it learns.** Every evaluation adds a secant pair to
   a window of `m = 8`. For a linear problem Anderson acceleration is equivalent
   to GMRES (Walker & Ni, 2011), so on this nearly linear map it already behaves
   like one long GMRES over the whole solve. Newton restarts its Krylov space at
   every step and discards it.

The "accurate initial guess" that Newton needs does not change this. The
explicit-Euler predictor starts at $\lVert F_0\rVert \approx 5\times10^{-4}$, and
the overhead appears near the solution, not far from it.

## Consequences

- Keep **Anderson with `abs_floor = 1e-8`** for FP32 production. Newton is not
  a cheaper route to the same solution at any floor tried.
- Quadratic convergence would need a residual map that is smooth at the scale
  of a Newton step, and probably an analytic Jacobian–vector product (forward
  mode through the pair sum) rather than a finite difference. Without those,
  Newton's per-step overhead outweighs what it gains.
- The FP64 histogram suggests one direction that could pay: a **hybrid** that
  runs Anderson and switches to Newton only when Anderson stagnates. Newton has
  no 68–150-evaluation tail, and that tail is where Anderson loses its FP64
  advantage. This is untested.

## Reproducing

```sh
# on a pod built from the collisionoperators-gpu template
bash /root/run_nk_ab.sh       # 10 runs, ~80 min on an RTX 3070
bash /root/run_nk_ablate.sh   # 3 runs, ~20 min, waits for the first script
```

Each run is a resume of the step-1500 checkpoint, for example:

```sh
julia main.jl parameters_LB_sq_d04.jl --collision_model=landau \
    --use_gpu=true --gpu_fp32=true --solver=newton --abs_floor=1e-8 \
    --snap_every=25 --suffix=<fresh-suffix> --N_STEPS=1700 --DT=0.005 \
    --resume=1500
```
