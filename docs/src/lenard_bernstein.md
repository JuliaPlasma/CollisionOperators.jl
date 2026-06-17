# Lenard–Bernstein operator (2D)

The Lenard–Bernstein (LB) operator is a linear model collision operator that
relaxes a distribution towards a Maxwellian while conserving mass, momentum and
energy. This page documents the structure-preserving *particle* discretisation
that lives at the
[repository root](https://github.com/JuliaPlasma/CollisionOperators.jl)
(`main_LB.jl`, `functions.jl`, `MantisWrappers.jl`, `Parameters.jl`,
`parameters_LB*.jl`). The construction follows Jeyakumar et al. (2024) for the
Landau operator, specialised to the LB kernel and written in 2D velocity space.

## Continuous operator

For a single species with density $f_s(\mathbf{v})$ and collision frequency
$\nu$, the operator is written in advective (continuity) form

```math
\partial_t f_s = \nabla_{\mathbf v}\cdot\!\big(\mathbf a\, f_s\big),
\qquad
\mathbf a = \nu\left(\frac{\nabla_{\mathbf v} f_s}{f_s} + \mathbf A + B\,\mathbf v\right),
```

where $\mathbf A\in\mathbb{R}^2$ and $B\in\mathbb{R}$ are Lagrange multipliers
fixed by the conservation constraints. The fixed point of the flow is the
Maxwellian for which $\nabla\log f_s + \mathbf A + B\mathbf v = 0$, and the
discrete entropy $H_h = -\int f_s\log f_s\,\mathrm d\mathbf v$ increases
monotonically.

## Particle representation

The density is carried by a weighted Dirac measure,

```math
f_s(\mathbf v) \approx \sum_\alpha w_\alpha\,\delta(\mathbf v - \mathbf v_\alpha),
```

with fixed weights $w_\alpha$ and moving markers $\mathbf v_\alpha$. By
continuity each marker is advected by $\dot{\mathbf v}_\alpha = -\mathbf a$:

```math
\dot{\mathbf v}_\alpha
  = -\nu\left(\frac{\nabla f_s(\mathbf v_\alpha)}{f_s(\mathbf v_\alpha)}
              + \mathbf A + B\,\mathbf v_\alpha\right).
```

In the pure-heat limit $\mathbf A = B = 0$ this is drift down the log-density
gradient — diffusion that increases $H_h$.

## Finite-element density and the gradient

To evaluate $\nabla f_s / f_s$ pointwise, a smooth density is reconstructed from
the markers by an $L^2$ projection onto a tensor-product B-spline space $X^0$
(degree `P_DEG`, regularity `K_REG`) on an anisotropic, possibly non-uniform
mesh with breakpoints `bp1`, `bp2`. The projection solves the mass system

```math
M\,\mathbf c = \mathbf b,\qquad
b_i = \sum_\alpha w_\alpha\,\varphi_i(\mathbf v_\alpha),
```

with the mass matrix prefactored once (`l2_project!`). The per-marker
log-gradient $\mathbf g_\alpha = \nabla f_s(\mathbf v_\alpha)/f_s(\mathbf v_\alpha)$
is then evaluated from the coefficients (`eval_loggrad_at_particles!`). Each
component is clamped to $\pm$`G_MAX` so that a Gibbs undershoot driving
$f_s\to 0^+$ cannot make $1/f_s$ blow up the implicit solve; conservation stays
exact because the multipliers are solved *from* the clamped gradient.

## Exact discrete conservation

The multipliers $\mathbf A=(A_1,A_2)$ and $B$ are chosen so that the discrete
momentum $\sum_\alpha w_\alpha\dot{\mathbf v}_\alpha$ and energy
$\sum_\alpha w_\alpha\,\mathbf v_\alpha\!\cdot\!\dot{\mathbf v}_\alpha$ vanish.
With the raw moments

```math
n = \sum_\alpha w_\alpha,\quad
U_k = \sum_\alpha w_\alpha v_{\alpha k},\quad
Q = \sum_\alpha w_\alpha |\mathbf v_\alpha|^2,
```

and $\mathbf S_g = \sum_\alpha w_\alpha \mathbf g_\alpha$,
$P = \sum_\alpha w_\alpha\,\mathbf v_\alpha\!\cdot\!\mathbf g_\alpha$, this is the
symmetric $3\times3$ system (`compute_drift_multipliers`)

```math
\begin{bmatrix} n & 0 & U_1 \\ 0 & n & U_2 \\ U_1 & U_2 & Q \end{bmatrix}
\begin{bmatrix} A_1 \\ A_2 \\ B \end{bmatrix}
= -\begin{bmatrix} S_{g,1} \\ S_{g,2} \\ P \end{bmatrix}.
```

Solving it before each velocity update guarantees momentum and energy are
conserved to round-off independent of mesh resolution or time step.

## Time integration

The markers are advanced with the **implicit midpoint** rule, which preserves
the conservation structure. Each step solves the fixed-point map

```math
\mathbf v^{n+1} = \mathbf v^{n}
  + \Delta t\,\dot{\mathbf v}\!\left(\tfrac12(\mathbf v^{n}+\mathbf v^{n+1})\right),
```

by Picard iteration, optionally accelerated by Anderson mixing
(`step_anderson!`) with damping decay and a stagnation-aware early exit. The
projection–gradient–multiplier–update sequence is re-evaluated at the midpoint
inside every iteration.

## Diagnostics

Streamed to `conservation_history_<suffix>.csv` every step:

- entropy $H_h = -\int f_s\log f_s\,\mathrm d\mathbf v$ (`compute_entropy`),
- energy and momentum (should be flat to round-off),
- inner-iteration count and residual $\lVert G(\mathbf v)-\mathbf v\rVert_2$,
- projection error $\lVert f_s - f_p\rVert_2$ against the element-constant
  histogram density (`compute_fs_minus_fp_l2`),
- negative-part $L^1$ norm $\int\max(-f_s,0)\,\mathrm d\mathbf v$ probing Gibbs
  undershoot (`compute_negative_part_l1`).

## Running

```sh
julia --project=. main_LB.jl parameters_LB2D_v3.jl
```

See the
[README](https://github.com/JuliaPlasma/CollisionOperators.jl#run)
for presets, overrides, checkpoint/resume and output formats.

## References

- A. Jeyakumar, M. Kraus, et al., *Structure-preserving particle methods for the
  Landau collision operator using the metriplectic framework* (2024).
