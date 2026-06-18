# Landau operator (2D, Gonzalez discrete gradient)

The Landau collision operator is the small-angle (Fokker–Planck) limit of the
Boltzmann operator for Coulomb interactions. It conserves mass, momentum and
energy and dissipates entropy (H-theorem), a *metriplectic* structure. This page
documents the structure-preserving *particle* discretisation that lives at the
[repository root](https://github.com/JuliaPlasma/CollisionOperators.jl)
(`main_Gonzalez.jl`, `functions.jl`, `MantisWrappers.jl`, `Parameters.jl`,
`parameters_*.jl`), advanced in time by the **Gonzalez discrete gradient** so
that the conservation laws and entropy production hold *exactly* at the discrete
level. The construction follows Jeyakumar et al. (2024), written here in 2D
velocity space.

## Continuous operator

For a single species with density $f(\mathbf v)$ the Landau operator is

```math
\partial_t f = \nabla_{\mathbf v}\cdot\!\int
  \mathsf{Q}(\mathbf v-\mathbf v')
  \big(f(\mathbf v')\,\nabla_{\mathbf v} f(\mathbf v)
     - f(\mathbf v)\,\nabla_{\mathbf v'} f(\mathbf v')\big)\,
  \mathrm d\mathbf v',
```

with the symmetric, positive-semidefinite projection kernel

```math
\mathsf{Q}(\mathbf u) = \frac{1}{|\mathbf u|}
  \Big(\mathsf{I} - \frac{\mathbf u\,\mathbf u^{\!\top}}{|\mathbf u|^2}\Big),
\qquad \mathsf{Q}(\mathbf u)\,\mathbf u = 0 .
```

Because $\mathsf Q$ annihilates $\mathbf u = \mathbf v-\mathbf v'$, the relative
velocity is in its kernel, which is exactly what makes momentum and energy
conserved while the entropy $H = -\int f\log f\,\mathrm d\mathbf v$ increases
monotonically.

## Particle representation and entropy gradient

The density is carried by a weighted Dirac measure,

```math
f(\mathbf v) \approx \sum_\alpha w_\alpha\,\delta(\mathbf v - \mathbf v_\alpha),
```

with fixed weights $w_\alpha$ and moving markers $\mathbf v_\alpha$. The
metriplectic structure is driven by the gradient of a *regularised* discrete
entropy: a smooth density $f_s$ is reconstructed from the markers by an $L^2$
projection onto a tensor-product B-spline space $X^0$ (degree `P_DEG`,
regularity `K_REG`) on an anisotropic, possibly non-uniform mesh with
breakpoints `bp1`, `bp2`. The projection solves the mass system

```math
M\,\mathbf c = \mathbf b,\qquad
b_i = \sum_\alpha w_\alpha\,\varphi_i(\mathbf v_\alpha),
```

with the mass matrix prefactored once (`l2_project!`). From $f_s$ the
entropy-gradient seed $r_i = \int \varphi_i\,(1+\log f_s)\,\mathrm d\mathbf v$ is
assembled (`compute_r!`), the field-side gradient is $L = M^{-1} r$, and its
value at each marker $G_\alpha = \nabla L(\mathbf v_\alpha)$ is evaluated
(`compute_G!`). The per-marker entropy gradient is then
$\partial H_h/\partial \mathbf v_\alpha = -w_\alpha\,G_\alpha$
(`compute_entropy_gradient!`).

## Discrete collision bracket

The pairwise Landau interaction is assembled marker-by-marker
(`compute_collision!`): with $\mathbf d = \mathbf v_\gamma-\mathbf v_\alpha$ and
$\mathbf g = G_\alpha-G_\gamma$,

```math
\dot{\mathbf v}_\gamma = \sum_{\alpha\neq\gamma} w_\alpha\,
  \frac{1}{|\mathbf d|}
  \Big(\mathbf g - \mathbf d\,\frac{\mathbf d\!\cdot\!\mathbf g}{|\mathbf d|^2}\Big),
```

i.e. the projection $\mathsf Q(\mathbf d)\,\mathbf g$. The antisymmetry in the
pair $(\gamma,\alpha)$ together with $\mathsf Q(\mathbf d)\mathbf d = 0$ gives
exact momentum and energy conservation, and the positive semidefiniteness gives
$\dot H_h \ge 0$ — at the *semi-discrete* level.

## Gonzalez discrete gradient in time

Preserving those structures *after* time discretisation needs more than a
generic implicit rule. The **Gonzalez discrete gradient** $\overline{\nabla} H$
is built to satisfy the discrete chain rule exactly,

```math
H_h^{\,n+1} - H_h^{\,n}
  = \overline{\nabla} H \cdot (\mathbf v^{n+1}-\mathbf v^{n}),
```

via the midpoint value plus a rank-one correction along
$\Delta\mathbf v = \mathbf v^{n+1}-\mathbf v^{n}$ (`picard_map!`):

```math
\overline{\nabla} H
  = \nabla H_h(\mathbf v_{\mathrm{mid}})
  + \frac{H_h^{\,n+1} - H_h^{\,n}
          - \nabla H_h(\mathbf v_{\mathrm{mid}})\!\cdot\!\Delta\mathbf v}
         {|\Delta\mathbf v|^2}\,\Delta\mathbf v,
\qquad
\mathbf v_{\mathrm{mid}} = \tfrac12(\mathbf v^{n}+\mathbf v^{n+1}).
```

Feeding $\overline{\nabla} H$ (rather than the raw gradient) through the
collision bracket evaluated at $\mathbf v_{\mathrm{mid}}$ yields the update

```math
\mathbf v^{n+1} = \mathbf v^{n}
  + \Delta t\;\widetilde{\mathsf G}(\mathbf v_{\mathrm{mid}})\,\overline{\nabla} H,
```

which conserves energy and momentum to round-off and reproduces the H-theorem
discretely, independent of mesh resolution or time step.

## Solving the implicit step

The relation above is a fixed-point map $\mathbf v^{n+1} = G(\mathbf v^{n+1})$.
It is solved by Picard iteration, optionally accelerated by Anderson mixing
(`step_anderson!`) with damping decay, a relative-plus-floor tolerance and a
stagnation-aware early exit; the best iterate seen is returned. The
projection–seed–gradient–bracket sequence is re-evaluated at the midpoint inside
every iteration.

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
julia --project=. main_Gonzalez.jl parameters_default.jl
```

See the
[README](https://github.com/JuliaPlasma/CollisionOperators.jl#run)
for presets, overrides, checkpoint/resume and output formats.

## References

- S. Jeyakumar, M. Kraus, M. J. Hole, and D. Pfefferlé, *Structure-preserving
  particle methods for the Landau collision operator using the metriplectic
  framework*, arXiv:2309.16894 (2024).
- O. Gonzalez, *Time integration and discrete Hamiltonian systems*, J. Nonlinear
  Sci. **6**, 449–467 (1996).
