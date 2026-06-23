# Mesh comparison: square (bp2 = bp1) vs rectangular (dense bp2)

Both runs use the **same anisotropic initial condition** (σ₁ = 4/3, σ₂ = 0.5),
the **same Gonzalez discrete-gradient** integrator, the **same ½·log f² entropy
trick** (`use_logsq = true`), DT = 0.001, 40 000 particles, Anderson(m=8),
seed 42. The **only** difference is the mesh in the v₂ direction.

## Setup

| | **Square mesh** (`aniso1000_bp2eqbp1_logsq`) | **Rectangular mesh** (`aniso100_logsq`) |
|---|---|---|
| `bp1` | `[-6,-5, LinRange(-4,4,17), 5,6]` | `[-6,-5, LinRange(-4,4,17), 5,6]` (same) |
| `bp2` | `[-6,-5, LinRange(-4,4,17), 5,6]` (= bp1) | `[-6, LinRange(-2.5,2.5,26), 6]` (dense, non-uniform) |
| Mesh cells | 400 (20 × 20, symmetric) | 540 (dense in v₂) |
| Δv₂ near core | 0.5 | 0.20 |
| Discrete gradient | Gonzalez | Gonzalez |
| Entropy integrand | ½·log f² | ½·log f² |

## Results at matched step 2000 (t = 2.0)

| Metric | Square (bp2 = bp1) | Rectangular (dense bp2) |
|---|---|---|
| σ₁ | 1.1500 | 1.1501 |
| σ₂ | 0.8398 | 0.8396 |
| **σ₁/σ₂** | **1.3693** | **1.3699** |
| Entropy Sₕ | 2.7957 | 2.7968 |
| Inner iterations | 14 | 11 |
| **∫(neg) — negative mass** | **3.29 × 10⁻⁴** | **1.52 × 10⁻³** |

→ **Physical relaxation is identical** (σ₁/σ₂ agree to 4 digits; both still
anisotropic at t = 2, target ratio = 1). The meshes differ **only** in the
negative-part (Gibbs ring): the rectangular dense-bp2 mesh carries **≈ 4.6×**
more negative mass at step 2000.

## Negative-part trajectory — the key difference

| step | Square ∫(neg) | Rectangular ∫(neg) |
|---|---|---|
| 400  | 4.60 × 10⁻⁴ | 2.19 × 10⁻⁴ |
| 800  | 3.62 × 10⁻⁴ | 2.41 × 10⁻⁴ |
| 1200 | 4.60 × 10⁻⁴ | 5.05 × 10⁻⁴ |
| 1600 | 3.97 × 10⁻⁴ | 9.44 × 10⁻⁴ |
| 2000 | 3.29 × 10⁻⁴ | 1.52 × 10⁻³ |
| 2375 | — (run ended at 2000) | 2.27 × 10⁻³ |

- **Square mesh:** negative mass **stays bounded**, oscillating in
  3.3–4.6 × 10⁻⁴, no long-time trend.
- **Rectangular mesh:** negative mass **grows monotonically** — ~7× from step
  800 to step 2000, still rising at step 2375 (~9× vs step 800).

The bulk distribution stays smooth in both cases (no central spike); the
growth is a diffuse negative halo spreading into the tails as the cloud
isotropizes. Energy is conserved (~10⁻¹¹), momentum ~10⁻¹³, entropy monotone,
solver healthy throughout (iter < 100, no Gonzalez |Δv|² blow-up).

## Wall-clock time

| Run | Steps | Wall time |
|---|---|---|
| **Square** (0 → 1000) | 1000 | 6 h 42 m |
| **Square** (1000 → 2000) | 1000 | 5 h 39 m |
| **Square total → 2000** | **2000** | **≈ 12 h 22 m** |
| **Rectangular** (0 → 100) | 100 | 52 m |
| **Rectangular** (100 → 400) | 300 | 3 h 01 m |
| **Rectangular** (400 → 1000) | 600 | 3 h 47 m |
| **Rectangular** (1000 → 2000) | 1000 | ≈ 5 h (segment of the ongoing 4000-step run) |
| **Rectangular total → 2000** | **2000** | **≈ 12 h 41 m** |

Per-step cost is essentially the same (~22 s/step on 8 threads); the small
difference reflects mesh size (540 vs 400 cells).

## Conclusion

With identical IC, solver and entropy trick, switching from the dense
non-uniform `bp2` to a **symmetric square mesh keeps the negative ring
bounded**, whereas the dense-bp2 mesh lets it **grow without saturating** at
long time. The relaxation physics (σ₁/σ₂) is unaffected by the mesh.

This pins the long-time growth of the Gibbs/honeycomb negativity to the
**dense / non-uniform v₂ mesh (projection)**, not to the Landau kernel or the
time integrator — consistent with the earlier 2-D LB probe finding that the
honeycomb tail is a mesh/projection artifact.

*Caveat (for Sandra): ½·log f² suppresses the bulk spike in both meshes, but it
does not eliminate the negative oscillation; on the production dense-bp2 mesh
the integrated negativity still grows slowly with time.*
