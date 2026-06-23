# Status update — Landau particle solver

Hi Sandra, summary of where things stand.

## 1. Which bracket are we using?

You're right: we use the **N² single-step bracket** — the direct particle
discretisation from the main text (Section 3) of the paper (arXiv 2404.18432v1),
**not** the two-step spline/K⁺ construction in Appendix A (which only appears in
the newer version of the manuscript).

Concretely, our collision routine is a direct double loop over particle pairs
(γ, α) that contracts the Landau projector U(v_γ − v_α) between particles:

```
for γ in 1:N, for α in 1:N:   dot_v_γ += w_α · U(v_γ − v_α) · (G_α − G_γ)
```

There is **no pseudo-inverse K⁺ and no M×M spline L-matrix** (Eq. 85/91) anywhere
in the code. The L²/spline projection is used **only** to build the discrete
entropy-gradient seed ∂S_h/∂v_α (via M⁻¹ and ∇φ_k at the particles), not the
metric bracket itself.

## 2. This is the reason it is slow

The pairwise bracket is **O(N²)**. With N = 40 000 particles, each collision
evaluation is ~1.6 × 10⁹ pair interactions, and this sits inside the
Anderson/Picard fixed-point loop (~11–90 inner iterations per step). Net cost is
**~22 s/step** for the Landau solver, versus **~1.5 s/step** for the
Lenard–Bernstein version on the identical mesh (same v3 dense bp2, 40k particles,
800 steps: LB finished in 20 min). So Landau is ~15× slower per step, and it is
entirely the N² bracket.

The Appendix A two-step bracket is precisely the documented fix: it is quadratic
in the spline degrees of freedom (M = 638 here) rather than in N, at the price of
one pseudo-inverse K⁺ per step — potentially ~thousands× fewer bracket
operations at this N. Worth considering if we want to push N higher.

## 3. The ½·log f² entropy trick, the negative ring, and its cause

We applied the identity log f = ½·log f² in the entropy and entropy-seed
quadrature (so that f_s < 0 Gibbs-undershoot quadrature points contribute via the
|f| guard, instead of being clamped to zero). It **suppresses the bulk spike**
strongly — the central pseudo-physics oscillation is gone and the bulk stays
smooth throughout.

It does **not** eliminate the negative oscillation. On the production
**dense non-uniform bp2 mesh**, the integrated negative part ∫max(−f_s,0) **grows
monotonically with time and does not saturate**:

| step | ∫(neg), dense bp2 | ∫(neg), coarse square (bp2 = bp1, Δ=0.5) |
|---|---|---|
| 800  | 2.4 × 10⁻⁴ | 3.6 × 10⁻⁴ |
| 1600 | 9.4 × 10⁻⁴ | 4.0 × 10⁻⁴ |
| 2400 | 2.3 × 10⁻³ | — |
| 3500 | 4.1 × 10⁻³ | — |

### Cause pinned down: the giant coarse outer cells of the non-uniform bp2 mesh

A high-contrast plot of the negative part with the mesh overlaid shows the
negative mass is **not** in the bulk: it sits in the two **giant coarse outer
cells** of the non-uniform mesh, `bp2 = [-6; linspace(-2.5,2.5,26); 6]`, i.e. the
cells [-6,-2.5] and [2.5,6], each **3.5 wide**, flanking the dense Δv₂ = 0.2 core.
As the cloud isotropises (σ₂: 0.5 → ~0.9) mass spreads into those coarse cells and
the projection there develops a large, growing Gibbs undershoot — which is exactly
why ∫(neg) grows with time. (Plus a fine cell-scale checkerboard in the f ≈ 0
corners, the irreducible B-spline projection floor.)

### Decisive test: re-project onto a dense *square* mesh

Taking the **same particle state at step 2000** and re-projecting it onto a dense
square mesh (bp1 = bp2, inner Δ = 0.2, **no giant outer cell**), then evolving
forward, the negative part **stops growing and decreases** — the opposite of the
rect mesh over the identical physical span:

| physical step | ∫(neg), dense rect bp2 | ∫(neg), dense square |
|---|---|---|
| 2000 | 1.5 × 10⁻³ | (re-proj 1.2 × 10⁻³) |
| 2200 | 1.8 × 10⁻³ | 4.2 × 10⁻⁴ |
| 2600 | 2.5 × 10⁻³ | 3.3 × 10⁻⁴ |
| 3000 | 3.1 × 10⁻³ ↑ | **2.7 × 10⁻⁴ ↓** |

At the same physical time the dense square is **~11× lower** and trending down.
The giant-cell negative lobes are gone; only the thin edge ring + corner
checkerboard (the projection floor) remain, and they do **not** accumulate.

**Conclusion:** the long-time growth of the negative ring is a **mesh / projection
artifact of the non-uniform bp2 mesh's coarse outer cells**, *not* a property of
the Landau kernel or the ½·log f² trick. A well-resolved square mesh keeps the
Gibbs negativity bounded and decreasing while the physical relaxation (σ₁/σ₂) is
unchanged.

## 4. Gonzalez discrete gradient — denominator is fine

We were worried the Gonzalez midpoint discrete gradient might break down as the
distribution approaches equilibrium, because its correction term divides by
|v^{n+1} − v^n|² (the per-step displacement), which → 0 at equilibrium. **In
practice it does not blow up.** The numerator (the second-order entropy remainder
S₁ − S₀ − Δv·∇S) is itself O(|Δv|²), so the ratio stays finite; a small floor
guards the exact-zero case.

Empirically, the max inner-iteration count is flat (~120–150) across every
500-step window from start to step 3500 — **no upward trend** as σ₁/σ₂ relaxes —
energy is conserved to ~10⁻¹¹, momentum to ~10⁻¹³, and entropy is monotone. The
run has stayed on Gonzalez the whole way; a plain-midpoint fallback was prepared
but never needed.

(One qualifier: at step 3500 the state is still anisotropic, σ₁/σ₂ ≈ 1.20, so the
most stringent test — full thermal equilibrium where Δv is pure noise — hasn't
been reached yet, but there is no sign of degradation approaching it.)

### Confirmed: plain midpoint gives the same result

To be sure the Gonzalez correction was not itself responsible for anything, we ran
a **plain-midpoint twin** (use_gonzalez = false) with the *same* IC, mesh and seed,
0 → 2000 steps. The two integrators agree to **~0.1 %** at every step — including
the negative-ring growth, which overlaps the Gonzalez curve almost exactly
(2.2 × 10⁻⁴ → 1.5 × 10⁻³, identical). Plain midpoint also conserves energy
(~10⁻¹¹) and momentum (~10⁻¹³). So the Gonzalez discrete-gradient correction is
**negligible** for this problem: plain midpoint is a perfectly good substitute with
no |Δv|² denominator at all, and the negative-ring growth is **integrator-
independent** (further confirming it is a mesh effect, Section 3).

## 5. Runs behind these results

All on anisotropic IC (σ₁ = 4/3, σ₂ = 0.5) → isotropic, ½·log f², 40k particles,
DT = 0.001, seed 42:

- **Gonzalez, dense rect bp2** — to step 3650 (σ₁/σ₂ relaxed 2.667 → ~1.20);
  ∫(neg) grew to 4 × 10⁻³, no saturation.
- **Gonzalez, dense square** — from the step-2000 state to +1000; ∫(neg) decreased
  to 2.7 × 10⁻⁴ (the decisive mesh test, Section 3).
- **Plain midpoint, dense rect bp2** — fresh 0 → 2000; matches Gonzalez to ~0.1 %
  (Section 4). Wall time 12 h 13 min (~22 s/step, the N² bracket).

**Bottom line:**
1. **Cost** = the N² single-step bracket.
2. **Negative ring growth** = a mesh/projection artifact of the non-uniform bp2
   mesh's giant coarse outer cells; a dense *square* mesh removes it (bounded and
   decreasing), with identical physics. ½·log f² fixes the separate *bulk* spike.
3. **Integrator** (Gonzalez vs plain midpoint) makes no difference — the Gonzalez
   |Δv|² denominator is well-behaved and the correction is negligible here.
