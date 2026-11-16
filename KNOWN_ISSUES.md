# Known issues

### K1 · The drift test of the port check does not catch a lost final Picard step.

- **location:** `scripts/anderson_port_check.jl:151`
- **evidence:** with `if exit_picard_step` in `fixed_point_solve!` replaced by `if false`, the
  after-run's largest `|dP1|`, `|dP2|`, `|dE_rel|` are 2.08e-16, 4.25e-17, 1.80e-13. Each stays
  under `max(before, ROUNDOFF)` (`ROUNDOFF = 10 * eps()`), and the summary still reads "met". The
  testset "fixed_point_solve!: the final Picard step" of `test/integration/main_helpers.jl` fails
  on that mutant (3 failures).
- **kind:** missing test
- **found:** 2026-11-16

### K2 · No unit test catches a history depth one above `m` in `fixed_point_solve!`.

- **location:** `test/integration/main_helpers.jl:123`
- **evidence:** the testset "fixed_point_solve!: Anderson converges, history depth m" compares
  only `m = 1` with `m = 2` (`its[2] < its[1]`). With `fixed_point_method(; m = m + 1, damping,
  use_anderson)` in `fixed_point_solve!`, every unit test passes; only the table of
  `scripts/anderson_port_check.jl` catches it (5025 evaluations, summary "not met").
- **kind:** missing test
- **found:** 2026-11-16
