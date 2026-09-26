using CollisionOperators
using Aqua
using Test

Aqua.test_all(CollisionOperators; stale_deps = false)

# Aqua's stale_deps check has no `broken` keyword, so it is marked here.
@test_broken isempty(Aqua.find_stale_deps(Base.PkgId(CollisionOperators)))  # issue #22
