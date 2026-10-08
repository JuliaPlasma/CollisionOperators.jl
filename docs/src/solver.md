```@meta
CurrentModule = Driver
```

# Driver API

The driver is a set of top-level scripts rather than part of the package module,
so `main.jl` would place these docstrings in `Main`, out of Documenter's reach.
The documentation build loads them into a `Driver` module instead (see
`docs/make.jl`). `plots.jl` is left out: it only wraps CairoMakie.

## The implicit solve

One Picard map of the Gonzalez discrete-gradient update, and the
Anderson-accelerated fixed-point iteration that solves it.

```@autodocs
Modules = [Driver]
Order = [:type, :function, :constant]
```

## Configuration and workspace

The run is configured through a single immutable `SimParameters` value,
built from a preset file plus `--key=value` overrides.

```@autodocs
Modules = [Driver.MantisWrappers]
Order = [:type, :function]
```
