# Dynamics API

## Player Specification

See [`PlayerSpec`](../problem_types/pdgnep.md) for the full constructor reference.

```@docs
PlayerSpec
```

## Dimension Accessors

```@docs
state_dim
control_dim
```

## ForwardDiff Compatibility

Nonlinear solvers (`iLQGames`, `ALGAMES`) differentiate dynamics using `ForwardDiff.jl`.
Dynamics functions must be compatible:

- Use generic arithmetic (`+`, `*`, etc.) rather than type-specific operations.
- Avoid `Float64`-typed allocations inside the dynamics; let Julia infer the element type.
- Do not use branches that depend on values — use smooth approximations instead.

```julia
# Good — ForwardDiff compatible
dyn = (x, u, p, t) -> [x[2]; u[1] - 0.1 * x[2]]

# Bad — Float64 literal breaks dual number propagation
dyn = (x, u, p, t) -> [x[2]; Float64(u[1])]
```
