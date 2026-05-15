# Solver Interface API

## Solve Entry Point

The primary entry point for all solvers:

```julia
sol = solve(game, SolverType(); verbose=false, warmstart=nothing)
```

See [Common Solver Options](../basics/solver_options.md) for the full option reference.

## Solver Reference Pages

| Solver | Description |
|--------|-------------|
| [`FNELQ`](../solvers/fnelq.md) | Exact LQ game solver |
| [`iLQGames`](../solvers/ilqgames.md) | Iterative nonlinear game solver |
| [`ALGAMES`](../solvers/algames.md) | Constrained game solver |

## Warmstart Data

See [`WarmstartData`](solutions.md) in the Solutions API.

## Implementing a Custom Solver

To add a new solver, define a struct and implement the `solve` method:

```julia
struct MySolver <: AbstractGameSolver
    max_iter::Int
    ε_conv::Float64
    MySolver(; max_iter=100, ε_conv=1e-6) = new(max_iter, ε_conv)
end

function DifferentialGames.solve(
    prob::GameProblem{T},
    solver::MySolver;
    verbose::Bool = false,
    warmstart = nothing,
) where {T}
    # ... your solver logic ...
    return GNEPSolution(...)
end

DifferentialGames.solver_capabilities(::Type{MySolver}) =
    [:nonlinear, :feedback_nash, :unconstrained]
```
