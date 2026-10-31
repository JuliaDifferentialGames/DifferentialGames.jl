# Solver Interface API

## Solve Entry Point

The primary entry point for all solvers:

```julia
sol = solve(game, solver; warmstart=nothing, verbose=false, check_compatibility=true)
```

- `warmstart` — a previous `GNEPSolution` or a [`WarmstartData`](solutions.md).
- `verbose` — passed to the solver; it is an argument of `solve`, **not** a field of the solver structs.
- `check_compatibility` — compare `required_capabilities(game)` with `solver_capabilities(typeof(solver))` and `@warn` on a mismatch. A warning does not stop the solve.

`GameSolver` is the abstract supertype of all solvers; [`WarmstartData`](solutions.md) is documented in the Solutions API.

```@docs
solve(::GameProblem, ::GameSolver)
solve(::AbstractGameProblem, ::GameSolver)
solver_capabilities
required_capabilities
CANONICAL_CAPABILITIES
```

## Dispatch Structure

`solve(game::GameProblem, solver::GameSolver; ...)` checks capabilities and then calls

```julia
_solve(game, solver, warmstart::Union{Nothing,WarmstartData}, verbose::Bool)
```

Solver packages **only** implement `_solve` (plus `solver_capabilities`); they do not add methods to `solve`. For problem types other than `GameProblem` (`LexicographicGameProblem`, `ConvexGameProblem`, `InverseGameProblem`, …) a second `solve(::AbstractGameProblem, ::GameSolver)` method forwards to `_solve` without the capability check.

## Solver Reference Pages

| Solver | Description |
|--------|-------------|
| [`FNELQ`](../solvers/fnelq.md) | Exact feedback Nash for LQ games |
| [`iLQGames`](../solvers/ilqgames.md) | Iterative nonlinear feedback-game approximation, unconstrained |
| [`ALGAMES`](../solvers/algames.md) | Constrained games, augmented Lagrangian |
| [`LIBR`](../solvers/libr.md), [`YiPavel`](../solvers/yipavel.md), [`InverseLQGames`](../solvers/invlq.md), [`RecedingHorizonNash`](../solvers/rhn.md) | See the individual pages |

## Implementing a Custom Solver

A minimal solver subtypes `GameSolver`, declares its capabilities from the closed vocabulary [`CANONICAL_CAPABILITIES`](@ref), and implements `_solve`:

```@example custom_solver
using DifferentialGames, LinearAlgebra
import DifferentialGamesBase: _solve, solver_capabilities

# A trivial "solver": zero controls for every player (not a Nash equilibrium!),
# used only to show the plumbing.
struct ZeroControlSolver <: GameSolver end

solver_capabilities(::Type{ZeroControlSolver}) = [:LQGame, :DiscreteTime, :UnconstrainedGame]

function _solve(game::GameProblem{T}, ::ZeroControlSolver,
                warmstart::Union{Nothing, GNEPSolution, WarmstartData},
                verbose::Bool) where {T}
    N  = n_steps(game)
    n  = state_dim(game)
    X  = zeros(T, n, N + 1)
    X[:, 1] .= game.initial_state
    A  = game.dynamics.A isa Vector ? game.dynamics.A[1] : game.dynamics.A
    for k in 1:N
        X[:, k+1] = A * X[:, k]
    end
    times = collect(range(zero(T), step = T(game.time_horizon.dt), length = N + 1))
    trajs = [Trajectory{T}(obj.player_id, X, zeros(T, control_dim(game, obj.player_id), N),
                           times, zero(T)) for obj in game.objectives]
    return GNEPSolution{T}(game, trajs, X, nothing, :Unknown, true, 0, 0.0, Dict{Symbol,Any}())
end

game = LQGameProblem(Matrix(0.9I(2)), [Matrix(1.0I(2))[:, 1:1], Matrix(1.0I(2))[:, 2:2]],
                     [Matrix(1.0I(2)), Matrix(1.0I(2))], [fill(0.1,1,1), fill(0.1,1,1)],
                     [Matrix(1.0I(2)), Matrix(1.0I(2))], [1.0, 1.0], 1.0; dt=0.1)
sol = solve(game, ZeroControlSolver())
sol.equilibrium_type
```

!!! warning "Annotate `warmstart` exactly"
    The fallback `_solve(::GameProblem, ::GameSolver, ::Union{Nothing,GNEPSolution,WarmstartData}, ::Bool)` in `DifferentialGamesBase` is more specific in the `warmstart` argument than an unannotated method. Leaving `warmstart` untyped makes your method **ambiguous** with it (`MethodError: _solve(...) is ambiguous`). Copy the `Union` annotation as above.

A full guide (registering with the solver catalog, tests, external packages) is planned in the extension guide.
