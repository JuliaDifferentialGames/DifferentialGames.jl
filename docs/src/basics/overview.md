# Overview & Architecture

DifferentialGames.jl follows the same problem/solver/solution separation used by DifferentialEquations.jl.

## The Three-Object Pattern

Every computation involves three objects:

```
game  = LQGameProblem(...)    # 1. Problem: pure specification, immutable
sol   = solve(game, FNELQ())  # 2. Solve:   returns solution
J1    = get_cost(sol, 1)      # 3. Solution: access results
```

**Problems are immutable.** A `GameProblem` contains dynamics, costs, constraints, and a time horizon — but no solver state. This means the same problem object can be solved with multiple solvers, solved in parallel, or reused in an outer optimization loop without copying.

**Solvers own their state.** All iteration counts, caches, and convergence history belong to the solver dispatch, not the problem. The solution object captures only the final result.

## Package Layers

```
┌─────────────────────────────────────────────────────────────┐
│  DifferentialGames.jl         (you are here)                │
│  Re-exports Base + Solvers. Single import for end users.    │
├─────────────────────────────────────────────────────────────┤
│  DifferentialGamesBaseSolvers.jl                            │
│  FNELQ · iLQGames · ALGAMES                                 │
├─────────────────────────────────────────────────────────────┤
│  DifferentialGamesBase.jl                                   │
│  GameProblem · PlayerSpec · dynamics · costs                │
│  constraints · solutions · solver interface                 │
└─────────────────────────────────────────────────────────────┘
```

**Solver authors** depend only on `DifferentialGamesBase`. They implement `_solve` and `solver_capabilities` and never touch the solver library.

**End users** import only `DifferentialGames` and get everything.

## Type Hierarchy

```
AbstractGameProblem
├── AbstractDeterministicGame{T}
│   └── GameProblem{T}           ← all concrete problems
├── AbstractStochasticGame{T}    ← future
├── AbstractPartiallyObservableGame{T}  ← future
├── AbstractInverseGameProblem{T}
│   └── InverseGameProblem{T}    ← inverse games
└── AbstractPotentialGame{T}     ← future
```

The numeric type `T` (typically `Float64`) is propagated through the entire type hierarchy for type stability.

## Equilibrium Concepts

DifferentialGames.jl currently targets:

| Concept | Strategy | Solver |
|---------|----------|--------|
| Feedback Nash (closed-loop) | `FeedbackStrategy` | FNELQ, iLQGames |
| Open-loop Nash | `OpenLoopStrategy` | ALGAMES |

The `GNEPSolution` records which equilibrium type was computed in `sol.equilibrium_type`.

## Key Design Decisions

**Separable vs. coupled dynamics.** `PDGNEProblem` creates games where each player has its own state vector (`SeparableDynamics`). This enables per-player linearization and is the default. `CoupledNonlinearDynamics` handles games with a shared state — richer but more expensive to solve.

**Cost-term DSL vs. raw matrices.** For LQ costs, pass `LQStageCost(Q, R)` directly. For composable nonlinear costs, use the `minimize`, `track_goal`, and `avoid_proximity` DSL. Both paths produce `PlayerObjective` objects with identical solver-facing interfaces.

**Constraint representation.** Constraints are typed as `AbstractPrivateConstraint` or `AbstractSharedConstraint` and carry metadata (which players they involve, their dimension). Solvers query this metadata to build the correct KKT structures.
