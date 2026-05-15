# DifferentialGames.jl

**DifferentialGames.jl** is a Julia package for computing Nash equilibria in N-player differential games. It provides a unified interface — modelled on the [SciML](https://sciml.ai) ecosystem — for specifying multi-agent optimal control problems, swapping numerical solvers, and extracting solutions.

!!! warning "Work in progress"
    The API is stabilizing. Breaking changes may occur before v1.0.0.

## Statement of Need

Differential games model situations where multiple decision-makers (players) each optimize their own objective subject to shared dynamics. They arise across:

- **Autonomous vehicles** — modeling interactions at intersections, merging, and overtaking
- **Robotics** — multi-robot coordination, pursuit-evasion, formation control
- **Aerospace** — spacecraft rendezvous, pursuit-guidance, air-traffic separation
- **Economics** — dynamic oligopoly, resource extraction, mechanism design

Computing Nash equilibria in these settings requires solving coupled optimality conditions that grow exponentially in complexity with player count and horizon length. Existing software is either problem-specific (hardcoded to a single formulation), research-grade (no stable API), or restricted to small linear-quadratic problems.

**DifferentialGames.jl fills this gap.** It provides:

- A composable, type-stable problem specification covering linear, nonlinear, constrained, and inverse games
- A common `solve(game, solver)` interface so algorithms can be swapped without rewriting the problem
- A growing library of numerical solvers (FNELQ, iLQGames, ALGAMES)
- An inverse game framework for cost recovery from observed trajectories

### Relation to Existing Work

| Package | Language | LQ | Nonlinear | Constraints | Inverse | Interface |
|---------|----------|-----|-----------|-------------|---------|-----------|
| **DifferentialGames.jl** | Julia | ✓ | ✓ | ✓ | ✓ | SciML |
| [iLQGames.jl](https://github.com/lassepe/iLQGames.jl) | Julia | ✓ | ✓ | ✗ | ✗ | custom |
| [ALGAMES.jl](https://github.com/simon-lc/Algames.jl) | Julia | ✓ | ✓ | ✓ | ✗ | custom |
| [ilqgames](https://github.com/HJReachability/ilqgames) | C++ | ✓ | ✓ | ✗ | ✗ | custom |
| [OpenSpiel](https://github.com/google-deepmind/open_spiel) | Python | ✗ | ✗ | ✗ | ✗ | RL-focused |

DifferentialGames.jl is inspired by the design of [TrajectoryOptimization.jl](https://github.com/RoboticExplorationLab/TrajectoryOptimization.jl) (problem/solver separation) and the [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) ecosystem (unified `solve` interface, composable problem types).

## Package Layout

```
DifferentialGames.jl                 ← public API (this package)
├── DifferentialGamesBase.jl         ← problem types, costs, constraints,
│                                      dynamics, solutions, solver interface
└── DifferentialGamesBaseSolvers.jl  ← FNELQ, iLQGames, ALGAMES
```

End users import only `DifferentialGames`. Package authors implementing new solvers depend on `DifferentialGamesBase`.

## Quick Example

```julia
using DifferentialGames, LinearAlgebra

# Two-player discrete-time LQ game
n = 4                                          # state dim
A  = 0.95 * I(n)
B  = [I(n)[:, 1:1], I(n)[:, 3:3]]             # one actuator per player
Q  = [diagm(ones(n)), diagm(ones(n))]
R  = [fill(0.1, 1, 1), fill(0.1, 1, 1)]
Qf = [10.0 * diagm(ones(n)), 10.0 * diagm(ones(n))]
x0 = [1.0, 0.0, -1.0, 0.0]

game = LQGameProblem(A, B, Q, R, Qf, x0, 3.0; dt=0.1)
sol  = solve(game, FNELQ())

println("converged: ", sol.converged)
println("J₁ = ", get_cost(sol, 1))
println("J₂ = ", get_cost(sol, 2))
```

## Contents

- **[Getting Started](getting_started/installation.md)** — install and run your first game in minutes
- **[Tutorials](tutorials/lq_games.md)** — worked examples from spacecraft to autonomous vehicles
- **[Basics](basics/overview.md)** — the problem/solver/solution pattern
- **[Problem Types](problem_types/lq_game.md)** — reference for each game formulation
- **[Solver Algorithms](solvers/fnelq.md)** — when to use each solver and what options it accepts
- **[API Reference](api/problems.md)** — complete docstring reference
- **[Contributing](contributing.md)** — how to add solvers, report bugs, and get support
