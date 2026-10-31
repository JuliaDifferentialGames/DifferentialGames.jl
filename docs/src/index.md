```@raw html
---
layout: home

hero:
  name: DifferentialGames.jl
  text: A Library for N-Player Differential Games
  tagline: One problem interface. Many solvers. Equilibria you can check.
  actions:
    - theme: brand
      text: Get started
      link: getting_started/installation/index.html
    - theme: alt
      text: Choose a solver
      link: basics/solver_selection/index.html
  image:
    src: /logo.png
    dark: /logo-dark.png
    alt: DifferentialGames.jl

features:
  - icon: 🧩
    title: Unified interface
    details: Describe a game once (dynamics, costs, constraints, horizon) and call solve(game, solver). Swapping the solver is one word.
    link: basics/overview/index.html
  - icon: 🎯
    title: Explicit equilibrium concepts
    details: Open-loop, feedback, generalized and variational Nash equilibria are defined precisely, and every solver page states which one it targets and what it does and does not guarantee.
    link: math/equilibrium_concepts/index.html
  - icon: 🧭
    title: Solver selection guide
    details: A capability matrix generated from solver_capabilities maps problem class, strategy type and constraints to FNELQ, iLQGames, ALGAMES, LIBR, YiPavel and more.
    link: basics/solver_selection/index.html
  - icon: 📚
    title: Mathematics with references
    details: Notation, Bellman recursions, Riccati steps and KKT systems are documented next to the code that implements them, with citations to the original papers.
    link: math/differential_games/index.html
  - icon: ✅
    title: Executable documentation
    details: Tutorials and examples run during the documentation build against the documented version, and check constraint satisfaction and residuals rather than printing numbers.
    link: tutorials/lq_games/index.html
  - icon: 🔌
    title: Extensible
    details: Implement _solve and solver_capabilities for your own solver, in your own package, and it plugs into the same interface.
    link: api/solver_interface/index.html
---
```

```@docs
DifferentialGames
```

!!! warning "Work in progress"
    The API is stabilizing. Breaking changes may occur before v1.0.0.

## Statement of Need

Differential games model situations where multiple decision-makers (players or agents) each optimize an objective with respect to constraints that may involve the decision variables of other agents. They arise across:

- **Autonomous vehicles** — modeling interactions at intersections, merging, and overtaking
- **Robotics** — multi-robot coordination, pursuit-evasion, formation control
- **Aerospace** — spacecraft rendezvous, pursuit-guidance, air-traffic separation
- **Economics** — dynamic oligopoly, resource extraction, mechanism design

The class of problems primarily considered for this package are Nash Equilibrium Problems (NEPs). There is a significant lack of open source implementations of many differential game solvers and no common standard across the field. If software does exist, it is generally for a single method developed in research (no stable API or a very limited API) or does not allow easy extension for new solvers.

**DifferentialGames.jl fills this gap.** It provides:

- A composable, type-stable problem specification covering linear, nonlinear, constrained, and inverse games
- A common `solve(game, solver)` interface so algorithms can be swapped without rewriting the problem
- A growing library of modern numerical solvers 

### Relation to Existing Work

| Package | Language | LQ | Nonlinear | Constraints | Inverse | Players|
|---------|----------|-----|-----------|-------------|---------|-----------|
| **DifferentialGames.jl** | Julia | ✓ | ✓ | ✓ | ✓ | N |
| [iLQGames.jl](https://github.com/lassepe/iLQGames.jl) | Julia | ✓ | ✓ | ✗ | ✗ | N |
| [ALGAMES.jl](https://github.com/simon-lc/Algames.jl) | Julia | ✓ | ✓ | ✓ | ✗ | N |
| [ilqgames](https://github.com/HJReachability/ilqgames) | C++ | ✓ | ✓ | ✗ | ✗ | N |
| [Nashpy](https://github.com/drvinceknight/Nashpy) | Python | ✓ | ✓ | ✓ | ✗ | 2 |

DifferentialGames.jl is inspired by the design of [TrajectoryOptimization.jl](https://github.com/RoboticExplorationLab/TrajectoryOptimization.jl) (problem/solver separation) and the [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) ecosystem (unified `solve` interface, composable problem types).

## Package Layout

```
DifferentialGames.jl                 ← umbrella and documentation (this package)
├── DifferentialGamesBase.jl         ← problem types, costs, constraints, dynamics,
│                                      solutions, solver interface, uncertainty
└── DifferentialGamesBaseSolvers.jl  ← FNELQ, SFNELQ, iLQGames, ALGAMES, LIBR, YiPavel,
                                       InverseLQGames, RecedingHorizonNash, DynamicTubeMPC

DifferentialGamesMARL.jl             ← multi-agent RL (MADDPG, guided policy search)
DifferentialGamesBenchmarks.jl       ← benchmark cases, challenges, evaluation, plots
```

End users import only `DifferentialGames` for model-based work; learning and benchmarking are separate packages. Package authors implementing new solvers depend on `DifferentialGamesBase` only. See [The Ecosystem](ecosystem/overview.md) for how the five packages fit together.

## Quick Example

```@example index
using DifferentialGames, LinearAlgebra

# Two-player discrete-time LQ game.
# Explicit Matrix(...) conversions keep this working with older DifferentialGamesBase versions;
# current versions also accept I(n) and Diagonal directly.
n  = 4                                                    # state dim
A  = Matrix(0.95 * I(n))
B  = [Matrix(1.0I(n))[:, 1:1], Matrix(1.0I(n))[:, 3:3]]  # one actuator per player
Q  = [Matrix(1.0I(n)), Matrix(1.0I(n))]
R  = [fill(0.1, 1, 1), fill(0.1, 1, 1)]
Qf = [Matrix(10.0I(n)), Matrix(10.0I(n))]
x0 = [1.0, 0.0, -1.0, 0.0]

game = LQGameProblem(A, B, Q, R, Qf, x0, 3.0; dt=0.1)
sol  = solve(game, FNELQ())

(converged = sol.converged, J₁ = get_cost(sol, 1), J₂ = get_cost(sol, 2))
```

## Contents

- **[Getting Started](getting_started/installation.md)** — install and run your first game in minutes
- **[Tutorials](tutorials/lq_games.md)** — worked examples from spacecraft to autonomous vehicles
- **[Basics](basics/overview.md)** — the problem/solver/solution pattern
- **[Problem Types](problem_types/lq_game.md)** — reference for each game formulation
- **[The Ecosystem](ecosystem/overview.md)** — how Base, Solvers, MARL and Benchmarks connect, and shared conventions
- **[Solver Selection Guide](basics/solver_selection.md)** — which solver supports which problem class, equilibrium concept and constraints
- **[Solver Algorithms](solvers/fnelq.md)** — FNELQ, iLQGames, ALGAMES, InverseLQGames, LIBR, RHN, and YiPavel
- **[API Reference](api/problems.md)** — complete docstring reference
- **[Contributing](contributing.md)** — how to add solvers, report bugs, and get support
