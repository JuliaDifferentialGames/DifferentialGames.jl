# DifferentialGames.jl

[![CI](https://github.com/JuliaDifferentialGames/DifferentialGames.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaDifferentialGames/DifferentialGames.jl/actions/workflows/CI.yml)
[![Docs Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaDifferentialGames.github.io/DifferentialGames.jl/stable)
[![Docs Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaDifferentialGames.github.io/DifferentialGames.jl/dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

A Julia ecosystem for solving N-player differential games using numerical methods, following the SciML interface conventions.

> ⚠️ **Work in progress** — API may change before v1.0.0.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaDifferentialGames/DifferentialGames.jl")
```

## Quick Start

```julia
using DifferentialGames, LinearAlgebra

# Two-player LQ game — feedback Nash equilibrium via FNELQ
n = 4
game = LQGameProblem(
    0.95 * I(n),
    [I(n)[:, 1:1], I(n)[:, 3:3]],
    [diagm(ones(n)), diagm(ones(n))],
    [fill(0.1, 1, 1), fill(0.1, 1, 1)],
    [10.0 * diagm(ones(n)), 10.0 * diagm(ones(n))],
    ones(n), 2.0; dt=0.1
)

sol = solve(game, FNELQ())
println("Player 1 cost: ", get_cost(sol, 1))
println("Player 2 cost: ", get_cost(sol, 2))
```

```julia
# Nonlinear game with collision avoidance — iLQGames
using DifferentialGames

dyn = (xi, ui, p, t) -> [xi[3]*cos(xi[4]); xi[3]*sin(xi[4]); ui[1]; ui[2]]
stage    = DiagonalLQStageCost([1.0, 1.0, 0.1, 0.1], [0.1, 0.1])
terminal = DiagonalLQTerminalCost([10.0, 10.0, 1.0, 1.0])

col = SharedInequality([1, 2];
    func = (x, u, p, t) -> [1.5^2 - sum((x[1:2] - x[5:6]).^2)],
    dim  = 1)

player1 = PlayerSpec(1, 4, 2, [0.0, 0.0, 1.0, 0.0], dyn,
                     PlayerObjective(1, stage, terminal))
player2 = PlayerSpec(2, 4, 2, [5.0, 0.0, -1.0, π], dyn,
                     PlayerObjective(2, stage, terminal))

game = PDGNEProblem([player1, player2], [col], 3.0, 0.1)
sol  = solve(game, iLQGames())
```

## Package Architecture

```
DifferentialGames.jl          ← umbrella (re-exports everything)
├── DifferentialGamesBase.jl  ← problem types, dynamics, costs, constraints,
│                                solution interface, inverse game framework
└── DifferentialGamesBaseSolvers.jl  ← FNELQ, iLQGames, ALGAMES
```

This structure lets downstream packages depend only on what they need: solver authors depend on `DifferentialGamesBase`; end users load `DifferentialGames`.

## Development Status

### Implemented

- ✅ Game problem specification (`GameProblem`, `LQGameProblem`, `PDGNEProblem`)
- ✅ Player-based API (`PlayerSpec`, `PlayerObjective`)
- ✅ Constraint system (private bounds, shared proximity, general nonlinear)
- ✅ Solution interface (`GNEPSolution`, `Trajectory`, feedback/open-loop strategies)
- ✅ Trajectory expansion (linearization + quadraticization for iterative solvers)
- ✅ Discrete-time feedback Nash equilibrium (FNELQ)
- ✅ Iterative LQ games (iLQGames)
- ✅ Augmented Lagrangian games (ALGAMES)
- ✅ Inverse game problem specification (`InverseGameProblem`, observation models, solver wrapper interface)

### In Progress

- 🚧 Inverse game solvers 
- 🚧 Callback/logging system

### Planned

- 📋 Stackelberg (leader-follower) games
- 📋 Stochastic differential games
- 📋 Mean field games
- 📋 Learning-based solvers (MADDPG, MAPPO)
- 📋 Benchmark suite

## Contributing

To add a new solver:

1. Define your solver struct inheriting from `GameSolver`
2. Implement `solver_capabilities(::Type{YourSolver})` to declare supported game types
3. Implement `_solve(game, solver, warmstart, verbose)` returning a `GNEPSolution`
4. Add tests comparing against known solutions or published benchmarks

See [FNELQ](https://github.com/JuliaDifferentialGames/DifferentialGamesBaseSolvers.jl/tree/main/src/solvers/FNELQ) as a reference implementation and [ExampleSolver](https://github.com/JuliaDifferentialGames/DifferentialGamesBaseSolvers.jl/tree/main/src/solvers/ExampleSolver) as a minimal template.

## Inspiration

This package follows the design principles of the SciML ecosystem and draws inspiration from:

- [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) — the `solve` interface pattern
- [TrajectoryOptimization.jl](https://github.com/RoboticExplorationLab/TrajectoryOptimization.jl) — trajectory representations
- [iLQGames.jl](https://github.com/lassepe/iLQGames.jl) — algorithm inspiration

## License

MIT License — see LICENSE file for details.

## Disclosure of Generative AI Usage

Generative AI (Claude Sonnet 4.5/4.6) was used in the creation of this library as a programming aid including guided code generation, assistance with performance optimization, and documentation. All code and documentation has been reviewed by the author(s) for accuracy.

## Acknowledgments

Bennet Outland thanks the Department of War Science, Math, and Research for Transformation (SMART) Scholarship for academic funding. Outland dedicates his contribution S.D.G.
