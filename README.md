# DifferentialGames

<!--
[![Build Status](https://github.com/BennetOutland/DifferentialGamesBase.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/BennetOutland/DifferentialGamesBase.jl/actions/workflows/CI.yml?query=branch%3Amain)
-->

[![CI](https://github.com/JuliaDifferentialGames/DifferentialGamesBase/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaDifferentialGames/DifferentialGamesBase/actions/workflows/CI.yml)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

A Julia package for solving N-player differential games using numerical methods, following the SciML interface standards.
Purpose

## Purpose
DifferentialGames.jl provides fast, flexible implementations of algorithms for computing Nash equilibria in differential games. The package supports both cooperative and non-cooperative games with applications to multi-agent control, robotics, and autonomous systems.

> ⚠️ **Work in progress**
>
> This repository is actively being developed and the API, files, and examples may change without notice. Use at your own risk.

## Installation

```julia
using Pkg
Pkg.add("https://github.com/JuliaDifferentialGames/DifferentialGames.jl.git")
```


## Quick Start
```julia
using DifferentialGames

# Define a 2-player linear-quadratic game
n, m = 4, 2  # state and control dimensions
A = randn(n, n)
B = [randn(n, m) for _ in 1:2]
Q = [diagm(ones(n)) for _ in 1:2]
R = [diagm(0.1*ones(m)) for _ in 1:2]
Qf = [diagm(10.0*ones(n)) for _ in 1:2]

game = LQGameProblem(A, B, Q, R, Qf, zeros(n), 10.0; dt=0.01)

# Solve for feedback Nash equilibrium
solver = FNELQ()
sol = solve(game, solver; verbose=true)

# Access solution
K_gains = sol.solver_info[:feedback_gains]  # Time-varying feedback gains
costs = sol.solver_info[:costs]              # Per-player costs
```

## Development Status
### Currently Implemented:

- ✅ Game problem specification (GameProblem, LQGameProblem)
- ✅ Solution interface (GameSolution, Trajectory)
- ✅ Discrete-time feedback Nash equilibrium (FNELQ)
- ✅ Solver capability system
- ✅ Warmstart support

### In Progress:

- 🚧 Open-loop Nash equilibrium solvers
- 🚧 Iterative LQ games (iLQGames)
- 🚧 Sampling-based methods (JAGUAR, see [DynamicPlanning.jl](https://github.com/JuliaDifferentialGames/DynamicPlanning.jl))
- 🚧 Callback system for monitoring
- 🚧 Inverse game theory algorithms

### Planned:

- 📋 Stackelberg games (leader-follower)
- 📋 Stochastic differential games
- 📋 Mean field games
- 📋 Learning-based solvers (MADDPG, MAPPO)
- 📋 Visualization tools
- 📋 Benchmark suite

## Architecture

The package is organized into modular components:
```
DifferentialGames.jl/
├── DifferentialGamesBase.jl      # Core problem types and interfaces
├── DifferentialGamesSolvers.jl    # Numerical solvers
```
This modular structure allows users to depend only on the components they need.


## Contributing

Contributions are welcome! The package is structured to make adding new algorithms straightforward:

- [ ] Define your solver struct inheriting from ```GameSolver```
- [ ] Implement ```solver_capabilities(::Type{YourSolver})``` to declare supported problem types
- [ ] Implement ```_solve(game::GameProblem, solver::YourSolver, warmstart, verbose)```
- [ ] Add tests comparing against known solutions or published benchmarks
- [ ] Submit a pull request

See [FNELQ](https://github.com/JuliaDifferentialGames/DifferentialGamesBaseSolvers.jl/tree/main/src/solvers/FNELQ) as a reference implementation.

## License
MIT License - see LICENSE file for details.

## Acknowledgments
This package follows the design principles of the SciML ecosystem and draws inspiration from:

DifferentialEquations.jl for the solve interface
Optimization.jl for algorithm dispatch patterns
TrajectoryOptimization.jl for trajectory representations
iLQGames.jl for interface inspiration and for algorithm implementation 

## Disclosure of Generative AI Usage

Generative AI, Claude Sonnet 4.5, was used in the creation of this library as a programming aid including guided code generation, assistance with performance optimization, and for assistance in writing documentation. All code and documentation included in this repository, whether written by the author(s) or generative AI, has been reviewed by the author(s) for accuracy and has completed a verification process upon release.
