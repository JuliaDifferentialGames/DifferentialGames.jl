# DifferentialGames.jl

<p align="center"><img src="docs/src/assets/logo.png" alt="DifferentialGames.jl logo" width="160"></p>

[![CI](https://github.com/JuliaDifferentialGames/DifferentialGames.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaDifferentialGames/DifferentialGames.jl/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaDifferentialGames.github.io/DifferentialGames.jl/dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20432309.svg)](https://doi.org/10.5281/zenodo.20432309)

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

# Two-player LQ game — feedback Nash equilibrium via FNELQ.
# `Matrix(...)` is optional with current DifferentialGamesBase (I(n), Diagonal are converted)
# but required by older versions.
n = 4
game = LQGameProblem(
    Matrix(0.95 * I(n)),
    [Matrix(1.0I(n))[:, 1:1], Matrix(1.0I(n))[:, 3:3]],
    [Matrix(1.0I(n)), Matrix(1.0I(n))],
    [fill(0.1, 1, 1), fill(0.1, 1, 1)],
    [Matrix(10.0I(n)), Matrix(10.0I(n))],
    ones(n), 2.0; dt=0.1
)

sol = solve(game, FNELQ())
println("Player 1 cost: ", get_cost(sol, 1))
println("Player 2 cost: ", get_cost(sol, 2))
```

```julia
# Nonlinear game without constraints — iLQGames (local feedback Nash)
using DifferentialGames

dyn = (xi, ui, p, t) -> [xi[3]*cos(xi[4]); xi[3]*sin(xi[4]); ui[1]; ui[2]]
stage    = DiagonalLQStageCost([0.1, 0.1, 1.0, 0.1], [0.1, 0.1])
terminal = DiagonalLQTerminalCost([10.0, 10.0, 1.0, 0.1])

player1 = PlayerSpec(1, 4, 2, [-3.0, 0.0, 1.0, 0.0],  dyn, PlayerObjective(1, stage, terminal))
player2 = PlayerSpec(2, 4, 2, [ 0.0, -3.0, 1.0, π/2], dyn, PlayerObjective(2, stage, terminal))

game = PDGNEProblem([player1, player2], 4.0, 0.1)
sol  = solve(game, iLQGames())
```

```julia
# Game with a shared collision constraint — ALGAMES.
# iLQGames does NOT support constraints; use ALGAMES for them.
using DifferentialGames

dyn = (xi, ui, p, t) -> ui                         # single integrator
mkplayer(i, x0, goal) = PlayerSpec(i, 2, 2, x0, dyn,
    PlayerObjective(i,
        # ALGAMES passes the JOINT state and control to cost functions; player i's blocks start at 2(i-1)
        NonlinearStageCost((x, u, p, t) -> 0.01 * sum(abs2, u[2(i-1)+1:2i]); is_separable=true),
        NonlinearTerminalCost((x, p) -> 10.0 * ((x[2(i-1)+1] - goal[1])^2 +
                                                (x[2(i-1)+2] - goal[2])^2))),
    [control_bounds(i; control_offset=2(i-1), control_dim=2,
                    lower=[-2.0, -2.0], upper=[2.0, 2.0])])

col = SharedInequality([1, 2];
    func = (x, u, p, t) -> [1.5^2 - sum(abs2, x[1:2] - x[3:4])],   # ‖p₁ - p₂‖ ≥ 1.5
    dim  = 1)

game = PDGNEProblem([mkplayer(1, [0.0, 2.0],  [5.0, -2.0]),
                     mkplayer(2, [0.0, -1.5], [5.0, 2.0])], [col], 4.0, 0.1)
sol  = solve(game, ALGAMES(; tol_opt=1e-2))
```

## License

MIT License — see LICENSE file for details.

## Disclosure of Generative AI Usage

Generative AI was used in the creation of this library as a programming aid including guided code generation, assistance with performance optimization, and documentation. All code and documentation has been reviewed by the author(s) for accuracy.

## Acknowledgments

Bennet Outland thanks the Department of War Science, Math, and Research for Transformation (SMART) Scholarship for academic funding. Outland dedicates his contribution S.D.G.

## Citation

If you used DifferentialGames.jl in your work, please cite using
```
@misc{outland_2026,
  author       = {Outland, Bennet},
  title        = {DifferentialGames.jl},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20432309},
  url          = {https://doi.org/10.5281/zenodo.20432309},
}
```
