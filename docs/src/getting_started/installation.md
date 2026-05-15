# Installation

## Requirements

- Julia 1.10 or later ([download](https://julialang.org/downloads/))
- No non-Julia dependencies — all numerical backends are pure Julia

## Installing DifferentialGames.jl

DifferentialGames.jl is not yet registered in the General registry. Install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaDifferentialGames/DifferentialGames.jl")
```

This installs the full ecosystem: `DifferentialGamesBase.jl` (problem types and interfaces) and `DifferentialGamesBaseSolvers.jl` (FNELQ, iLQGames, ALGAMES) are pulled in automatically.

## Verifying the Installation

```julia
using DifferentialGames, LinearAlgebra

n = 2
game = LQGameProblem(
    I(n), [I(n)[:, 1:1], I(n)[:, 2:2]],
    [I(n), I(n)], [fill(0.1,1,1), fill(0.1,1,1)],
    [I(n), I(n)], ones(n), 1.0; dt=0.1
)
sol = solve(game, FNELQ())
@assert sol.converged
println("Installation OK — J₁ = ", get_cost(sol, 1))
```

## Optional: Visualization

The animation extension is loaded automatically when `Plots.jl` is available:

```julia
using Pkg
Pkg.add("Plots")
```

```julia
using DifferentialGames, Plots

sol  = solve(game, iLQGames())
anim = animate_solution(sol)
gif(anim, "game.gif"; fps=20)
```

## Installing Sub-packages Separately

If you are building a solver package, depend only on the interface:

```julia
Pkg.add(url="https://github.com/JuliaDifferentialGames/DifferentialGamesBase.jl")
```

If you need only the existing solvers (not problem construction utilities):

```julia
Pkg.add(url="https://github.com/JuliaDifferentialGames/DifferentialGamesBaseSolvers.jl")
```

## Development Installation

To contribute or run the latest code:

```julia
using Pkg
Pkg.develop(url="https://github.com/JuliaDifferentialGames/DifferentialGames.jl")
```

Or clone and `dev` locally:

```bash
git clone https://github.com/JuliaDifferentialGames/DifferentialGames.jl
```

```julia
Pkg.develop(path="path/to/DifferentialGames.jl")
```
