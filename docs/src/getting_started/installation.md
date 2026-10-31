# Installation

## Requirements

- Julia **1.11** or later ([download](https://julialang.org/downloads/)). `DifferentialGamesBaseSolvers.jl` declares `julia = "1.11"` in its `[compat]`; the umbrella package's own `julia = "1.10"` bound is not sufficient on its own.
- No non-Julia dependencies — all numerical backends are pure Julia.

## Installing DifferentialGames.jl

None of the three packages is registered in the General registry yet, so they must be added **together, by URL, in a single `Pkg.add` call** so the resolver can see all three:

```julia
using Pkg
Pkg.add([
    PackageSpec(url="https://github.com/JuliaDifferentialGames/DifferentialGamesBase.jl"),
    PackageSpec(url="https://github.com/JuliaDifferentialGames/DifferentialGamesBaseSolvers.jl"),
    PackageSpec(url="https://github.com/JuliaDifferentialGames/DifferentialGames.jl"),
])
```

!!! warning "Adding only the umbrella package fails"
    `Pkg.add(url=".../DifferentialGames.jl")` on its own currently fails with
    `Unsatisfiable requirements detected for package DifferentialGamesBaseSolvers`,
    because its two dependencies are not in a registry. (Verified in a clean depot.)
    This page will be simplified once the packages are registered.

## Verifying the Installation

```@example install
using DifferentialGames, LinearAlgebra

n = 2
game = LQGameProblem(
    Matrix(1.0I(n)), [Matrix(1.0I(n))[:, 1:1], Matrix(1.0I(n))[:, 2:2]],
    [Matrix(1.0I(n)), Matrix(1.0I(n))], [fill(0.1, 1, 1), fill(0.1, 1, 1)],
    [Matrix(1.0I(n)), Matrix(1.0I(n))], ones(n), 1.0; dt=0.1
)
sol = solve(game, FNELQ())
@assert sol.converged
println("Installation OK — J₁ = ", get_cost(sol, 1))
```

## Optional: Visualization

`animate_solution` and `save_animation` live in a package extension that activates when `Plots.jl` is loaded alongside `DifferentialGamesBaseSolvers`:

```julia
using Pkg
Pkg.add("Plots")
```

```julia
using DifferentialGames, Plots

anim = animate_solution(sol)     # `sol` is a GNEPSolution
gif(anim, "game.gif"; fps=20)
```

`animate_solution(sol; fps, subsample, xy_indices)` works for any `GNEPSolution`: for `PDGNEProblem` solutions the first two components of each player's state block are plotted; for shared-state games pass `xy_indices=[(x_idx, y_idx), …]`, one tuple per player. `save_animation(anim, "game.gif")` writes the file. `plot(sol)` gives static trajectory and convergence panels.

## Installing Sub-packages Separately

If you are building a solver package, depend only on the interface package:

```julia
Pkg.add(url="https://github.com/JuliaDifferentialGames/DifferentialGamesBase.jl")
```

To use the solvers without the umbrella package, add `DifferentialGamesBase.jl` first and then `DifferentialGamesBaseSolvers.jl`.

## Development Installation

Clone the three repositories side by side and `develop` them together:

```bash
git clone https://github.com/JuliaDifferentialGames/DifferentialGamesBase.jl
git clone https://github.com/JuliaDifferentialGames/DifferentialGamesBaseSolvers.jl
git clone https://github.com/JuliaDifferentialGames/DifferentialGames.jl
```

```julia
using Pkg
Pkg.develop([
    PackageSpec(path="DifferentialGamesBase.jl"),
    PackageSpec(path="DifferentialGamesBaseSolvers.jl"),
    PackageSpec(path="DifferentialGames.jl"),
])
```
