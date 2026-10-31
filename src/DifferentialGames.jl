"""
    DifferentialGames

Umbrella package for N-player differential games. It reexports

- `DifferentialGamesBase`: problem types (`LQGameProblem`, `LTVLQGameProblem`, `PDGNEProblem`, …),
  costs, constraints, dynamics, solutions (`GNEPSolution`) and the solver interface
  (`GameSolver`, `solve`, `solver_capabilities`);
- `DifferentialGamesBaseSolvers`: `FNELQ`, `SFNELQ`, `iLQGames`, `ALGAMES`, `LIBR`, `YiPavel`,
  `InverseLQGames` and `RecedingHorizonNash`.

```julia
using DifferentialGames
sol = solve(game, FNELQ())
```
"""
module DifferentialGames

using Reexport

@reexport using DifferentialGamesBase
@reexport using DifferentialGamesBaseSolvers

end