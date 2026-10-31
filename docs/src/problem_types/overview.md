# Problem Types

Every game in the ecosystem is an `AbstractGameProblem{T}`. The concrete types fall into a small number of families; solvers declare which family they accept in `_solve` and `solver_capabilities`.

## Type hierarchy

```
AbstractGameProblem{T}
├── AbstractDeterministicGame{T}
│   ├── GameProblem{T}                         LQ · LTV · nonlinear · PD-GNEP · constrained
│   ├── RecedingHorizonNashProblem{T}          a GameProblem + initial state + number of steps
│   ├── AbstractConvexGame{T}
│   │   └── ConvexGameProblem{T}
│   ├── AbstractPotentialGame{T}
│   │   └── AbstractStatePotentialGame{T}
│   │       └── StatePotentialGameProblem{T,S}
│   └── AbstractOrdinalPotentialGame{T}
│       ├── OrdinalPotentialGameProblem{T,S}
│       └── AbstractLexicographicGame{T}
│           └── LexicographicGameProblem{T}
├── AbstractStochasticGame{T}
│   ├── StochasticGameProblem{T}               noise on top of a nominal game
│   └── RobustGameProblem{T}                   bounded disturbance on top of a nominal game
├── AbstractPartiallyObservableGame{T}
│   └── POGameProblem{T}
└── AbstractInverseGameProblem{T}
    ├── InverseGameProblem{T}
    └── InverseLQGame{T}
```

The extended problems (`Stochastic`, `Robust`, `PO`, `Inverse`) **wrap** a game through a named field rather than extending it; this keeps the solvers for the base game decoupled from the extensions.

## Catalog

| Type | Describes | Page | Solver(s) in this ecosystem |
|------|-----------|------|------------------------------|
| `GameProblem` via `LQGameProblem` | linear dynamics on a shared state, quadratic costs | [LQ](lq_game.md) | [FNELQ](../solvers/fnelq.md), iLQGames, ALGAMES |
| `GameProblem` via `LTVLQGameProblem` | the same, time varying | [LTV LQ](ltv_lq_game.md) | FNELQ |
| `GameProblem` via `PDGNEProblem` | separable dynamics, costs and constraints coupling players | [PD-GNEP](pdgnep.md) | [iLQGames](../solvers/ilqgames.md), [ALGAMES](../solvers/algames.md), [DynamicTubeMPC](../solvers/dtmpc.md) (wrapper) |
| `LexicographicGameProblem` | collision cost first, personal cost second | [Lexicographic](lexicographic.md) | [LIBR](../solvers/libr.md) |
| `ConvexGameProblem` | convex costs and constraint sets (Rosen) | [Convex](convex.md) | [YiPavel](../solvers/yipavel.md) |
| `StatePotentialGameProblem`, `OrdinalPotentialGameProblem` | finite state-based potential games | [Potential](potential.md) | none yet (problem definition only) |
| `StochasticGameProblem`, `RobustGameProblem`, `POGameProblem` | uncertainty wrappers | [Uncertainty](uncertainty.md) | [SFNELQ](../solvers/sfnelq.md) (additive noise); `DynamicTubeMPC` uses disturbance bounds |
| `InverseGameProblem`, `InverseLQGame` | recover cost parameters from observed play | [Inverse](inverse_game.md) | [InverseLQGames](../solvers/invlq.md) |
| `RecedingHorizonNashProblem` | closed-loop simulation of repeated finite-horizon solves | [RHN](../solvers/rhn.md) | [RecedingHorizonNash](../solvers/rhn.md) |

`required_capabilities(game)` reports the capability symbols a `GameProblem` needs; the [solver selection guide](../basics/solver_selection.md) lists what each solver declares.

## Choosing a formulation

1. Linear dynamics on a shared state, quadratic costs, no constraints → `LQGameProblem` (exact solution available).
2. Per-agent dynamics, costs that couple agents, optional constraints → `PDGNEProblem`.
3. Priorities between objectives (safety before performance) → `LexicographicGameProblem`.
4. Convex costs and a shared affine constraint, variational equilibrium wanted → `ConvexGameProblem`.
5. Noise, bounded disturbances or limited observation → wrap the nominal game; see [Uncertainty](uncertainty.md).
6. Observed trajectories and unknown costs → [`InverseGameProblem`](inverse_game.md).
