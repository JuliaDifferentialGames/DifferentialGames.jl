# Problem Types API

Quick-reference index for all problem constructors and their options.

## Shared-State LQ Games

| Constructor | Description |
|-------------|-------------|
| [`LQGameProblem`](../problem_types/lq_game.md) | LTI linear-quadratic game |
| [`LTVLQGameProblem`](../problem_types/ltv_lq_game.md) | LTV linear-quadratic game |

## Partially-Decoupled Games

| Constructor | Description |
|-------------|-------------|
| [`PDGNEProblem`](../problem_types/pdgnep.md) | Separable-dynamics game |
| [`PlayerSpec`](../problem_types/pdgnep.md#Building-a-Player) | Per-player specification |

## Inverse Games

| Constructor | Description |
|-------------|-------------|
| [`InverseGameProblem`](../problem_types/inverse_game.md) | Base inverse game type |
| [`InversePDGNEProblem`](../problem_types/inverse_game.md#InversePDGNEProblem-Constructor) | PD-GNEP inverse problem |

## Utilities

| Function | Description |
|----------|-------------|
| [`n_players(game)`](../problem_types/pdgnep.md) | Number of players |
| [`n_steps(game)`](../problem_types/lq_game.md) | Number of time steps |
| [`state_dim(game)`](../problem_types/pdgnep.md#Joint-State-Layout) | Joint state dimension |
| [`state_dim(game, i)`](../problem_types/pdgnep.md#Joint-State-Layout) | Player i's state dimension |
