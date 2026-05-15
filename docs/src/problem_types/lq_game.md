# LQGameProblem

A shared-state, finite-horizon, discrete-time LQ game. All players share a single joint state vector with linear dynamics and quadratic costs.

## Constructor

```@docs
LQGameProblem
```

## Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `A` | `Matrix{T}` | State transition matrix (n×n) |
| `B` | `Vector{Matrix{T}}` | Control input matrices, one per player |
| `Q` | `Vector{Matrix{T}}` | State cost matrices, one per player |
| `R` | `Vector{Matrix{T}}` | Control cost matrices, one per player |
| `Qf` | `Vector{Matrix{T}}` | Terminal state cost matrices |
| `x0` | `Vector{T}` | Initial state |
| `tf` | `T` | Final time |
| `dt` | `T` | Time step (keyword, default 0.01) |

## Example

```julia
using DifferentialGames, LinearAlgebra

n, m = 4, 1   # 4 states, 1 control per player
A  = [1.0 0.1 0.0 0.0;
      0.0 1.0 0.0 0.0;
      0.0 0.0 1.0 0.1;
      0.0 0.0 0.0 1.0]
B  = [reshape([0.0, 0.1, 0.0, 0.0], n, 1),
      reshape([0.0, 0.0, 0.0, 0.1], n, 1)]
Q  = [diagm([1.0, 0.0, 0.0, 0.0]), diagm([0.0, 0.0, 1.0, 0.0])]
R  = [fill(0.1, 1, 1), fill(0.1, 1, 1)]
Qf = [10.0 * Q[1], 10.0 * Q[2]]
x0 = [2.0, 0.0, -2.0, 0.0]

game = LQGameProblem(A, B, Q, R, Qf, x0, 3.0; dt=0.1)
sol  = solve(game, FNELQ())
```

## Notes

- `B[i]` has shape `(n, m_i)` where `m_i` is player `i`'s control dimension
- All matrices must have compatible numeric type `T`; `Float64` is the default
- For time-varying dynamics, see [`LTVLQGameProblem`](ltv_lq_game.md)
- For separable (per-player state) games, see [`PDGNEProblem`](pdgnep.md)

## Compatible Solvers

- [`FNELQ`](../solvers/fnelq.md) — exact; recommended for all LQ games
- [`iLQGames`](../solvers/ilqgames.md) — convergent on LQ problems, but FNELQ is faster
