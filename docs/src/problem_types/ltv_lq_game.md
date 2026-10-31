# LTVLQGameProblem

A **linear time-varying** (LTV) LQ game where the dynamics matrices `A`, `B` and the cost matrices `Q`, `R` can vary at each time step. Useful for games linearized around a reference trajectory, or systems with periodic forcing.

## Constructor

```@docs
LTVLQGameProblem
```

## Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `A_seq` | `Vector{Matrix{T}}` | State matrices at each step; `length(A_seq) == round(tf/dt)` |
| `B_seq` | `Vector{Vector{Matrix{T}}}` | `B_seq[i][k]` is player i's input matrix at step k |
| `Q_seq` | `Vector{Vector{Matrix{T}}}` | `Q_seq[i][k]` is player i's state cost at step k |
| `R_seq` | `Vector{Vector{Matrix{T}}}` | `R_seq[i][k]` is player i's control cost at step k |
| `Qf` | `Vector{Matrix{T}}` | Terminal cost matrices (one per player) |
| `x0` | `Vector{T}` | Initial state |
| `tf` | `T` | Final time |
| `dt` | `T` | Time step (keyword) |

## Example

```@example ltv
using DifferentialGames, LinearAlgebra, Random
Random.seed!(1)

N = 20                 # number of time steps (tf / dt)
n = 4

# Time-varying A (e.g., from linearization along a reference); must be Matrix{Float64}
A_seq = [Matrix(0.95I(n)) + 0.01 * randn(n, n) for _ in 1:N]

# Per-player sequences: X_seq[i][k] is player i's matrix at step k
B_seq = [[reshape(Matrix(1.0I(n))[:, 1], n, 1) for _ in 1:N],
         [reshape(Matrix(1.0I(n))[:, 3], n, 1) for _ in 1:N]]
Q_seq = [[Matrix(1.0I(n)) for _ in 1:N] for _ in 1:2]
R_seq = [[fill(0.1, 1, 1) for _ in 1:N] for _ in 1:2]
Qf    = [Matrix(10.0I(n)), Matrix(10.0I(n))]
x0    = ones(n)

game = LTVLQGameProblem(A_seq, B_seq, Q_seq, R_seq, Qf, x0, 2.0; dt=0.1)
sol  = solve(game, FNELQ())
(converged = sol.converged, costs = get_costs(sol))
```

## Notes

The indexing convention is:
- `A_seq[k]` — state matrix at step `k`
- `B_seq[i][k]` — player `i`'s input matrix at step `k`
- `Q_seq[i][k]` — player `i`'s state cost at step `k`
- `R_seq[i][k]` — player `i`'s control cost at step `k`

## Compatible Solvers

- [`FNELQ`](../solvers/fnelq.md) — handles LTV automatically
