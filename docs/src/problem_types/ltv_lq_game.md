# LTVLQGameProblem

A **linear time-varying** (LTV) LQ game where the dynamics matrices `A`, `B` and the cost matrices `Q`, `R` can vary at each time step. Useful for games linearized around a reference trajectory, or systems with periodic forcing.

## Constructor

```@docs
LTVLQGameProblem
```

## Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `A_seq` | `Vector{Matrix{T}}` | State matrices at each step (length N) |
| `B_seq` | `Vector{Vector{Matrix{T}}}` | `B_seq[k][i]` is player i's input matrix at step k |
| `Q_seq` | `Vector{Vector{Matrix{T}}}` | `Q_seq[i][k]` is player i's state cost at step k |
| `R_seq` | `Vector{Vector{Matrix{T}}}` | `R_seq[i][k]` is player i's control cost at step k |
| `Qf` | `Vector{Matrix{T}}` | Terminal cost matrices (one per player) |
| `x0` | `Vector{T}` | Initial state |
| `tf` | `T` | Final time |
| `dt` | `T` | Time step (keyword) |

## Example

```julia
using DifferentialGames, LinearAlgebra

N = 20     # number of time steps
n, m = 4, 1

# Time-varying A (e.g., from linearization along a reference)
A_seq = [0.95 * I(n) + 0.01 * randn(n, n) for _ in 1:N]

# Constant B, Q, R (can be made time-varying analogously)
B_seq = [[reshape(I(n)[:, 1], n, 1)], [reshape(I(n)[:, 3], n, 1)]]
B_seq = [B_seq for _ in 1:N]   # replicate across time — wrong shape, see note

# Proper format:
B_at_k   = [[reshape(I(n)[:, 1], n, 1), reshape(I(n)[:, 3], n, 1)] for _ in 1:N]
Q_player = [diagm(ones(n)) for _ in 1:N]
Q_seq    = [Q_player, Q_player]
R_player = [fill(0.1, 1, 1) for _ in 1:N]
R_seq    = [R_player, R_player]
Qf       = [10.0 * diagm(ones(n)), 10.0 * diagm(ones(n))]
x0       = ones(n)

# Note: B_seq indexing is B_seq[k][i] not B_seq[i][k]
game = LTVLQGameProblem(
    A_seq,
    [[B_at_k[k][1] for k in 1:N], [B_at_k[k][2] for k in 1:N]],
    Q_seq, R_seq, Qf, x0, 2.0; dt=0.1
)
sol = solve(game, FNELQ())
```

## Notes

The indexing convention is:
- `B_seq[i][k]` — player `i`'s input matrix at step `k`
- `Q_seq[i][k]` — player `i`'s state cost at step `k`
- `R_seq[i][k]` — player `i`'s control cost at step `k`

## Compatible Solvers

- [`FNELQ`](../solvers/fnelq.md) — handles LTV automatically
