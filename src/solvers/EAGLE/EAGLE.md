# **E**fficient **A**lgorithm for **G**ame-theoretic **L**Q **E**quilibria

A dynamic programming solver for finite-horizon linear-quadratic differential games.
Computes feedback Nash equilibrium strategies using backward Riccati recursion with:
- Schur complement method for exploiting block-diagonal structure
- Joseph-form updates for numerical stability
- Efficient factorization to minimize allocations

## Fields
- `dt::Float64` : discretization timestep
- `max_schur_iters::Int` : maximum iterations for Schur complement method (default: 3)
- `schur_tol::Float64` : convergence tolerance for Schur iterations (default: 1e-8)

## Example
```julia
solver = EAGLE(0.1)  # 0.1 second timestep
solution = solve(game, solver)
```

## References

- iLQGames.jl
- Dynamic Noncooperative Game Theory