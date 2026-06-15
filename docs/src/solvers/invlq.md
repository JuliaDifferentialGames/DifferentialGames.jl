# InverseLQGames

**Inverse Linear-Quadratic Differential Game Solver**. Computes the canonical parameter set characterising all cost-function matrices consistent with an observed Nash equilibrium.

## Overview

This solver implements the algebraic solution-set algorithm from:

> Inga, J., Bischoff, E., Molloy, T.L., Flad, M., Hohmann, S. (2019). *Solution sets for inverse non-cooperative linear-quadratic differential games.* IEEE Control Systems Letters, 3(4), 871–876. [DOI:10.1109/LCSYS.2019.2919271](https://doi.org/10.1109/LCSYS.2019.2919271)

Given an observed Nash equilibrium strategy (feedback gains K*), InverseLQGames finds **all** possible cost matrices (Qᵢ, Rᵢⱼ) that would make K* a valid feedback Nash equilibrium for the infinite-horizon LQ differential game.

## Problem Statement

For a system:
```math
ẋ(t) = A x(t) + \sum_{i=1}^N B_i u_i(t)
```

With feedback strategies:
```math
u_i(t) = -K_i x(t)
```

And costs:
```math
J_i = \frac{1}{2} \int_0^\infty (x^T Q_i x + \sum_j u_j^T R_{ij} u_j) dt
```

Given K* = (K₁*, …, Kₙ*), the solver finds all (Qᵢ, Rᵢ₁, …, RᵢN) consistent with the Nash equilibrium.

## Algorithm

The algorithm computes the canonical parameter set Θ = ∩ᵢ ker(Mᵢ) with Rᵢᵢ ≻ 0:

1. **Compute closed-loop matrix**: F = A − Σᵢ BᵢKᵢ*
2. **Form Kronecker sum**: F⊕ = Fᵀ ⊗ Iₙ + Iₙ ⊗ Fᵀ
3. **For each player i**:
   - Compute Sᵢ = (Iₙ ⊗ Bᵢᵀ) F⊕⁻¹
   - Build Mᵢ using Kronecker products of feedback gains
   - Compute ker(Mᵢ) via SVD
4. **Result**: The intersection of all ker(Mᵢ) gives all consistent cost parameters

## Usage

```julia
# With exact K* (known feedback gains)
sol = solve(inverse_problem, InverseLQGames())

# With observed trajectories (estimate K* via least squares)
sol = solve(inverse_problem, InverseLQGames(;
    tol       = 1e-6,
    K_ridge   = 0.0,
    svd_tol   = 1e-8,
    verbose   = false
))
```

## Solver Type Documentation

```@docs
InverseLQGames
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `tol` | `1e-6` | Residual threshold; converged when all ‖Mᵢθᵢ‖ ≤ tol |
| `K_ridge` | `0.0` | Tikhonov regularization for K estimation from trajectories |
| `svd_tol` | `1e-8` | Relative singular-value threshold for null-space computation |

## Capabilities

```@docs
solver_capabilities(::Type{InverseLQGames})
```

## Input Modes

### Mode 1: Exact K* Known

If the Nash equilibrium feedback gains K* are known exactly, provide them in the `InverseLQGame` problem:

```julia
prob = InverseLQGame(A, B, K_star)
sol = solve(prob, InverseLQGames())
```

### Mode 2: Trajectory Data

If only trajectory data (X, U) is available, the solver estimates K* via regularized least squares:

```julia
prob = InverseLQGame(A, B, state_trajectories, control_trajectories)
sol = solve(prob, InverseLQGames(K_ridge=1e-3))  # Use ridge for noisy data
```

The estimation solves:
```math
K̂_i = \arg\min \sum_k \|K_i x^{[k]} + u_i^{[k]}\|_2^2
```

Which has the closed-form solution:
```math
K̂_i = -U_i X^T (X X^T + λI)^{-1}
```

## Output

The solution contains:

- Per-player constraint matrices Mᵢ
- Null-space bases for each ker(Mᵢ)
- Minimum-norm representatives θᵢ
- Residuals for each player
- Convergence status

## Requirements

- The closed-loop matrix F must be **Hurwitz** (all eigenvalues have negative real parts) for F⊕ to be invertible (Lemma 1 of Inga et al. 2019)
- The null space must be non-empty for each player (otherwise no consistent cost parameters exist)

## Applications

- **Inverse game theory**: Inferring player objectives from observed behavior
- **Strategy verification**: Checking if observed behavior is consistent with LQ Nash equilibrium
- **Cost function design**: Understanding what costs would produce observed strategies

## Notes

- When K* is not known exactly, estimation quality depends on trajectory quality and regularization
- The Hurwitz condition is necessary for the Kronecker sum to be invertible
- The solver warns if F is not Hurwitz or if any ker(Mᵢ) is empty

## References

Inga, J., Bischoff, E., Molloy, T.L., Flad, M., Hohmann, S. (2019). *Solution sets for inverse non-cooperative linear-quadratic differential games.* IEEE Control Systems Letters, 3(4), 871–876.

See also the implementation notes in [DifferentialGamesBaseSolvers.jl](https://github.com/JuliaDifferentialGames/DifferentialGamesBaseSolvers.jl).
