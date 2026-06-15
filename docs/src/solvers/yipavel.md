# YiPavel

**Distributed Primal-Dual GNE Solver** based on Yi & Pavel (2017). Computes the variational Generalized Nash Equilibrium (v-GNE) for convex games with shared coupling constraints.

## Overview

This solver implements Algorithm 1 from:

> Yi, P. & Pavel, L. (2017). *A distributed primal-dual algorithm for computation of generalized Nash equilibria via operator splitting.* IEEE CDC 2017, pp. 3841–3846.

YiPavel targets `ConvexGameProblem{T}`: an N-player convex game with:
- Separable dynamics
- Convex private feasible sets Ωᵢ
- A shared affine coupling constraint Ax ≤ b (or Ax ≥ b)

## Algorithm

Each iteration simultaneously updates all players' primal decisions, auxiliary consensus variables (z), and local copies of the coupling multiplier (λ):

### Primal Update (simultaneous):
```math
x_{i,k+1} = P_{Ω_i}[ x_{i,k} - τ_i(∇_{x_i}f_i(x_k) - A_i^T λ_{i,k}) ]
```

### Auxiliary Update (consensus on λ):
```math
z_{i,k+1} = z_{i,k} + ν_i ∑_{j≠i}(λ_{i,k} - λ_{j,k})
```

### Dual Update (projected gradient ascent):
```math
λ_{i,k+1} = P_{ℝ_+^m}[ λ_{i,k} - σ_i(
    A_i(2x_{i,k+1} - x_{i,k}) - b_i
    + ∑_{j≠i}[2(z_{i,k+1}-z_{j,k+1}) - (z_{i,k}-z_{j,k})]
    + ∑_{j≠i}(λ_{i,k} - λ_{j,k}) ) ]
```

## Usage

```julia
sol = solve(game, YiPavel())

# With options:
sol = solve(game, YiPavel(;
    max_iter = 5000,
    tol      = 1e-6,
    τ        = 0.05,
    ν        = 0.02,
    σ        = 0.05,
    verbose  = false
))
```

## Solver Type Documentation

```@docs
YiPavel
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `max_iter` | `5000` | Maximum number of iterations |
| `tol` | `1e-6` | Convergence tolerance: ‖Δx‖_∞ < tol |
| `τ` | `0.05` | Primal step size (same for all players) |
| `ν` | `0.02` | Auxiliary (z) step size |
| `σ` | `0.05` | Dual step size |

## Capabilities

```@docs
solver_capabilities(::Type{YiPavel})
```

## Problem Type

YiPavel solves `ConvexGameProblem{T}` with:

- **Separable dynamics**: Each player i controls uᵢ ∈ Ωᵢ ⊂ ℝ^{mᵢ}
- **Convex private constraints**: Ωᵢ are convex sets
- **Shared affine coupling constraint**: Ax ≤ b (or equivalently Ax ≥ b)
- **Coupled costs**: fᵢ(u₁, …, uₙ) may couple across players

## Input Requirements

When calling `solve`, you must provide:

```julia
sol = solve(game, YiPavel();
    cost_fns    = [f1, f2, ...],    # f_i(seg_1,...,seg_N) -> Real
    coupling_A  = A,                # m × Σdⱼ matrix
    coupling_b  = b,                # m-vector
    coupling_leq = true,            # true = Ax ≤ b (default), false = Ax ≥ b
    lb          = [lb_1, ...],      # per-player lower bounds
    ub          = [ub_1, ...])      # per-player upper bounds
```

### Arguments:
- `cost_fns`: Vector of cost functions, one per player. Each function accepts the full joint control vector segmented by player
- `coupling_A`: Coupling constraint matrix (m constraints × total control dimension)
- `coupling_b`: Coupling constraint right-hand side
- `coupling_leq`: If `true`, constraint is Ax ≤ b; if `false`, constraint is Ax ≥ b (converted internally to standard form)
- `lb`, `ub`: Per-player box constraints (can be `nothing` for unbounded players)

## Convergence

Convergence to the **variational GNE (v-NE)** is guaranteed under Assumptions 1–3 of Yi & Pavel (2017) when step-sizes satisfy Theorem 2 (Lemma 3).

Convergence is declared when:
```math
\max_i \|x_{i,k+1} - x_{i,k}\|_\infty < \text{tol}
```

## Algorithm Details

### Communication Graph

The algorithm assumes a **fully-connected communication graph**. The auxiliary variables z and dual variables λ enable consensus across all players.

### Coupling Constraint Form

Internally, the coupling constraint is stored as Ax ≥ b (standard form for the algorithm). If the user provides Ax ≤ b (the natural form for capacity constraints), it is automatically converted via sign flip.

### Projections

- **Primal**: Projected onto the private feasible set Ωᵢ
- **Dual**: Projected onto ℝ_+^m (non-negative orthant) for inequality constraints

## Applications

- **Resource allocation**: Multiple agents sharing limited resources
- **Power systems**: Generators with shared capacity constraints
- **Traffic management**: Vehicles with shared road capacity
- **Market equilibrium**: Producers with shared production constraints

## Notes

- The algorithm requires that cost functions are convex and differentiable
- Convergence rate depends on step-size selection (τ, ν, σ)
- The auxiliary variable z accumulates the Laplacian of λ to enforce consensus
- For better performance, step sizes should be tuned based on problem scale and conditioning

## Step-Size Tuning

The default step sizes (τ=0.05, ν=0.02, σ=0.05) work well for many problems, but may need adjustment:

- **Small τ**: Slower convergence but more stable
- **Large τ**: Faster convergence but may oscillate or diverge
- **ν ≈ τ/2.5**: Good rule of thumb for the auxiliary step size
- **σ ≈ τ**: Good starting point for the dual step size

## References

Yi, P. & Pavel, L. (2017). *A distributed primal-dual algorithm for computation of generalized Nash equilibria via operator splitting.* IEEE CDC 2017, pp. 3841–3846.

See also the implementation notes in [DifferentialGamesBaseSolvers.jl](https://github.com/JuliaDifferentialGames/DifferentialGamesBaseSolvers.jl).
