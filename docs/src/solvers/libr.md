# LIBR

**Lexicographic Iterative Best Response** solver for games with lexicographic preferences.

## Overview

This solver implements Algorithm 1 from:

> Miller, K. & Mitra, S. (2022). *Multi-agent motion planning using differential games with lexicographic preferences.* IEEE CDC 2022, pp. 5751–5757.

LIBR computes a **Lexicographic Nash Equilibrium (L-NE)** for games where each player has a lexicographic cost structure: Jᵢ(z) = (Jᵢᶜᵒˡ, Jᵢᵖᵉʳ) ∈ ℝ² ordered lexicographically.

## Algorithm

Each outer IBR (Iterative Best Response) iteration updates every agent sequentially with two gradient-descent phases:

1. **Phase 1**: Minimize collision cost Jᵢᶜᵒˡ with all others' strategies fixed; record J*_col
2. **Phase 2**: Minimize personal cost Jᵢᵖᵉʳ subject to Jᵢᶜᵒˡ ≤ J*_col + slack using a quadratic penalty

The algorithm converges to an L-NE, guaranteed to exist as a pure strategy (Proposition 2 of Miller & Mitra 2022).

## Usage

```julia
sol = solve(game, LIBR())

# With options:
sol = solve(game, LIBR(;
    max_iter     = 100,
    tol          = 1e-4,
    inner_iter   = 200,
    step_size    = 0.01,
    ls_beta      = 0.5,
    ls_max_iter  = 15,
    ls_armijo    = 1e-4,
    col_penalty  = 1e3,
    col_slack    = 1e-6,
    verbose      = false
))
```

## Solver Type Documentation

```@docs
LIBR
```

## Options

### IBR Loop
| Option | Default | Description |
|--------|---------|-------------|
| `max_iter` | `100` | Maximum outer IBR iterations |
| `tol` | `1e-4` | Convergence: max change in any player's control |

### Per-Phase Gradient Descent
| Option | Default | Description |
|--------|---------|-------------|
| `inner_iter` | `200` | Gradient steps per phase per player per IBR iteration |
| `step_size` | `0.01` | Initial step size α₀ for backtracking line search |
| `ls_beta` | `0.5` | Step contraction factor for line search |
| `ls_max_iter` | `15` | Maximum backtracking steps |
| `ls_armijo` | `1e-4` | Armijo sufficient-decrease constant |

### Phase 2 Penalty
| Option | Default | Description |
|--------|---------|-------------|
| `col_penalty` | `1e3` | Quadratic penalty weight ρ for Jᵢᶜᵒˡ > J*_col |
| `col_slack` | `1e-6` | Absolute slack: Phase 2 target is Jᵢᶜᵒˡ ≤ J*_col + δ |

## Capabilities

```@docs
solver_capabilities(::Type{LIBR})
```

## Problem Type

LIBR operates on `LexicographicGameProblem{T}`, which wraps a `PDGNEProblem` (with `SeparableDynamics`) and adds per-player lexicographic cost structure.

Each player i has a lexicographic cost:
```math
J_i(z) = (J_i^{col}, J_i^{per}) \in \mathbb{R}^2
```

ordered such that (a, b) < (c, d) if a < c or (a == c and b < d).

## Trajectory Representation

The solver uses a private state-control trajectory representation for each player:
- `X_i` ∈ ℝ^{nᵢ × (N+1)} — private state trajectory (from Euler rollout)
- `U_i` ∈ ℝ^{mᵢ × N} — private control trajectory (optimization variable)

User-supplied cost functions must accept this representation: `z_i = (X_i, U_i)`.

## Integration

The dynamics are integrated using Euler's method:
```math
x_{k+1} = x_k + dt \cdot f_i(x_k, u_k, nothing, (k-1)\cdot dt)
```

The implementation is ForwardDiff-compatible, allowing dual numbers to propagate through rollout and costs.

## Convergence

Convergence is declared when:
```math
\max_i \|U_i^{(l+1)} - U_i^{(l)}\|_\infty < \text{tol}
```

## Applications

- **Multi-agent motion planning**: Where collision avoidance (Jᶜᵒˡ) has higher priority than reaching a goal (Jᵖᵉʳ)
- **Safety-critical systems**: Where safety constraints must be satisfied before optimizing performance
- **Hierarchical objectives**: Any scenario where players have prioritized objectives

## Notes

- The lexicographic structure ensures that collision avoidance (or other critical objectives) are prioritized over personal costs
- The algorithm requires that cost functions are differentiable
- The Phase 2 constraint is enforced via a quadratic penalty, not hard constraints
- Convergence to L-NE is guaranteed under the conditions of Miller & Mitra (2022)

## References

Miller, K. & Mitra, S. (2022). *Multi-agent motion planning using differential games with lexicographic preferences.* IEEE CDC 2022, pp. 5751–5757.

See also the implementation notes in [DifferentialGamesBaseSolvers.jl](https://github.com/JuliaDifferentialGames/DifferentialGamesBaseSolvers.jl).
