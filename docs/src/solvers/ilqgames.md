# iLQGames

**Iterative Linear Quadratic Games** solver. Computes a local feedback Nash equilibrium for finite-horizon nonlinear games by repeatedly linearizing the dynamics and solving the resulting LQ subgame.

## Algorithm

iLQGames extends the iterative LQR (iLQR) idea to the multi-player setting. Each outer iteration consists of:

1. **Forward pass** — roll out the current strategy from the initial state to obtain a nominal trajectory ``(\bar{x}_0, \ldots, \bar{x}_N)``.
2. **Linearization** — compute first-order Taylor expansions of each player's dynamics and second-order expansions of each player's cost along the nominal trajectory.
3. **LQ subgame** — solve the resulting time-varying LQ game exactly with FNELQ to obtain gain corrections ``\delta K_i^k``.
4. **Line search** — step from the nominal trajectory toward the corrected trajectory, backtracking until a sufficient decrease in the total Nash residual is found.

Convergence is declared when the maximum change in any element of the trajectory falls below ``\varepsilon_\text{conv}``.

## Usage

```julia
sol = solve(game, iLQGames())

# With options:
sol = solve(game, iLQGames(;
    max_iter  = 100,
    ε_conv    = 1e-6,
    α_min     = 1e-4,
    verbose   = false,
))
```

```@docs
iLQGames
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `max_iter` | `100` | Maximum number of outer iterations |
| `ε_conv` | `1e-6` | Convergence threshold on ``\max_k \|x_k^{(j+1)} - x_k^{(j)}\|_\infty`` |
| `α_min` | `1e-4` | Minimum line-search step size before aborting |
| `verbose` | `false` | Print per-iteration log |

## Applicable Problem Types

| Problem Type | Supported |
|--------------|-----------|
| `LQGameProblem` | Yes (converges in 1 iter) |
| `LTVLQGameProblem` | Yes |
| `PDGNEProblem` | Yes |
| Nonlinear dynamics / costs | Yes |
| Shared / private constraints | No — use `ALGAMES` |

## Convergence

iLQGames finds a **local** feedback Nash equilibrium. There is no guarantee of global optimality, and the solution found may depend on the initial strategy (zero by default). For nonlinear problems it is common to run from several random initializations and take the best.

Convergence criterion:

```math
\max_k \|x_k^{(j+1)} - x_k^{(j)}\|_\infty < \varepsilon_{\text{conv}}
```

## Notes

- Dynamics and stage costs must be differentiable with respect to state and control. The solver uses `ForwardDiff.jl` for automatic differentiation; closures must be ForwardDiff-compatible (avoid type-specific branches or non-differentiable operations).
- For **constrained** games, use [`ALGAMES`](algames.md) instead, optionally warmstarted from an iLQGames solution.
- For **LQ** games, [`FNELQ`](fnelq.md) is faster (direct solver, no iteration).
